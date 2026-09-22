"""Delete uploads and rendered artefacts after a fixed period (WP-07).

Until this landed, an accepted upload stayed on disk forever. Rejected ones were
already cleaned up by `_discard`, which made the gap easy to miss: the failure
path was tidy and the success path was not.

There are exactly two places a job's files live, and that is worth stating
because a sweep that misses one is a promise that is quietly false:

    UPLOAD_ROOT/<job_id>/   input.<ext>, voice.<ext>
    OUTPUT_ROOT/<job_id>/   angles.json, annotated.mp4, worst.jpg, best.jpg,
                            coaching.json

There is no third. The encoder pipes ffmpeg's stderr into `tempfile.TemporaryFile`
(encoder.py), which is anonymous - unlinked immediately on POSIX, opened
O_TEMPORARY on Windows - so the OS removes it on close and there is nothing on
disk for a sweep to find. Every other write in the pipeline goes to one of the
two roots above. Checked by walking the writes rather than assumed.

Age is the directory's own mtime. It moves as artefacts are written and then
stops when the job finishes, so it is the time the job completed. Serving a file
out of /results does not touch it, which is the behaviour wanted here: retention
is measured from when the data was created, not from when someone last looked at
it.

RETENTION_HOURS is the single source. The consent copy and the UI read it off
`GET /exercises`, the same way the upload limits are served, so the
period a user is told cannot drift from the period the code enforces.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

log = logging.getLogger("coach.retention")


RETENTION_HOURS = float(os.environ.get("RETENTION_HOURS") or 72)

# How often the background sweep runs. Well under the TTL so a job is never much
# older than the stated period by the time it goes.
SWEEP_INTERVAL_SECONDS = 15 * 60

# job ids are "<exercise>_<YYYYMMDD>_<HHMMSS>_<6 hex>" (runner.new_job_id). The
# delete-now route puts a caller-supplied string on the end of a filesystem path,
# so it is matched exactly rather than sanitised - ".." never matches this and
# neither does anything else with a separator in it.
JOB_ID = re.compile(r"^[a-z0-9]+_\d{8}_\d{6}_[0-9a-f]{6}$")


def retention_note(hours: float = None) -> str:
    """The sentence the UI and the consent form show. Built from the number the
    sweep actually uses, so the two cannot say different things."""
    h = RETENTION_HOURS if hours is None else hours
    if h >= 48 and h % 24 == 0:
        return (f"Your video, voice note and results are deleted automatically "
                f"after {int(h // 24)} days.")
    if h == 24:
        return ("Your video, voice note and results are deleted automatically "
                "after 24 hours.")
    return (f"Your video, voice note and results are deleted automatically after "
            f"{h:g} hours.")


@dataclass
class SweepResult:
    """What the sweep did, so it can be logged and asserted on."""
    deleted: List[str] = field(default_factory=list)      # job ids
    paths_removed: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    scanned: int = 0
    cutoff_epoch: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.failed


def _job_dirs(root: Path) -> Iterable[Path]:
    """Directories under `root` that are jobs, matched on the id format.

    Not "every directory here". OUTPUT_ROOT is data/outputs/, which the benchmark
    and harness scripts also write into - penn_eval/, penn_eval_pushup/,
    faithfulness/ - and the first run of this sweep deleted two of them along
    with 116 real jobs, because they were older than a day and nothing said they
    were not jobs. Recoverable, since the benchmark reproduces, but a sweep that
    removes evidence is a worse failure than one that leaves a clip behind.
    """
    if not root.is_dir():
        return []
    return (p for p in root.iterdir() if p.is_dir() and JOB_ID.match(p.name))


def expired_jobs(roots: Iterable[Path], cutoff: float) -> List[Path]:
    """Job directories last written before `cutoff`, across every root."""
    out = []
    for root in roots:
        for d in _job_dirs(root):
            try:
                if d.stat().st_mtime < cutoff:
                    out.append(d)
            except OSError:
                # vanished between listing and stat, which is fine - something
                # else removed it and that is the outcome we wanted anyway
                continue
    return sorted(out)


def sweep(roots: Iterable[Path], now: Optional[float] = None,
          hours: Optional[float] = None) -> SweepResult:
    """Remove every job directory older than the retention period.

    `now` and `hours` are injectable so a test can age a directory without
    sleeping through a real TTL.
    """
    roots = [Path(r) for r in roots]
    now = time.time() if now is None else now
    hours = RETENTION_HOURS if hours is None else hours
    cutoff = now - hours * 3600

    result = SweepResult(cutoff_epoch=cutoff)
    result.scanned = sum(1 for r in roots for _ in _job_dirs(r))

    for d in expired_jobs(roots, cutoff):
        try:
            shutil.rmtree(d)
        except OSError as exc:
            # Logged rather than raised: one undeletable directory must not stop
            # the sweep reaching the rest of them.
            result.failed.append(str(d))
            log.warning("retention: could not remove %s: %s", d, exc)
            continue
        result.paths_removed.append(str(d))
        if d.name not in result.deleted:
            result.deleted.append(d.name)

    if result.paths_removed or result.failed:
        log.info("retention: removed %d director%s for %d job(s) older than %gh"
                 "%s", len(result.paths_removed),
                 "y" if len(result.paths_removed) == 1 else "ies",
                 len(result.deleted), hours,
                 f", {len(result.failed)} failed" if result.failed else "")
    return result


def delete_job(roots: Iterable[Path], job_id: str) -> SweepResult:
    """Delete one job now, whatever its age. Backs the delete-now path.

    Returns an empty result rather than raising when the job is not there:
    "delete this" and "this is already gone" are the same outcome from the
    caller's point of view, and distinguishing them would tell an unauthenticated
    caller which job ids exist.
    """
    result = SweepResult()
    if not JOB_ID.match(job_id or ""):
        raise ValueError(f"not a job id: {job_id!r}")

    for root in roots:
        d = Path(root) / job_id
        result.scanned += 1
        if not d.is_dir():
            continue
        try:
            shutil.rmtree(d)
        except OSError as exc:
            result.failed.append(str(d))
            log.warning("retention: could not remove %s: %s", d, exc)
            continue
        result.paths_removed.append(str(d))

    if result.paths_removed:
        result.deleted.append(job_id)
        log.info("retention: deleted %s on request (%d director%s)", job_id,
                 len(result.paths_removed),
                 "y" if len(result.paths_removed) == 1 else "ies")
    return result
