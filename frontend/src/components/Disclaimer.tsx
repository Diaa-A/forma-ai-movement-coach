/**
 * Not-medical-advice notice.
 *
 * Rendered on every screen and deliberately not dismissible. The system measures
 * joint angles and reads them against calibrated heuristics; it does not diagnose
 * anything, and BUILD_REFERENCE.md 3 makes that a locked constraint rather than a
 * preference. Keeping it in one component means there's no screen where someone
 * forgot to add it.
 */
export default function Disclaimer() {
  return (
    <p className="disclaimer">
      This is general movement feedback from an automated system, not medical
      advice. It cannot diagnose injuries or conditions. If something hurts, or
      you are unsure whether an exercise is right for you, speak to a qualified
      professional.
    </p>
  )
}
