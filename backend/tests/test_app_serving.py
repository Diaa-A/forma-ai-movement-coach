"""The PWA and the API share an origin, so the static mount must not eat the API.

Mounting StaticFiles at "/" matches every path, and Starlette resolves routes in
the order they were added. Get that order wrong and /analyze starts returning
index.html with a 200, which is a genuinely horrible thing to debug — the client
sees a success and a body it can't parse.

These run whether or not frontend/dist has been built, because a clean checkout
won't have it.
"""
from fastapi.testclient import TestClient

from backend.main import app, FRONTEND_DIST


client = TestClient(app)


def test_api_routes_survive_the_spa_mount():
    """The one that matters. If this fails, the mount was added too early."""
    assert client.get('/exercises').status_code == 200
    assert client.get('/health').json() == {'status': 'ok'}


def test_analyze_still_reaches_the_handler_not_the_static_files():
    """A bad request should come back as the route's own 400, not as index.html."""
    r = client.post('/analyze', data={'exercise_type': 'nope'},
                    files={'video': ('a.mp4', b'x', 'video/mp4')})
    assert r.status_code == 400
    assert 'nope' in r.json()['detail']


def test_root_serves_the_app_when_built_and_explains_itself_when_not():
    r = client.get('/')
    assert r.status_code == 200
    if FRONTEND_DIST.is_dir():
        assert '<div id="root">' in r.text
    else:
        # no build — should say so rather than 404 or serve an empty page
        assert 'frontend/dist not found' in r.json()['note']


def test_api_index_is_reachable_either_way():
    """/api keeps working even when the SPA has taken over /."""
    body = client.get('/api').json()
    assert 'POST /analyze' in body['endpoints']
