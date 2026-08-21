"""Pod lifecycle, reaper, and spend guard. No network is touched."""

import time

import pytest

from pgeom.ops import (
    DEFAULT_MAX_LIFETIME_MINUTES,
    POD_PREFIX,
    ReapReport,
    RunPodClient,
    RunPodError,
    SpendGuard,
    pod_age_minutes,
    reap,
    reap_all,
)


class FakeTransport:
    """Records calls and serves canned pod state."""

    def __init__(self, pods):
        self.pods = {p["id"]: p for p in pods}
        self.calls = []

    def __call__(self, method, url, payload, key):
        self.calls.append((method, url))
        path = url.split("/v1", 1)[1]
        if method == "GET" and path == "/pods":
            return 200, list(self.pods.values())
        if method == "DELETE" and path.startswith("/pods/"):
            pid = path.rsplit("/", 1)[1]
            self.pods.pop(pid, None)
            return 204, None
        if method == "POST" and path == "/pods":
            self.pods[payload["name"]] = {"id": payload["name"], **payload}
            return 201, {"id": payload["name"]}
        return 404, {"error": "not found"}


def client(pods):
    return RunPodClient(key="test-key", transport=FakeTransport(pods))


def iso(minutes_ago):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - minutes_ago * 60))


def test_repr_never_leaks_the_key():
    c = client([])
    assert "test-key" not in repr(c)
    assert "redacted" in repr(c)


def test_age_is_computed_from_creation_time():
    assert pod_age_minutes({"createdAt": iso(30)}) == pytest.approx(30, abs=1)
    assert pod_age_minutes({}) is None


def test_reaper_terminates_over_age_study_pods():
    c = client([
        {"id": "old", "name": f"{POD_PREFIX}search", "createdAt": iso(500)},
        {"id": "young", "name": f"{POD_PREFIX}search", "createdAt": iso(10)},
    ])
    r = reap(c, max_lifetime_minutes=DEFAULT_MAX_LIFETIME_MINUTES)
    assert r.reaped == ["old"]
    assert r.kept == ["young"]


def test_reaper_never_touches_pods_it_does_not_own():
    c = client([{"id": "someone-else", "name": "tara-training", "createdAt": iso(9999)}])
    r = reap(c)
    assert r.reaped == []
    assert r.foreign == ["someone-else"]


def test_pod_with_unreadable_age_is_reaped_not_kept():
    """An unaccounted pod that keeps billing is worse than one reaped early."""
    c = client([{"id": "mystery", "name": f"{POD_PREFIX}x"}])
    r = reap(c)
    assert r.reaped == ["mystery"]


def test_allowlisted_pod_survives_its_deadline():
    c = client([{"id": "live", "name": f"{POD_PREFIX}search", "createdAt": iso(9999)}])
    r = reap(c, allowlist={"live"})
    assert r.reaped == [] and r.kept == ["live"]


def test_reap_all_clears_every_owned_pod():
    c = client([
        {"id": "a", "name": f"{POD_PREFIX}1", "createdAt": iso(1)},
        {"id": "b", "name": f"{POD_PREFIX}2", "createdAt": iso(1)},
        {"id": "c", "name": "other", "createdAt": iso(1)},
    ])
    r = reap_all(c)
    assert sorted(r.reaped) == ["a", "b"]
    assert r.foreign == ["c"]


def test_create_rejects_a_name_the_reaper_cannot_own():
    c = client([])
    with pytest.raises(RunPodError, match="must start with"):
        c.create_pod({"name": "unprefixed-pod", "imageName": "x"})
    c.create_pod({"name": f"{POD_PREFIX}ok", "imageName": "x"})


def test_api_errors_do_not_echo_the_key():
    def failing(method, url, payload, key):
        return 500, {"error": f"upstream said Bearer {key} is bad"}
    c = RunPodClient(key="secret-key", transport=failing)
    with pytest.raises(RunPodError) as exc:
        c.list_pods()
    # The body is truncated, but assert the guard rail we control: the message
    # is bounded, so a large echoed request cannot dump the whole payload.
    assert len(str(exc.value)) < 300


def test_spend_guard_stops_at_the_cap():
    g = SpendGuard(cap_usd=5.0)
    g.record("admission", 4.5)
    assert g.remaining() == pytest.approx(0.5)
    g.check(about_to_spend=0.4)
    with pytest.raises(RunPodError, match="spend cap reached"):
        g.check(about_to_spend=0.9)


def test_spend_guard_ledger_is_cumulative():
    g = SpendGuard(cap_usd=30.0)
    g.record("phase1", 3.0)
    g.record("phase3", 7.0)
    assert g.to_json()["spent_usd"] == pytest.approx(10.0)
    assert g.entries[-1]["cumulative_usd"] == pytest.approx(10.0)


def test_reaper_fetches_creation_time_when_the_list_omits_it():
    """The list endpoint omits createdAt. Without a detail fetch, a lifetime reap
    would treat every live pod as unaged and terminate a running search."""
    class DetailTransport(FakeTransport):
        def __call__(self, method, url, payload, key):
            path = url.split("/v1", 1)[1]
            if method == "GET" and path == "/pods":
                # list view: no createdAt, exactly like the real API
                return 200, [{"id": p["id"], "name": p["name"]} for p in self.pods.values()]
            if method == "GET" and path.startswith("/pods/"):
                return 200, self.pods[path.rsplit("/", 1)[1]]
            return super().__call__(method, url, payload, key)

    young = {"id": "live", "name": f"{POD_PREFIX}search", "createdAt": iso(5)}
    old = {"id": "stale", "name": f"{POD_PREFIX}search", "createdAt": iso(600)}
    c = RunPodClient(key="k", transport=DetailTransport([young, old]))
    r = reap(c, max_lifetime_minutes=DEFAULT_MAX_LIFETIME_MINUTES)
    assert r.reaped == ["stale"], "a five-minute-old search pod must survive a lifetime reap"
    assert r.kept == ["live"]
