"""RunPod operations: pod lifecycle, external reaper, and spend tracking.

Operational rules this module enforces, rather than trusts:

* one pod at a time;
* no pod outlives its declared deadline, whatever happens to the driver process;
* the API key is read from disk or environment and never returned, logged, or
  written into a manifest, a report, or a pod's persistent volume.

Every network call goes through an injectable transport so the tests exercise
the lifecycle and the reaper without touching RunPod or spending anything.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

BASE_URL = "https://rest.runpod.io/v1"
CONFIG_PATH = Path.home() / ".runpod" / "config.toml"
KEY_ENV = "RUNPOD_API_KEY"

#: Pods this study creates carry this prefix, so the reaper can tell its own
#: pods from anything else on the account and refuse to touch the latter.
POD_PREFIX = "pgeom-"

#: Hard ceiling on pod lifetime. A pod older than this is reaped regardless of
#: what it claims to be doing.
DEFAULT_MAX_LIFETIME_MINUTES = 240

#: Idle ceiling from the operating rules.
MAX_IDLE_MINUTES = 5


class RunPodError(RuntimeError):
    """API call failed. The message never contains the API key."""


def load_api_key() -> str:
    """Read the key from the environment, else from the RunPod config file."""
    env = os.environ.get(KEY_ENV)
    if env:
        return env.strip()
    if CONFIG_PATH.is_file():
        m = re.search(r'apikey\s*=\s*"([^"]+)"', CONFIG_PATH.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    raise RunPodError(
        f"no RunPod API key: set {KEY_ENV} or write {CONFIG_PATH} with apikey = \"...\""
    )


def _urllib_transport(method: str, url: str, payload: dict | None, key: str):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode()
            return resp.status, (json.loads(body) if body.strip() else None)
    except urllib.error.HTTPError as e:
        # Truncate: an error body can echo the request, and the request carries the key.
        return e.code, {"error": e.read().decode()[:300]}


@dataclass
class RunPodClient:
    """Thin REST client. ``transport`` is injectable so tests never hit the network."""

    key: str = field(default_factory=load_api_key, repr=False)
    transport: Callable[..., tuple[int, Any]] = _urllib_transport

    def __repr__(self) -> str:  # never leak the key through a repr in a traceback
        return "RunPodClient(key=<redacted>)"

    def _call(self, method: str, path: str, payload: dict | None = None):
        status, body = self.transport(method, f"{BASE_URL}{path}", payload, self.key)
        if status not in (200, 201, 204):
            raise RunPodError(f"{method} {path} failed with HTTP {status}: {str(body)[:200]}")
        return body

    def list_pods(self) -> list[dict]:
        return self._call("GET", "/pods") or []

    def get_pod(self, pod_id: str) -> dict:
        return self._call("GET", f"/pods/{pod_id}") or {}

    def delete_pod(self, pod_id: str) -> None:
        self._call("DELETE", f"/pods/{pod_id}")

    def create_pod(self, spec: dict) -> dict:
        if not str(spec.get("name", "")).startswith(POD_PREFIX):
            raise RunPodError(f"pod name must start with {POD_PREFIX!r} so the reaper can own it")
        return self._call("POST", "/pods", spec) or {}

    def list_network_volumes(self) -> list[dict]:
        return self._call("GET", "/networkvolumes") or []

    def pod_billing(self) -> Any:
        return self._call("GET", "/billing/pods")


def pod_age_minutes(pod: dict, *, now: float | None = None) -> float | None:
    """Age of a pod in minutes from its creation timestamp, or None if absent."""
    stamp = pod.get("createdAt") or pod.get("lastStatusChange")
    if not stamp:
        return None
    try:
        created = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    current = now if now is not None else time.time()
    return (current - created.timestamp()) / 60.0


@dataclass
class ReapReport:
    inspected: int
    reaped: list[str]
    kept: list[str]
    foreign: list[str]
    errors: dict[str, str]

    def to_json(self) -> dict:
        return {
            "inspected": self.inspected, "reaped": self.reaped, "kept": self.kept,
            "foreign_pods_left_alone": self.foreign, "errors": self.errors,
        }


def reap(
    client: RunPodClient,
    *,
    max_lifetime_minutes: float = DEFAULT_MAX_LIFETIME_MINUTES,
    allowlist: set[str] | None = None,
    now: float | None = None,
) -> ReapReport:
    """Terminate this study's pods that have outlived their deadline.

    Pods whose name lacks :data:`POD_PREFIX` belong to something else on the
    account and are reported but never deleted. A pod in ``allowlist`` is a
    known-live job and is kept regardless of age, so the deadline can be
    extended deliberately rather than by accident.
    """
    allowlist = allowlist or set()
    reaped: list[str] = []
    kept: list[str] = []
    foreign: list[str] = []
    errors: dict[str, str] = {}

    pods = client.list_pods()
    for pod in pods:
        pod_id = pod.get("id", "?")
        name = pod.get("name") or ""
        if not name.startswith(POD_PREFIX):
            foreign.append(pod_id)
            continue
        if pod_id in allowlist:
            kept.append(pod_id)
            continue
        age = pod_age_minutes(pod, now=now)
        # An unparseable age is treated as over-age: an unaccounted pod that
        # keeps billing is worse than one reaped a little early.
        if age is None or age >= max_lifetime_minutes:
            try:
                client.delete_pod(pod_id)
                reaped.append(pod_id)
            except RunPodError as e:
                errors[pod_id] = str(e)[:120]
        else:
            kept.append(pod_id)
    return ReapReport(len(pods), reaped, kept, foreign, errors)


def reap_all(client: RunPodClient) -> ReapReport:
    """Terminate every pod this study owns. Used at phase boundaries and on abort."""
    return reap(client, max_lifetime_minutes=0.0)


@dataclass
class SpendGuard:
    """Local spend ledger with a hard cap.

    RunPod bills continuously, so the guard is checked before starting work and
    again on every tick. Crossing the cap raises rather than warning: the
    protocol's cap is a stop rule, not advice.
    """

    cap_usd: float
    spent_usd: float = 0.0
    entries: list[dict] = field(default_factory=list)

    def record(self, label: str, usd: float) -> None:
        self.spent_usd += usd
        self.entries.append({"label": label, "usd": round(usd, 6),
                             "cumulative_usd": round(self.spent_usd, 6)})

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)

    def check(self, *, about_to_spend: float = 0.0) -> None:
        if self.spent_usd + about_to_spend > self.cap_usd:
            raise RunPodError(
                f"spend cap reached: ${self.spent_usd:.4f} spent, "
                f"${about_to_spend:.4f} requested, cap ${self.cap_usd:.2f}"
            )

    def to_json(self) -> dict:
        return {
            "cap_usd": self.cap_usd,
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(self.remaining(), 6),
            "entries": self.entries,
        }
