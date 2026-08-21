#!/usr/bin/env python3
"""Pod inventory, reaper, and spend reporting.

Run `pods.py list` before and after every phase, as the operating rules require.
`pods.py reap` is the external reaper: it is safe to run at any time and never
touches a pod this study does not own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgeom.ops import (  # noqa: E402
    DEFAULT_MAX_LIFETIME_MINUTES,
    RunPodClient,
    RunPodError,
    pod_age_minutes,
    reap,
    reap_all,
)


def cmd_list(client: RunPodClient) -> int:
    pods = client.list_pods()
    print(f"pods: {len(pods)}")
    for p in pods:
        age = pod_age_minutes(p)
        age_s = f"{age:.1f}m" if age is not None else "unknown"
        print(f"  {p.get('id'):16s} {str(p.get('name'))[:28]:30s} "
              f"{str(p.get('desiredStatus')):10s} age={age_s:>9s} "
              f"${p.get('costPerHr') or 0}/hr")
    vols = client.list_network_volumes()
    print(f"network volumes: {len(vols)}")
    for v in vols:
        print(f"  {v.get('id'):16s} {str(v.get('name'))[:24]:26s} "
              f"{v.get('size')}GB  {v.get('dataCenterId')}")
    return 0


def cmd_reap(client: RunPodClient, args) -> int:
    report = reap_all(client) if args.all else reap(
        client, max_lifetime_minutes=args.max_lifetime_minutes
    )
    print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    return 1 if report.errors else 0


def cmd_spend(client: RunPodClient) -> int:
    try:
        print(json.dumps(client.pod_billing(), indent=2, sort_keys=True)[:4000])
    except RunPodError as e:
        print(f"billing unavailable: {e}")
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("reap")
    r.add_argument("--max-lifetime-minutes", type=float, default=DEFAULT_MAX_LIFETIME_MINUTES)
    r.add_argument("--all", action="store_true", help="terminate every pod this study owns")
    sub.add_parser("spend")
    args = ap.parse_args(argv)

    try:
        client = RunPodClient()
    except RunPodError as e:
        print(str(e))
        return 2
    if args.cmd == "list":
        return cmd_list(client)
    if args.cmd == "reap":
        return cmd_reap(client, args)
    return cmd_spend(client)


if __name__ == "__main__":
    sys.exit(main())
