#!/usr/bin/env python3
"""Verify that artifacts survive pod destruction on the shared network volume.

Checks, from docs/DURABILITY.md:

1. a file written under /workspace/pgeom survives destroying the pod and
   reappears on a second pod with an unchanged SHA-256;
2. a truncated checkpoint line is detectable, so resume can take the last
   *complete* checkpoint rather than a half-written one;
3. the reaper leaves nothing billing afterwards.

CPU pods get no public IP, so results come back over RunPod's HTTP proxy and
only the second pod needs that channel. Pods are torn down in a finally block.
Run this backgrounded: a foreground timeout that kills the driver would leave a
pod billing.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgeom.ops import POD_PREFIX, RunPodClient, RunPodError  # noqa: E402

VOLUME_ID = "89x1h5z4nj"          # tara_gcg
DATACENTER = "US-KS-2"
STUDY_DIR = "/workspace/pgeom"
DUR = f"{STUDY_DIR}/durability"
IMAGE = "runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204"

WRITE_CMD = f"""
set -e
mkdir -p {DUR}
head -c 1048576 /dev/urandom > {DUR}/payload.bin
sha256sum {DUR}/payload.bin | cut -d' ' -f1 > {DUR}/payload.sha256
printf '{{"update": 7, "suffix_ids": [101,202,303,404], "loss": 1.25}}\\n' > {DUR}/ckpt_complete.jsonl
printf '{{"update": 8, "suffix_ids": [101,202' > {DUR}/ckpt_truncated.jsonl
sync
sleep 45
"""

VERIFY_CMD = f"""
mkdir -p /tmp/out
python3 - <<'EOF' > /tmp/out/verdict.json 2>/tmp/out/err.txt
import hashlib, json, os
d = "{DUR}"
res = {{}}
try:
    with open(f"{{d}}/payload.bin","rb") as fh:
        res["readback_sha256"] = hashlib.sha256(fh.read()).hexdigest()
    res["expected_sha256"] = open(f"{{d}}/payload.sha256").read().strip()
    res["survived"] = res["readback_sha256"] == res["expected_sha256"]
except Exception as e:
    res["error"] = f"{{type(e).__name__}}: {{e}}"
    res["survived"] = False

def valid_lines(path):
    n = 0
    try:
        for line in open(path):
            try:
                json.loads(line); n += 1
            except Exception:
                pass
    except Exception:
        return -1
    return n

res["complete_checkpoint_valid_lines"] = valid_lines(f"{{d}}/ckpt_complete.jsonl")
res["truncated_checkpoint_valid_lines"] = valid_lines(f"{{d}}/ckpt_truncated.jsonl")
res["truncation_detectable"] = res["truncated_checkpoint_valid_lines"] == 0
res["volume_toplevel_names"] = sorted(os.listdir("/workspace"))[:40]
res["study_dir_names"] = sorted(os.listdir(d))
print(json.dumps(res, indent=2, sort_keys=True))
EOF
cd /tmp/out && python3 -m http.server 8080
"""


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_pod(client, name, cmd, http_port=False):
    spec = {
        "name": f"{POD_PREFIX}{name}",
        "imageName": IMAGE,
        "computeType": "CPU",
        "cpuFlavorIds": ["cpu3g"],
        "vcpuCount": 2,
        "dataCenterIds": [DATACENTER],
        "networkVolumeId": VOLUME_ID,
        "volumeMountPath": "/workspace",
        "containerDiskInGb": 20,
        "dockerStartCmd": ["bash", "-lc", cmd],
    }
    if http_port:
        spec["ports"] = ["8080/http"]
    return client.create_pod(spec)


def fetch_verdict(pod_id, timeout_s=420):
    url = f"https://{pod_id}-8080.proxy.runpod.net/verdict.json"
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=12) as r:
                return json.loads(r.read().decode())
        except Exception:
            attempt += 1
            if attempt % 5 == 0:
                log(f"  waiting for verdict ({attempt} tries)")
            time.sleep(10)
    return None


def main() -> int:
    client = RunPodClient()
    results = {"datacenter": DATACENTER, "volume_id": VOLUME_ID, "study_dir": STUDY_DIR}
    created = []
    try:
        log("pod 1: writing payload and checkpoints")
        pod1 = make_pod(client, "durability-write", WRITE_CMD)
        created.append(pod1["id"])
        time.sleep(150)                      # let the start command run and sync
        log("destroying pod 1")
        client.delete_pod(pod1["id"])
        created.remove(pod1["id"])
        time.sleep(25)

        log("pod 2: reading back on fresh hardware")
        pod2 = make_pod(client, "durability-read", VERIFY_CMD, http_port=True)
        created.append(pod2["id"])
        verdict = fetch_verdict(pod2["id"])
        if verdict is None:
            results["error"] = "verdict never became reachable over the pod proxy"
        else:
            results.update(verdict)
    finally:
        for pod_id in list(created):
            try:
                client.delete_pod(pod_id)
                log(f"deleted {pod_id}")
            except RunPodError as e:
                log(f"DELETE FAILED {pod_id}: {e}")
        time.sleep(4)
        results["pods_after"] = len(client.list_pods())
        log(f"pods after: {results['pods_after']}")

    out = Path(__file__).resolve().parents[1] / "reports" / "durability_verification.json"
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    ok = results.get("survived") and results.get("truncation_detectable") and results.get("pods_after") == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
