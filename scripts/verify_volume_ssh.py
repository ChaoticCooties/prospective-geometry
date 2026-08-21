#!/usr/bin/env python3
"""Durability readback over SSH on a GPU pod, which also validates the exact
pod configuration Phase 1 will use.

Pod 1 already wrote a payload under /workspace/pgeom/durability and was
destroyed, so a successful read here *is* the survive-pod-destruction check.
The CPU HTTP-proxy channel failed twice, and SSH is needed to drive the searches
anyway, so it is solved once here rather than worked around.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgeom.ops import POD_PREFIX, RunPodClient, RunPodError  # noqa: E402

VOLUME_ID = "89x1h5z4nj"
DATACENTER = "US-KS-2"
# The REST API accepts a different enum than GraphQL reports, so candidates are
# tried in price order and the first with capacity in this datacenter wins.
GPU_CANDIDATES = (
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A6000",
    "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA A40",
    "NVIDIA RTX 6000 Ada Generation",
)
IMAGE = "runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204"
DUR = "/workspace/pgeom/durability"
PUBKEY = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wait_ssh(client, pod_id, timeout_s=600):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = client.get_pod(pod_id)
        ip = pod.get("publicIp") or ""
        pms = pod.get("portMappings") or {}
        port = None
        if isinstance(pms, dict):
            port = pms.get("22")
        elif isinstance(pms, list):
            port = next((p.get("publicPort") for p in pms if str(p.get("privatePort")) == "22"), None)
        if ip and port:
            return ip, int(port)
        log(f"  waiting: status={pod.get('desiredStatus')} ip={ip!r} ports={pms}")
        time.sleep(15)
    raise RunPodError("pod never exposed SSH")


def ssh(host, port, cmd, timeout=180):
    p = subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


VERIFY = f"""
python3 - <<'EOF'
import hashlib, json, os
d = "{DUR}"
r = {{}}
try:
    r["readback_sha256"] = hashlib.sha256(open(f"{{d}}/payload.bin","rb").read()).hexdigest()
    r["expected_sha256"] = open(f"{{d}}/payload.sha256").read().strip()
    r["survived"] = r["readback_sha256"] == r["expected_sha256"]
except Exception as e:
    r["error"] = f"{{type(e).__name__}}: {{e}}"; r["survived"] = False
def valid(p):
    n = 0
    try:
        for line in open(p):
            try: json.loads(line); n += 1
            except Exception: pass
    except Exception: return -1
    return n
r["complete_checkpoint_valid_lines"] = valid(f"{{d}}/ckpt_complete.jsonl")
r["truncated_checkpoint_valid_lines"] = valid(f"{{d}}/ckpt_truncated.jsonl")
r["truncation_detectable"] = r["truncated_checkpoint_valid_lines"] == 0
r["volume_toplevel_names"] = sorted(os.listdir("/workspace"))[:40]
try: r["gpu"] = os.popen("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader").read().strip()
except Exception: pass
print("VERDICT_JSON:" + json.dumps(r, sort_keys=True))
EOF
"""


def main() -> int:
    client = RunPodClient()
    created = []
    results = {"datacenter": DATACENTER, "volume_id": VOLUME_ID}
    try:
        pod = None
        attempts = {}
        for gpu in GPU_CANDIDATES:
            log(f"trying {gpu}")
            try:
                pod = client.create_pod({
                    "name": f"{POD_PREFIX}durability-ssh",
                    "imageName": IMAGE,
                    "gpuTypeIds": [gpu],
                    "gpuCount": 1,
                    "cloudType": "SECURE",
                    "dataCenterIds": [DATACENTER],
                    "networkVolumeId": VOLUME_ID,
                    "volumeMountPath": "/workspace",
                    "containerDiskInGb": 30,
                    "ports": ["22/tcp"],
                    "supportPublicIp": True,
                    "env": {"PUBLIC_KEY": PUBKEY},
                })
                results["gpu_used"] = gpu
                break
            except RunPodError as e:
                attempts[gpu] = str(e)[:140]
                log(f"  unavailable: {str(e)[:110]}")
        results["gpu_attempts"] = attempts
        if pod is None:
            results["error"] = "no candidate GPU had capacity in this datacenter"
            raise RunPodError(results["error"])
        created.append(pod["id"])
        results["pod_id"] = pod["id"]
        host, port = wait_ssh(client, pod["id"])
        log(f"ssh {host}:{port}")
        for attempt in range(10):
            rc, out = ssh(host, port, "echo ready")
            if rc == 0:
                break
            log(f"  sshd not up yet ({attempt}): {out[:80]}")
            time.sleep(15)
        rc, out = ssh(host, port, VERIFY)
        log(f"verify rc={rc}")
        line = next((l for l in out.splitlines() if l.startswith("VERDICT_JSON:")), None)
        if line:
            results.update(json.loads(line.split(":", 1)[1]))
        else:
            results["error"] = out[:400]
    finally:
        for pid in created:
            try:
                client.delete_pod(pid)
                log(f"deleted {pid}")
            except RunPodError as e:
                log(f"DELETE FAILED {pid}: {e}")
        time.sleep(4)
        results["pods_after"] = len(client.list_pods())
        log(f"pods after: {results['pods_after']}")

    out_path = Path(__file__).resolve().parents[1] / "reports" / "durability_verification.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results.get("survived") and results.get("pods_after") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
