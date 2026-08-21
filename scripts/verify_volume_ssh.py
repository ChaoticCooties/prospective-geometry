#!/usr/bin/env python3
"""Verify durability on the study's own volume, over SSH on a GPU pod.

Full cycle: write on pod A, destroy pod A, read back on pod B. A successful
read is the survive-pod-destruction check. The run also validates the exact pod
configuration Phase 1 will use, including SSH, the mount path, and the GPU.

Run backgrounded. A foreground timeout that kills the driver would skip the
teardown and leave a pod billing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pgeom.ops import POD_PREFIX, RunPodClient, RunPodError  # noqa: E402

VOLUME_ID = "5l9knlrkk1"          # pgeom-clean
DATACENTER = "EU-RO-1"
GPU_CANDIDATES = ("NVIDIA RTX PRO 4500 Blackwell", "NVIDIA GeForce RTX 4090")
IMAGE = "runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204"
STUDY_DIR = "/workspace/pgeom"
DUR = f"{STUDY_DIR}/durability"
PUBKEY = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def create(client, suffix):
    last = None
    for gpu in GPU_CANDIDATES:
        try:
            pod = client.create_pod({
                "name": f"{POD_PREFIX}dur-{suffix}",
                "imageName": IMAGE,
                "gpuTypeIds": [gpu], "gpuCount": 1,
                "dataCenterIds": [DATACENTER],
                "networkVolumeId": VOLUME_ID,
                "volumeMountPath": "/workspace",
                "containerDiskInGb": 30,
                "ports": ["22/tcp"],
                "supportPublicIp": True,
                "env": {"PUBLIC_KEY": PUBKEY},
            })
            log(f"  created on {gpu}: {pod['id']}")
            return pod, gpu
        except RunPodError as e:
            last = str(e)[:120]
            log(f"  {gpu} unavailable")
    raise RunPodError(f"no GPU available in {DATACENTER}: {last}")


def wait_ssh(client, pod_id, timeout_s=600):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = client.get_pod(pod_id)
        ip = pod.get("publicIp") or ""
        pms = pod.get("portMappings") or {}
        port = pms.get("22") if isinstance(pms, dict) else next(
            (p.get("publicPort") for p in pms if str(p.get("privatePort")) == "22"), None)
        if ip and port:
            return ip, int(port)
        log(f"  waiting: {pod.get('desiredStatus')} ip={ip!r}")
        time.sleep(15)
    raise RunPodError("pod never exposed SSH")


def ssh(host, port, cmd, timeout=240):
    p = subprocess.run(["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def wait_sshd(host, port, tries=12):
    for i in range(tries):
        rc, out = ssh(host, port, "echo ready", timeout=45)
        if rc == 0:
            return True
        log(f"  sshd not up ({i}): {out[:60]}")
        time.sleep(15)
    return False


WRITE = f"""
set -e
mkdir -p {DUR}
head -c 4194304 /dev/urandom > {DUR}/payload.bin
sha256sum {DUR}/payload.bin | cut -d' ' -f1 > {DUR}/payload.sha256
printf '{{"update": 7, "suffix_ids": [101,202,303,404], "loss": 1.25}}\\n' > {DUR}/ckpt_complete.jsonl
printf '{{"update": 8, "suffix_ids": [101,202' > {DUR}/ckpt_truncated.jsonl
sync
echo "WROTE=$(cat {DUR}/payload.sha256)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | tail -1
"""

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
print("VERDICT_JSON:" + json.dumps(r, sort_keys=True))
EOF
"""


def main() -> int:
    client = RunPodClient()
    results = {"datacenter": DATACENTER, "volume_id": VOLUME_ID, "study_dir": STUDY_DIR}
    created = []
    try:
        log("pod A: write")
        podA, gpu = create(client, "write")
        created.append(podA["id"])
        results["gpu_used"] = gpu
        host, port = wait_ssh(client, podA["id"])
        if not wait_sshd(host, port):
            raise RunPodError("sshd never came up on pod A")
        rc, out = ssh(host, port, WRITE)
        log(f"write rc={rc}")
        print(out)
        results["write_output"] = out[-600:]
        results["wrote_sha256"] = next(
            (l.split("=", 1)[1] for l in out.splitlines() if l.startswith("WROTE=")), None)

        log("destroying pod A")
        client.delete_pod(podA["id"])
        created.remove(podA["id"])
        time.sleep(30)

        log("pod B: read back on fresh hardware")
        podB, _ = create(client, "read")
        created.append(podB["id"])
        host2, port2 = wait_ssh(client, podB["id"])
        if not wait_sshd(host2, port2):
            raise RunPodError("sshd never came up on pod B")
        rc, out = ssh(host2, port2, VERIFY)
        log(f"verify rc={rc}")
        line = next((l for l in out.splitlines() if l.startswith("VERDICT_JSON:")), None)
        if line:
            results.update(json.loads(line.split(":", 1)[1]))
        else:
            results["error"] = out[:500]
    except Exception as e:
        results.setdefault("error", f"{type(e).__name__}: {str(e)[:300]}")
    finally:
        for pid in list(created):
            try:
                client.delete_pod(pid)
                log(f"deleted {pid}")
            except RunPodError as e:
                log(f"DELETE FAILED {pid}: {e}")
        time.sleep(5)
        results["pods_after"] = len(client.list_pods())
        log(f"pods after: {results['pods_after']}")

    out_path = Path(__file__).resolve().parents[1] / "reports" / "durability_verification.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    ok = results.get("survived") and results.get("truncation_detectable") and results.get("pods_after") == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
