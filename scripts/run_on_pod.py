#!/usr/bin/env python3
"""Provision a pod, sync the repo, run a remote script, fetch artifacts, reap.

Credential handling, which is the part worth being careful about:

* the repository is rsynced over SSH rather than cloned, so no GitHub
  credential ever reaches the pod;
* the Hugging Face token is written to the pod's own disk over an SSH stdin
  pipe, never passed as a command-line argument (visible in `ps`) and never
  placed in the pod's `env` record (stored by the API) or on the network volume
  (which outlives the pod);
* nothing is echoed back into logs.

The pod is always reaped in a finally block. Run this backgrounded: a foreground
timeout that kills the driver would skip teardown and leave a pod billing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pgeom.ops import POD_PREFIX, RunPodClient, RunPodError  # noqa: E402

VOLUME_ID = "5l9knlrkk1"          # pgeom-clean
DATACENTER = "EU-RO-1"
# RTX 4090 is Ada (sm_89), which the image's torch 2.6.0+cu124 has kernels for.
# The RTX PRO 4500 Blackwell is deliberately excluded: it schedules fine and
# reports CUDA available, then fails at the first matmul with "no kernel image is
# available for execution on the device", because Blackwell needs CUDA 12.8 and a
# newer torch than the pinned runtime.
GPU_CANDIDATES = ("NVIDIA GeForce RTX 4090",)
IMAGE = "runpod/pytorch:0.7.0-cu1241-torch260-ubuntu2204"
REMOTE_REPO = "/workspace/pgeom/repo"
HF_CACHE = "/workspace/pgeom/hf"      # model weights live on the volume, cached across pods
POD_TOKEN = "/root/.hf_token"         # pod-local, never the volume: the volume outlives the pod
PUBKEY = (Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()
TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=240"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def create(client, name):
    last = None
    for gpu in GPU_CANDIDATES:
        try:
            pod = client.create_pod({
                "name": f"{POD_PREFIX}{name}",
                "imageName": IMAGE,
                "gpuTypeIds": [gpu], "gpuCount": 1,
                "dataCenterIds": [DATACENTER],
                "networkVolumeId": VOLUME_ID,
                "volumeMountPath": "/workspace",
                "containerDiskInGb": 40,
                "ports": ["22/tcp"],
                "supportPublicIp": True,
                "env": {"PUBLIC_KEY": PUBKEY},   # deliberately no secrets here
            })
            log(f"pod {pod['id']} on {gpu}")
            return pod, gpu
        except RunPodError as e:
            last = str(e)[:120]
            log(f"  {gpu} unavailable")
    raise RunPodError(f"no GPU available in {DATACENTER}: {last}")


def wait_ssh(client, pod_id, timeout_s=900):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = client.get_pod(pod_id)
        ip = pod.get("publicIp") or ""
        pms = pod.get("portMappings") or {}
        port = pms.get("22") if isinstance(pms, dict) else next(
            (p.get("publicPort") for p in pms if str(p.get("privatePort")) == "22"), None)
        if ip and port:
            return ip, int(port)
        time.sleep(12)
    raise RunPodError("pod never exposed SSH")


def ssh(host, port, cmd, *, timeout=3600, stdin_text=None, stream=False):
    argv = ["ssh", *SSH_OPTS, "-p", str(port), f"root@{host}", cmd]
    if stream:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE if stdin_text else None,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if stdin_text:
            proc.stdin.write(stdin_text)
            proc.stdin.close()
        out = []
        for line in proc.stdout:
            print(line.rstrip(), flush=True)
            out.append(line)
        proc.wait(timeout=timeout)
        return proc.returncode, "".join(out)
    p = subprocess.run(argv, input=stdin_text, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def wait_sshd(host, port, tries=20):
    for i in range(tries):
        rc, out = ssh(host, port, "echo ready", timeout=45)
        if rc == 0:
            return True
        time.sleep(15)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="path under scripts/remote/")
    ap.add_argument("--name", default="run")
    ap.add_argument("--args", default="")
    ap.add_argument("--fetch", default="reports/remote", help="local dir for artifacts")
    ap.add_argument("--timeout", type=int, default=7200)
    args = ap.parse_args()

    client = RunPodClient()
    created = []
    status = {"script": args.script, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        pod, gpu = create(client, args.name)
        created.append(pod["id"])
        status["pod_id"], status["gpu"] = pod["id"], gpu
        host, port = wait_ssh(client, pod["id"])
        log(f"ssh {host}:{port}")
        if not wait_sshd(host, port):
            raise RunPodError("sshd never came up")

        log("syncing repo")
        # tar over SSH rather than rsync: the pod image has tar but not rsync,
        # and this needs nothing installed on either end.
        # COPYFILE_DISABLE stops macOS tar emitting AppleDouble "._*" entries,
        # and --no-same-owner stops GNU tar on the pod failing on ownership it
        # cannot set. Without both, the transfer exits non-zero on warnings.
        sync = subprocess.run(
            f"COPYFILE_DISABLE=1 tar czf - -C {ROOT} "
            f"--exclude=.git --exclude=reports/remote --exclude=audit/artifacts_full "
            f"--exclude=__pycache__ --exclude=.pytest_cache --exclude='._*' . | "
            f"ssh {' '.join(SSH_OPTS)} -p {port} root@{host} "
            f"'mkdir -p {REMOTE_REPO} && tar xzf - --no-same-owner -C {REMOTE_REPO}'",
            shell=True, capture_output=True, text=True, timeout=900,
        )
        if sync.returncode != 0:
            raise RunPodError(f"repo sync failed: {sync.stderr[:300]}")

        # Token over stdin: never a command-line argument, never in the pod's env
        # record, never on the network volume.
        ssh(host, port, f"cat > {POD_TOKEN} && chmod 600 {POD_TOKEN}",
            stdin_text=TOKEN_PATH.read_text().strip(), timeout=60)

        log("installing pinned dependencies")
        # Pinned to the versions the study was frozen against. The chat-template
        # hash check would catch a template change, but pinning avoids relying on
        # that as the only guard. Wheels cache on the volume so a later pod is fast.
        rc, out = ssh(host, port,
            f"export PIP_CACHE_DIR=/workspace/pgeom/pipcache && "
            f"python3 -m pip install -q --disable-pip-version-check "
            f"'transformers==4.49.0' 'tokenizers>=0.21,<0.22' 'accelerate>=1.0' "
            f"'safetensors>=0.4' 'huggingface_hub>=0.30' 2>&1 | tail -5",
            timeout=1800)
        if rc != 0:
            raise RunPodError(f"dependency install failed: {out[:300]}")
        rc, ver = ssh(host, port,
            "python3 -c 'import torch,transformers;"
            "print(torch.__version__, transformers.__version__, torch.cuda.is_available())'",
            timeout=300)
        log(f"remote versions: {ver.strip()}")
        status["remote_versions"] = ver.strip()

        log("running remote script")
        remote_cmd = (
            f"set -o pipefail; mkdir -p {HF_CACHE} && "
            # HF_HUB_CACHE, not HF_HOME: HF_HOME also redirects the token lookup,
            # which would put the credential on the network volume that outlives
            # the pod. The token is read from a pod-local file and expanded inside
            # the pod's shell, so its value never appears in the ssh command line.
            f'export HF_HUB_CACHE={HF_CACHE} HF_TOKEN="$(cat {POD_TOKEN})" '
            f"PYTHONPATH={REMOTE_REPO}/src PYTHONUNBUFFERED=1 && "
            f"cd {REMOTE_REPO} && python3 -u scripts/remote/{args.script} {args.args}"
        )
        rc, out = ssh(host, port, remote_cmd, timeout=args.timeout, stream=True)
        status["remote_rc"] = rc

        log("fetching artifacts")
        local = ROOT / args.fetch
        local.mkdir(parents=True, exist_ok=True)
        fetch = subprocess.run(
            f"ssh {' '.join(SSH_OPTS)} -p {port} root@{host} "
            f"'tar czf - -C /workspace/pgeom/artifacts . 2>/dev/null' | "
            f"tar xzf - --no-same-owner -C {local}",
            shell=True, capture_output=True, text=True, timeout=900,
        )
        status["fetch_rc"] = fetch.returncode
    except Exception as e:
        status["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        log(status["error"])
    finally:
        for pid in created:
            try:
                client.delete_pod(pid)
                log(f"deleted {pid}")
            except RunPodError as e:
                log(f"DELETE FAILED {pid}: {e}")
        time.sleep(5)
        status["pods_after"] = len(client.list_pods())
        log(f"pods after: {status['pods_after']}")

    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status.get("remote_rc") == 0 and status.get("pods_after") == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
