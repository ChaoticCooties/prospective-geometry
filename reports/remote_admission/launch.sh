#!/usr/bin/env bash
set -uo pipefail
mkdir -p /workspace/pgeom/hf /workspace/pgeom/artifacts
export HF_HUB_CACHE=/workspace/pgeom/hf
export HF_TOKEN="$(cat /root/.hf_token)"
export PYTHONPATH=/workspace/pgeom/repo/src PYTHONUNBUFFERED=1
cd /workspace/pgeom/repo
python3 -u scripts/remote/fit_and_admit.py  > /workspace/pgeom/artifacts/run.log 2>&1
echo $? > /workspace/pgeom/artifacts/run.done
