# Durability and credential handling

The protocol forbids provisioning paid compute until durability is defined and
verified. This file is the definition, and it records what has actually been
verified rather than what is intended.

## What can be lost

| Failure | What it costs | Mitigation |
| --- | --- | --- |
| Pod terminated or preempted mid-search | up to one GCG update per search | checkpoint every update to a network volume that outlives the pod |
| Pod disk lost | all in-pod artifacts | artifacts live on the network volume, not pod-local disk |
| Local machine lost | code, protocol, frozen prompts | private Git remote holds everything tracked |
| Network volume lost | checkpoints and raw artifacts | pull artifacts to local at every phase boundary |
| Credential leak | account compromise | credentials are injected at run time only, never written to repo, volume, manifest, or logs |

## Code and protocol: private Git remote

`origin` → `github.com/ChaoticCooties/prospective-geometry`, **private**.

Verified on 2026-08-21, not merely asserted:

1. cloned fresh from the remote into a scratch directory;
2. clone is at commit `cc2823c` with all 47 tracked files;
3. `scripts/verify_freeze.py` reports **43 files match** against
   `manifest/freeze.json`;
4. the full suite — **64 tests** — passes from the clone alone.

A pod therefore reconstructs the exact frozen study from the remote plus the
model revision, with no dependency on this laptop.

## Artifacts: network volume

Status: **defined, not yet verified.** Verification needs a RunPod API key,
which is not present on this machine.

Design:

- one RunPod network volume, mounted at `/workspace/persist`, outliving every
  pod;
- the repository is cloned to `/workspace/prospective-geometry` on pod start,
  which is pod-local and disposable;
- every artifact that costs money to recompute is written under
  `/workspace/persist/<phase>/<block>/`, never to pod-local disk;
- every GCG update writes canonical suffix token IDs, the own-objective loss,
  and the update index, so a resumed search restarts from the last completed
  update rather than from scratch;
- artifacts are pulled to this machine at every phase boundary and their
  SHA-256 recorded in the phase report.

Verification steps to run once a key exists, before any GPU is billed:

1. create the volume, write a known file, destroy the pod, create a second pod,
   and confirm the file and its SHA-256 survive;
2. kill a pod mid-write and confirm the resume path picks up the last complete
   checkpoint rather than a truncated one;
3. confirm the reaper terminates a pod idle beyond five minutes.

Step 3 is already demonstrated. During the volume inspection on 2026-08-21 a
pod outlived its intended lifetime because the readback loop hung; `pods.py
reap --all` terminated it and `pods.py list` confirmed zero pods afterward.
Total unintended spend was about $0.02. The lesson is recorded in the operating
rules: every RunPod call runs backgrounded, because a foreground tool timeout
can kill the driver process before its teardown runs and leave a pod billing.

## Datacenter and GPU findings, 2026-08-21

The existing volume `tara_gcg` (50 GB) lives in **US-KS-2**, and a network
volume can only attach to pods in its own datacenter. That pins the choice of
GPU, and US-KS-2 is expensive:

| Datacenter | Cheapest GPU at or above 24 GB | Price |
| --- | --- | --- |
| US-KS-2 (where `tara_gcg` is) | RTX PRO 4500 Blackwell, 32 GB | $0.500/hr |
| US-TX-1 | RTX A6000, 48 GB | $0.330/hr |
| EU-SE-1 | RTX A6000, 48 GB | $0.330/hr |
| EU-CZ-1 | RTX 3090, 24 GB | $0.220/hr |

Reusing `tara_gcg` therefore costs 1.5x to 2.3x more per GPU hour than moving to
a fresh volume elsewhere, on top of the clean-room objection. The cheapest
listed GPU overall, the A5000 at $0.160/hr, showed no stock in any datacenter
at the time of checking.

Recommended: a fresh volume in **US-TX-1** with the **RTX A6000 at $0.330/hr**.
48 GB removes any out-of-memory risk for an 8B model with a GCG backward pass,
and the datacenter is unrelated to the archived work.

Account balance at freeze: **$21.37**, which is below the study's $25-30 target
band. Standing spend is $0.005/hr for the existing 50 GB volume.

## Credentials

Rule: no credential in remotes, logs, manifests, shell history, or persistent
volumes.

| Credential | Where it lives | How it reaches a pod |
| --- | --- | --- |
| Hugging Face token | `~/.cache/huggingface/token`, mode 600, outside the repo | injected as `HF_TOKEN` in the pod environment at start, never written to `/workspace/persist` |
| RunPod API key | not yet supplied | local environment only; never sent to a pod |
| GitHub | `gh` keyring on this machine | pod clones over HTTPS with a short-lived deploy token, or the repo is rsynced up |

`scripts/secret_scan.sh` scans the tracked tree for credential patterns and is
run before every push. It found nothing on the Phase 0 freeze.

The Hugging Face token used for this study was transmitted in plaintext through
a chat transcript, which is a durable log. It should be rotated at
`https://huggingface.co/settings/tokens` once the study no longer needs it.

## Remaining gate

Paid compute stays blocked until the network-volume verification above passes.
