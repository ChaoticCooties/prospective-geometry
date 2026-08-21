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

Status: **verified on 2026-08-22.** Verification needs a RunPod API key,
which is not present on this machine.

Decision: reusing `tara_gcg` was proposed and then withdrawn as unexecutable —
US-KS-2 has no GPU capacity, and a volume only attaches within its own
datacenter. See withdrawn deviation D1. The study uses a dedicated volume.

Design:

- the dedicated `pgeom-clean` network volume (30 GB, EU-RO-1), mounted at
  `/workspace`, outliving every pod;
- this study writes **only** under `/workspace/pgeom`; `pgeom.firewall` raises
  on any other path under `/workspace`;
- the study never writes an `ARCHIVE_MANIFEST.md` onto the volume, which would
  make its own subdirectory fail closed;
- the repository is cloned to `/workspace/prospective-geometry` on pod start,
  which is pod-local and disposable;
- every artifact that costs money to recompute is written under
  `/workspace/persist/<phase>/<block>/`, never to pod-local disk;
- every GCG update writes canonical suffix token IDs, the own-objective loss,
  and the update index, so a resumed search restarts from the last completed
  update rather than from scratch;
- artifacts are pulled to this machine at every phase boundary and their
  SHA-256 recorded in the phase report.

Verified results, volume `pgeom-clean` (`5l9knlrkk1`, 30 GB, EU-RO-1), on an
RTX PRO 4500 Blackwell with 32,623 MiB:

1. **Survives pod destruction.** A 4 MB random payload written under
   `/workspace/pgeom/durability` on pod A read back byte-identical on pod B
   after pod A was destroyed. SHA-256
   `ec1429f7e79a0637d81888f9acece5595e73e47f38305f9b6d9316ee67a2be63` on both
   sides.
2. **Truncation is detectable.** A complete checkpoint line parses (1 valid
   line); a deliberately truncated one does not (0 valid lines). Resume can
   therefore take the last *complete* checkpoint rather than a half-written one.
3. **Nothing left billing.** `pods_after` was 0, confirmed by a separate
   `pods.py list`.

The volume's top level contains only `pgeom`, confirming it is dedicated to this
study and holds nothing else.

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

Capacity, not preference, settled this. Probing every storage-capable
datacenter found GPUs in only three of twenty:

| Datacenter | GPUs with capacity |
| --- | --- |
| EU-RO-1 | RTX PRO 4500 Blackwell (32 GB), RTX 4090 |
| US-CA-2 | RTX 4090 |
| US-IL-1 | RTX 4090 |

The study runs in **EU-RO-1** on the **RTX PRO 4500 Blackwell, 32 GB**, chosen
for VRAM headroom on the GCG backward pass at the same listed price as a 24 GB
4090. Phase 2 measures the real hourly rate from billing rather than trusting
the listed price, and forecasts from that.

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
