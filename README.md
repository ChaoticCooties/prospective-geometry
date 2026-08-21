# Prospective token-reachable geometry

A clean-room, preregistered study asking whether fresh four-token hard suffixes
optimized toward two independently fitted and causally admitted internal safety
objectives reach reproducibly different token geometries under one matched
output-blind optimizer, and whether that geometry predicts held-out behavior.

Model: `meta-llama/Llama-3.1-8B-Instruct` at revision
`0e9e39f249a16976918f6564b8830bc894c89659`.

**Start with [PROTOCOL.md](PROTOCOL.md)** for the frozen design and
[CLAIMS.md](CLAIMS.md) for what prior work already occupies and what this study
may therefore claim.

## Status

Phase 0 is complete and frozen. No GPU has been provisioned, no model weights
have been loaded, and no model outcome exists.

| Gate | Result |
| --- | --- |
| Prompts overlapping the archive | 0 of 1,848 |
| Cross-role intent-identity pairs | 0 |
| Null simulation, complete pipeline false pass | 0.00000, CI95 [0, 0.00019] |
| Null simulation ceiling | 0.05 — accepted |
| Geometry power under a real effect | 0.766 |
| Tests | 64 passing |

## Clean-room rule

The archived Renegade experiments are read-only scientific history. Scientific
programs fail closed on any input that resolves inside the archive, checked two
ways so that moving the archive cannot silently disable the guard:

```python
from pgeom.firewall import guard_path
path = guard_path(user_supplied, role="prompt file")  # raises ArchiveAccessError
```

Permitted archive access is limited to reading `ARCHIVE_MANIFEST.md`, the
standalone corpus-overlap audit that emits hashes and counts only, and copying
generic infrastructure with source path, commit, and SHA-256 recorded. No
infrastructure has been imported so far.

## Layout

| Path | Contents |
| --- | --- |
| `PROTOCOL.md` | the preregistration: every frozen value and stop rule |
| `CLAIMS.md` | prior work, the Mody placebo scope, and the claim ceiling |
| `src/pgeom/` | firewall, provenance, tokenization, candidates, admission, geometry, null simulation |
| `prompts/` | builder, provenance, and the eight frozen role files |
| `audit/` | archive corpus-overlap audit, role audit, and their artifacts |
| `reports/` | null simulation and power report |
| `manifest/freeze.json` | machine-readable freeze with SHA-256 of every tracked file |
| `tests/` | 64 tests over the firewall, replay contract, gates, and prompt integrity |

## Reproducing Phase 0

```bash
python -m pytest tests/ -q                     # 64 tests
python prompts/build_prompts.py                # rebuild the eight roles
python audit/role_audit.py                     # role-versus-role audit
python -m pgeom.nullsim --n-sims 20000 --out reports/nullsim_report.json
python scripts/build_manifest.py               # regenerate and diff the manifest
```

The archive harvest is separate because it is the one program that reads archive
content:

```bash
python audit/corpus_overlap_audit.py harvest \
  --archive-root /Users/chaoticcooties/Desktop/renegade/archive \
  --out-dir audit/artifacts_full --include-git-history
python audit/corpus_overlap_audit.py check \
  --candidates prompts/roles/_all_prompts.jsonl \
  --hash-dir audit/artifacts_full \
  --out audit/artifacts_full/new_prompt_overlap.json --containment-frac 0.5
```

Set `PGEOM_ARCHIVE_ROOTS` if the archive lives elsewhere.

## What this study cannot claim

Not the first internal-target attack, prompt cube, interaction decomposition,
basin, manifold, mechanism, causal route, global landscape, or strategy knob.
[CLAIMS.md](CLAIMS.md) records who occupies each.
