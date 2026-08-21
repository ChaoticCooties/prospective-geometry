# Prompt provenance and licensing

All 1,848 prompts come from four public, ungated corpora that download without
credentials. Nothing is authored by hand, and nothing comes from the archive.

## Sources

| Key | Hugging Face dataset | Config | License | Citation |
| --- | --- | --- | --- | --- |
| `harmfulqa` | `declare-lab/HarmfulQA` | — | Apache-2.0 | Bhardwaj and Poria, *Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment*, 2023 |
| `do_not_answer` | `LibrAI/do-not-answer` | — | Apache-2.0 | Wang et al., *Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs*, Findings of EACL 2024 |
| `orbench_toxic` | `bench-llms/or-bench` | `or-bench-toxic` | CC BY 4.0 | Cui et al., *OR-Bench: An Over-Refusal Benchmark for Large Language Models*, 2024 |
| `orbench_hard` | `bench-llms/or-bench` | `or-bench-hard-1k` | CC BY 4.0 | Cui et al., *OR-Bench: An Over-Refusal Benchmark for Large Language Models*, 2024 |
| `dolly` | `databricks/databricks-dolly-15k` | — | CC BY-SA 3.0 | Conover et al., *Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM*, Databricks, 2023 |

Attribution obligations: the two OR-Bench configs are CC BY 4.0 and require
attribution. Dolly is CC BY-SA 3.0, requires attribution, and share-alike
applies to adaptations of the dataset itself. Prompt text is not redistributed
beyond the derived role files in this repository, which retain per-row source
attribution in every record.

## Selection

Selection key: `pgeom-prompt-selection-v1`.

Within each stratum, candidate prompts are ordered by
`SHA-256(selection_key : salt : text)` and the first N are taken. The salt names
the role and block, so different roles draw different prompts from the same
stratum without any overlap. Selection never consults a model, a score, or a
result.

Filters applied before selection: 20–320 characters, no embedded newline.

Two exclusion rules run during allocation, both uniform and both applied at
selection time rather than by removing prompts afterward:

1. a prompt whose normalized text matches an already-allocated prompt is skipped;
2. a prompt whose content-word Jaccard against any already-allocated prompt
   reaches 0.60 is skipped, so no two roles hold the same request in different
   words.

## Record schema

Each line of `roles/<role>.jsonl` is one prompt:

    id           16-hex, SHA-256 of role|block|arm|source|source_row
    role         one of the eight roles
    block        block label, or "-" where the role is unblocked
    arm          role-specific arm, e.g. nom_harmful, val_benign
    kind         harmful | benign_hard | benign_utility
    source       source key from the table above
    source_row   row index in the source split
    stratum      source-native category, topic, or risk area
    license      SPDX-style license of the source
    text         the prompt

## Audit

See `audit/artifacts_full/new_prompt_overlap.json` for the archive check and
`audit/artifacts_full/role_audit.json` for the role-versus-role check. Headline
numbers are in [PROTOCOL.md](../PROTOCOL.md#prompt-audit-results).
