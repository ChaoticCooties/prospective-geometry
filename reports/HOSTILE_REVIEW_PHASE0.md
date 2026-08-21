# Hostile review: Phase 0

Written against this study, not for it. Every item is a way the design could
produce a result that does not mean what the paper would say it means, or could
fail for a reason unrelated to the hypothesis. Nothing here is resolved by
argument; the ones that are not fixed are recorded as limits on the claim.

## Attacks that would land

**1. Block independence is thinner than "six independent blocks" suggests.**
The six search blocks come from two corpora, three strata each, not six
independent corpora. A corpus-level artifact — annotation style, question
framing, topic breadth — is shared by three blocks at a time. With N = 6 and a
required 6-of-6, three correlated blocks is a material fraction. *Status:
unfixed, because no third suitable ungated harmful corpus with a matched benign
counterpart was available without credentials. Recorded as a limitation; the
claim is scoped to "six blocks over two corpora", never "six independent
sources".*

**2. A large template family sits inside the prompt set.** 537 of 1,848
prompts (29%) share a leading four-word skeleton with at least four others, the
largest being "how can i use" at 84 prompts. Prompts inside a family are not
independent draws. *Status: unfixed by design. It is the reason blocks, not
prompts, are N, and it is reported in the protocol rather than smoothed away.*

**3. Layer 14 is asserted, not derived.** Package A's site is fixed a priori
from public literature. If layer 14 is a mediocre refusal readout on this exact
revision, admission fails and the study stops — and the paper would have
learned nothing about the hypothesis, only about layer 14. *Status: accepted
deliberately. Allowing a layer search would convert a preregistered site into a
selected one, which is the larger sin. The failure mode is disclosed in advance.*

**4. k = 8 is arbitrary.** Package B fixes eight heads because a small fixed k
was required, not because eight is indicated. A different k could admit where
eight fails, or vice versa. *Status: accepted. B is required to be one exact
object; any data-driven k would be a menu.*

**5. Admission can fail for lack of power, not lack of effect.** At a 22-point
effect against an 8-point placebo, simulated admission power is 0.104. A real,
moderate, genuinely causal objective will usually be rejected. A NO-GO is
therefore weak evidence against the objectives — it is mostly evidence that the
gate is severe. *Status: unfixed and intended. The report must say "did not
clear a severe gate", never "the objectives are not causal".*

**6. The null simulation only covers nulls I thought of.** Its generative model
is additive position effects plus Gaussian noise with a block-scale term. Real
suffix landscapes have epistasis the simulation does not produce. A false-pass
of 0.00000 is a statement about this model, not about reality. *Status:
unfixed, inherent. Mitigated by the statistic being *exactly* zero — not
approximately — whenever both objectives weight positions identically, which is
an algebraic property rather than a simulated one.*

**7. The archive audit cannot catch paraphrase.** It keeps only hashes, so it
tests exact text, normalized text, and 8-gram containment. A prompt that
restates an archived prompt with different wording and no shared 8-gram passes.
The content-word Jaccard test that would catch this was run *among the new
roles* only, because running it against the archive would require retaining
archive text, which the clean-room rule forbids. *Status: unfixed, and a genuine
gap. It is bounded by the fact that the new prompts come from corpora the
archive's markers do not name, and that only 5 of 1,848 prompts share even one
8-gram with the archive.*

**8. Controls are handed more selection freedom than the cross faces.**
Same-objective and control faces choose orientation and path on discovery; cross
faces have orientation fixed and choose only the path. This biases against the
hypothesis, so a pass is safe — but a real, weak effect could be masked by a
control that got a better draw from a larger search space. *Status: deliberate.
Asymmetry in the conservative direction is the correct error to make.*

**9. The behavior phase confounds corpus with harm.** Sealed harmful prompts
come from each block's own corpus; sealed benign prompts come from OR-Bench hard,
because neither HarmfulQA nor Do-Not-Answer has a benign counterpart. Any
harmful-versus-benign contrast in Phase 5 is therefore partly a corpus contrast.
*Status: unfixed. Mitigated by treating benign prompts as a utility-collapse
check at the level of a single condition, not as a discrimination test, and by
the fact that Phase 1's matched contrast — OR-Bench toxic against OR-Bench hard,
same category — is within-corpus, which is where matching actually matters.*

**10. The margin statistic reads only three interior vertices per path.** With
width four there are exactly three informative points on a monotone path. The
statistic is low-resolution and a single noisy vertex moves it. *Status:
inherent to width four, which is fixed by the research question. Mitigated by
averaging two cross faces per block and requiring all six blocks.*

**11. WITHDRAWN — the searches do not share storage with the previous study.**
Originally logged because artifacts were to live on the `tara_gcg` volume
alongside archived work. That plan proved unexecutable: US-KS-2 has no GPU
capacity, and a volume only attaches within its own datacenter. The study runs
on a dedicated volume in EU-RO-1, so this attack no longer lands. The original
text is kept below for the record.

**11 (original). The searches run on storage that holds the previous suffix study.**
Artifacts live under `/workspace/pgeom` on the `tara_gcg` volume, which also
holds archived Renegade work. `pgeom.firewall` fails closed on any path under
`/workspace` outside the study subdirectory, so this study's code cannot resolve
into the prior work — but containment is a guard on *our* code, not proof that
nothing was read, and it does not bind libraries, shell commands, or later
edits. For a paper whose selling point is being fully prospective, this is the
cheapest available attack on it, and it is the one item in this review I cannot
close. Recorded as deviation D1 in the protocol. *Status: accepted by the
operator after the alternative was declined twice. Disclosed rather than
mitigated away.*

## Attacks that would not land

**"You picked the containment threshold after seeing the overlap."** No. The
0.5 threshold is the default in `audit/corpus_overlap_audit.py` and was written
before the check ran. The five prompts with nonzero overlap were left in
precisely so that no threshold moved after the fact.

**"You removed the prompts that failed the intent audit."** No. The 25
cross-role intent pairs were fixed by adding a uniform deduplication rule to the
allocator and rebuilding all 1,848 prompts deterministically. No prompt was
individually deleted, and the change predates the freeze.

**"The geometry statistic drifts positive under path selection."** It does not.
Maximizing over all 24 monotone paths on a face where both objectives weight the
four positions identically returns exactly 0.0, verified as an equality in
`tests/test_geometry.py`, not as a tolerance.

**"Blocks that pass different conditions were combined into a 7-of-8."** They
cannot be. `evaluate_admission` implements one conjunction per block;
`test_different_conditions_cannot_be_satisfied_by_different_block_subsets`
constructs exactly that attack and asserts it yields zero passing blocks.

**"The candidate bank is a sorted-id prefix, so token id confounds everything."**
It is keyed-hash sampled, asserted to differ from the sorted prefix and to be
uniform across id quartiles in `tests/test_candidates.py`.

## What would make this paper stronger, at cost

- A third harmful corpus with a matched benign counterpart, giving six blocks
  over three corpora. Needs credentials for a gated dataset.
- A paraphrase-level archive audit, which needs a text-retaining audit process
  and an explicit exception to the clean-room rule.
- Width six or eight suffixes, giving 2⁶ or 2⁸ vertices and a higher-resolution
  margin. Multiplies search and scoring cost well past the $30 cap.
