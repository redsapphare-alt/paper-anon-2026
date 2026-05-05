# 2nd/ data quality report (2026-05-03)

## TL;DR

The 31 new result files in `2nd/` (Gemini cross-model + math/reasoning Tier-2)
have **systematic extraction bugs** that overstate accuracy. Stored
`is_correct` values cannot be trusted at face value. Independent re-extraction
disagrees with stored values on 10-92% of records depending on the file.

## Per-file verdict

| Bucket | Files | Status |
|---|---|---|
| ✅ Usable as-is | 1 | reasoning_gpt_4o_1agent (agree 99.4%) |
| ⚠️ Borderline (low effective signal but agreement OK) | 3 | medical_gemini_{2,3,4}agent (≈25%, agree>95%) |
| ❌ Stored numbers wrong (Claude math/reasoning, Gemini legal/logic/long_context, GPT math/reasoning N≥2) | 26 | Need re-extraction |
| ❌ Incomplete | 1 | medical_gemini_1agent (200/5000) |

## Root causes

1. **Substring matching**: Claude math/reasoning multi-agent files evaluate
   `is_correct` by checking if `correct_answer` appears anywhere in the
   response text. Long multi-step reasoning frequently contains the gold
   number as an intermediate calculation, producing false positives. The
   reported ~80% accuracy on Claude math is essentially noise.

2. **Refusal mistaken for answer**: Gemini was given the LegalBench
   yes-no task with a prompt that mentioned A/B/C/D options. Gemini refused
   ("the answer choices are missing — please provide them") and the substring
   "Relevant" inside the apology was scored as the gold label.

3. **Format mismatch**: Gemini logic/long_context use numeric ground truth
   (`gt=3`) but Gemini outputs letters (`Answer: D`). The pipeline never
   reconciled the two. Accuracy sits at random-chance (24-28%) for every
   `N`, confirming the comparison is essentially random.

4. **Broken extraction (heading match)**: Several files normalised the
   ground-truth letter to substring-search the response for the *option
   text* of that letter, causing arbitrary matches inside reasoning.

## What this means for the paper

The current paper (`paper/latex_v2/agent_count_neurips_2026.tex`) uses only
data from `results/merged/` (the original "trustable" cells). Those cells
were sanity-checked previously and are not affected by this report.

**Do not import 2nd/ accuracy numbers into the paper without re-extraction.**

## Recovery path (if we want to use 2nd/)

Apply `scripts/reextract_2nd.py`'s domain-aware extractors as the ground
truth, then recompute:

- per-question `is_correct` from the raw model text
- per-cell accuracy, paired McNemar, bootstrap CIs
- curves and ∆₁→₄

Caveats:
- For multi-agent Claude files, `predicted_answer` already stores the
  *aggregated* output (post-coordination). We only have the *aggregated*
  answer, not per-agent answers, so we cannot re-derive p (per-agent
  accuracy) for the decomposition framework. We can still report
  aggregate accuracy and ∆.
- The math-extraction is harder than letter-extraction; expect some
  irreducible disagreement (~5%) even with a clean extractor.

## Files emitted

- `analysis/reextract_2nd_report.json` — per-file disagreement counts
- `scripts/sanity_check_2nd.py` — first-pass sanity (blob/clean/near-random)
- `scripts/reextract_2nd.py` — independent re-extraction + agreement check
