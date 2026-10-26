# bytebrain — agent entry point

- Current line of work (2026-10-22): `research/pycode/README.md` — own byte-LM for Python guided by measured
  "truth" (mirror), GPU syntax checker in the generation loop, population-graded self-training.
  Read it before touching `research/pycode/` or `train.py`; update its section 7 (status/next) at the end of a session.
- Cross-project handbook: `../META/handbook-llm/README.md`.

## Request discipline (pycode line) — read before anything else
1. First read research/pycode/USER-REQUESTS.md, then README section 7.
2. Any idea or request from Leonardo becomes a ledger row, his words quoted, in the SAME turn.
3. Every work turn starts with "Working on #N". OPEN items run oldest-first unless he reorders.
4. Before his OPEN items are built, the only allowed work is blocking bug fixes, a smoke test
   and review of HIS item's code. Your own experiment must name the # it serves and fit
   in 15 min / one GPU slot.
5. "Next step" proposals of yours go into the ledger as CLAUDE-IDEA rows, below his items.
6. DONE = path + command + output line for exactly what he asked. A lookup, oracle or rule
   is not a "generator/writer/model": mark PARTIAL and name the missing piece.
7. His idea failed? Check own bugs (padding, ctx overflow, judge, arm budget), adapt, rerun,
   mark FAILED-ADAPT. Never drop without his quoted OK.
8. Comparisons: same split, decoding, k, selector, training budget, params per arm.
   Otherwise write "not comparable". Never put selected/pass@k next to plain pass@1.
9. Reviews, controls and errata are gates on HIS work, not a reason to postpone it.
10. End of session: update ledger statuses and README section 7. Report OPEN items first.

## Pre-run gates (from review #44, research/pycode/REVIEW-44.md) — check BEFORE every GPU slot
21 of 26 incidents on 10-21..26 were predicted by a lesson already on disk. The machine gates (VRAM, disk,
threads) were checked; the inference gates were not. Write the answers into the run script header or the ledger.
1. Baseline row: constant answer, `return arg0`, one-cue rule. A number that does not beat them is not a result.
2. Cue audit of LLM-made data: count cue words per class; one-class cues become shortcuts (main => formula).
3. Where does gold enter? grep the accept/score path. A loop that needs gold to accept a fix is a ceiling, not a loop.
4. Disjointness by provenance (who wrote it, from what). A set used to pick a variant is a dev set from then on.
5. Positive control through the real CLI path; the smoke must reproduce a known number that is not a floor/ceiling.
6. Train/eval identity: same format, ctx, max_len, padding; longest gold < generation cap.
7. Prediction arithmetic: the predicted number must be reachable from the stated mechanism and the ceiling.
8. Equal budgets per arm; best value at a grid edge => extend the grid; a control with an empty log => no verdict.
9. Bug found => grep the same bug class across the whole line, not only this file.
10. setsid + per-arm result files; the verdict goes into the ledger in the same turn the run ends.
11. Before DONE, name the part that does the work (lookup? rule? selected seed?).
12. One GPU process at a time; latency only with a runtime whose output matches HF greedy (parity checked).
