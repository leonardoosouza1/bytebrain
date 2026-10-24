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
