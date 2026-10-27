# Master queue — 10-27 afternoon (ledger #56)

Leonardo is out for ~4 h. The order is: run everything he asked, then integrate. Errors get noted here, not hidden. Each line has a status:
- TODO — not started
- RUN — running
- DONE <number>
- FAIL <why> (+ adapt)

GPU: one job at a time via gpuq/runner.sh, which picks pending/*.sh in name order. CPU: ≤ 6 threads in total, nice 5, load < 12.

## Stage 1 — finish his open items (GPU queue order)
| # | item | where | status |
|---|---|---|---|
| G0 | #7 selftrain on Qwen = MY variant (CLAUDE-IDEA, #57), finishing; then #12 addr2d | q7c (pid 8691) | RUN |
| G1 | #13 liquid planner cells + #54A tau-bank stress arm | gpuq 10_liquid13.sh | RUN (agent building arm) |
| G2 | #51 extract Python from Coder-3B vs random slice vs native 1.5B/0.5B | gpuq 20_py_extract.sh | queued |
| G3 | #49 bigger understander: s5 recipe on Coder-1.5B / 3B, 3 seeds | gpuq 25_understander49.sh | RUN (agent building) |
| G4 | #24 guess-and-derive from I/O examples | gpuq 30_guess_derive24.sh | review running |
| G5 | #53 relief frozen config on GPU seeds | gpuq 40_relief53.sh | queued |
| G6 | #54B relief-guided repair in the generation loop | 50_relief_repair54.sh or CPU | RUN (agent) |
| C1 | #8 stress gate, cycle wait 3600 steps | CPU | RUN |

## Stage 2 — integration ("juntar", his #55/#56)
| # | item | depends | status |
|---|---|---|---|
| I1 | Belt with the BIGGER understander: best of #49 / #51 plugged into understander → planner → writer → bwrap; A / B / v2pool | G2, G3 | TODO |
| I2 | Mixture of understanders (MoE): 0.5B s5 + 1.5B + 3B (+ extracted). Router = agreement vote / check_spec pass / token confidence; equal-call budget vs the best single model | G3 | TODO |
| I3 | Planner with the tau-bank stress cell inside the real belt (if G1 shows a gain) | G1 | TODO |
| I4 | Repair loop in the belt: tests fail → relief/MIX line → regenerate unit (G6) + saci-code early signal (#46) | G6 | TODO |
| I5 | sciops wired into multifn behind a flag → pt-BR "calculadora científica" end to end | q7c done | TODO |
| I6 | #55 target "mini-interpretador" (or stock CLI): hierarchical spec JSON (#33) + statement-level writer — the missing piece | build | TODO |
| I7 (folded into #58 eco58) | HIS #37 formulation: Qwen+Sonnet judges point at the error → retrain OUR planner/writer (not Qwen) on len5/6 + I6 interpreter failures | G1, I6 | TODO |

## Stage 3 — remaining PARTIAL rows (#2 #4 #10 #16 #35 #41)
To do after Stage 2, or whenever the CPU/GPU is idle.

## Log
- 12:32 G3 #49: wrapper built, CPU smoke ok, step-0 loss 2.6078 = s5. Total 14.6 GPU-h. Split: seed 1 (25_*) now, seeds 2-3 (80_*) at the end. Opus review running; script in hold/ until GO.
- 12:34 G4 #24: Opus review NO-GO (grid edges in every tune cell -> no verdict; derive overcharged ×S in budget; 3/81 cross-length leak). Fixer resumed with 4 fixes. AMENDMENT C: primary budget = real compute (3*NOP per soft step).
- 12:37 I2 #56/#25: synth/moe_understander56.py (registered 12:30). CPU route on cached raws (moe56/route_cpu.json): PC 100% per-request parity, s5 175/179 149/174 441/593. M6 (6x 0.5B LoRA) vote/agree: A 98.3/98.9 (s5 97.8), B 93.7/93.7 (best single sfG 89.1, +4.6 CI [1.1,8.0]), v2pool 91.7/90.2 (best single sfO 93.9, vote -2.2 CI [-4.2,-0.2]); s5-SC@6 v2pool 75.9; oracle@6 (bound) 98.9/99.4/97.3. GPU part (conf router, tuned cascade, q3b v2pool/DEV, s5-SC A/B) queued gpuq/pending/70_moe56.sh; re-queue after #49 adapters land to add q15/q30.
- 12:38 I2 MoE (CPU offline routing, unreviewed): vote of 6 x 0.5B experts B 93.7 vs best single 89.1 (+4.6 CI 1.1..8.0); v2pool 91.7 vs 93.9 (-2.2); A 98.3 vs 97.8. Positive control s5 exact. Contamination check of sf* experts pending in Opus review; 70_moe56.sh in hold.
- 12:45 I2 MoE review GO, numbers recomputed exactly.
  - Correction: on B the best single is tied (sfG = sfO = 155/174). Vote M6 163/174 is +4.6 vs sfG [1.1, 8.0] but vs sfO [-1.1, 10.3]: NOT clear of zero. agree M6 gets the same 163 at 2.45 calls.
  - v2pool is not clean for any arm (all 593 golds are in the s5 train), so it is a dev set.
  - The vote is not an equal-budget comparison until s5-SC@6 lands (GPU).
  - 70_moe56.sh is in pending. I fixed waitgpu: plain fuser /dev/kfd also counted GPU-hidden CPU jobs.
- 12:51 G0 #7 (my variant) PASS on dev-ish v2pool only; A drops ~9 pts for T/O/Ts; B flat. Not a clean gain.
- 12:59 G1 #54A: stress arm built. Own init bug B11 found+fixed (gate began closed -> no learning). Old arms bit-identical; e2e control exact. Risk: gate stays open (~CfC). Opus review running; 10_liquid13 in hold.
- 13:00 G3 #49 review GO. Reviewer found the wrapper CPU shim corrupted the model (step-0 loss 6.11 vs 2.61), fixed. 0.5B control now runs FIRST with a gate (loss ±.02 and A ≥ .95), else 1.5B/3B are skipped. Queued 25_* (~5.4 h, seed 1) + 80_* (~9.8 h, seeds 2-3). 3B VRAM est. 78%.
- 13:05 G1 #54A review GO. Amendment C: if the gate stays open (≥4/7 seeds collapse), the verdict is tagged 'MECHANISM NOT ENGAGED' and not killed. Startup guard added. Back in pending. Commit 0bfca24 locks the code and predictions.
- 13:13 G6 #54B relief-guided repair DONE (CPU, my recompute from v2/rows.json, R=8, 126 cells/arm). d32_ter: none .631, random .646, relief .649, mix .648, mf .649, full resample .638, oracle .668. d32: none .890, random .900, relief .901, oracle .906. Relief ≈ random (+0.2). Oracle line only +3.7 over none: the writer, not localization, is the bottleneck. Adapt in I4: aim repair at writer quality (resample with confidence/temperature, bigger writer), not at finding the line.
- 13:13 #53 adapt (CPU, Opus GO, agent-recomputed): pure water arms lose to relief; relief + 2×gap-pond water (W2 = line-junction surprisal) +1.7 pp [+0.7, +2.6] on EVAL seeds 1-4, +2.6 on unseen seeds 3-4, +1.2 [-0.1, 2.5] on seed 6. Gains on foreign_token/rename. Frozen config relief53_adapt/frozen.json; GPU confirm = 40_relief53.
- 13:13 G4 #24 fixer DONE: held-out leak measured (len3 119/200 test problems equivalent to a trained skeleton); primary = no-equivalent subset (len2 n=173, len3 n=81); grid widened, g_polish + r_derive_gd arms; GPU smoke gate. Opus re-review next, then pending.
- 13:15 I6 interp55 scripts moved pending -> hold (60/61) until the Opus review lands. I7 (his #37/#57: Qwen+Sonnet judges -> retrain OUR writer/planner on their natural failures, arms J/R/N/Q-only/O, 3 seeds) and I5 (sciops wired via new files, pt-BR sci calculator e2e + Coder-3B direct baseline) builders launched (Opus); #24 Opus re-review launched. q12 addr2d s0 done 13:14, s1 running.
- 13:21 I6 interp55 built (family 18,432 variants, 176 unseen eval units, 160 held-out programs; floors: reference 1.0, nn 0.034/0.081, mutants 0.188/0). Author's Opus review: 2 blockers fixed (hidden repetition penalty in Qwen gen config; VRAM probe at real size). Scripts 60 (~5-7 h) / 61 (~10-12 h) in hold until my independent Opus review lands. Predictions 13:06: flat .03-.15 per fn; 0.5B LoRA .15-.40; 3B 2-shot .20-.45.
- 13:40 I7 #37/#57 built (synth/judge_retrain57.py, hold/65): our planner/writer fail on len5/6 -> Qwen-3B judge -> Sonnet final (rule, serial, cap) -> gold-free accepted fixes -> rows routed to blamed module; arms N/J/R/Q/O, 3 paired seeds. Smoke: positive control test_a .8703 exact; Sonnet stage+location 7/7. Blocker found by me: N (plain continue-training) drops test_a .870 -> .784 after 40 steps -> recipe damages the writer; sent back for root cause + 2nd Opus review. Also: flag rule matched Sonnet on stage blame in smoke (judges may add only location+fix).
- 13:40 #24 re-review NO-GO (tune ~11 h/seed vs 4 h timeout, no partial save; cross-length leak file stale). Fixer resumed.
- 13:43 I5 sci calculator built (synth/multifn_sci.py, sci_belt56.py): old ops keep indices, 5 sci ops appended; writers W0/WE64/WF64/WF128 + understander US vs UC + Coder-3B direct baseline. CPU positive control A .9777 / B .8563 exact. Author-spawned Opus review GO, 8 fixes applied. Queued pending/55_sci_belt.sh (~5.3 h). Pred: belt >=75-80% on held-out pt-BR sci requests; Coder-3B direct 40-75%.
- 14:03 G4 #24 re-review GO (grid 345 cells, resume, timeouts from measured cell; len3 primary n=78, len2 n=173). Queued as pending/85_guess_derive24.sh (last: up to ~19.5 h). Pre-run: no stale tune partials.
- 14:35 I7 N-drop root cause: lr 1e-3 fresh AdamW (3.3x base final lr) flips ternary thresholds + coarse test_a. Fix for all arms: lr 1e-4 + Guard57 (train-only regression guard, keep last passing snapshot). N-B test_a s0/s5/s6 +4.2/+0.3/0.0. Finding: N alone lifts the planner on len5/6 (+30/+10/+62.5 pp) -> base planner was under-trained, headroom for J is mostly the writer. 2nd Opus review GO-with-fixes applied. Queued pending/65_judge_retrain57.sh (~2-4.5 h).
- 14:41 #58 (user): ecosystem = everything trains together, TOP PRIORITY. 65_judge_retrain57 moved to hold; eco58 builder launched; eco58 script goes FIRST in the GPU queue (05_) after #12.
- 14:50 #12 addr2d (HIS row x column address) DONE 2 seeds: test_a pass@1 .945/.9625 vs addrctl .56/.785, addrseg .584/.9075, addrrk .791/.9025; test_num addr2d .9625/.8762. Prediction was .02-.35 -> his idea beat it by far. Limit: test_b (unseen comps) addr2d .08/.07 vs addrrk .38/.40; test_len4 .04. Seed 2 queued pending/07_q12_seed2.sh (~40 min).
- 15:49 HANDOFF (usage limit). GPU runner keeps going on its own: 10_liquid13 (running) -> 07_q12_seed2 -> 20 -> 25 -> 40 -> 55 -> 70 -> 80 -> 85. In hold: 05_eco58 (TOP PRIORITY #58; builder agent still smoke-testing -> needs my diff read + review, then move to pending; it must run before the rest), 60/61 interp55 (waiting on independent Opus review), 65_judge_retrain57 (folded into eco58). Next session: read this log, check eco58 output in eco58/, queue it, then write verdicts as runs land.
