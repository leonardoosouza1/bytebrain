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
| I7 | HIS #37 formulation: Qwen+Sonnet judges point at the error → retrain OUR planner/writer (not Qwen) on len5/6 + I6 interpreter failures | G1, I6 | TODO |

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
