"""Ledger #13: a LIQUID neuron as the planner's predictor over several inputs at once (registered 2026-10-27 10:00,
before any run of this file; time from `date`).

His words (#13): "ter grupos de bytes... vinculados/amarrados por uma outra tabela de enderecamento, como se tivessemos
modelos pequenos que unidos fazem a magica... de acordo com as entradas de texto/bytes qual e a previsao? e sabe quem
poderia lidar com varias entradas ao mesmo tempo? neuronio liquido, se ele preve mercado financeiro... ele pode tentar
prever o que vai, ele ja viu muitas funcoes".

What is built (only the decoder cell of the address planner changes; everything else is blockaddr.Net):
  byte groups      emb(name bytes) -> 2 local convs -> learned boundary gate g_i            (blockaddr, unchanged)
  address table    a_i = cumsum(g): each byte group's address; decoder counter c_t          (blockaddr, unchanged)
  table entry      r_t = sum_i softmax(-(a_i - c_t)^2 / s) h_i: the group addressed now    (blockaddr, unchanged)
  PREDICTOR (new)  a liquid cell that takes the inputs as SEPARATE streams, each through its own small head
                   ("modelos pequenos que unidos"): stream 1 = r_t (byte-group entry read from the address table),
                   stream 2 = emb(previous block symbol, the op token); each head sees [stream, x_{t-1}] -> U units
                   (LeCun tanh, as ncps), the heads' outputs are concatenated into the backbone B_t.
  arms
    gru   blockaddr.Net("addr", gate_bias=-4) unchanged: GRUCell over cat[r_t, sym]. CONTROL. Same init, same math;
          only the forward/greedy are re-written to put tensors on the model's device (BA.Net creates its zeros on
          CPU). `parity` checks it equals BA.Net bit for bit.
    cfc   Hasani et al. 2022 (Nature MI) closed-form continuous-time cell, eq. (10):
              x_t = sigma(-f(B) * dt) * g(B) + (1 - sigma(-f(B) * dt)) * h(B),  g, h = tanh(linear), f = linear
          dt = 1 per decoder step. ONE time head f (bridge_cpu used ta(b)*1 + tb(b): two redundant matrices, see
          own-bug list). No extra leak on top (bridge_cpu added s += (c - s) / tau).
    ltc   liquid time-constant neuron (Hasani et al. 2021), stepped with its EXACT solution for an input held
          constant over dt (closed form, no ODE solver):
              dx/dt = -(w_tau + D(B)) x + D(B) A(B)  ->  x_t = x_ss + (x_{t-1} - x_ss) exp(-(w_tau + D) dt),
              x_ss = D A / (w_tau + D);  D = softplus(linear) >= 0 = total synaptic conductance (input-dependent
              time constant: the "liquid" part; bias init +1, B10), A = tanh(linear) = conductance-weighted reversal potential,
              w_tau = softplus(param) = leak, init as a log-spaced tau bank 1..128 (ties #8). D -> 0 gives tau -> 1/w_tau
              (hold), D large gives a fast write: the cell carries x_{t-1} additively, which pure CfC does not.
    lstm  LSTMCell over cat[r_t, sym] (extra recurrent control), hidden sized to match parameters.
  params  every arm within 5% of the GRU planner (149 220 in the synth2 world), asserted.

Worlds (same data, splits, steps, lr schedule, batch 48, greedy decoding, grader as the original GRU planner):
  synth2    blockaddr.main path: bridge_cpu.Data, 1500 steps, splits val/test_a/test_num/len3..len6, synth2.grade
            (strict bwrap). The recorded GRU seeds 0-6 at gate bias -4 live in $R/blockaddr/blockaddr_addr_s*_gb-4.json.
  dispatch  belt planner #1 (multifn.train_planner path, e2e_pipeline `train`), 2000 steps, MF.Data, INTERLEAVED plans;
            graded by multifn.evaluate with an ORACLE writer (block_text), so pass1 isolates the planner.
  expr      belt planner #2 (compose.train_planner path), 2000 steps, CP.Data, compose.evaluate, oracle writer.

Own-bug check of the old CfC code (synth/bridge_cpu.py CfC, read 10-27; that file is NOT edited):
  B1 redundant time head: gate = sigmoid(ta(b) + tb(b)) is ncps' sigmoid(t_a * ts + t_b) with ts = 1 fixed, so ta and
     tb collapse into ONE linear map; 16 512 of the CfC arm's 98 944 "equal" non-embedding params (17%) were dead
     weight, and that gate's effective lr was 2x. Fixed here: one f head, the params moved into the heads.
  B2 extra leak: s += (c - s) * rate is not Hasani's CfC; with the fixed bank rate = 1/tau a group writes only 1/tau of
     each new input (tau=128: 0.8%), so 7 of 8 groups were near-silent at K=0 (write/hold coupled; adapt 2 named it).
  B3 no carry in pure CfC: cfc_tau1 (rate 1 = the true CfC) passes state only through the silu backbone (no additive
     path): a vanilla-RNN memory, test_a .006 at K=0 vs GRU .993. That is a property of CfC without its mixed-memory
     option, not of liquid neurons; the `ltc` arm keeps the additive carry with an input-dependent time constant.
  B4 lr: the 10-22 grid put the CfC best at its top edge (1e-2) while GRU's best was 3e-3: the CfC was under-tuned;
     the #45 rerun fixed the grid. Here every arm runs at the planner's registered lr 3e-3 (primary) AND at 1e-2
     (sensitivity, all arms, never used to select).
  B5 gate lr 10x on a matrix (#45 adapt 3): not present here (no special param groups at all).
  B6 silu backbone instead of ncps' LeCun tanh 1.7159 tanh(0.666 x): not a bug, but here the heads use LeCun tanh.
  B7 init: PyTorch default uniform for every linear (as the GRU); no saturation at init: f ~ 0 -> gate .5, D bias +1
     -> softplus 1.31 (B10). Gate saturation is measured at eval (`sat` fields), not assumed.
  B8 normalisation: none in any arm (the GRU planner has none). Not added (rule 8).
  B10 (MY bug, found by the CPU smoke 10-27 ~10:25, fixed before any GPU run): ltc with conductance init
     D bias -2 (softplus .13) never trained (seed 0 1500 steps: loss 1.0, test_a .009; the boundary gate collapsed,
     address range ~30 = every byte a boundary). Cause: x_ss = D/(w_tau + D) * A couples the write AMPLITUDE to
     the conductance, so at init fast units wrote ~1/10 of A (the old B2 write/hold coupling again, in my code).
     400-step train loss, seed 0 (train split only, no held-out split looked at): D bias -2 -> 1.20, 0 -> .010,
     +1 -> .006, +2 -> .006. Chosen +1 (D = 1.31: every unit writes >= 57% amplitude, keep <= .27 at init; holding
     is still reachable by D -> 0, where keep -> exp(-w_tau)). The init line is the only change.
  B9 device: BA.Net.forward/greedy allocate zeros/full on CPU -> the original cannot run on GPU. Re-written here with
     device=..., same math (`parity` subcommand: bit-exact on CPU).

PREDICTION (written 2026-10-27 10:00, before any run; GPU full run, 7 seeds 0-6, lr 3e-3, mean over seeds):
  P0 positive control: gru synth2 seed 0 on CPU, 2 threads, reproduces the recorded eval exactly (len6 .645);
     gru expr seed 0 reproduces the belt meta calls8 plan_exact .9333 (thread count may move it: then within 1 ex).
  P1 synth2 len6: gru mean .70-.85 (recorded .76, SD .31, min .13; GPU numerics differ per seed);
     cfc .70 (.45-.90); ltc .75 (.50-.95); lstm within 10 pp of gru. test_a and test_num >= .98 for every arm.
  P2 mechanism / seed variance (his "predicts what comes next, it has seen many functions"): the liquid cells get the
     table entry r_t as a separate driven input, so their state is re-driven by what the address points at each step
     instead of integrating its own history; prediction: SD over seeds of len6 for the best liquid arm <= .20
     (gru .31) and its worst seed >= .40 (gru .13). Confidence ~55%.
  P3 belt worlds (oracle writer): dispatch test_comb, len4, len8 >= .95 for gru and both liquid arms (gru is at
     ceiling there: not discriminative, reported only); expr calls8 gru .93-1.0, cfc and ltc >= .85.
  Verdict rule (synth2 len6 and len5, lr 3e-3, 7 seeds; margin must exceed 3 pp AND 2x the SE of the difference):
     PASS for "a liquid predictor in the planner works at least as well": best of {cfc, ltc} mean >= gru - 3 pp.
     PASS+ "and is more stable": additionally SD <= 0.65 x gru SD and min seed >= gru min.
     KILL (liquid predictor loses here): cfc AND ltc both <= gru - 10 pp mean on len6 -> own-bug check, then adapt.
  AMENDMENT A (2026-10-27 10:16, before any GPU run; from the independent Opus review of 10-27): the rule above mixed a
     non-inferiority test with a superiority margin and took the max of two noisy arms. Replaced by (computed by
     `report`, not by hand): primary = synth2 len6, lr 3e-3, 7 seeds; each of cfc and ltc tested SEPARATELY vs gru,
     paired by seed, Bonferroni over the 2 arms (two-sided 97.5% paired-t CI on mean(X - gru), t_.9875,6 = 2.969).
       BETTER: CI low > 0.   PASS (non-inferior): CI low > -0.10 (3 pp is unreachable at n=7 with gru SD .31; the
       margin is widened to 10 pp HERE, before data).   KILL: CI high < -0.10 for BOTH arms, and only if neither
       liquid arm's best lr on len6 sits at the grid edge 1e-2 (grid is now 1e-3,3e-3,1e-2 for every arm).
       Otherwise INCONCLUSIVE.   PASS+ (descriptive): SD <= .65 x gru SD and min seed >= gru min.
     len5 and the other lrs are reported, not used for the verdict. report asserts 7 seeds per primary arm.
     Known: the seed-0 CPU smoke (cfc .995 vs gru .645) is one of gru's bad seeds; it is not evidence.
  Baselines (gate 1): constant plan (most common train plan) and "<eos> only" are scored in the report.

AMENDMENT B (2026-10-27 12:13 from `date`, written before any run of the `stress` arm; ledger #54(A)). His words: put
  the tau bank ("banco T", slow liquid leak, written only under "stress"/surprise) INSIDE the planner, the measured
  bottleneck (gold plan -> writer 100% on len3-6; e2e len6 .756 +- .31 over 7 planner seeds; planner plan_exact len6
  .565/1/.93/.975/1/.513/.038, writer7seed/summary.json).
  arm stress  StressCell = the cfc arm's cell (same Heads, same eq. 10 candidate c_t) plus stressgate.py's surprise-gated
          leak on a tau bank: H units in 8 groups, tau_g = 1, 2, 4 .. 128 (log bank 1..128, as ltc/stressgate);
              x_t = x_{t-1} + (c_t - x_{t-1}) * rate_g
              group tau=1: rate 1 (plain CfC write: the fast path, == the cfc arm's update for those units);
              slow groups: rate_g = sigmoid(a_g * z_t + b_g), init see B11 (stressgate's a=1, b=logit(1/tau) failed)
              z_t = log(mean((P x_{t-1}.detach() - R e_t)^2) + 1e-3).detach()   one scalar per sequence and step
              e_t = cat[r_t, emb(prev sym)].detach() (the cell's two input streams), R a fixed random 2d x 16 buffer
              (seed 1234, not trained), P = Linear(H, 16). --pred-detach semantics: the aux loss 0.1 * mean mse over
              the real decoder inputs (y_in != pad) trains ONLY P; the gate scalars a, b get gradient only through
              the CE loss (z is detached). a, b: no weight decay (as stressgate's wg); everything else unchanged.
          params matched to gru like the other arms (U searched), and asserted within 5% of the cfc arm.
  e2e (added for every synth2 run, all arms): planner(name) -> plan -> REAL byte writer (bytewriter d32 fp, writer
      seeds 0,1,2, trained once on CPU with BW.train_writer exactly as writer7seed.py and cached as text tables in
      OUT/writers/) -> source -> bwrap strict grade, via writer7seed.score (cached path whose equality to
      bytewriter.evaluate was the writer7seed gate). Splits test_a, test_num, len3..len6. Fields: e2e[d32_wK][split]
      = {exec, pass1, plan_exact, text_exact}; e2e_pass1_mean = mean over the 3 writer seeds.
      Positive control (CPU, P0b): gru synth2 seed 0 e2e pass1 must equal writer7seed d32_w0/1/2 e2e["0"] exactly
      (test_num .86/.58/.7171, len5 .9, len6 .645) and plan_exact = block_exact (.565 len6).
  Verdict B (computed by `report` -> verdict_b; synth2, lr 3e-3, 7 seeds 0-6, paired by seed):
      PRIMARY metric = plan_exact (block_exact) len6. Comparisons added: stress vs gru, stress vs cfc; Bonferroni
      over these 2 -> two-sided 97.5% paired-t CI, t_.9875,6 = 2.969 (as Amendment A). Same calls as A:
      BETTER CI low > 0; PASS CI low > -0.10; LOSES CI high < -0.10; else INCONCLUSIVE; PASS+ descriptive
      (SD <= .65 x ref SD and min seed >= ref min). KILL of "stress in the planner" = LOSES vs gru AND stress best lr
      on len6 not at the grid edge 1e-2. Secondary (same CI, reported, not used for the verdict): plan_exact len5,
      e2e_pass1_mean len5/len6. Amendment A (pass1, cfc/ltc vs gru) is unchanged and still reported.
  PREDICTION B (12:13): mechanism: in decoding, emb(prev sym) is the net's own last output, so after training P
      predicts it from x_{t-1} -> z low -> holders freeze on every predictable step and write when the address read
      r_t brings something new (block boundary). That keeps a stable memory of the early plan, but the len6 failures
      of gru are address/count drift past the train lengths, which a frozen memory does not fix; the fast path is only
      1/8 of the units (cfc has all H). Numbers (GPU, lr 3e-3, 7 seeds): stress plan_exact len6 mean .72 (.45-.92),
      SD .25 (gru .37), min seed .15 (gru .04); len5 .87. vs gru: INCONCLUSIVE or PASS (P(BETTER) ~20%); vs cfc:
      mean diff -.05, INCONCLUSIVE (P(BETTER) ~10%). e2e_pass1_mean len6 = plan_exact len6 + .02..+.06 (wrong plans
      that still pass, as gru .756 vs .717), test_num ~.72 for every arm (writer digit copy, planner-independent).
      Gate: on len6 greedy, slow groups' mean rate < .10 and the tau>=32 groups have >30% of steps with rate < .02
      (freeze on predictable input); learned a_g > 0 for most slow groups (write on surprise, as stressgate).
  B11 (MY bug, CPU smoke 10-27 ~12:25, fixed 12:42 before any GPU run; the prediction above is unchanged): with
      stressgate's init (a_g = 1, b_g = logit(1/tau_g)) the arm never trained: synth2 seed 0, 1500 steps, CE loss .985
      (gru .0003), every held-out split ~0 (aux mse 3.65, z median 1.07: the decoder input r_t is never well
      predicted, so z stays ~+1 and slow groups wrote ~1/tau of each input = the B2/B10 write/hold coupling again).
      Train-loss-only diagnosis (500 steps, seed 0, train split): b = logit(1/tau) 1.086; b = 0 .008; b = 2 .0031;
      all groups rate 1 .0021. Fix keeps the tau bank but on the HOLD side: a_g, b_g init so that rate = .88 at z = 1
      (surprised) and rate = 1/tau_g at z = log 1e-3 (input perfectly predicted): a_g = (2 - logit(1/tau_g)) / 7.91,
      b_g = 2 - a_g. 500-step train loss .0034. The init lines are the only change. Held-out numbers seen so far are
      those of the broken init only. [Correction by the 10-27 13:01 review: NOT true. smoke_cpu_b/stress_tiny_b11
      (12:54, after B11) evaluated the held-out splits: seed 0, 250 steps, lr 3e-3, len6 block_exact .9925, e2e
      .9925; gate rate .92-.93 on every slow group, z median 1.37. Seed 0 is a bad gru seed (.565) and a good cfc
      seed (.995), so it does not move Prediction B; it is declared here so nobody counts it as unseen.]

AMENDMENT C (2026-10-27 13:01 from `date`, independent review, before any full stress run; code changes are
  stress-only + report-only, the other arms are bit-identical).
  C1 what B11 changes in the hypothesis: stressgate's init sat on the HOLD side (rate ~1/tau at z ~ 0, write needs
     high z). B11 sits on the WRITE side: for every z the init can produce (z >= log 1e-3) the slow rate is in
     [1/tau, .88]; at the z actually seen (~1.3) it is ~.92. The default is therefore "leaky CfC", and his mechanism
     (hold, write only under surprise) exists only if training CLOSES the slow gates. A result of the stress arm
     with the gate still open is a CfC result, not a test of #54A.
  C2 measured, not assumed (stress_stats on the longest held-out split): slow-group rate over ALL greedy steps (as
     registered) and over the steps before the first <eos> (new: post-eos steps are trivially predictable and would
     make the gate look closed); FORCED-OPEN ablation = same trained net, slow groups forced to rate 1 at eval, its
     plan_exact on that split vs the unforced one.
  C3 verdict_b adds, per seed (lr 3e-3): prereg_gate_met = Prediction B's gate (slow mean rate < .10 AND each of
     the tau >= 32 groups has rate < .02 on > 30 % of steps); collapsed = pre-eos slow mean rate > .80 AND
     |plan_exact - plan_exact_forced_open| <= .01 on len6. If collapsed in >= 4 of 7 seeds the verdict is tagged
     "MECHANISM NOT ENGAGED" (the comparison stands as stated but is not evidence for or against his tau bank) and
     KILL is not issued: own-bug check + adaptation (rule 7) instead.

AMENDMENT D (2026-10-27 17:00 from `date`, ledger #54(A), new arm `stress2`; the 5 old arms are bit-identical:
  fixed-seed init + teacher-forced logits + greedy + 20 train steps hashed before/after the edit, all equal).
  Why: the full run (full/report13.json, verdict_amendment_B.mechanism) measured the `stress` slow rate at .87-.90
  pre-eos in 7/7 seeds (collapsed 5/7): it is a CfC, not his "T so large the data stays there and only changes under
  stress". stress2 makes the hold the DEFAULT and keeps learning alive:
  arm stress2  StressCell2. State = [fast H/2 | slow H/2 in 4 groups, tau 16, 64, 256, 1024]; same Heads + eq. 10
          candidate c_t as cfc. Fast units: x_f = c_f (cfc write, rate 1: learning never depends on the slow gate,
          which was the B11 failure with only 1/8 fast units). Slow group g:
              s += (c_s - s) * (1/tau_g + (1 - 1/tau_g) * open_g)      (closed -> leak 1/tau_g = hold)
              open_g = 1[logit_g > 0] (HARD, in training and eval: no train/eval mismatch), straight-through
                       gradient via sigmoid(logit_g): a closed gate still receives gradient (B11 avoided)
              logit_g = softplus(w1_g) (surprise_t - 1) + softplus(w2_g) (log(dis_g + 1e-3) - log .1) + gb_g
              surprise_t = -log p_{t-1}(prev symbol) under the planner's own previous output (log V at START):
                       in greedy decoding prev is its own argmax -> no gold at eval; dis_g = mean_g (c_s - s)^2
                       (what arrives contradicts what is held). Both detached (signals, not paths).
              w >= 0 by softplus: opening is MONOTONE in stress; training tunes sensitivity/threshold only, it
              cannot learn a content write gate (that would be an LSTM gate, not his stress). Init w = 1, gb = -2:
              closed when the input was predicted (surprise ~0) unless dis is ~e^2 x .1.
          Rate penalty: RATE_W * relu(open_rate - RATE_TARGET), open_rate = straight-through mean over real decoder
              steps and slow groups, RATE_TARGET .2, RATE_W 1. Gate scalars w1, w2, gb: no weight decay (as stress).
          Choice vs alternatives: hard-concrete / Bernoulli-ST sample in training but threshold at eval (mismatch,
              and the eval store would be written by a different rule than the trained one); a soft gate with a bias
              init near 0 is the B11 gradient-starved state. Deterministic hard gate + ST keeps write/hold binary as
              he described and the backward alive.
          params: U matched to gru (149 788 vs 149 220, +0.38%; cfc 149 776), asserted within 2% of gru and cfc.
  Instruments (stress2_stats, greedy on len6[:256]): slow_rate_all, slow_rate_pre_eos (incl. first eos),
      slow_rate_pre_eos_t>=1 (t0 always reads START = maximal surprise: the "birth" write), per-group rates, and for
      t>=1 pre-eos steps the correlation of the open fraction with post-hoc gold proxies (never fed to the net):
      wrong_now / wrong_prev (greedy token != gold), first_wrong_now, block_start_now/prev (op:/fin: emitted), and
      with surprise. Ablations on len3..len6 plan_exact: zero_slow (store zeroed every step), closed_after_t0 (only
      the START write, then hold: isolates the stress rewrites), force_open (slow rate 1).
  Seen before this prediction (CPU smoke, 2 threads, 1500 steps; declared, not a verdict): seed 0 lr 3e-3 len6
      plan_exact .4175 (cfc s0 .995, gru s0 .565 at the same lr on CPU), seed 0 lr 1e-3 .9875, seed 1 lr 3e-3 .99;
      slow_rate_all .050-.056, pre_eos .079-.091, t>=1 .0003-.010; zero_slow -> 0.0 on every length (all 3 runs);
      force_open len6 .01-.11; gate params moved < .15 from init.
  PREDICTION D (GPU, synth2, 7 seeds 0-6, 1500 steps, lrs 1e-3/3e-3/1e-2, plan_exact len6 mean; ref numbers = the
      full run: gru .9925/.7936/.6696, cfc .9993/.9296/.8654, stress .9971/.9264/.8411):
    D1 gate: slow_rate_pre_eos .06-.12 in 7/7 seeds at every lr (< .3 always, the requirement); t>=1 < .03: in
       practice a store written at birth (START) and then held, rarely rewritten. corr(open, wrong_prev) > 0 in
       >= 5/7 seeds at lr 3e-3 (open steps are uncertain steps); corr with block_start ~0 (|r| < .1).
    D2 accuracy: lr 1e-3 stress2 .97 (.90-1.0), INCONCLUSIVE or PASS vs all three (ceiling); lr 3e-3 stress2 .85
       (.65-.95), SD .15, below cfc/stress by ~.08 (half the units write per step), above gru mean; vs gru
       INCONCLUSIVE, vs cfc INCONCLUSIVE (P(LOSES) ~25%), vs stress INCONCLUSIVE; lr 1e-2 stress2 .75 (.5-.9).
       Confidence ~50% on the 3e-3 interval.
    D3 mechanism (his test): zero_slow hurts EVERY length to ~0 (the net is built around the birth snapshot), so
       "hurts len5/6 more than len3/4" in absolute drop is NOT discriminative there (floor); predicted true in <= 3/7
       seeds (drops tie at ~1.0). closed_after_t0: |drop| <= .01 on every length in >= 5/7 seeds = the stress
       rewrites after t0 do no work; if instead closed_after_t0 drops len5/6 by > .05 more than len3/4 in >= 4/7
       seeds, the stress-gated rewrites carry long-function state (his mechanism engaged beyond the birth write).
    Verdict (report54, per lr, paired by seed, Bonferroni over 3 comparisons, t_.99167,6 = 3.287, margin .10, same
       BETTER/PASS/LOSES/INCONCLUSIVE calls as Amendment A). LOSES vs cfc at 3e-3 with D3 "rewrites do no work" =
       the tau bank is a snapshot here, not a stress memory -> own-bug check + adapt (rule 7), not a kill.

AMENDMENT E (2026-10-27 17:13 from `date`, ledger #54(A), registered BEFORE any training on the new world; the 6 old
  arms gru/cfc/ltc/lstm/stress/stress2 are bit-identical: bitid.py hashes (init, logits, greedy, 20 train steps) equal
  before/after, scratchpad bitid54e_before.json == bitid54e_after.json).
  Why: on synth2 every fact the planner needs is in the input at step 0, so a mid-sequence rewrite is never needed
  (stress2 smoke: open at t0, t>=1 rate .0006; only-t0-write == normal). His example is a fact that CHANGES: "def sum"
  whose body has one sign flipped. World `amend` (synth/amend54.py, docstring = full spec): item 0 states an operator
  (f<v>), x<v> items later CORRECT it (half of them to the sign partner add<->sub ...), `ret` items must output the
  CURRENT operator, n<d> fillers are copied. The first `ret` after a correction sits D items later (train D 1-16;
  test_long D 17-32); before the correction the old value must be emitted. Splits val/test_iid (train dist),
  test_pos (amends only at item index % 4 == 0, never in train), test_long (D 17-32), test_multi (3-4 amends, up to 86
  items vs train <= 60), test_hold (control: no amend, D 1-32). Steps 2000 for every arm (amend54.STEPS), batch 48, same
  lr schedule/clip/AdamW as every world. Params (amend vocab 21): gru 146 904; cfc 147 460, ltc 146 996, lstm 147 412,
  stress 147 232, stress2 147 472, stress3 147 232 (all within 0.4%).
  arm stress3 (review M2, 10-27: at greedy eval stress2's surprise is its own uncertainty, never "I was wrong"; trained
      gb -2.1 / w .9 need surprise >~ 2, measured greedy max .81, so stress2 may never open after t0 on ANY world):
      StressCell3 = stress2 + input surprise: P = Linear(H, 16) predicts a fixed random projection of the address read
      r_t from x_{t-1} (aux mse, AUX_W .1, trains only P); zin = log(mse_t + 1e-3) - log(running mean_{k<t} mse + 1e-3)
      (causal, per sequence, detached, 0 at t0); logit += softplus(w3) zin, softplus(w3) init 2 (open when the input is
      ~e^1.5 = 4.5x less predictable than this sequence's average, other terms neutral). Rate penalty mask excludes t0
      (stress2 keeps its mask). Still monotone in stress (w3 >= 0): no content gate. stress2 runs UNCHANGED as well.
  Baselines (amend54.py baselines, same scorer; oracle on every non-`ret` position; measured 17:1x before this text,
      data only): primary test_long amend_first_use: const .116, never_update 0, recent4/8_else_first 0, oracle 1.
      test_iid amend_first_use: const .14, never_update 0, recent4 .24, recent8 .49; hold_long 1.0 for never_update.
  PRIMARY = test_long amend_first_use (first `ret` after the correction, D 17-32), lr 3e-3, 7 seeds 0-6, GPU.
  PREDICTION E (means over 7 seeds, lr 3e-3; confidence ~40% on each interval):
    E1 accuracy primary: gru .75 (.40-.95), lstm .80 (.45-.97), cfc .30 (.10-.60: no additive carry, B3), ltc .50,
       stress .40, stress2 .30 (.10-.60: slow store closed after t0 -> holds the BIRTH value, the amend must ride on the
       fast cfc half), stress3 .70 (.35-.95). test_iid amend_long (D 10-16): gru .95, cfc .60, stress2 .55, stress3 .90.
       test_hold hold_long: stress2 .90 and stress3 .90 (birth write + hold), cfc .50, gru .95.
    E2 gate (cell_stats, test_iid/test_long greedy): stress2 amend_window open rate < .10 and ratio amend/elsewhere
       < 2 or undefined in >= 4/7 seeds (not engaged). stress3: amend_window open rate >= .50, elsewhere <= .10,
       ratio >= 5 in >= 5/7 seeds; amend_event_hit >= .70.
    E3 ablations (stress3, test_long/test_hold): freeze_after_t0 (slow rate exactly 0 after t0) drops amend_first_use
       by >= .30 while hold_long drops <= .10 (the store is used AND rewritten); zero_slow drops both by >= .30;
       oracle_gate (open only at t0, amend, amend+1) within .05 of normal; force_open drops hold_long by >= .20.
  Verdict E (amend54.py report -> report54e.json; per variant stress2 and stress3, paired by seed vs gru and vs cfc,
    Bonferroni over those 2, two-sided 97.5% paired-t CI, t_.9875,6 = 2.969, margin .10, calls as Amendment A):
    SHOWN = BETTER (CI low > 0) vs gru AND vs cfc on the primary at lr 3e-3.
    Otherwise KILL RULE E: "his mechanism not shown on this world", EXCEPT (checked in this order):
      WORLD AT CEILING (gru and cfc both >= .95 on the primary: not discriminative, no kill);
      MECHANISM NOT ENGAGED (t>=1 open rate < .03 on test_iid AND closed_after_t0 |drop| <= .01 on amend_first_use in
      >= 4/7 seeds: no kill, own-bug check + adapt, rule 7). Best lr (by primary and by val amend_long) at a grid edge
      is flagged (gate 8: extend the grid before a final call). corr(open, amend window) only if >= 20 open events.
  Seen before this text: data stats and baselines only; no net has been trained on `amend`. The CPU smoke after this
  text (1 seed, gru/cfc/stress2 [+stress3], short budget) is a learnability check, not evidence.
AMENDMENT E1 (2026-10-27 17:24 from `date`, after the E CPU smoke, before any GPU run and before any net saw the new
  splits). CPU smoke, seed 0, lr 3e-3, 2000 steps, 2 threads (smoke_cpu/amend/*.json; not a verdict):
    test_long amend_first_use: gru 1.0, cfc 1.0, stress3 .986, stress2 0.0 (stress2 never trained: loss 2.12, filler
    copy .15; open .19). By rule E the world is AT CEILING on its primary (gru, cfc >= .95): not discriminative.
    stress3 mechanism, seed 0: open at the amend step .92 vs .024 elsewhere (ratio 20, event hit .99); freeze_after_t0
    amend_first_use .998 -> .016 (iid) while hold_long stays 1.0; zero_slow -> .09/0; oracle_gate .996/1.0;
    force_open hold_long 1.0 -> .38. (E2/E3 predicted this; one seed, CPU.)
  Added (amend54.py SPECS, generated AFTER train: every E split incl. train is byte-identical, md5 of repr(sets)
  57bc5785 before and after): test_xlong (n_amend 1, pre-gap 1-8, D 64-160 after the amend: 4-10x the longest train
  gap) and test_xhold (no amend, D 64-160). MAX_OUT 91 -> 180 (greedy only; training unchanged). Ablations and gate
  stats also on test_xlong/test_xhold.
  VERDICT ORDER (registered now): E decides; ONLY if E is at ceiling (gru and cfc >= .95 on test_long at lr 3e-3, 7
    seeds) does E1 decide, with the SAME rule on test_xlong amend_first_use (SHOWN = BETTER vs gru AND cfc, t 2.969,
    margin .10; WORLD AT CEILING / MECHANISM NOT ENGAGED exceptions as E).
  PREDICTION E1 (GPU, lr 3e-3, 7 seeds, test_xlong amend_first_use mean; confidence ~35%): gru .60 (.20-1.0),
    cfc .50 (.10-1.0), lstm .65, ltc .40, stress .40, stress2 .10 (training failure seen in the smoke, expected in
    >= 3/7 seeds at 3e-3), stress3 .80 (.40-1.0; the value must survive in the tau 256/1024 groups: the tau 16/64
    groups leak it within the gap). test_xhold hold_long: stress3 .85, gru .70, cfc .60. stress3 SHOWN on E1: ~30%.
  Own-bug fixed during the E smoke (amend54.py, before any GPU run): oracle_gate wrapper reused the previous chunk's
  input-surprise baseline at t0 (shape error, crashed: no number produced); report's grid-edge flag compared an lr to
  mean values (would have mis-flagged; no report had been run).
  Seen AFTER E1 was written (CPU, seed 0, smoke_cpu_e1/; declared so nobody counts them as unseen, not a verdict):
    test_xlong amend_first_use: gru .904, cfc 1.0, stress2 (lr 1e-3) .628, stress3 .574; test_xhold hold_long: gru .898,
    cfc 1.0, stress2 .47, stress3 .664. stress2 trains at lr 1e-3 (test_long .984; its lr 3e-3 seed-0 failure is
    lr-specific, as its weak synth2 seed 0 at 3e-3) but opens at amend only 34% of events (fast half does the rewrite;
    closed_after_t0 .628 -> .528). stress3 xlong failures are HOLD decay, not missed rewrites: event hit .98, xhold .66;
    the closed-gate leak 1/tau (tau 16..1024) replaces the tau 16/64 groups within a 64-160 gap.

AMENDMENT F (2026-10-27 17:43 from `date`; ledger #54A, coordinator request after the E/E1 smoke; registered BEFORE any
  net was trained on world amendk or with arm stress4; seen before this text: amendk data stats + baselines only).
  arm stress4 (his literal formulation: "T tao grande que o dado fica la e so muda se houver um estresse"): StressCell4
      = stress3 with tau = infinity when closed: slow group rate = open_g (hard ST gate, as stress3), i.e. closed ->
      the slow store is copied exactly, open -> full rewrite. Same params as stress3 (147 232 amend; asserted <= 2% of
      gru and cfc in cmd_run). Rate penalty / t0 mask / input-surprise term / aux loss identical to stress3.
  world amendk (amendk54.py; K held slots a.. in birth order, K ~ U[4, 6]; `x<s><v>` corrects ONE slot, half of the
      time to its sign partner = "only one sign changed"; `r<s>` asks one named slot; 0-2 uses of OTHER slots inside
      each gap (2 in the long tests) so the last value in the stream is not the answer). Splits val / test_iid (train
      dist, D 1-16), test_long (D 17-32), test_xlong (D 64-160), test_hold (D 1-32), test_xhold (D 64-160),
      test_multi (4-5 amends), train 20000; random.Random(55). STEPS 2000 (same as amend), batch 48, same schedule.
      Baselines (same scorer, oracle on non-use items), test_xlong amend_first_use: const .148, never_update 0,
      recent4/8_else_birth 0, last_seen_any (one-slot memory of a correct stream) .126, oracle 1.0.
      Escalation (registered now): if the CPU smoke (seed 0, lr 3e-3) gives cfc >= .95 on the primary, world amendk8
      (K ~ U[6, 8]) is smoked next; the grid tried is reported. Only the world whose smoke has gru AND cfc < .95 on
      the primary enters q54e.sh.
  PRIMARY F: test_xlong amend_first_use (D 64-160 after the correction of one of K slots).
  Verdict F (amendk54.py report -> report54f_<world>.json; per variant stress2/stress3/stress4, paired by seed vs gru
      and vs cfc, lr 3e-3, 7 seeds, t 2.969, margin .10): SHOWN = BETTER vs gru AND vs cfc on the primary; else
      WORLD AT CEILING (gru and cfc >= .95), MECHANISM NOT ENGAGED (as E), else NOT SHOWN (kill rule, as E). No
      fallback metric. On world amend, stress4 is judged by the E/E1 rule as a third variant.
  PREDICTIONS F (GPU, lr 3e-3, 7 seeds, means; confidence ~35%):
    F1 stress4 on amend: nothing fades, so test_xlong amend_first_use ~= test_long amend_first_use (|diff| <= .05) and
       test_xhold hold_long ~= test_hold hold_long (|diff| <= .05); levels .90 (.60-1.0); stress3 drops by >= .25
       from test_long to test_xlong (the E1 smoke mechanism). freeze_after_t0 == closed_after_t0 for stress4 (both
       rate 0) on every split (sanity, exact).
    F2 amendk test_xlong amend_first_use: gru .45 (.15-.85), cfc .55 (.20-.95), lstm .45, ltc .35, stress .35,
       stress2 .30, stress3 .35, stress4 .75 (.40-1.0). test_xlong hold_after_amend: stress4 .80, gru .45, cfc .55.
       test_long amend_first_use: gru .80, cfc .85, stress4 .85. stress4 SHOWN on F: ~35%.
    F3 stress4 mechanism (amendk test_xlong): amend event hit >= .80, elsewhere open <= .10; force_open drops
       test_xhold hold_long by >= .30; freeze_after_t0 drops amend_first_use to <= .20.
  The CPU smoke after this text (1 seed, gru/cfc/stress3/stress4) is a learnability / ceiling check, not evidence.
  Seen AFTER F was written (18:06 from `date`; CPU, seed 0, lr 3e-3, 2 threads; smoke_cpu_f*/; not a verdict):
    amendk (2000 steps) test_xlong amend_first_use: gru .824, cfc .106, stress3 .626, stress4 .790 -> gru and cfc
    < .95: amendk enters q54e.sh, amendk8 not needed (grid tried: amendk only). All arms undertrained at 2000 steps
    (test_iid seq_exact .02-.07; hold_long .18-.50). Learnability: gru at 6000 steps (smoke_cpu_f6k, not the
    registered budget) test_iid seq_exact .53, test_xlong amend_first_use .952, hold_after_amend .756, hold_long .858.
    The registered budget stays 2000 steps (equal for every arm); at 6000 steps gru would sit at the ceiling of the
    primary, so F measures the 2000-step regime.
    stress4 mechanism on amendk: amend event hit .99, elsewhere open .021; freeze_after_t0 == closed_after_t0 exactly
    (F1 sanity holds); freeze drops amend_first_use .79 -> .108.
    stress4 on amend (F1): test_long amend_first_use 1.0, test_xlong .478, test_xhold hold_long .448 -> F1 as stated
    FAILS on seed 0. Decomposition: on sequences whose filler copies are all right (address healthy) test_xlong
    amend_first_use is .970 (n 203; stress3 .708, n 404), but stress4's address drifts on long inputs (xlong filler
    .814, all-filler-ok .406 vs stress3 .993/.808). So the value no longer fades (his claim, conditional), but the
    exact-hold state derails the address counter; own-bug check pending (rule 7).

AMENDMENT G (2026-10-27 18:24 from `date`; review NO-GO of F; registered BEFORE any net was trained on world amendg;
  seen before this text: amendg data stats + baselines only). F's world `amendk` and PRIMARY F are SUPERSEDED (B1:
  the first use after an amend always asked the amended slot, so the slot-blind rule last_x_else_birth scored 1.0 on
  amend_first_use; F smoke numbers above stay as history, not evidence for G).
  world amendg (amendk54.py docstring; K ~ U[4, 6]; amendg8 K ~ U[6, 8] = escalation only): every amend segment asks
      a PAIR (amended slot a, other slot o) in random order (first query = a with p .5, train and test); distractor
      uses avoid a and the first-asked slot. random.Random(56).
  PRIMARY G: test_xlong post_amend_all = per amend event, the amended-slot use AND the other-slot use all correct
      (D 33-64, held-out GAP; lengths inside the train range: train p95 97 items, 0% of post-amend uses beyond it;
      25% of train sequences carry 4-6 amends for that, review M1). CO-PRIMARY: test_xlong post_amend_all_addr_ok
      (events whose segment fillers are all right). Reported per arm: post_amend_use/{amended|other},
      amend_first_use, filler_acc, eos_ok. Length extrapolation is SECONDARY only: test_xxlong / test_xxhold (D
      64-160, 24% of uses beyond the train max length).
  Baselines (same scorer), test_xlong post_amend_all: const .002, never_update 0, last_seen_any .014,
      last_x_else_birth .014 (its amend_first_use 1.0 = the B1 shortcut, other-slot uses .139), recent4/8_else_birth
      0, oracle 1.0. test_iid: recent8 .336, recent4 .141, last_x .060.
  BUDGET G (review M2; chosen on VAL only, declared now): gru seed 0 lr 3e-3, CPU, at 2000 and at 4000 steps (two
      full cosine schedules); a script prints only the val split. STEPS = 4000 if val post_amend_all(4000) -
      val post_amend_all(2000) >= .05, else 2000. Equal for every arm. 6000+ not probed: per step GPU ~= CPU here
      (launch-bound decoder), and a 7-seed grid above 4000 steps does not fit the 6 h cap; if 4000 is picked and its
      val still rises, the plateau is NOT verified (said in the report). CPU smokes after the pick run at STEPS for
      every arm whose projected CPU wall is < 15 min; an arm that does not fit is smoked at 2000 and labelled so.
  Verdict G (amendk54.py report -> report54g_<world>.json; per variant stress2/stress3/stress4, paired by seed vs gru
      and vs cfc, lr 3e-3, 7 seeds, margin .10, FAMILY-WISE t = t_{1-.05/12, 6} = 3.863 over 3 variants x 2 refs
      (review m2; the same t now also applies to the E/E1 rule on world amend, replacing 2.969: a declared change)):
      SHOWN = BETTER vs gru AND vs cfc on the primary AND the co-primary not LOSES vs either; else WORLD AT CEILING
      (gru and cfc >= .95 on the primary), MECHANISM NOT ENGAGED (review m1: hard open rate at t >= the births < .03
      AND |drop| of closed_after_births <= .01 on the primary, in >= 4/7 seeds), else NOT SHOWN (kill rule, as E).
  PREDICTIONS G (GPU, lr 3e-3, 7 seeds, means, at STEPS; confidence ~30%):
    G1 test_xlong post_amend_all: gru .45 (.15-.80), cfc .30 (.05-.70), lstm .45, stress2 .30, stress3 .40,
       stress4 .55 (.20-.90). post_amend_use/other >= post_amend_use/amended - .10 for every arm (the other slot is
       older but untouched). test_long post_amend_all >= test_xlong + .10 for gru/cfc/stress3; stress4 within .10.
    G2 stress4 SHOWN on G: ~25%. stress3 SHOWN: ~10%. stress2: ~5%.
    G3 mechanism (stress4, test_xlong): open rate at amend items >= .80, elsewhere after births <= .10;
       closed_after_births drops post_amend_all by >= .30; force_open drops test_xhold hold_long by >= .30.
    G4 secondary xxlong: every arm drops >= .15 from test_xlong (length extrapolation), stress4 by the most
       (F smoke: its address counter drifts on long inputs).
  The CPU smoke after this text (1 seed) is a learnability / ceiling check, not evidence.
  Seen AFTER G was written (18:38 from `date`; CPU, seed 0, lr 3e-3, 2 threads; not a verdict):
    BUDGET pick (val only): gru val post_amend_all 2000 steps .097, 4000 steps .111 (gain .013 < .05) -> STEPS 2000.
    Val use_acc .361/.380, final train loss .257/.233 (both runs stuck on the loss plateau F's gru left between 2000
    and 4000 steps of a 6000-step schedule). Disclosure: the run's stdout line also printed test seq_exact of the
    4000 probe (iid .037, xlong .004), seen while reading the val pick.
    gru smoke at STEPS (= the 2000 probe): test_xlong post_amend_all .014 = the slot-blind baseline last_x_else_birth
    exactly (amended .724, other .243; filler 1.0, eos 1.0); test_iid post_amend_all .119. gru learned the
    "last correction" shortcut, not slot binding: the world sits at the FLOOR at 2000 steps. Added after this (a
    no-kill label only, declared here before any other arm's result was read): WORLD AT FLOOR = gru and cfc means
    <= best slot-blind baseline + .05 (.064) on the primary -> not discriminative, no kill (checked after CEILING).
    Report-code fixes (19:01, after ALL seed-0 smokes below were read; found by a synthetic 7-seed dry run of the
    report, not by any GPU result): (i) a variant BETTER than gru and cfc but itself <= .064 is "NOT SHOWN ... slot-
    blind floor; no kill" (no SHOWN at the floor); (ii) co-primary undefined (no address-healthy event) counts as 0
    for a reference arm and as NO VERDICT for the variant, and a variant NO VERDICT on the co-primary blocks SHOWN.
    Smokes (2000 steps) test_xlong post_amend_all / filler_acc / final loss: gru .014/1.0/.257, lstm .020/1.0/.273,
    stress2 .036/1.0/.270, stress3 .012/1.0/.308, cfc .000/.111/2.48, stress4 .010/.039/2.45 (cfc and stress4 never
    left the initial plateau: no address, no eos). Every arm <= the floor: at 2000 steps world amendg is NOT
    discriminative; the grid as registered is expected to return WORLD AT FLOOR on G (P2 `amend` stays informative).

Usage:
  liquid13.py parity                                   # BA.Net == this gru arm, bit-exact on CPU
  liquid13.py run --world synth2 --arm cfc --seeds 0,1 --lr 3e-3 --out DIR [--steps N --device cuda --no-e2e]
  (arms gru cfc ltc lstm stress stress2; synth2 runs also score e2e through the real d32 writer, Amendment B)
  liquid13.py report54 --out DIR --ref liquid13/full      # Amendment D: stress2 vs gru/cfc/stress
  liquid13.py run --world amend --arm stress3 ...; amend54.py baselines|report --out DIR   # Amendment E
  liquid13.py run --world amendk --arm stress4 ...; amendk54.py baselines|report --world amendk --out DIR   # F
  liquid13.py report --out DIR
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import blockaddr as BA  # noqa: E402
import bridge_cpu as BC  # noqa: E402
import compose as CP  # noqa: E402
import multifn as MF  # noqa: E402
import amend54 as AM  # noqa: E402  (Amendment E world `amend`)
import amendk54 as AK  # noqa: E402  (Amendment G worlds `amendg`, `amendg8`; F `amendk` superseded)

ARMS = ["gru", "cfc", "ltc", "lstm", "stress", "stress2", "stress3", "stress4"]
AMW = {"amend": AM, **{w: AK for w in AK.CONFIGS}}           # Amendment E/F worlds -> module
AUX_W = 0.1                 # stress: weight of the next-input prediction loss (stressgate.AUX_W)
RATE_TARGET, RATE_W = 0.2, 1.0   # stress2: hinge penalty RATE_W * relu(open rate - RATE_TARGET), open rate straight-through
WORLDS = {"synth2": 1500, "dispatch": 2000, "expr": 2000}       # default steps = the original planner's
NAMES = {"synth2": ["val", "test_a", "test_num", "len3", "len4", "len5", "len6"],
         "dispatch": ["test_comb", "len4", "len5", "len6", "len7", "len8", "basic", "full"],
         "expr": ["test_same", "calls4", "calls5", "calls6", "calls8"]}


def lecun_tanh(x):
    return 1.7159 * torch.tanh(0.666 * x)


class Heads(nn.Module):
    """one small net per input stream ("modelos pequenos"), each sees [stream, x_prev]; outputs concatenated."""

    def __init__(self, dims, H, U):
        super().__init__()
        self.hs = nn.ModuleList(nn.Linear(d + H, U) for d in dims)

    def forward(self, streams, x):
        return torch.cat([lecun_tanh(h(torch.cat([s, x], 1))) for h, s in zip(self.hs, streams)], 1)


class CfCCell(nn.Module):
    """Hasani 2022 eq. (10): x_t = sig(-f dt) g + (1 - sig(-f dt)) h; one time head; no extra leak."""

    def __init__(self, dims, H, U):
        super().__init__()
        self.hidden_size = H
        self.heads = Heads(dims, H, U)
        k = U * len(dims)
        self.f, self.g, self.h = nn.Linear(k, H), nn.Linear(k, H), nn.Linear(k, H)
        self.stat = None

    def forward(self, streams, x, dt=1.0):
        b = self.heads(streams, x)
        gate = torch.sigmoid(-self.f(b) * dt)
        if self.stat is not None:
            self.stat.append(gate.detach())
        return gate * torch.tanh(self.g(b)) + (1 - gate) * torch.tanh(self.h(b))


class LTCCell(nn.Module):
    """LTC neuron stepped with its exact solution for input held over dt; input-dependent time constant."""

    def __init__(self, dims, H, U, taus=(1, 128)):
        super().__init__()
        self.hidden_size = H
        self.heads = Heads(dims, H, U)
        k = U * len(dims)
        self.D, self.A = nn.Linear(k, H), nn.Linear(k, H)
        nn.init.constant_(self.D.bias, 1.0)       # B10: -2 broke training (see header); +1 -> D 1.31 at init
        tau = torch.logspace(math.log10(taus[0]), math.log10(taus[1]), H)
        self.w_raw = nn.Parameter(torch.log(torch.expm1(1.0 / tau)))      # softplus(w_raw) = 1/tau
        self.stat = None

    def forward(self, streams, x, dt=1.0):
        b = self.heads(streams, x)
        lam = F.softplus(self.w_raw) + F.softplus(self.D(b))
        keep = torch.exp(-lam * dt)
        if self.stat is not None:
            self.stat.append(keep.detach())
        x_ss = (F.softplus(self.D(b)) / lam) * torch.tanh(self.A(b))
        return x_ss + (x - x_ss) * keep


class StressCell(nn.Module):
    """Amendment B: the cfc candidate (eq. 10) written into a tau bank through a surprise-gated leak (stressgate.py
    cfc_stressgate, --pred-detach). Group tau=1 writes at rate 1; slow groups at sigmoid(a z + b)."""

    def __init__(self, dims, H, U, taus=(1, 128), groups=8, proj=16):
        super().__init__()
        assert H % groups == 0
        self.hidden_size, self.g = H, groups
        self.heads = Heads(dims, H, U)
        k = U * len(dims)
        self.f, self.g_, self.h = nn.Linear(k, H), nn.Linear(k, H), nn.Linear(k, H)
        tau = torch.logspace(math.log10(taus[0]), math.log10(taus[1]), groups)
        self.register_buffer("fast", tau <= 1.0 + 1e-6)
        # B11 (own bug, 10-27 12:40): b = logit(1/tau), a = 1 (stressgate's init) made the slow groups write only
        # ~1/tau at the init surprise (z ~ +1): the B2/B10 write/hold coupling; train loss stuck at ~1.0. Init now
        # puts the bank on the HOLD side: rate = sigmoid(2) = .88 at z_hi = 1 (surprised) and exactly 1/tau when the
        # input is perfectly predicted (z_lo = log 1e-3): a slow leak of time constant tau, written under stress.
        z_hi, z_lo, w_hi = 1.0, math.log(1e-3), 2.0
        a0 = (w_hi - torch.logit((1.0 / tau).clamp(max=0.99))) / (z_hi - z_lo)
        self.a = nn.Parameter(a0)
        self.b = nn.Parameter(w_hi - a0 * z_hi)
        self.pred = nn.Linear(H, proj)
        gen = torch.Generator().manual_seed(1234)          # own generator: the global RNG is not consumed
        self.register_buffer("R", torch.randn(sum(dims), proj, generator=gen) / math.sqrt(sum(dims)))
        self.stat = None                                    # eval diagnostic: [(rate [B, g], z [B])]
        self.aux = None                                     # train: per-step prediction mse [B]
        self.force_open = False                             # Amendment C2: eval-only ablation, slow groups rate 1

    def forward(self, streams, x, dt=1.0):
        b = self.heads(streams, x)
        gate = torch.sigmoid(-self.f(b) * dt)
        c = gate * torch.tanh(self.g_(b)) + (1 - gate) * torch.tanh(self.h(b))
        y = torch.cat(streams, 1).detach() @ self.R
        mse = ((self.pred(x.detach()) - y) ** 2).mean(-1)   # surprise of this input given x_{t-1}; trains only P
        if self.aux is not None:
            self.aux.append(mse)
        z = torch.log(mse.detach() + 1e-3)[:, None]
        rate = torch.where(self.fast, torch.ones_like(z), torch.sigmoid(z * self.a + self.b))     # [B, g]
        if self.force_open:
            rate = torch.ones_like(rate)
        if self.stat is not None:
            self.stat.append((rate.detach(), z[:, 0].detach()))
        return x + (c - x) * rate.repeat_interleave(self.hidden_size // self.g, -1)


class StressCell2(nn.Module):
    """Amendment D (#54A, 10-27): his "banco de T": the slow store HOLDS (leak 1/tau, tau 16..1024) and is rewritten
    only when a run-time stress signal is high. State = [fast H/2 | slow H/2 in G groups]. Fast units = plain cfc
    write (rate 1). Slow group g: s += (c_s - s) * (1/tau_g + (1 - 1/tau_g) * open_g), open_g = HARD step
    1[logit_g > 0] in train and eval (straight-through: backward through sigmoid(logit_g), so a closed gate still
    gets gradient: the B11 failure mode), logit_g = softplus(w1_g) (surprise - 1) + softplus(w2_g) (log(dis_g + 1e-3)
    - log .1) + gb_g. Monotone in stress by construction (w >= 0): training can tune sensitivity and threshold,
    never write on content alone. surprise = -log p_{t-1}(prev symbol) of the planner's own previous prediction
    (log V at START), dis_g = mean (c_s - s)^2 over group g ("what arrives contradicts what is held"); both detached.
    No gold at eval: in greedy decoding prev is the planner's own argmax."""

    def __init__(self, dims, H, U, taus=(16, 1024), groups=4):
        super().__init__()
        assert H % 2 == 0 and (H // 2) % groups == 0
        self.hidden_size, self.G, self.Hf = H, groups, H // 2
        self.heads = Heads(dims, H, U)
        k = U * len(dims)
        self.f, self.g_, self.h = nn.Linear(k, H), nn.Linear(k, H), nn.Linear(k, H)
        self.register_buffer("inv_tau", 1.0 / torch.logspace(math.log10(taus[0]), math.log10(taus[1]), groups))
        sp1 = math.log(math.e - 1.0)                       # softplus(sp1) = 1
        self.w1 = nn.Parameter(torch.full((groups,), sp1))
        self.w2 = nn.Parameter(torch.full((groups,), sp1))
        self.gb = nn.Parameter(torch.full((groups,), -2.0))   # closed when the input was predicted and agrees
        self.stat = None                                    # eval diagnostic: [(hard [B,G], soft [B,G], surp [B], dis [B,G])]
        self.pen = None                                     # train: per-step mean open (straight-through) [B]
        self.force_open = False                             # eval ablation: slow groups rate 1 (= cfc on all units)
        self.zero_slow = False                              # eval ablation: slow store zeroed every step
        self.closed_after_t0 = False                        # eval ablation: only the START write, then hold

    def forward(self, streams, x, surprise, start=None, dt=1.0):
        b = self.heads(streams, x)
        gate = torch.sigmoid(-self.f(b) * dt)
        c = gate * torch.tanh(self.g_(b)) + (1 - gate) * torch.tanh(self.h(b))
        Hf, G = self.Hf, self.G
        s, cs = x[:, Hf:], c[:, Hf:]
        dis = ((cs - s) ** 2).detach().view(x.shape[0], G, -1).mean(-1)                         # [B, G]
        logit = (F.softplus(self.w1) * (surprise.detach()[:, None] - 1.0)
                 + F.softplus(self.w2) * (torch.log(dis + 1e-3) - math.log(0.1)) + self.gb)
        soft = torch.sigmoid(logit)
        hard = (logit > 0).float()
        op = hard + soft - soft.detach()                    # straight-through: forward hard, backward sigmoid
        if self.force_open:
            op = torch.ones_like(op)
        if self.closed_after_t0:
            op = op * start.float()[:, None]
        rate = self.inv_tau + (1 - self.inv_tau) * op
        s = s + (cs - s) * rate.repeat_interleave((self.hidden_size - Hf) // G, -1)
        if self.zero_slow:
            s = torch.zeros_like(s)
        if self.pen is not None:
            self.pen.append(op.mean(-1))
        if self.stat is not None:
            self.stat.append((hard.detach(), soft.detach(), surprise.detach(), dis))
        return torch.cat([c[:, :Hf], s], 1)


class StressCell3(StressCell2):
    """Amendment E (#54A, review M2): stress2 + an INPUT-surprise term, so that a correction arriving in the input can
    reach the gate at run time without gold. P = Linear(H, 16) predicts R r_t (fixed random projection of the address
    read, seed 1234 own generator) from x_{t-1}.detach(); mse_t trains only P (aux loss AUX_W, as the stress arm).
    zin_t = log(mse_t + 1e-3) - log(mean_{k<t} mse_k + 1e-3) (surprise relative to this sequence's own running
    baseline, causal, detached; 0 at t0). logit_g = stress2 logit + softplus(w3_g) zin_t, w3 init softplus 2.
    Rate penalty excludes t0 (Planner.forward). stress2 itself is untouched (bit-identity)."""

    NO_LEAK = False

    def __init__(self, dims, H, U, taus=(16, 1024), groups=4, proj=16):
        super().__init__(dims, H, U, taus, groups)
        self.pred = nn.Linear(H, proj)
        gen = torch.Generator().manual_seed(1234)
        self.register_buffer("R", torch.randn(dims[0], proj, generator=gen) / math.sqrt(dims[0]))
        self.w3 = nn.Parameter(torch.full((groups,), math.log(math.expm1(2.0))))    # softplus(w3) = 2
        self.aux = None                                     # train: per-step input-prediction mse [B]
        self._sum, self._n = None, 0                        # running input-surprise baseline (reset at START)

    def forward(self, streams, x, surprise, start=None, dt=1.0):
        b = self.heads(streams, x)
        gate = torch.sigmoid(-self.f(b) * dt)
        c = gate * torch.tanh(self.g_(b)) + (1 - gate) * torch.tanh(self.h(b))
        Hf, G = self.Hf, self.G
        s, cs = x[:, Hf:], c[:, Hf:]
        dis = ((cs - s) ** 2).detach().view(x.shape[0], G, -1).mean(-1)                         # [B, G]
        mse = ((self.pred(x.detach()) - streams[0].detach() @ self.R) ** 2).mean(-1)            # [B]
        if self.aux is not None:
            self.aux.append(mse)
        m = mse.detach()
        if self._sum is None or bool(start.all()):          # t0 reads START in every row: new sequence
            zin = torch.zeros_like(m)
            self._sum, self._n = m, 1
        else:
            zin = torch.log(m + 1e-3) - torch.log(self._sum / self._n + 1e-3)
            self._sum, self._n = self._sum + m, self._n + 1
        logit = (F.softplus(self.w1) * (surprise.detach()[:, None] - 1.0)
                 + F.softplus(self.w2) * (torch.log(dis + 1e-3) - math.log(0.1)) + self.gb
                 + F.softplus(self.w3) * zin[:, None])
        soft = torch.sigmoid(logit)
        hard = (logit > 0).float()
        op = hard + soft - soft.detach()
        if self.force_open:
            op = torch.ones_like(op)
        if self.closed_after_t0:
            op = op * start.float()[:, None]
        rate = op if self.NO_LEAK else self.inv_tau + (1 - self.inv_tau) * op     # stress4: closed = exact hold
        s = s + (cs - s) * rate.repeat_interleave((self.hidden_size - Hf) // G, -1)
        if self.zero_slow:
            s = torch.zeros_like(s)
        if self.pen is not None:
            self.pen.append(op.mean(-1))
        if self.stat is not None:
            self.stat.append((hard.detach(), soft.detach(), surprise.detach(), dis, zin))
        return torch.cat([c[:, :Hf], s], 1)


class StressCell4(StressCell3):
    """Amendment F (#54A literal: "T tao grande que o dado fica la e so muda se houver um estresse"): stress3 with
    tau = infinity when closed. Slow group g: s += (c_s - s) * open_g (closed -> rate exactly 0, nothing fades; open ->
    full rewrite as stress3). Same parameters as stress3 (inv_tau buffer kept, unused)."""
    NO_LEAK = True


class Planner(BA.Net):
    """blockaddr.Net with a swappable decoder cell; encoder/reader/heads identical (same init order)."""

    def __init__(self, arm, size=None, gate_bias=-4.0):
        super().__init__("addr", gate_bias=gate_bias)        # consumes the RNG exactly as BA.Net
        self.kind = arm
        d = self.emb.embedding_dim
        if arm == "gru":
            return
        del self.cell
        if arm == "lstm":
            H = size
            self.cell = nn.LSTMCell(2 * d, H)
            self.out, self.adv = nn.Linear(H, len(BA.SYM)), nn.Linear(H, 1)
        else:
            H = 2 * d
            self.cell = {"cfc": CfCCell, "ltc": LTCCell, "stress": StressCell, "stress2": StressCell2,
                         "stress3": StressCell3, "stress4": StressCell4}[arm]([d, d], H, size)
        self.aux_loss = None

    @property
    def dev(self):
        return self.emb.weight.device

    def step(self, h, a, m, c, prev, st):
        w = -(a - c[:, None]) ** 2 / self.log_s.exp() ** 2
        w = torch.softmax(w.masked_fill(~m, -1e9), 1)
        r = (w[..., None] * h).sum(1)
        y = self.sym(prev)
        if self.kind == "gru":
            st = self.cell(torch.cat([r, y], 1), st)
            o = st
        elif self.kind == "lstm":
            st = self.cell(torch.cat([r, y], 1), st)
            o = st[0]
        elif self.kind in ("stress2", "stress3", "stress4"):   # surprise of prev under the previous prediction
            with torch.no_grad():
                V = self.out.out_features
                sp = -F.log_softmax(self.out(st), -1).gather(1, prev.clamp(max=V - 1)[:, None])[:, 0]
                sp = torch.where(prev >= V, torch.full_like(sp, math.log(V)), sp)     # START: nothing predicted yet
            st = self.cell([r, y], st, sp, prev >= self.out.out_features)
            o = st
        else:
            st = self.cell([r, y], st)
            o = st
        return self.out(o), torch.sigmoid(self.adv(o))[:, 0], st

    def init_state(self, B):
        H = self.cell.hidden_size
        z = torch.zeros(B, H, device=self.dev)
        return (z, torch.zeros(B, H, device=self.dev)) if self.kind == "lstm" else z

    def forward(self, x, y_in):
        x, y_in = x.to(self.dev), y_in.to(self.dev)
        h, a, m = self.encode(x)
        B = x.shape[0]
        c, st, outs = torch.zeros(B, device=self.dev), self.init_state(B), []
        if self.kind == "stress":
            self.cell.aux = []
        if self.kind == "stress2":
            self.cell.pen = []
        if self.kind in ("stress3", "stress4"):
            self.cell.pen, self.cell.aux = [], []
        for t in range(y_in.shape[1]):
            lo, p, st = self.step(h, a, m, c, y_in[:, t], st)
            outs.append(lo)
            c = c + p
        if self.kind == "stress":
            mk = (y_in != 0).float()                           # real decoder inputs (START or a symbol), not pad
            self.aux_loss = (torch.stack(self.cell.aux, 1) * mk).sum() / mk.sum()
            self.cell.aux = None
        if self.kind == "stress2":
            mk = (y_in != 0).float()
            op = (torch.stack(self.cell.pen, 1) * mk).sum() / mk.sum()        # slow open rate, straight-through
            self.open_rate = op.item()
            self.aux_loss = RATE_W * F.relu(op - RATE_TARGET)
            self.cell.pen = None
        if self.kind in ("stress3", "stress4"):                # Amendment E: t0 excluded from the rate penalty
            mk = (y_in != 0).float()
            mk0 = mk.clone()
            mk0[:, 0] = 0
            op = (torch.stack(self.cell.pen, 1) * mk0).sum() / mk0.sum().clamp(min=1)
            pm = (torch.stack(self.cell.aux, 1) * mk).sum() / mk.sum()
            self.open_rate = op.item()
            self.aux_loss = RATE_W * F.relu(op - RATE_TARGET) + AUX_W * pm
            self.cell.pen = self.cell.aux = None
        return torch.stack(outs, 1)

    @torch.no_grad()
    def greedy(self, x):
        x = x.to(self.dev)
        h, a, m = self.encode(x)
        B = x.shape[0]
        c, st = torch.zeros(B, device=self.dev), self.init_state(B)
        prev, out = torch.full((B,), BA.START, device=self.dev), []
        for _ in range(BA.MAX_OUT):
            lo, p, st = self.step(h, a, m, c, prev, st)
            prev = lo.argmax(-1)
            out.append(prev)
            c = c + p
        return torch.stack(out, 1).cpu()


def n_params(net):
    return sum(p.numel() for p in net.parameters())


def matched(arm, target):
    """size (U for cfc/ltc, H for lstm) whose total params are closest to target; asserted within 5%."""
    if arm == "gru":
        return None
    best = None
    for s in range(8, 400):
        n = n_params(Planner(arm, s))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (s, n)
        if n > target * 1.1:
            break
    assert abs(best[1] - target) <= 0.05 * target, (arm, best, target)
    return best[0]


# ------------------------------------------------------------------------------------------------- worlds

class OracleWriter:
    """block text by rule: isolates the planner in the belt graders (multifn/compose evaluate)."""

    def write(self, c):
        return [MF.block_text(MF.SYM[i]) for i in c.tolist()]


def setup_world(world):
    """-> (data, name_bytes, targets, train list). Sets BA.SYM/START/MAX_OUT exactly as the original path."""
    if world == "amend":                            # Amendment E (#54A): held fact + mid-sequence corrections
        data = AM.Data()
        BA.SYM, BA.START, BA.MAX_OUT = AM.SYM, AM.START, data.max_out
        return data, AM.name_bytes, AM.targets
    if world in AK.CONFIGS:                         # Amendment F: K held slots, one-slot corrections
        data = AK.Data(world)
        BA.SYM, BA.START, BA.MAX_OUT = AK.SYM, AK.START, data.max_out
        return data, AK.name_bytes, AK.targets
    if world == "synth2":
        BA.SYM, BA.START, BA.MAX_OUT = BC.SYM, len(BC.SYM), BC.MAX_OUT
        data = BC.Data()
        return data, BA.name_bytes, BA.targets
    if world == "dispatch":                         # e2e_pipeline.py:57-58 + cmd_train: interleaved, MAX_OUT 49
        MF.PLAN_FORMAT = "interleaved"              # (smoke 10:1x caught it: "grouped" reproduced the old
        MF.MAX_OUT = max(34, len(MF.plan_of(tuple(MF.OPS))) + 4)   # multifn_s0.json, not the belt's meta)
        BA.SYM, BA.START, BA.MAX_OUT = MF.SYM, len(MF.SYM), MF.MAX_OUT
        return MF.Data(), MF.name_bytes, MF.targets
    data = CP.Data()
    BA.SYM, BA.START = CP.SYM, len(CP.SYM)
    BA.MAX_OUT = max(len(CP.plan_of(s)) for k in data.sets for s in data.sets[k]) + 4   # e2e compose_max_out
    return data, CP.name_bytes, CP.targets


def train(world, arm, size, data, nb, tg, seed, steps, lr, dev, log=None):
    """the loop of blockaddr.main / multifn.train_planner / compose.train_planner (identical among them)."""
    torch.manual_seed(seed)
    net = Planner(arm, size).to(dev)
    if arm in ("stress", "stress2", "stress3", "stress4"):  # gate scalars without weight decay (stressgate wg)
        nd = ({"cell.a", "cell.b"} if arm == "stress" else {"cell.w1", "cell.w2", "cell.gb"}
              | ({"cell.w3"} if arm in ("stress3", "stress4") else set()))
        opt = torch.optim.AdamW([{"params": [p for n, p in net.named_parameters() if n not in nd]},
                                 {"params": [p for n, p in net.named_parameters() if n in nd], "weight_decay": 0.0}],
                                lr=lr, weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    r, tr, losses, auxs = random.Random(seed + 100), data.sets["train"], [], []
    for s in range(steps):
        for gp in opt.param_groups:
            gp["lr"] = lr * min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
        ts = [tr[r.randrange(len(tr))] for _ in range(48)]
        y_in, y = tg(ts)
        loss = F.cross_entropy(net(nb(ts), y_in).transpose(1, 2), y.to(dev), ignore_index=0)
        losses.append(loss.item())
        if arm == "stress":
            auxs.append(net.aux_loss.item())
            loss = loss + AUX_W * net.aux_loss
        elif arm in ("stress2", "stress3", "stress4"):
            auxs.append(net.open_rate)
            loss = loss + net.aux_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if (s + 1) % 250 == 0 and log:
            log({"step": s + 1, "loss": round(sum(losses[-100:]) / 100, 4),
                 **({("open" if arm in ("stress2", "stress3", "stress4") else "aux"): round(sum(auxs[-100:]) / 100, 4)} if auxs else {})})
    net.aux_last100 = round(sum(auxs[-100:]) / 100, 5) if auxs else None
    return net, losses


def evaluate(world, net, data):
    if world in AMW:
        return AMW[world].evaluate(net, data)
    if world == "synth2":
        return BA.evaluate(net, data, NAMES[world])
    ev = (MF if world == "dispatch" else CP).evaluate(net, OracleWriter(), data, NAMES[world])
    return {k: {m: x for m, x in v.items() if not m.startswith("example")} for k, v in ev.items()}


@torch.no_grad()
def saturation(world, net, data, nb):
    """liquid gate / keep statistics on the longest held-out split (greedy pass)."""
    if world in AMW:                                         # Amendment E/F instruments (stress2/3/4; else None)
        return AMW[world].cell_stats(net, data)
    if net.kind not in ("cfc", "ltc", "stress", "stress2", "stress3", "stress4"):
        return None
    k = {"synth2": "len6", "dispatch": "len8", "expr": "calls8"}[world]
    if net.kind == "stress":
        return stress_stats(world, net, data, nb, k)
    if net.kind in ("stress2", "stress3", "stress4"):
        return stress2_stats(world, net, data, nb, k)
    net.cell.stat = []
    net.greedy(nb(data.sets[k][:256]))
    v = torch.cat([s.flatten() for s in net.cell.stat]).float().cpu()
    net.cell.stat = None
    name = "gate sigma(-f dt)" if net.kind == "cfc" else "keep exp(-lambda dt)"
    return {"what": name, "split": k, "mean": round(v.mean().item(), 4), "lt_.02": round((v < .02).float().mean().item(), 4),
            "gt_.98": round((v > .98).float().mean().item(), 4),
            "quantiles_10_50_90": [round(x, 4) for x in torch.quantile(v[:1_000_000], torch.tensor([.1, .5, .9])).tolist()]}


@torch.no_grad()
def split_plan_exact(world, net, data, k):
    """plan_exact (block_exact) on split k through the same graded path as evaluate()."""
    if world == "synth2":
        v = BA.evaluate(net, data, [k])[k]
    else:
        v = (MF if world == "dispatch" else CP).evaluate(net, OracleWriter(), data, [k])[k]
    net.eval()
    return v.get("plan_exact", v.get("block_exact"))


def stress_stats(world, net, data, nb, k):
    """per-group write rate and surprise z over the greedy pass on split k: all decoding steps (as registered) and
    the steps up to and including the first <eos> (Amendment C2); plus the forced-open ablation on the full split."""
    net.cell.stat = []
    gen = net.greedy(nb(data.sets[k][:256]))                                # [B, T] on cpu
    rate = torch.cat([r for r, _ in net.cell.stat]).float().cpu()          # [steps*B, g], step-major
    z = torch.cat([q for _, q in net.cell.stat]).float().cpu()
    net.cell.stat = None
    eos = BA.SYM.index("<eos>")
    done_before = torch.cumsum((gen == eos).long(), 1) - (gen == eos).long() > 0     # eos emitted at an earlier step
    alive = (~done_before).t().reshape(-1)                                  # step-major, matches rate rows
    ra = rate[alive]
    try:
        pe_norm = split_plan_exact(world, net, data, k)
        net.cell.force_open = True
        pe_open = split_plan_exact(world, net, data, k)
    finally:
        net.cell.force_open = False
    return {"what": "stress write rate per group (tau 1..128)", "split": k,
            "rate_mean": [round(v, 4) for v in rate.mean(0).tolist()],
            "rate_lt_.02": [round(v, 4) for v in (rate < .02).float().mean(0).tolist()],
            "rate_mean_pre_eos": [round(v, 4) for v in ra.mean(0).tolist()],
            "rate_lt_.02_pre_eos": [round(v, 4) for v in (ra < .02).float().mean(0).tolist()],
            "pre_eos_frac": round(alive.float().mean().item(), 4),
            "plan_exact": pe_norm, "plan_exact_forced_open": pe_open,
            "z_mean": round(z.mean().item(), 3),
            "z_quantiles_10_50_90": [round(x, 3) for x in torch.quantile(z, torch.tensor([.1, .5, .9])).tolist()],
            "a": [round(v, 3) for v in net.cell.a.tolist()], "b": [round(v, 3) for v in net.cell.b.tolist()]}


def _corr(a, b):
    """Pearson (phi for 0/1) correlation of two flat float tensors; None if either is constant."""
    a, b = a.float() - a.float().mean(), b.float() - b.float().mean()
    d = (a.pow(2).sum() * b.pow(2).sum()).sqrt().item()
    return round((a * b).sum().item() / d, 4) if d > 0 else None


@torch.no_grad()
def stress2_stats(world, net, data, nb, k):
    """Amendment D instruments: hard open rate of the slow groups over all greedy steps and before the first <eos>;
    correlation of the per-step open fraction with stress proxies (scored with gold AFTER decoding, never fed to the
    net); ablations zero_slow / force_open on len3..len6 (synth2) or split k."""
    cell = net.cell
    tasks = data.sets[k][:256]
    cell.stat = []
    gen = net.greedy(nb(tasks))                                              # [B, T] cpu
    hard = torch.stack([q[0] for q in cell.stat], 1).float().cpu()          # [B, T, G]
    soft = torch.stack([q[1] for q in cell.stat], 1).float().cpu()
    surp = torch.stack([q[2] for q in cell.stat], 1).float().cpu()          # [B, T]
    dis = torch.stack([q[3] for q in cell.stat], 1).float().cpu()           # [B, T, G]
    cell.stat = None
    eos = BA.SYM.index("<eos>")
    alive = ~(torch.cumsum((gen == eos).long(), 1) - (gen == eos).long() > 0)    # up to and incl. first eos
    of = hard.mean(-1)                                                       # open fraction of slow groups per step
    out = {"what": "stress2 slow-group hard open rate (tau 16..1024 bank)", "split": k,
           "slow_rate_all": round(hard.mean().item(), 4), "slow_rate_pre_eos": round(of[alive].mean().item(), 4),
           "slow_rate_pre_eos_per_group": [round(v, 4) for v in hard[alive].mean(0).tolist()],
           "soft_mean_pre_eos": round(soft[alive].mean().item(), 4),
           "open_at_t0": round(of[:, 0].mean().item(), 4),
           "slow_rate_pre_eos_t>=1": round(of[:, 1:][alive[:, 1:]].mean().item(), 4),
           "surprise_q10_50_90_pre_eos": [round(v, 3) for v in torch.quantile(surp[alive], torch.tensor([.1, .5, .9])).tolist()],
           "dis_mean_pre_eos": round(dis[alive].mean().item(), 4),
           "w1": [round(v, 3) for v in F.softplus(cell.w1).tolist()], "w2": [round(v, 3) for v in F.softplus(cell.w2).tolist()],
           "gb": [round(v, 3) for v in cell.gb.tolist()]}
    if world == "synth2":                                                    # stress proxies from gold, post hoc
        _, y = BA.targets(tasks)
        T = gen.shape[1]
        gold = torch.zeros(len(tasks), T, dtype=torch.long)
        gold[:, :min(T, y.shape[1])] = y[:, :T]
        wrong = (gen != gold)
        first = wrong & (torch.cumsum(wrong.long(), 1) == 1)                 # the first wrong token of the plan
        wrong_prev = torch.cat([torch.zeros(len(tasks), 1, dtype=torch.bool), wrong[:, :-1]], 1)
        bstart = torch.tensor([[BA.SYM[j].startswith(("op:", "fin:")) for j in row] for row in gen.tolist()])
        bstart_prev = torch.cat([torch.zeros(len(tasks), 1, dtype=torch.bool), bstart[:, :-1]], 1)
        m = alive.clone()
        m[:, 0] = False                                                      # t0 always reads START: excluded
        o = of[m]
        px = {"wrong_now": wrong, "wrong_prev": wrong_prev, "first_wrong_now": first,
              "block_start_now": bstart, "block_start_prev": bstart_prev}
        out["proxy_t>=1_pre_eos"] = {
            n: {"corr": _corr(o, v[m]), "frac": round(v[m].float().mean().item(), 4),
                "p_open_if": round(o[v[m]].mean().item(), 4) if v[m].any() else None,
                "p_open_if_not": round(o[~v[m]].mean().item(), 4) if (~v[m]).any() else None}
            for n, v in px.items()}
        out["corr_open_surprise"] = _corr(o, surp[m])
    abl = {}
    for sp in (["len3", "len4", "len5", "len6"] if world == "synth2" else [k]):
        row = {}
        for mode in ("normal", "zero_slow", "force_open", "closed_after_t0"):
            cell.zero_slow, cell.force_open, cell.closed_after_t0 = (mode == "zero_slow", mode == "force_open",
                                                                     mode == "closed_after_t0")
            try:
                row[mode] = split_plan_exact(world, net, data, sp)
            finally:
                cell.zero_slow = cell.force_open = cell.closed_after_t0 = False
        abl[sp] = row
    out["ablation_plan_exact"] = abl
    net.eval()
    return out


# ------------------------------------------------------------------------------------------------- e2e (synth2)
E2E_SPLITS = ["test_a", "test_num", "len3", "len4", "len5", "len6"]          # writer7seed.SPLITS
WRITER_SEEDS = [0, 1, 2]


def writer_tables(root, data):
    """{ws: {(op, dig): text}} for the d32 fp byte writer, trained on CPU exactly as writer7seed.py
    (BW.train_writer(train, 32, False, ws)), written for every (op, digit) key a plan can produce; cached on disk."""
    import bytewriter as BW
    import writer7seed as W7
    from synth2 import FINAL, UNITS
    os.makedirs(os.path.join(root, "writers"), exist_ok=True)
    keys = sorted({BW.block_key(u, False) for u in UNITS} | {BW.block_key(f, True) for f in FINAL})
    out = {}
    for ws in WRITER_SEEDS:
        f = os.path.join(root, "writers", f"d32_w{ws}.json")
        if not os.path.exists(f):
            t0 = time.perf_counter()
            writer, _, wloss = BW.train_writer(data.sets["train"], 32, False, ws)
            tx = W7.writer_texts(writer, keys)
            json.dump({"writer_seed": ws, "writer_loss": wloss, "train_wall_s": round(time.perf_counter() - t0, 1),
                       "texts": [[k[0], k[1], v] for k, v in tx.items()]}, open(f, "w"))
        out[ws] = {(r[0], r[1]): r[2] for r in json.load(open(f))["texts"]}
    return out


def e2e(net, data, tables):
    """planner -> plan -> real writer text -> bwrap strict grade (writer7seed.score, cached by source)."""
    import writer7seed as W7
    res = {}
    for k in E2E_SPLITS:
        plans = W7.planner_plans(net, data.sets[k])
        for ws, tx in tables.items():
            res.setdefault(f"d32_w{ws}", {})[k] = W7.score(data.sets[k], plans, tx)
    net.train()
    return res


def cmd_run(a):
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    if dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.3)
    data, nb, tg = setup_world(a.world)
    torch.manual_seed(0)
    target = n_params(BA.Net("addr", gate_bias=-4.0))
    size = matched(a.arm, target)
    if a.arm == "stress":                                  # Amendment B: also within 5% of the cfc arm
        nc = n_params(Planner("cfc", matched("cfc", target)))
        ns = n_params(Planner("stress", size))
        assert abs(ns - nc) <= 0.05 * nc, (ns, nc)
    if a.arm in ("stress2", "stress3", "stress4"):         # Amendment D/E/F: within 2% of gru AND of cfc
        nc = n_params(Planner("cfc", matched("cfc", target)))
        ns = n_params(Planner(a.arm, size))
        assert abs(ns - target) <= 0.02 * target and abs(ns - nc) <= 0.02 * nc, (ns, target, nc)
    steps = a.steps or (AMW[a.world].STEPS if a.world in AMW else WORLDS[a.world])
    out = os.path.join(a.out, a.world)
    os.makedirs(out, exist_ok=True)
    tables = writer_tables(a.out, data) if a.world == "synth2" and not a.no_e2e else None
    for seed in [int(s) for s in a.seeds.split(",")]:
        f = os.path.join(out, f"{a.arm}_lr{a.lr:g}_s{seed}.json")
        if os.path.exists(f) and not a.force:
            print(json.dumps({"skip": f}), flush=True)
            continue
        t0 = time.perf_counter()
        logf = open(f[:-5] + ".log", "w")
        log = lambda d: (logf.write(json.dumps(d) + "\n"), logf.flush(), print(json.dumps({"seed": seed, **d}), flush=True))
        net, losses = train(a.world, a.arm, size, data, nb, tg, seed, steps, a.lr, dev, log)
        tw = time.perf_counter() - t0
        net.eval()
        ev = evaluate(a.world, net, data)
        res = {"world": a.world, "arm": a.arm, "seed": seed, "lr": a.lr, "steps": steps, "size": size,
               "params": n_params(net), "gru_params": target, "device": str(dev), "threads": a.threads,
               "train_wall_s": round(tw, 1), "total_wall_s": round(time.perf_counter() - t0, 1),
               "final_loss": round(sum(losses[-100:]) / 100, 5), "eval": ev, "sat": saturation(a.world, net, data, nb),
               "max_vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if dev.type == "cuda" else None}
        if net.aux_last100 is not None:
            res["open_rate_last100" if a.arm in ("stress2", "stress3", "stress4") else "aux_mse_last100"] = net.aux_last100
        if tables is not None:
            t1 = time.perf_counter()
            res["e2e"] = e2e(net, data, tables)
            res["e2e_pass1_mean"] = {k: round(sum(res["e2e"][w][k]["pass1"] for w in res["e2e"]) / len(res["e2e"]), 4)
                                     for k in E2E_SPLITS}
            res["e2e_wall_s"] = round(time.perf_counter() - t1, 1)
            net.eval()
        if a.world == "synth2":
            res["gate_examples"] = [BA.boundaries(net.cpu(), data, t) for t in data.sets["len4"][:3]]
        log({"result": {k: (v.get("pass1"), v.get("plan_exact", v.get("block_exact"))) for k, v in ev.items()},
             "wall": res["total_wall_s"], **({"e2e_pass1_mean": res["e2e_pass1_mean"]} if "e2e" in res else {})})
        logf.close()
        json.dump(res, open(f, "w"), indent=1)


def cmd_parity(a):
    """the gru arm == BA.Net: same init (state dict), same teacher-forced logits, same greedy, on CPU, 3 worlds."""
    torch.set_num_threads(a.threads)
    out = {}
    for world in WORLDS:
        data, nb, tg = setup_world(world)
        torch.manual_seed(3)
        ref = BA.Net("addr", gate_bias=-4.0)
        torch.manual_seed(3)
        mine = Planner("gru")
        sd_eq = all(torch.equal(p, q) for p, q in zip(ref.state_dict().values(), mine.state_dict().values()))
        ts = data.sets["train"][:48]
        y_in, _ = tg(ts)
        held = data.sets[NAMES[world][-1]][:64]
        out[world] = {"state_dict_equal": sd_eq,
                      "logits_equal": torch.equal(ref(nb(ts), y_in), mine(nb(ts), y_in)),
                      "greedy_equal": torch.equal(ref.greedy(nb(held)), mine.greedy(nb(held)))}
        for arm in ARMS[1:]:
            size = matched(arm, n_params(ref))
            out[world][f"{arm}_size_params"] = (size, n_params(Planner(arm, size)), n_params(ref))
    print(json.dumps(out, indent=1))
    ok = all(v["state_dict_equal"] and v["logits_equal"] and v["greedy_equal"] for v in out.values())
    print("PARITY", "PASS" if ok else "FAIL")


def cmd_report(a):
    import statistics as st
    rows = {}
    for world in WORLDS:
        d = os.path.join(a.out, world)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                r = json.load(open(os.path.join(d, fn)))
                rows.setdefault((world, r["arm"], r["lr"]), []).append(r)
    rep = {}
    for (world, arm, lr), rs in sorted(rows.items()):
        rs.sort(key=lambda r: r["seed"])
        key = "pass1"
        e = {}
        for k in NAMES[world]:
            v = [r["eval"][k][key] for r in rs]
            e[k] = {"mean": round(sum(v) / len(v), 4), "sd": round(st.stdev(v), 4) if len(v) > 1 else None,
                    "min": min(v), "seeds": v}
        rep[f"{world}/{arm}/lr{lr:g}"] = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs],
                                          "params": rs[0]["params"], "pass1": e,
                                          "wall_s_mean": round(sum(r["total_wall_s"] for r in rs) / len(rs), 1)}
        if world == "synth2":                              # Amendment B metrics: plan_exact and real-writer e2e
            ms = {"plan_exact": lambda r, k: r["eval"][k]["block_exact"]}
            if all("e2e_pass1_mean" in r for r in rs):
                ms["e2e_pass1_mean"] = lambda r, k: r["e2e_pass1_mean"][k]
            for mname, fn in ms.items():
                rep[f"{world}/{arm}/lr{lr:g}"][mname] = {
                    k: {"mean": round(st.mean(v), 4), "sd": round(st.stdev(v), 4) if len(v) > 1 else None,
                        "min": min(v), "seeds": v}
                    for k in ("len5", "len6", "test_num") for v in [[fn(r, k) for r in rs]]}
    rep["verdict_amendment_A"] = verdict_a(rows)
    rep["verdict_amendment_B"] = verdict_b(rows)
    json.dump(rep, open(os.path.join(a.out, "report13.json"), "w"), indent=1)
    print("VERDICT (amendment A):", json.dumps(rep["verdict_amendment_A"]))
    print("VERDICT (amendment B):", json.dumps(rep["verdict_amendment_B"]))
    for k, v in rep.items():
        if k.startswith("verdict"):
            continue
        print(k, v["n_seeds"], v["params"], {s: (x["mean"], x["sd"], x["min"]) for s, x in v["pass1"].items()})


def verdict_a(rows, world="synth2", key="len6", lr=3e-3, tq=2.969, margin=0.10):
    """Amendment A rule (see header). rows: {(world, arm, lr): [run dicts]}."""
    import statistics as st

    def by_seed(arm, l):
        return {r["seed"]: r["eval"][key]["pass1"] for r in rows.get((world, arm, l), [])}
    g = by_seed("gru", lr)
    out = {"world": world, "key": key, "lr": lr, "gru_n": len(g)}
    if len(g) != 7:
        out["verdict"] = f"NO VERDICT: gru has {len(g)} seeds, need 7"
        return out
    calls = {}
    for arm in ("cfc", "ltc"):
        x = by_seed(arm, lr)
        if sorted(x) != sorted(g):
            calls[arm] = {"call": f"NO VERDICT: seeds {sorted(x)} != gru {sorted(g)}"}
            continue
        d = [x[s] - g[s] for s in sorted(g)]
        m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
        lo, hi = m - tq * se, m + tq * se
        lrs = sorted({l for (w, a, l) in rows if w == world and a == arm})
        best = max(lrs, key=lambda l: st.mean(by_seed(arm, l).values()) if by_seed(arm, l) else -1)
        call = "BETTER" if lo > 0 else "PASS" if lo > -margin else "LOSES" if hi < -margin else "INCONCLUSIVE"
        xs, gs = list(x.values()), list(g.values())
        calls[arm] = {"call": call, "mean_diff": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
                      "best_lr": best, "best_lr_at_edge": best == max(lrs) and len(lrs) > 1,
                      "pass_plus": st.stdev(xs) <= .65 * st.stdev(gs) and min(xs) >= min(gs)}
    out["arms"] = calls
    if all(c.get("call") == "LOSES" and not c.get("best_lr_at_edge") for c in calls.values()):
        out["verdict"] = "KILL"
    elif any(c.get("call") in ("BETTER", "PASS") for c in calls.values()):
        out["verdict"] = "PASS (" + ",".join(a for a, c in calls.items() if c.get("call") in ("BETTER", "PASS")) + ")"
    else:
        out["verdict"] = "INCONCLUSIVE"
    return out


def paired_call(x, g, tq, margin):
    """paired-t CI of mean(x - g) over common seeds -> BETTER / PASS / LOSES / INCONCLUSIVE (Amendment A rule)."""
    import statistics as st
    d = [x[s] - g[s] for s in sorted(g)]
    m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
    lo, hi = m - tq * se, m + tq * se
    call = "BETTER" if lo > 0 else "PASS" if lo > -margin else "LOSES" if hi < -margin else "INCONCLUSIVE"
    xs, gs = list(x.values()), list(g.values())
    return {"call": call, "mean_diff": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
            "per_seed_diff": [round(v, 4) for v in d],
            "pass_plus": st.stdev(xs) <= .65 * st.stdev(gs) and min(xs) >= min(gs)}


def verdict_b(rows, world="synth2", lr=3e-3, tq=2.969, margin=0.10):
    """Amendment B rule (see header): stress vs gru and stress vs cfc, Bonferroni over these 2 comparisons.
    Primary = plan_exact len6; secondary (reported only) = plan_exact len5, e2e_pass1_mean len5/len6."""
    import statistics as st
    mets = {"plan_exact_len6": lambda r: r["eval"]["len6"]["block_exact"],
            "plan_exact_len5": lambda r: r["eval"]["len5"]["block_exact"],
            "e2e_len6": lambda r: r["e2e_pass1_mean"]["len6"],
            "e2e_len5": lambda r: r["e2e_pass1_mean"]["len5"]}

    def by_seed(arm, l, fn):
        try:
            return {r["seed"]: fn(r) for r in rows.get((world, arm, l), [])}
        except KeyError:                                   # e2e missing in a run file
            return {}
    out = {"world": world, "lr": lr, "primary": "plan_exact_len6", "t": tq, "margin": margin}
    s = by_seed("stress", lr, mets["plan_exact_len6"])
    if len(s) != 7:
        out["verdict"] = f"NO VERDICT: stress has {len(s)} seeds, need 7"
        return out
    lrs = sorted({l for (w, a, l) in rows if w == world and a == "stress"})
    best = max(lrs, key=lambda l: st.mean(by_seed("stress", l, mets["plan_exact_len6"]).values()))
    out["stress_best_lr"], out["stress_best_lr_at_edge"] = best, best == max(lrs) and len(lrs) > 1
    for ref in ("gru", "cfc"):
        for mname, fn in mets.items():
            x, g = by_seed("stress", lr, fn), by_seed(ref, lr, fn)
            if len(g) != 7 or sorted(x) != sorted(g):
                out[f"vs_{ref}/{mname}"] = {"call": f"NO VERDICT: seeds {sorted(x)} vs {ref} {sorted(g)}"}
                continue
            out[f"vs_{ref}/{mname}"] = paired_call(x, g, tq, margin)
    mech = mechanism_b([r for r in rows.get((world, "stress", lr), [])])            # Amendment C3
    out["mechanism"] = mech
    not_engaged = mech["n_collapsed"] >= 4
    cg = out.get("vs_gru/plan_exact_len6", {}).get("call", "")
    if cg.startswith("NO VERDICT"):
        out["verdict"] = cg
    elif cg == "LOSES" and not out["stress_best_lr_at_edge"] and not not_engaged:
        out["verdict"] = "KILL"
    else:
        out["verdict"] = f"vs gru {cg}; vs cfc {out.get('vs_cfc/plan_exact_len6', {}).get('call')}"
    if not_engaged:
        out["verdict"] += (f" | MECHANISM NOT ENGAGED: gate open (collapsed) in {mech['n_collapsed']}/7 seeds -> "
                           "not evidence for or against the tau bank; own-bug check + adapt (rule 7), no KILL")
    else:
        out["verdict"] += f" | mechanism: prereg gate met {mech['n_prereg_gate_met']}/7, collapsed {mech['n_collapsed']}/7"
    return out


def mechanism_b(rs):
    """Amendment C3, per seed from sat (len6): slow groups = tau > 1 (groups 1..7), tau >= 32 = groups 5..7."""
    per, n_met, n_col = {}, 0, 0
    for r in sorted(rs, key=lambda r: r["seed"]):
        s = r.get("sat") or {}
        if "rate_mean_pre_eos" not in s:
            per[r["seed"]] = "no Amendment-C stats in this run file"
            continue
        slow_all = sum(s["rate_mean"][1:]) / 7
        slow_pre = sum(s["rate_mean_pre_eos"][1:]) / 7
        met = slow_all < .10 and all(v > .30 for v in s["rate_lt_.02"][5:8])
        col = slow_pre > .80 and abs(s["plan_exact"] - s["plan_exact_forced_open"]) <= .01
        n_met, n_col = n_met + met, n_col + col
        per[r["seed"]] = {"slow_rate_all": round(slow_all, 4), "slow_rate_pre_eos": round(slow_pre, 4),
                          "plan_exact": s["plan_exact"], "forced_open": s["plan_exact_forced_open"],
                          "prereg_gate_met": met, "collapsed": col}
    return {"per_seed": per, "n_prereg_gate_met": n_met, "n_collapsed": n_col}


def cmd_report54(a):
    """Amendment D: stress2 (runs in --out/synth2) vs gru / cfc / stress (runs in --ref/synth2), synth2, per lr,
    paired by seed, Bonferroni over the 3 comparisons per lr (t_.99167,6 = 3.287). Primary = plan_exact len6."""
    import statistics as st
    rows = {}
    for root, arms in ((a.out, ("stress2",)), (a.ref, ("gru", "cfc", "stress"))):
        d = os.path.join(root, "synth2")
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if fn.endswith(".json"):
                r = json.load(open(os.path.join(d, fn)))
                if r["arm"] in arms:
                    rows.setdefault((r["arm"], r["lr"]), {})[r["seed"]] = r
    pe = lambda r, k: r["eval"][k]["block_exact"]
    rep = {"what": "Amendment D (#54A stress2), synth2, plan_exact = block_exact", "out": a.out, "ref": a.ref}
    for lr in sorted({l for (_, l) in rows}):
        row = {}
        for arm in ("stress2", "gru", "cfc", "stress"):
            rs = rows.get((arm, lr), {})
            if rs:
                row[arm] = {k: {"mean": round(st.mean(v), 4), "sd": round(st.stdev(v), 4) if len(v) > 1 else None,
                                "min": min(v), "n": len(v)}
                            for k in ("len3", "len4", "len5", "len6") for v in [[pe(rs[s], k) for s in sorted(rs)]]}
        x = rows.get(("stress2", lr), {})
        for ref in ("gru", "cfc", "stress"):
            g = rows.get((ref, lr), {})
            if len(x) == 7 and sorted(x) == sorted(g):
                row[f"vs_{ref}/plan_exact_len6"] = paired_call({s: pe(x[s], "len6") for s in x},
                                                               {s: pe(g[s], "len6") for s in g}, 3.287, 0.10)
            else:
                row[f"vs_{ref}/plan_exact_len6"] = {"call": f"NO VERDICT: seeds {sorted(x)} vs {ref} {sorted(g)}"}
        mech = {}
        for sd in sorted(x):
            m = x[sd].get("sat") or {}
            ab = m.get("ablation_plan_exact", {})
            drop = {mode: {"short_len3_4": round(st.mean(ab[k]["normal"] - ab[k][mode] for k in ("len3", "len4")), 4),
                           "long_len5_6": round(st.mean(ab[k]["normal"] - ab[k][mode] for k in ("len5", "len6")), 4)}
                    for mode in ("zero_slow", "closed_after_t0", "force_open") if ab and mode in ab["len3"]}
            mech[sd] = {"slow_rate_all": m.get("slow_rate_all"), "slow_rate_pre_eos": m.get("slow_rate_pre_eos"),
                        "slow_rate_pre_eos_t>=1": m.get("slow_rate_pre_eos_t>=1"),
                        "corr_open_wrong_prev": m.get("proxy_t>=1_pre_eos", {}).get("wrong_prev", {}).get("corr"),
                        "corr_open_block_start_now": m.get("proxy_t>=1_pre_eos", {}).get("block_start_now", {}).get("corr"),
                        "drop": drop}
        if mech:
            v = [m["slow_rate_pre_eos"] for m in mech.values() if m["slow_rate_pre_eos"] is not None]
            row["mechanism"] = {"per_seed": mech, "slow_rate_pre_eos_max": max(v) if v else None,
                                "n_long_drop_gt_short_zero_slow": sum(
                                    m["drop"]["zero_slow"]["long_len5_6"] > m["drop"]["zero_slow"]["short_len3_4"]
                                    for m in mech.values() if "zero_slow" in m["drop"]),
                                "n_closed_after_t0_no_effect": sum(
                                    max(map(abs, m["drop"]["closed_after_t0"].values())) <= .01
                                    for m in mech.values() if "closed_after_t0" in m["drop"])}
        rep[f"lr{lr:g}"] = row
    json.dump(rep, open(os.path.join(a.out, "report54.json"), "w"), indent=1)
    print(json.dumps({k: ({a_: v[a_]["len6"] for a_ in ("stress2", "gru", "cfc", "stress") if a_ in v}
                          | {c: v[c]["call"] for c in v if c.startswith("vs_")}) if k.startswith("lr") else v
                      for k, v in rep.items()}, indent=1))


def cmd_baselines(a):
    """gate 1: constant plan (most common train plan) and '<eos>' only, graded by the same path."""
    out = {}
    for world in WORLDS:
        data, nb, tg = setup_world(world)

        class Const:
            def __init__(self, plan):
                self.row = [BA.SYM.index(s) for s in plan] + [0] * (BA.MAX_OUT - len(plan))

            def eval(self):
                pass

            def train(self):
                pass

            def greedy(self, x):
                return torch.tensor([self.row] * x.shape[0])
        plan_fn = {"synth2": lambda t: BC.blocks_of(*t), "dispatch": MF.plan_of, "expr": CP.plan_of}[world]
        from collections import Counter
        common = Counter(tuple(plan_fn(t)) for t in data.sets["train"]).most_common(1)[0][0]
        for nm, plan in (("const_plan", list(common)), ("eos_only", ["<eos>"])):
            net = Const(plan)
            net.kind = "const"
            ev = evaluate(world, net, data)
            out[f"{world}/{nm}"] = {k: v["pass1"] for k, v in ev.items()}
    print(json.dumps(out, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "baselines13.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--world", required=True, choices=list(WORLDS) + list(AMW))
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--seeds", default="0")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-e2e", action="store_true", help="synth2: skip the real-writer e2e eval")
    p = sub.add_parser("parity")
    p.add_argument("--threads", type=int, default=2)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p = sub.add_parser("report54")
    p.add_argument("--out", required=True)
    p.add_argument("--ref", required=True, help="dir with the gru/cfc/stress synth2 runs (liquid13/full)")
    p = sub.add_parser("baselines")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "parity": cmd_parity, "report": cmd_report, "report54": cmd_report54,
     "baselines": cmd_baselines}[a.cmd](a)


if __name__ == "__main__":
    main()
