# User requests ledger (pycode / byte line)

Rule: every idea or request from Leonardo goes here the moment he says it, with status. An agent starting work
reads this first and works the OPEN items in order before proposing its own. Own ideas are allowed only after
the open items, and must say which request they serve. A failed idea is re-checked for our bugs, then adapted —
never dropped silently.

Status: OPEN · RUNNING · PARTIAL (what exists / what is missing) · DONE (path + command + output line)
· FAILED-ADAPT (bug classes checked; adaptation #) · DROPPED (his quoted OK) · CLAUDE-IDEA (serves #N)
"plan"/"planner" is never DONE for a request that asked for generation.

| # | Date | Request (his words, short) | Status | Evidence / next |
|---|---|---|---|---|
| 1 | 10-21 | Clean dataset: only functions, no comments; domain Python instead of PT-BR | DONE | synth/, synth2 corpora |
| 2 | 10-21 | Score the error ("how much did it miss"), mirror/orientation not the answer | DONE (partial) | critic.py AUC 0.82; selector by execution |
| 3 | 10-21 | Corpus in VRAM as index/mirror (what comes before/after a word) during generation | OPEN | not built as a generation-time index |
| 4 | 10-21 | GPU-side validation in the loop (parser/lint idea on GPU) | DONE (partial) | GPU syntax checker in loop (README) |
| 5 | 10-21 | Frozen coder + own small validator + selection; real executable Python; scientific calculator | DONE (partial) | Hybrid Codegen 39/39 calculator (Coder-3B) |
| 6 | 10-22 | .dna idea: code as byte matrices, row/column products; what signals indentation/newline | OPEN | not run |
| 7 | 10-22 | Multi-model loop: meaning + structure + text models; ERRO as teacher; agents point errors | OPEN | critic exists; loop not built |
| 8 | 10-22 | tau-bank liquid memory: huge tau holds data, changes only under "stress" (sum vs a-b sign) | DONE (as address integrator) | blockaddr.py: address = cumsum(gate), changes only on boundary |
| 9 | 10-22 | Forest/water image models as insight for code | OPEN | only read prior results; no run |
| 10 | 10-22 | Leave literal bytes: blocks of 256 forming a chain, structured hybrid, bridges (CfC/bytes) | DONE | blockaddr.py planner + bytewriter.py byte writer (see #17) |
| 11 | 10-22 | 5 agents with distinct ideas, tested; no bugs (false signal) | DONE | scratchpad sonnet_proposals / opus_audits |
| 12 | 10-23 | Position = address (row, column) of code text | DONE | units.py addrseg 1.75 -> 58% (2 units) |
| 13 | 10-23 | Groups of bytes tied by an addressing table, small models united, liquid neuron predicting | DONE | planner blockaddr.py + writer bytewriter.py: exec/pass 100% len3-6 (planner seed 4, writer d64 s0) |
| 14 | 10-23 | Opus agents after Sonnets to catch gaps | DONE | opus_audits.md |
| 15 | 10-23 | CPU binary / ternary / 4-state: pros, cons; DDR4 3000 | DONE (speed) / RUNNING (quality on GPU queue) | tern2f bench; blockaddr ternary 100% |
| 16 | 10-23 | Hybrid that carries everything: tokens, ternary, CfC, bytes | OPEN | planner done; byte writer + token input missing |
| 17 | 10-23 | Byte WRITER guided by the address (not lookup table) | DONE (1 seed) | `bytewriter.py --out .../bytewriter` -> exec 1.0 / pass1 1.0 on test_a, len3-6; test_num 0.86 (unseen op+digit text). Bug fixed: max_len 80 < running_max 88 bytes |
| 18 | 10-23 | Smaller model generates more? Must generate EXECUTABLE code; correct battery | RUNNING | writer d16..256 x fp/ter x 3 seeds, same planner ckpt, greedy k=1, no selector |
| 19 | 10-23 | Opus agent: process so I never drift from his requests again | DONE | protocol added to bytebrain/CLAUDE.md |
| 20 | 10-23 | Save everything he asks and do it (run later if needed, but do it) | DONE (this file) | |
| 21 | 10-23 | "desenha os modelos que estamos fazendo, o fluxo de entrada de dados e saidas, conversão e afins" — visual of where everything passes | OPEN | diagram artifact |
| 22 | 10-23 | "função que depende de outra, algo maior... calculadora basica e avançada... main que chama outras funções" — see limits and what is needed | OPEN | multi-function synth set (helpers + main) through planner+writer |
