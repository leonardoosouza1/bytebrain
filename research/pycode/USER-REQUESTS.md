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
| 18 | 10-23 | Smaller model generates more? Must generate EXECUTABLE code; correct battery | DONE | bytewriter18/*.json, 30 runs: d32 fp (13k writer) = 100% exec+pass on test_a, len3-6 (3/3 seeds), test_num 72; d64 ter 95.3, d128 ter 100 on test_num; d16 fails (fp 48 len6, ter 0: writer loss 0.59). Smallest fully executable: d32. Ternary = STE fake-quant, no speed claim. R2 plain byte seq2seq 190k no address (3 seeds): test_a 99.0, test_num 95.3, len3 17.1, len4 4.1, len6 2.0 -> address gives length |
| 19 | 10-23 | Opus agent: process so I never drift from his requests again | DONE | protocol added to bytebrain/CLAUDE.md |
| 20 | 10-23 | Save everything he asks and do it (run later if needed, but do it) | DONE (this file) | |
| 21 | 10-23 | "desenha os modelos que estamos fazendo, o fluxo de entrada de dados e saidas, conversão e afins" — visual of where everything passes | DONE | https://claude.ai/artifact/QAXC69pSiRGZiBTKTFb23y (pipeline, planner address, writer, #22 deps, learned vs rule) |
| 22 | 10-23 | "função que depende de outra, algo maior... calculadora basica e avançada... main que chama outras funções" — see limits and what is needed | FAILED-ADAPT -> RUNNING | grouped plan (all defs, then all dispatch): test_comb 99-100, len4 29-41, len5-8 0, full 0 (3 seeds; exec ~99 = runs but wrong). Own bug class: plan format needs 2 reads of the name but planner counter only moves forward. Adaptation interleaved (per op: deps, def, disp; one pass): 100% pass/exec/plan_exact on test_comb, len4-8 and FULL 15-op calculator (45 symbols, 3x longest train plan), 3/3 seeds. multifn_il/*.json. compose.py (main = call tree, train 1-3 calls): calls4/5 100, calls6 99.3-100, calls8 93.3-100 (3 seeds; plan 33 symbols vs 23 train). main return line rendered by rule from the plan. Limit: the name spells the tree in prefix |
| 23 | 10-23 | "realmente gera código de verdade e funciona? o quanto comparado por uma LLM? ou qwen pequeno? ... é o writer que salva o modelo? e o restante?" | PARTIAL | qwen05_full.json (100/split, CPU): test_a 46, test_num 53, len3 26, len4 8, len5 9, len6 10 vs ours 100. 3B run died at start (session end 10-23); not rerun. Not done: multifn/compose arms, failure analysis |
| 24 | 10-23 | "isso pode virar um modelo 'matematico' algo que dependendo do problema ele resolve ou encontra o melhor caminho... podemos usar derivada" | PARTIAL | idea: input = problem (I/O examples), not the name; search the program by gradient over a relaxed (soft) plan, bwrap as judge |
| 25 | 10-23 | "ele pode gerar em python... isso já é uma junção de modules, mistura de experts, então da para encaixar mais coisas" | OPEN | gradsearch.py slots = soft mixture over unit-experts; next: plug planner (name/desc prior) + calc ops + compose tree as more experts in the same differentiable executor |
| 26 | 10-23 | "se ele chuta e deriva, a derivada vai apontar a direção, basta o modelo seguir e derivar, até chegar em algum momento que a distância da derivação quase não muda" — loop em nano/micro segundos | OPEN | GUESS-AND-DERIVE.md: planner guess + gradient inner loop until |dloss|~0 + bwrap judge |
| 27 | 10-26 | "vc não me falou o fluxo completo, de como treina... na hora do treino não deveria ajustar os pesos? o banco de T infinito cadê ele?" | PARTIAL | ARCHITECTURE.md sec 1 (explained; verified in code) — next: multi-tau per level not built (blockaddr gate cumsum); put it in ARCHITECTURE doc |
| 28 | 10-26 | "precisa ser um especialista em python... entender macro e micro, local e global" | PARTIAL | ARCHITECTURE.md sec 3.2 design only; local conv + global address; agent design doc |
| 29 | 10-26 | "pedir algo em pt-br que faça código... usar o qwen para entender isso e 'transcrever' para linguagem que o modelo entenda" | PARTIAL | ARCHITECTURE.md sec 3.0-3.1 + milestone S1 design only; Qwen as translator pt-BR -> internal spec (name/plan DSL) -> planner/writer |
| 30 | 10-26 | "usar o qwen como validador ou até o claude code com sonnet que olhe e julga... o quanto dela está correta?" | PARTIAL | partial_score.py built (untested); LLM judge stub, needs calibration; graded partial correctness: per-test fraction, per-block, LLM judge (Qwen / Sonnet) calibrated against bwrap |
| 31 | 10-26 | "não rode nada, só ajuste o que precisa com agents em paralelo e vamos debater" | DONE | gradsearch.py (abs/reverse bugs, io grade, ambiguity, anneal, stop-dloss, guess arms), partial_score.py + qwen_compare.py (--chat/--full/--suite), ARCHITECTURE.md; py_compile only, nothing run |
