# Plan optymalizacji mini_az (Etap 1–4)

Referencyjny snapshot planu i stanu wdrożenia, żeby można było podjąć
wątek z innego komputera. Ostatnia aktualizacja: 2026-04-21.

**Cel nadrzędny**: ~3–5× wall-clock throughput (iter/4h) bez zmian w semantyce
RL. Baseline ~80 iter w 4h na Azure D96s → docelowo ~250–350 iter.

---

## Etap 1 — Quick wins w inference — UKOŃCZONY (commit `0c9bf58`)

Oczekiwany zysk: ~1.5–1.8× throughput.

- **1.1 bfloat16 autocast** w `policy_value_single` / `forward_policy_value`.
  EPYC 9004 ma AVX-512-BF16 → ~2× na matmul. Value/softmax pozostaje w fp32.
  *Uwaga*: `torch.compile` pominięty — na CPU bez AMX bf16 autocast potrafi być
  10–13× wolniejszy, więc domyślnie wyłączony i włączany flagą
  `--bf16_inference` (commit `992a9db`).
- **1.2 Vectorize `_encode_pieces`** — bitboard → `np.unpackbits` → 8×8 plane,
  jedna konwersja per (typ bierki × kolor). ~13× szybszy encode, ~25% mniej
  CPU/iter na workerach.
- **1.3 Shared-memory wagi** — `/dev/shm/mini_az_weights_*.pt` + `mp.Value('q')`
  licznik wersji zamiast 92× pickle-into-Queue. Broadcast 2 GB/iter → ~0.01s
  + atomic tmp+rename.

**Pliki**: `mini_az/encoding.py`, `mini_az/network.py`, `mini_az/selfplay.py`,
`mini_az/main.py`, `mini_az/trainer.py`.

---

## Etap 2 — MCTS leaf batching — UKOŃCZONY (commit `1adf96e`)

Oczekiwany zysk: dodatkowe ~2–3× → łącznie ~3–5× vs baseline.

- **2.1 Centralny inference server — POMINIĘTY (decyzja świadoma)**. Analiza
  pokazała, że na 92 rdzeniach × 1-core workery rozproszone batche K=8 per
  worker (~350k leafs/s) biją konsolidację w jeden proces na 24 wątkach
  (~15–20k leafs/s). Oszczędność RAM z jednej kopii wag (~2 GB) nieistotna przy
  386 GB na VM.
- **2.2 Virtual-loss leaf batching w MCTS** — nowe `ChessNet.policy_value_batch`
  (N boardów → N priors+v w jednym forwardzie, bf16 autocast).
  `policy_value_single` to teraz batch-of-1. `mcts_search(..., leaf_batch_size=K)`
  zbiera K nierozwiniętych liści pod virtual-loss (`EdgeStats.N_virt`),
  robi batched forward, commituje wartości odwracając VL. Terminale i
  transposition-cache pomijają eval. CLI: `--mp_leaf_batch N` (default 1 =
  identyczne z poprzednim zachowaniem).

**Pliki**: `mini_az/network.py`, `mini_az/mcts.py`, `mini_az/selfplay.py`,
`mini_az/main.py`, `mini_az/trainer.py`.

---

## Etap 3 — Async training loop i eval off-critical-path — UKOŃCZONY

Oczekiwany zysk: +1.3× (workery trzymają 100% rdzeni, bez pauz treningu/eval).

- **3.1 Trener w osobnym procesie** (commit `8ba953d`). `mini_az/trainer.py`
  działa jako `mp.Process` równolegle do 92 workerów; własny `ChessNet` +
  `opt` + `scheduler` + `ReplayBuffer`; drenuje `out_q` w pętli, robi
  `steps_per_iter` kroków gradientu per iter, zapisuje snapshoty,
  publikuje nowe wagi do `/dev/shm` i bumpuje `weights_version`, raportuje
  metryki do orchestratora przez `metrics_q`. **Kluczowa zmiana**: `pause_ev`
  NIE jest już ustawiany podczas treningu.
- **3.2 Eval w osobnym procesie** (commit `5deef2a`). `mini_az/evaluator.py`
  z `run_eval_job()` — one-shot proc spawnowany na granicy `--eval_every`;
  bierze snapshot z `/dev/shm`, gra SF/random/vs-opponent, decyduje o
  promocji, atomicznie zapisuje `best.pt`. Liczba wątków: `--eval_threads`
  (default auto = `max(4, workers//12)`). Jeśli poprzedni eval wciąż działa
  gdy wpada kolejny — skip (nie piętrzymy SF subprocessów). Workery i trener
  nie pauzują.

**Pliki**: `mini_az/trainer.py` (nowy), `mini_az/evaluator.py` (nowy),
`mini_az/train_helpers.py` (nowy), `mini_az/main.py`, `mini_az/selfplay.py`.

---

## Ulepszenie eval gating (commit `ec5c2ce`, 2026-04-21)

Po 4h uczenia zauważono, że promocja modelu do `best.pt` potrafiła regresować
do łatwiejszej bramki (np. z `sf_score` na `rnd`). Wprowadzony **system tierów**:

```
rnd(0) < self(1) < sf_easy_win(2) < sf_easy_score(3) < sf_win(4) < sf_score(5)
```

- `prev_kind == gate_kind` → porównanie na tej samej skali (stare zachowanie).
- `tier(gate_kind) > tier(prev_kind)` → zawsze promuj i zablokuj wyższy tier
  (zdobyliśmy twardszy sygnał).
- `tier(gate_kind) < tier(prev_kind)` → porównaj metrykę z bieżącego evalu na
  skali `prev_kind` (nie schodzimy do łatwiejszej bramki).

Dodatkowo: gra kończąca na `max_plies` w `play_vs_stockfish` liczy się jako
**draw** (nie adjudykacja przez materiał) — zapobiega fałszywym „wygranym" z SF.

**Parametry eval odchudzone** (kompromis między szybkością a wariancją):
- `--eval_games 6 --eval_sims 200` (było 10/400)
- `--rand_eval_games 10 --rand_eval_sims 128` (było 20/200)
- `--self_eval_games 8` (było 12)
- `--sf_eval_elo_easy -1` (wyłączony)
- `train_continue_v3.sh` zachowuje replay buffer (bez `--clear_buffer`).
- `tmux_run_learning.sh` domyślnie odpala `train_continue_v3.sh`.

**Te parametry są nietykalne — zostają.**

---

## Etap 4 — Stockfish cache + szersza sieć + replay diskless

### 4.1 Cache SF teacher policy — UKOŃCZONY (commit `506aa75`, 2026-04-21)

Oczekiwany zysk: ~20–30% redukcja czasu gry na workerach z SF (hit rate ~20–30%
w otwarciach/mid-game).

- `SfTeacherCache` — per-worker LRU keyed by `(transposition_key, depth,
  multipv, cp_cap)`. Przechowuje intermediate `{move: cp}` przed softmaxem.
- `cp_soft_scale` i `eps` **celowo poza kluczem** — wpływają tylko na
  post-softmax shaping, re-softmax na hicie (~100 μs vs ~50 ms engine.analyse).
- Movetime path **nie jest cacheowany** (wall-clock-limited → niedeterministyczny).
- CLI: `--sf_teacher_cache_size` (default 10000).
- Worker 0 loguje `hits/misses/size/hit_rate` co 60 s.
- Testy: `tests/test_stockfish_cache.py` — 7 testów (hit identyczny, depth/multipv
  w kluczu, eps/cp_soft_scale poza kluczem, LRU eviction, movetime uncached,
  `cache=None` disabled).

### 4.2 Batched SF evals — UKOŃCZONY (commit `d632e41`, 2026-04-21)

Zamiast engine pool (2× SF per worker → za dużo RAM/CPU) wybrano *prefetch
w tle, dzielący silnik SF z głównym wątkiem przez RLock*.

- `SfTeacherPrefetcher` — wątek tła. Po `board.push` w self-play wysyła
  spekulatywne zapytanie o pozycję N+1; wynik zapisuje w cache. Gdy main
  dochodzi do ply N+1 → cache hit → pomijamy ~50 ms engine call.
- Silnik SF jest thread-unsafe → wspólny `RLock`. `sf_teacher_policy_legal`
  i `sf_eval_cp_white` dostały opcjonalny `lock=` param.
- Kolejka bounded (default 2), `submit()` dropuje cicho przy przepełnieniu —
  prefetching to best-effort.
- CLI: `--sf_teacher_prefetch` (flaga on/off).
- Worker 0 loguje dodatkowo `queued/done/dropped/skipped_hit`.
- Testy: `tests/test_stockfish_prefetch.py` — 5 testów (background fill →
  main hit, skip when cached, movetime refused, lock-serialised main,
  bounded-queue drops).

### 4.3 Szersza sieć — DO ROZWAŻENIA (jeszcze nie zrobione)

Tylko **jeśli iter-time < 30 s** po E1–E4.2 i mamy czas. Zmiana architektury
= inkompatybilność z obecnymi checkpointami (wymagany `train_fresh_v3.sh`
lub distillation z szerszej sieci).

- Obecnie: 96 kanałów × 10 bloków = 5.6 M params; bottleneck `board_fc`
  (3.1 M z 5.6 M).
- Kandydaci: `128×12 ≈ 10 M` lub `160×10 ≈ 14 M`. 386 GB RAM pomieści.
- **Pliki**: `mini_az/network.py:51`, `mini_az/main.py:151`, `mini_az/selfplay.py:488`.

### 4.4 Replay diskless — DO ROZWAŻENIA (jeszcze nie zrobione)

Zamiast co N iter zapisywać `replay.pkl.gz` na dysk, trzymać w shared-memory
arenie. Kandydat do rozważenia po obserwacji czasów I/O przy większym buforze.

---

## Planowany porządek wdrożenia (historyczny)

1. **Day 1–2**: Etap 1 (quick wins). Mierzymy wall-time per iter przed/po.
2. **Day 3–5**: Etap 2.1 inference server (POMINIĘTY) → Etap 2.2 leaf batching.
3. **Day 6**: Etap 2.2 — wymaga 2.1, MCTS refaktor. (Zrobione bez 2.1.)
4. **Day 7–8**: Etap 3 (async training + eval).
5. **Day 9+**: Etap 4 wybiórczo — SF cache (tanio), potem szersza sieć.

---

## Aktualny stan (snapshot — 2026-04-21)

| Etap | Status | Commit |
|---|---|---|
| 1.1 bf16 autocast (opt-in) | ✅ | `0c9bf58` / `992a9db` |
| 1.2 vectorize encoding | ✅ | `0c9bf58` |
| 1.3 shared-mem weights | ✅ | `0c9bf58` |
| 2.1 central inference server | ⏭️ świadomie pominięty | — |
| 2.2 virtual-loss leaf batching | ✅ | `1adf96e` |
| 3.1 async trainer process | ✅ | `8ba953d` |
| 3.2 async eval process | ✅ | `5deef2a` |
| eval gating tier system + param tuning | ✅ | `ec5c2ce` |
| 4.1 SF teacher LRU cache | ✅ | `506aa75` |
| 4.2 SF teacher prefetch (tło + RLock) | ✅ | `d632e41` |
| 4.3 szersza sieć (128×12 / 160×10) | ⬜ do rozważenia | — |
| 4.4 replay diskless | ⬜ do rozważenia | — |

**Testy**: 112/112 zielone (100 dotychczasowych + 7 dla 4.1 cache + 5 dla 4.2 prefetch).

---

## Walidacja po każdym etapie

```bash
python -m pytest tests/ -x -q --tb=short     # 112 zielonych
python bench_v3.py                           # throughput searches/s
python elo_benchmark.py --net mini_az.pt --games 10   # Elo regression
```

Sygnały alarmowe:
- `bench_parallel(workers=92, sims=200)` < 250 searches/s → coś nie działa
  (baseline ~60, docelowo > 250).
- Elo spadł > 50 pkt po etapie optymalizacji → rollback (bug w RL, nie
  performance).
