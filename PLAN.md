# Plan optymalizacji mini_az (Etap 1–4)

Referencyjny snapshot planu i stanu wdrożenia, żeby można było podjąć
wątek z innego komputera. Ostatnia aktualizacja: 2026-04-21 (Etap 5 dodany).

**Cel nadrzędny**: ~3–5× wall-clock throughput (iter/4h) bez zmian w semantyce
RL. Baseline ~80 iter w 4h na Azure D96s → docelowo ~250–350 iter.

**Stan po 4h runu (iter 86→161)**: ~76 iter/4h. Throughput 2× vs baseline,
ale poniżej docelowego 118 (eval overhead + nieoptymalne timings). Model
zaczął zdobywać punkty vs SF 1320 (sf_score=0.083 na iter 140). Policy i value
head się uczą (pol_loss 2.29→2.07, vz_corr 0.85→0.87). Główny problem:
rosnące draw rate i threefold repetitions.

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

## Etap 5 — Training quality & throughput tuning (bez zmian arch.)

Obserwacje z 4h runu (iter 86→161, ~76 iteracji):
- iter_time stabilizuje się na ~135–165s (cel planu: ~122s, delta wynika z eval overhead)
- pol_loss: 2.29 → 2.07 ✅ (systematyczny spadek — model się uczy)
- vz_corr: 0.85 → 0.87 ✅ (value head lepiej koreluje z wynikiem gry)
- sf_score: 0.0 → 0.083 na iter 140 ✅ (pierwsze punkty vs SF 1320 Elo!)
- threefold_per_game: 10 → 14 ⚠️ (rosnące — model nadmiernie powtarza)
- rep_plies_per_game: 7 → 9 ⚠️ (j.w.)
- draws: ~26% → ~31% ⚠️ (lekki wzrost — model uczy się "nie przegrywać")

**4.4 replay diskless** — po analizie danych wyceniony jako **niskopriorytowy**.
Bufor żyje w RAM trenera, dump na dysk co 10 iter to safety backup zajmujący
~2–5s (przy 386 GB RAM + fast SSD). Nie wpływa na iter_time.

Poniższe punkty nie ruszają architektury sieci i są kompatybilne z istniejącymi
checkpointami. Posortowane od najłatwiejszego / najwyższy gain first.

---

### 5.1 Collate vectorization — OCZEKIWANY ZYSK: −15–25% train_step time

**Problem**: `collate()` w `training.py` iteruje per-sample w Pythonie:
```python
for i, s in enumerate(samples):
    L = s.moves_fs.shape[0]
    fs[i, :L] = torch.tensor(s.moves_fs, ...)
    ...
```
Przy batch=512 × ~30 legal moves = 15k indywidualnych tensor copy w Pythonie.
Cały train_step trwa ~1.0s (150 steps × 1.0s = 150s → ~50% iter_time).

**Zmiana**: Przebudować `collate()` na bulk numpy stack → single `torch.from_numpy`:
1. Pre-alokacja numpy arrays `(B, Lmax)` dla fs/ts/pr/mask/target_pi.
2. `np.copyto(dst[i, :L], s.moves_fs)` zamiast per-element torch copy.
3. Jeden `torch.from_numpy(...).to(device)` na końcu zamiast B×5 tensorów.
4. `flip_sample_lr` wektorowo na batchu (batch-level LR flip).

**Pliki**: `mini_az/training.py:collate()`, `mini_az/training.py:flip_sample_lr()`.

**Ryzyko**: Żadne — pure refaktor, identyczne wyniki liczbowe.

---

### 5.2 Anti-repetition tuning — OCZEKIWANY ZYSK: ~15–25% mniej draws

**Problem**: Dane z 4h runu: `threefold_per_game` rośnie 10→14,
`rep_plies_per_game` rośnie 7→9, draws 26→31%. Model uczy się shufflować
zamiast konwertować. Obecne kary rep_penalty:
- `adv >= 3`: 0.10 (agresywne — OK)
- `adv >= 1`: 0.25
- `else`: 0.50

**Zmiany w `selfplay.py:make_game_samples_unified`**:

A) **Ostrzejsze kary za powtórzenie** (nie daj się do drawu):
   - `adv >= 3`: 0.05 (prawie blokuj powtórzenie przy dużej przewadze)
   - `adv >= 1`: 0.12
   - `else`: 0.35

B) **Draw-by-repetition penalty w z-targetze**: jeśli gra kończy się
   remisem (z=0.0) a strona miała `adv >= 2` w ostatnich 10 ply, daj
   `z = -0.15` zamiast `0.0` (karanie za zmarnowaną przewagę). Symetrycznie
   dla strony w tyle: `z = +0.10` (nagrodzenie za obronę).

C) **Monotonic progress bonus**: dodaj mały bonus (+0.02) do pi_target dla
   ruchów które zwiększają `material_score` o ≥1 — zachęcanie do wymiany
   na korzyść. Opcjonalnie gated na `adv >= 1`.

**Pliki**: `mini_az/selfplay.py` (pętla główna + post-game z-relabeling).

**Ryzyko**: Niskie. Karanie drawu przy przewadze to standard w AZ-variants.
Wymaga monitorowania: jeśli draws spadną < 15% przy wzroście resign → za agresywne.

---

### 5.3 Larger effective batch — OCZEKIWANY ZYSK: stabilniejsze gradienty

**Problem**: Batch=512, steps=150 → 76,800 training samples/iter. Przy
`recent_frac=0.70, recent_window=200k` te same pozycje widziane 2–3× w trakcie
jednego itera. Mniejszy noise w gradientach → szybsza konwergencja, szczególnie
value head (val_loss plateau ~0.65).

**Zmiana**: `--batch 1024 --steps_per_iter 80`. Ten sam łączny compute
(81,920 vs 76,800 samples), ale 2× mniejsza wariancja gradientów. Alternatywnie
gradient accumulation 2×512 jeśli batch=1024 nie mieści się w pamięci (mało
prawdopodobne na CPU z 386 GB).

**Pliki**: Tylko `train_continue_v3.sh` (zmiana parametrów CLI).

**Ryzyko**: Minimalny. Gdyby val_loss się pogorszyło (underfitting z mniejszej
ilości steps), cofnij na 512/150.

---

### 5.4 Per-worker persistent transposition cache — OCZEKIWANY ZYSK: −5–10% selfplay time

**Problem**: `pv_cache` w `mcts_search()` jest tworzony na nowo dla każdego
searcha (20k entries max). Otwarcia (1.e4, 1.d4 itd.) są re-ewaluowane od zera
w każdej grze. Przy 200 sims i ~10 identycznych pozycji otwierania = ~10 forward
passes zmarnowanych per game × 80 games/iter × 68 workerów.

**Zmiana**: Worker trzyma `worker_pv_cache: dict[Hashable, PV]` (LRU, 50k entries)
persistujący między grami. `mcts_search()` dostaje opcjonalny `ext_cache=` param.
Na początku każdego searcha: lookup ext_cache → seed pv_cache. Na końcu:
merge wyników do ext_cache (write-through). Entries starsze niż N=100 gier → evict
(żeby stale wagi nie zanieczyszczały cache).

**Pliki**: `mini_az/mcts.py` (parametr + merge logic), `mini_az/selfplay.py`
(worker trzyma cache, przekazuje do mcts_search).

**Ryzyko**: Niskie — stale wagi mogą dawać nieaktualne evaluacje, ale eviction
po N grach temu zapobiega. Odwrócenie: `--persistent_pv_cache 0` (off).

---

### 5.5 Smooth temperature annealing — OCZEKIWANY ZYSK: lepszy training signal

**Problem**: Obecna temperatura w selfplay ma skokowe progi:
```
ply < 12:  temp = 1.0
ply 12–28: temp = 0.35
ply >= 28: temp = 0.08
```
Skok z 1.0 → 0.35 przy ply 12 jest dyskontynuacją — ruchy tuż przed progiem
mają zupełnie inny noise-level od ruchów tuż po. To utrudnia value head:
pozycje w okolicy ply 12 mają niespójny training signal.

**Zmiana**: Liniowe przejście zamiast skoków:
```python
if tply < 12:
    temperature = 1.0
elif tply < 30:
    temperature = 1.0 - (1.0 - 0.08) * (tply - 12) / 18  # 1.0 → 0.08 liniowo
else:
    temperature = 0.08
```

**Pliki**: `mini_az/selfplay.py` (sekcja temp w pętli głównej).

**Ryzyko**: Żadne — kosmetyczna zmiana, kompatybilna z checkpoint. Monitoruj
`avg_pi_ent` — powinno spadać bardziej gładko.

---

### 5.6 Adaptive SF teacher depth — OCZEKIWANY ZYSK: −10–15% SF CPU time

**Problem**: SF teacher depth=8 to ~11ms/call (bench). W otwarciach (ply<20)
depth=6 daje prawie identyczny wynik: otwierania są w book/dobrze known,
a depth=8 marnuje ~5ms. W późnej grze depth=10 daje lepszy sygnał (pozycje
taktyczne, mating nets).

**Zmiana**: Zamiast stałego `depth=8`:
```python
if tply < 20:
    sf_depth = max(6, sf_teacher_depth - 2)
elif tply > 80:
    sf_depth = sf_teacher_depth + 2
else:
    sf_depth = sf_teacher_depth
```
CLI: `--sf_teacher_depth_adaptive` (flaga on/off, default off).

**Pliki**: `mini_az/selfplay.py` (wewnątrz `use_teacher` bloku).

**Ryzyko**: Minimalne. Shallow depth w otwarciach to standard. Gdyby policy
quality spadła (wyższy pol_loss) → wyłącz flagę.

---

### 5.7 Moves-left search integration — OCZEKIWANY ZYSK: lepsze play, mniej shufflingu

**Problem**: Model ma wytrenowaną `ml_head` (moves-left prediction), ale
jest używana TYLKO w loss. MCTS search jej nie wykorzystuje. Tymczasem
info o "ile ply do końca" może pomóc w selekcji ruchów: preferuj linie
które szybciej kończą grę (= mniej shufflingu, mniej remisów).

**Zmiana**: W `mcts_search`, po batched eval oprócz `priors, v` pobierz też
`ml_pred`. Zmodyfikuj backup `commit()`: dodaj dyskonto `v * gamma^(ml_pred/scale)`
z małym `gamma=0.998` — linie z mniejszym predicted moves_left mają lekko
wyższą efektywną wartość. To standardowa technika z MuZero.

Implementacja:
1. `policy_value_batch` zwraca `List[(priors, v, ml)]` zamiast `List[(priors, v)]`.
2. `mcts_search` opcjonalnie dyscountuje: `v_eff = v * (gamma ** (ml / scale))`.
3. CLI: `--mcts_ml_discount 0.998` (1.0 = wyłączony).

**Pliki**: `mini_az/network.py:policy_value_batch`, `mini_az/mcts.py`.

**Ryzyko**: Umiarkowane. Discount może karać wygrywające ale długie linie
(np. endgame K+R vs K). Dlatego `gamma` musi być blisko 1.0 i kontrolowalne.
Start z 0.998 i `scale=100` → efekt ledwie zauważalny, rośnie dopiero przy
dużym `ml`.

---

### 5.8 Dynamic games_per_iter — OCZEKIWANY ZYSK: −5–8% wasted wait time

**Problem**: `games_per_iter=80` jest stałe. Trainer blokuje się w
`out_q.get(timeout=10)` czekając na 80-tą grę. Jeśli 79 gier jest gotowych
po 70s ale 80-ta trwa wolno (długa gra, SF heavy) — trainer czeka.

**Zmiana**: Zamiast stałego progu, trainer zbiera gry przez `min_selfplay_sec`
(np. 90s) i potem drainuje ile jest w kolejce:
```python
games_collected = 0
deadline = time.time() + min_selfplay_sec
while time.time() < deadline or games_collected < min_games:
    ...drain...
# po deadline: drain all remaining (non-blocking)
while not out_q.empty():
    ...drain...
```
CLI: `--min_selfplay_sec 90 --min_games 40`.

**Pliki**: `mini_az/trainer.py` (pętla drainowania).

**Ryzyko**: Niskie. Gwarantujemy minimum `min_games` (np. 40), więc nie
będzie pustego treningu. Adaptive buffer fill = mniejszy jitter w iter_time.

---

### Sugerowana kolejność implementacji Etapu 5

| # | Punkt | Trudność | Gain | Priorytet |
|---|---|---|---|---|
| 1 | 5.1 Collate vectorization | Niska | −15–25% train time | 🔴 NAJWYŻSZY |
| 2 | 5.2 Anti-repetition tuning | Niska | −15–25% draws | 🔴 WYSOKI |
| 3 | 5.5 Smooth temp annealing | Trivial | lepszy signal | 🟡 ŚREDNI |
| 4 | 5.3 Larger batch | Trivial (CLI) | stabilne grady | 🟡 ŚREDNI |
| 5 | 5.4 Persistent PV cache | Średnia | −5–10% selfplay | 🟡 ŚREDNI |
| 6 | 5.6 Adaptive SF depth | Niska | −10–15% SF CPU | 🟢 NISKI |
| 7 | 5.8 Dynamic games_per_iter | Średnia | −5–8% wait | 🟢 NISKI |
| 8 | 5.7 ML search integration | Wysoka | mniej shufflingu | 🟢 EKSPERYMENT |

Rekomendacja: **5.1 + 5.2 + 5.5 + 5.3** w jednym commicie → następny 4h run →
walidacja na danych. Potem 5.4 + 5.6. Punkt 5.7 (ML search) zostawić na eksperyment
po walidacji powyższych.

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
| 4.4 replay diskless | ⏭️ niskopriorytetowe (analiza → brak bottlenecku) | — |
| 5.1 collate vectorization | ⬜ następny | — |
| 5.2 anti-repetition tuning | ⬜ następny | — |
| 5.3 larger effective batch (1024) | ⬜ następny | — |
| 5.4 persistent per-worker PV cache | ⬜ po walidacji 5.1–5.3 | — |
| 5.5 smooth temperature annealing | ⬜ następny | — |
| 5.6 adaptive SF teacher depth | ⬜ po walidacji 5.1–5.3 | — |
| 5.7 moves-left search integration | ⬜ eksperyment | — |
| 5.8 dynamic games_per_iter | ⬜ po walidacji 5.1–5.3 | — |

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
