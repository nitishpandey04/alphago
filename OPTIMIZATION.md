# Optimization plan — making training fast enough for bigger boards

This is roadmap item **#1**: speed. The goal is to make self-play fast enough
that 7×7 and 9×9 training (items #2 and #3) become practical. This doc is a plan,
not a changelog — update it as work lands.

## The problem: MCTS is CPU-bound, the GPU is idle

At 5×5 the network is ~314k params, so a forward pass is dominated by fixed
launch/transfer overhead, not compute. The real cost is **pure-Python tree
work**: `board.clone()`, `board.play()`, `legal_actions()` (a flood-fill suicide
check at *every* empty point), and the `select_child` loop. Measured GPU
utilization during self-play is ~3%.

Two independent things are slow:
1. **Per-node cost** — each MCTS node does too much Python work.
2. **Single-core** — `selfplay.py` steps 128 games in *lockstep on one core*;
   only the GPU forward pass is batched. That looks parallel but isn't
   multi-core. All tree work runs serially.

These multiply. Fixing both stacks: e.g. a 3× per-node win × ~10× cores ≈ 30×.

## Step 0: profile first (don't guess)

Before optimizing, measure where per-node time actually goes and whether the net
should even be on the GPU at this size.

- `cProfile` / `py-spy` a single-process self-play round; confirm the hot
  functions (`legal_actions`, `_group_and_liberties`, `play`, `clone`).
- Benchmark **CPU vs GPU inference** at batch sizes 1, 8, 32. If CPU inference is
  competitive for the tiny net, multiprocessing gets *much* simpler (no GPU
  coordination — see below).

## Lever A: multi-core via multiprocessing (biggest single win)

The GIL serializes pure-Python, so `threading` won't help — **separate
processes** each get their own GIL. The natural axis is **data-parallel
self-play actors**: N workers each generate games with their own copy of the
net; samples flow back to the trainer.

Two architectures:
- **(a) Independent actors** — each worker does CPU tree search *and* its own
  forward passes. Simplest. Workers contend for the GPU and lose global
  cross-game batching.
- **(b) Inference server** — workers do only CPU tree work and send leaf-eval
  requests to one GPU process that batches across *all* workers. Best of both,
  more plumbing (queues + a batching loop). This is what KataGo/Leela do.

**Recommendation:** start with (a). If the step-0 benchmark shows CPU inference
is competitive, run inference *on CPU inside each worker* — then it's
embarrassingly parallel with zero GPU coordination. Move to (b) only if GPU
contention shows up as the bottleneck.

### Expected speedup on the target CPU (Intel Core Ultra 7 265K)

8 P-cores + 12 E-cores = **20 threads, no hyperthreading**. E-cores run ~60–70%
of P-core throughput; leave 1–2 cores for the trainer/OS. Aggregate ≈
`8 + 12×0.65 ≈ 16` P-core-equivalents. After IPC/aggregation overhead and
Amdahl (training + arena are serial phases), expect a realistic **~8–14×**
self-play throughput, not a clean 20×.

### Gotchas (these bite)

- **Use the `spawn` start method, not `fork`** — CUDA contexts don't survive
  `fork`.
- **`torch.set_num_threads(1)` in every worker** — otherwise each process spawns
  its own intra-op thread pool and they oversubscribe cores, slowing everything
  down.
- **Per-worker RNG seeding** — derive distinct, reproducible seeds per worker so
  games differ but runs are repeatable.
- **CUDA context per process** (~few hundred MB each) — fine on 16 GB, but a
  reason CPU inference can be cleaner.

## Lever B: cut per-node cost (single-thread, composes with A)

- **Vectorize board/group logic with NumPy** — replace Python-set flood fill in
  `_group_and_liberties` / `legal_actions` with array operations. Biggest
  single-thread win; `legal_actions` is O(N² × floodfill) per node today.
- **Lazy expansion** — don't create a child (with a full `board.play`) for every
  legal move on first visit; expand children on demand. Cuts work and memory,
  matters a lot at 9×9 where the branching factor explodes.
- **Virtual loss** — lets multiple leaves *within one game* be selected and
  batched into a single forward pass, instead of one leaf per game per sim.
- **Incremental legality / group tracking** — maintain liberties/groups across
  moves instead of recomputing from scratch each `play`.

## Lever C: sample efficiency (fewer games for the same strength)

- **Dihedral symmetry augmentation** — 8 rotations/reflections of each
  (state, policy) sample. ~8× effective data for near-free; not a speed fix but
  reduces how many games training needs.

## Suggested attack order

1. **Profile** + CPU-vs-GPU inference benchmark (step 0).
2. **Independent-actor multiprocessing** (Lever A) — simplest, biggest lever.
3. **Vectorize per-node cost** (Lever B: NumPy board, lazy expansion).
4. **Virtual loss** and/or **inference server** (Lever A-b) only if GPU
   contention becomes the bottleneck after 2–3.
5. **Symmetry augmentation** (Lever C) — independent, can land anytime.

Benchmark before/after at each step (self-play positions/sec) so gains are
measured, not assumed. Once self-play is fast, move to roadmap #2 (bigger
boards) and #3 (experiments on them).
