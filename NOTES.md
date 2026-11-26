# How the code works (notes for the report and the oral)

## Files
| file | what |
|---|---|
| `src/kernels.h` | the work for ONE row / point / leaf / chunk. Shared by all versions. |
| `src/seq.c` `omp.c` `pth.c` `cilk.c` | the four versions: only the loops differ (who runs which chunk). |
| `src/tree.c` | quadtree memory and geometry. |
| `src/main.c` | reads data, times each phase (fastest of `-r` runs), checks vs reference, saves results. |
| `scripts/bench.sh` | every measurement for the report → `results/*.csv`. |

## Phase by phase

**P (step 1).** Row i: bisection on beta_i so the entropy of p_{j|i} = log min(u, d_i).
Work with l_k = log(w_k/w_max) ≤ 0, so exp(beta·l_k) ≤ 1, no overflow.
H(beta) = log S − beta·Σe_k l_k / S. Uniform rows (u ≥ d_i, or equal weights) skip the bisection: beta = 0.
Then symmetrize: p_ij = (p_{j|i} + p_{i|j}) / 2n, finding (j, i) by binary search in row j.
*Parallel:* rows are independent → dynamic chunks of 256 rows. Two loops with a barrier between
(symmetrizing row i reads p_{i|j} from row j).
*Cost:* for k49, bisection (≈ 30 iterations × degree per row). For unweighted graphs every row is uniform,
so P is just the symmetrization: nnz binary searches, memory-bound.

**Attraction (step 2).** Sparse × dense: F_i = Σ_j p_ij/(1+d_ij²)·(y_i − y_j). O(nnz).
Reads `col`, `P` once (streaming) and `Y[j]` at random (gather) → **memory bound**, few flops per byte.
`col` is int32 to halve index traffic. *Parallel:* dynamic chunks of 256 rows, because the
degree varies (DBLP: 1 to 3,299). Static split by rows would give one thread far more nonzeros.

**Quadtree build (step 3).** Fixed depth L: a 2^L × 2^L grid of leaves; level l is a dense array of 4^l cells.
Cells numbered by **Morton (Z-order) code**, so children of cell c are 4c..4c+3, and the code at level l
is the leaf code shifted right by 2(L−l). Points sorted by leaf with a **counting sort** → the points
of every cell (any level) are contiguous in memory.
*Parallel counting sort (stable):* (1) bbox per block of points, (2) per-block histograms of leaves,
(3) totals per block of leaves + short sequential scan, (4) offsets: each block's first slot in each leaf,
(5) scatter. Stable ⇒ same order as sequential ⇒ identical results.
*Cost:* O(n + p·4^L): the histograms grow with threads and with L (why large L slows the build).

**Centers of mass.** Leaves from their (contiguous) points, then level by level up: parent = weighted
mean of its 4 children. Each level is a parallel loop; a barrier between levels (level l needs l+1).
Top levels (< 4096 cells) sequential: too little work to split.

**Walk (step 3).** For each point, an explicit stack from the root. Cell accepted when
diag/dist < theta (compare squares, no sqrt). Leaf: direct sum, skip the point itself by its sorted index.
The point is never inside an accepted cell: dist ≤ diag there, so diag/dist ≥ 1 > theta.
*Parallel:* points in **sorted (Morton) order**, dynamic chunks of 256: neighbouring points walk almost
the same cells → good cache reuse; dense regions cost more → dynamic scheduling.
Compute-bound: the phase that scales best.

**Z, determinism.** Z is summed per chunk of 256 points, then the chunk sums in order. The same in every
version ⇒ all four give **bit-identical** results for any thread count (check: `-cmp results/<g>-seq.mat`).
`-ffp-contract=off` stops gcc and clang from fusing multiply-adds differently.

## The three models
- **OpenMP:** `parallel for schedule(dynamic, chunk)`; `single` for the scans; implicit barriers.
- **Pthreads:** threads created per loop; work handed out by an atomic counter (`__atomic_fetch_add`),
  join = barrier. `par_blocks` for one-block-per-thread loops.
- **OpenCilk:** `cilk_for` + `#pragma cilk grainsize 1` over chunks; work stealing balances. Threads via
  `CILK_NWORKERS` (set before the runtime starts).

## Why the speedup stops (to measure and confirm on your machine)
- 8 cores / 16 hardware threads: beyond 8, two threads share a core (SMT): small gains for compute,
  ~none for memory-bound phases.
- Attraction and P on unweighted graphs: memory bandwidth, not cores.
- Build: tiny, and the p × 4^L histograms grow with p.
- Centers of mass: top levels sequential + one barrier per level.
- Walk: best scaling (compute-bound, independent points).

## Things you may be asked to change live
- Count cells visited per point: add a counter in `walk_point` (`kernels.h`), increment per popped cell.
- Scheduling: change `schedule(dynamic, ...)` / `ROW_CHUNK` / `WALK_CHUNK` (`tsne.h`) / grainsize.
- Cell criterion: `T->diag2[l] < theta2 * d2` in `walk_point`.
