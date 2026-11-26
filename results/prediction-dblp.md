# Prediction for coPapersDBLP, made BEFORE running it
(from the k49_norm_10NN and com-Amazon runs on the Ryzen 7 5700X, OpenMP; commit this file before running `bench.sh dblp`)

coPapersDBLP: n = 540,486 (1.61x com-Amazon), nnz = 30,491,458 (16.5x com-Amazon), unweighted, L = 9 (as com-Amazon).

| phase | model | from | 1 thread | 16 threads |
|---|---|---|---|---|
| P | ∝ nnz (unweighted: only the symmetrization) | Amazon: 25 ns/nnz, speedup 6.8x | 0.77 s | 0.15 s (speedup ~5x: arrays 0.6 GB, no longer fit in the 32 MB L3) |
| attraction | ∝ nnz | Amazon: 3.1 ns/nnz | 0.09 s | 0.015 s (memory bandwidth) |
| build | a·n + b·4^L | k49, Amazon: 9 ns/point, 4.4 ns/leaf | 6 ms | 5 ms (p x 4^L histograms, no gain) |
| centers of mass | ∝ 4^L + n | Amazon | 2 ms | 0.6 ms |
| walk | c · n · log2 n | k49: c = 0.110 µs, Amazon: c = 0.115 µs | 1.16 s | 0.14 s (Amazon speedup 8.6x) |
| gradient | ∝ n | Amazon | 0.8 ms | 0.6 ms |
| **total** | | | **≈ 2.0 s** | **≈ 0.31 s** |

Main risk: P and the attraction. com-Amazon's arrays fit in cache; coPapersDBLP's do not, and its
degrees range from 1 to 3,299, so per-nonzero costs may be higher than predicted.
