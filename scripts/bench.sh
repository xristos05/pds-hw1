#!/usr/bin/env bash
# Run every measurement of the report and save the raw data in results/.
#
#   bash scripts/bench.sh small     # k49_norm_10NN and com-Amazon (run this first)
#   bash scripts/bench.sh dblp      # coPapersDBLP (run AFTER writing your prediction)
#   bash scripts/bench.sh all       # both
#
# For clean timings: plug in the laptop / close other programs, and run nothing
# else meanwhile. The whole thing takes a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

WHAT=${1:-small}
case "$WHAT" in
  small) GRAPHS="k49_norm_10NN com-Amazon" ;;
  dblp)  GRAPHS="coPapersDBLP" ;;
  all)   GRAPHS="k49_norm_10NN com-Amazon coPapersDBLP" ;;
  *) echo "usage: $0 small|dblp|all"; exit 1 ;;
esac

THREADS="1 2 4 6 8 12 16"
REPS=5
mkdir -p results

make -s all

# The machine, for the report
{
  echo "date: $(date -Is)"
  lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|L2 cache|L3 cache|MHz"
  free -h | head -2
  gcc --version | head -1
  clang --version | head -1
  grep -E "^CFLAGS" Makefile
  uname -a
} > results/machine.txt

for g in $GRAPHS; do
  echo "=== $g: correctness (all versions must be identical to seq) ==="
  ./tsne_seq data/$g -r 1 -o results | tee -a results/log.txt
  for v in omp pth cilk; do
    ./tsne_$v data/$g -r 1 -t 16 -o results -cmp results/$g-seq.mat | tee -a results/log.txt
  done

  echo "=== $g: time against threads ==="
  ./tsne_seq data/$g -r $REPS -csv results/timings.csv | grep time
  for v in omp pth cilk; do
    for p in $THREADS; do
      ./tsne_$v data/$g -r $REPS -t $p -csv results/timings.csv | grep time | sed "s/^/  $v p=$p /"
    done
  done

  echo "=== $g: tree depth L (seq and omp, 16 threads) ==="
  for L in 5 6 7 8 9 10 11; do
    ./tsne_seq data/$g -r 3 -L $L -csv results/depth.csv > /dev/null
    ./tsne_omp data/$g -r 3 -L $L -t 16 -csv results/depth.csv | head -2 | tail -1 | sed "s/^/  L=$L /"
  done

  echo "=== $g: theta (omp, 16 threads): speed vs accuracy ==="
  for th in 0.1 0.25 0.5 0.75 1.0; do
    ./tsne_omp data/$g -r 3 -t 16 -theta $th -csv results/theta.csv | grep -E "vs reference" | sed "s/^/  theta=$th /"
  done
done

if [[ " $GRAPHS " == *" k49_norm_10NN "* ]]; then
  echo "=== k49: direct O(n^2) vs Barnes-Hut, and the crossover n ==="
  for p in 1 16; do
    ./tsne_omp data/k49_norm_10NN -r 3 -t $p -direct -csv results/direct.csv | grep -E "direct"
    ./tsne_omp data/k49_norm_10NN -r 5 -t $p -crossover results/crossover.csv | tail -12
  done
fi

echo
echo "Done. Raw data in results/: timings.csv depth.csv theta.csv direct.csv crossover.csv machine.txt log.txt"
