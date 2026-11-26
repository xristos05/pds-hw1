# Build all versions:   make            (or: make seq / omp / pth / cilk)
# OpenCilk's clang must be on the PATH for the cilk version (env/setup.sh does that);
# or give its path:     make CILKCC=/opt/opencilk-3.0.0/bin/clang
CC       ?= gcc
CILKCC   ?= clang
CFLAGS   ?= -O3 -march=native -ffp-contract=off -Wall -Wno-misleading-indentation
MATIO_CFLAGS := $(shell pkg-config --cflags matio 2>/dev/null)
MATIO_LIBS   := $(shell pkg-config --libs matio 2>/dev/null || echo -lmatio)

COMMON  = src/main.c src/tree.c starter/data.c
HEADERS = src/tsne.h src/kernels.h starter/data.h

all: tsne_seq tsne_omp tsne_pth tsne_cilk
seq: tsne_seq
omp: tsne_omp
pth: tsne_pth
cilk: tsne_cilk

tsne_seq: $(COMMON) src/seq.c $(HEADERS)
	$(CC) $(CFLAGS) $(MATIO_CFLAGS) -o $@ $(COMMON) src/seq.c $(MATIO_LIBS) -lm

tsne_omp: $(COMMON) src/omp.c $(HEADERS)
	$(CC) $(CFLAGS) -fopenmp $(MATIO_CFLAGS) -o $@ $(COMMON) src/omp.c $(MATIO_LIBS) -lm

tsne_pth: $(COMMON) src/pth.c $(HEADERS)
	$(CC) $(CFLAGS) -pthread $(MATIO_CFLAGS) -o $@ $(COMMON) src/pth.c $(MATIO_LIBS) -lm

tsne_cilk: $(COMMON) src/cilk.c $(HEADERS)
	$(CILKCC) $(CFLAGS) -fopencilk $(MATIO_CFLAGS) -o $@ $(COMMON) src/cilk.c $(MATIO_LIBS) -lm

clean:
	rm -f tsne_seq tsne_omp tsne_pth tsne_cilk
.PHONY: all seq omp pth cilk clean
