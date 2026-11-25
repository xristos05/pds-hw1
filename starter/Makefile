# The starter: a reader for the data, and a program that prints a summary.
#   make && ./readdata ../data/k49_norm_10NN
CC      ?= gcc
CFLAGS  ?= -O2 -Wall
MATIO_CFLAGS := $(shell pkg-config --cflags matio 2>/dev/null)
MATIO_LIBS   := $(shell pkg-config --libs matio 2>/dev/null || echo -lmatio)

readdata: readdata.c data.c data.h
	$(CC) $(CFLAGS) $(MATIO_CFLAGS) -o $@ readdata.c data.c $(MATIO_LIBS)

clean:
	rm -f readdata
.PHONY: clean
