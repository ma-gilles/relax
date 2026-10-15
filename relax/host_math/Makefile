# librelax_host_math.so: the C library's scalar acos/atan2 over arrays (relax/sampling/libm.py). Host code only: no
# CUDA, no libcuda; no -ffast-math and no tree vectorization, so no libmvec variant replaces the scalar calls.
CC ?= cc
LIB ?= librelax_host_math.so
CFLAGS := -O2 -fPIC -fno-fast-math -fno-tree-vectorize -fno-builtin -std=c99
# Accepted and ignored: the recovar NativeLibrary build passes PYTHON and RECOVAR_CUDA_INCLUDE to every make.
PYTHON ?= python
RECOVAR_CUDA_INCLUDE ?=

all: $(LIB)

$(LIB): relax_host_math.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -shared -o $@ $< -lm

clean:
	rm -f $(LIB)

.PHONY: all clean
