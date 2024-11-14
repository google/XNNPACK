#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x8-packw/scalar.c.in -D NR=2  -D KBLOCK=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x2-gemm-goi-scalar-u2.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=4  -D KBLOCK=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x4-gemm-goi-scalar-u2.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=8  -D KBLOCK=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x8-gemm-goi-scalar-u2.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=16 -D KBLOCK=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x16-gemm-goi-scalar-u2.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=32 -D KBLOCK=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x32-gemm-goi-scalar-u2.c &

tools/xngen src/x8-packw/scalar.c.in -D NR=2  -D KBLOCK=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x2-gemm-goi-scalar-u4.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=4  -D KBLOCK=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x4-gemm-goi-scalar-u4.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=8  -D KBLOCK=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x8-gemm-goi-scalar-u4.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=16 -D KBLOCK=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x16-gemm-goi-scalar-u4.c &
tools/xngen src/x8-packw/scalar.c.in -D NR=32 -D KBLOCK=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x32-gemm-goi-scalar-u4.c &

### C4 packing
tools/xngen src/x8-packw/kr-scalar.c.in -D NR=8  -D KR=4 -D TYPE=int8_t -D IZP=0 -o src/qs8-packw/gen/qs8-packw-x8c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-scalar.c.in -D NR=16 -D KR=4 -D TYPE=int8_t -D IZP=0 -o src/qs8-packw/gen/qs8-packw-x16c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-scalar.c.in -D NR=32 -D KR=4 -D TYPE=int8_t -D IZP=0 -o src/qs8-packw/gen/qs8-packw-x32c4-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-scalar.c.in -D NR=64 -D KR=4 -D TYPE=int8_t -D IZP=0 -o src/qs8-packw/gen/qs8-packw-x64c4-gemm-goi-scalar.c &

### C8 packing
tools/xngen src/x8-packw/kr-scalar.c.in -D NR=8  -D KR=8 -D TYPE=int8_t -D IZP=0   -o src/qs8-packw/gen/qs8-packw-x8c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-scalar.c.in -D NR=16 -D KR=8 -D TYPE=int8_t -D IZP=0   -o src/qs8-packw/gen/qs8-packw-x16c8-gemm-goi-scalar.c &

tools/xngen src/x8-packw/kr-scalar.c.in -D NR=8  -D KR=8 -D TYPE=int8_t -D IZP=128 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x8c8-gemm-goi-scalar.c &
tools/xngen src/x8-packw/kr-scalar.c.in -D NR=16 -D KR=8 -D TYPE=int8_t -D IZP=128 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x16c8-gemm-goi-scalar.c &

### AVXVNNI micro-kernels
### C8 packing
tools/xngen src/x8-packw/kr-c4-avxvnni.c.in -D NR=8  -D KR=4 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=0   -D AVX=2  -D PREFETCH=0 -o src/qs8-packw/gen/qs8-packw-x8c4-gemm-goi-avxvnni.c &
tools/xngen src/x8-packw/kr-c4-avxvnni.c.in -D NR=8  -D KR=4 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=0   -D AVX=2  -D PREFETCH=1 -o src/qs8-packw/gen/qs8-packw-x8c4-gemm-goi-avxvnni-prfm.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=0   -D AVX=2  -D PREFETCH=0 -o src/qs8-packw/gen/qs8-packw-x8c8-gemm-goi-avxvnni.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=0   -D AVX=2  -D PREFETCH=1 -o src/qs8-packw/gen/qs8-packw-x8c8-gemm-goi-avxvnni-prfm.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=128 -D AVX=2  -D PREFETCH=0 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x8c8-gemm-goi-avxvnni.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=128 -D AVX=2  -D PREFETCH=1 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x8c8-gemm-goi-avxvnni-prfm.c &

tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=0   -D AVX=10 -D PREFETCH=0 -o src/qs8-packw/gen/qs8-packw-x8c8-gemm-goi-avx256vnni.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=0   -D AVX=10 -D PREFETCH=1 -o src/qs8-packw/gen/qs8-packw-x8c8-gemm-goi-avx256vnni-prfm.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=128 -D AVX=10 -D PREFETCH=0 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x8c8-gemm-goi-avx256vnni.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=128 -D AVX=10 -D PREFETCH=1 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x8c8-gemm-goi-avx256vnni-prfm.c &

tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=0   -D AVX=2  -D PREFETCH=0 -o src/qs8-packw/gen/qs8-packw-x16c8-gemm-goi-avxvnni.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=0   -D AVX=2  -D PREFETCH=1 -o src/qs8-packw/gen/qs8-packw-x16c8-gemm-goi-avxvnni-prfm.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=128 -D AVX=2  -D PREFETCH=0 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x16c8-gemm-goi-avxvnni.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=128 -D AVX=2  -D PREFETCH=1 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x16c8-gemm-goi-avxvnni-prfm.c &

tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=0   -D AVX=10 -D PREFETCH=0 -o src/qs8-packw/gen/qs8-packw-x16c8-gemm-goi-avx256vnni.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=0   -D AVX=10 -D PREFETCH=1 -o src/qs8-packw/gen/qs8-packw-x16c8-gemm-goi-avx256vnni-prfm.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=128 -D AVX=10 -D PREFETCH=0 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x16c8-gemm-goi-avx256vnni.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=QS8 -D TYPE=int8_t -D IZP=128 -D AVX=10 -D PREFETCH=1 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x16c8-gemm-goi-avx256vnni-prfm.c &

# X8 packing
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=X8  -D TYPE=int8_t -D IZP=0   -D AVX=2  -D PREFETCH=0 -o src/x8-packw/gen/x8-packw-x8c8-gemm-goi-avx2.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=X8  -D TYPE=int8_t -D IZP=0   -D AVX=2  -D PREFETCH=1 -o src/x8-packw/gen/x8-packw-x8c8-gemm-goi-avx2-prfm.c &

tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=X8  -D TYPE=int8_t -D IZP=0   -D AVX=10 -D PREFETCH=0 -o src/x8-packw/gen/x8-packw-x8c8-gemm-goi-avx256skx.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=8  -D KR=8 -D DATATYPE=X8  -D TYPE=int8_t -D IZP=0   -D AVX=10 -D PREFETCH=1 -o src/x8-packw/gen/x8-packw-x8c8-gemm-goi-avx256skx-prfm.c &

tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=X8  -D TYPE=int8_t -D IZP=0   -D AVX=2  -D PREFETCH=0 -o src/x8-packw/gen/x8-packw-x16c8-gemm-goi-avx2.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=X8  -D TYPE=int8_t -D IZP=0   -D AVX=2  -D PREFETCH=1 -o src/x8-packw/gen/x8-packw-x16c8-gemm-goi-avx2-prfm.c &

tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=X8  -D TYPE=int8_t -D IZP=0   -D AVX=10 -D PREFETCH=0 -o src/x8-packw/gen/x8-packw-x16c8-gemm-goi-avx256skx.c &
tools/xngen src/x8-packw/kr-avxvnni.c.in -D NR=16 -D KR=8 -D DATATYPE=X8  -D TYPE=int8_t -D IZP=0   -D AVX=10 -D PREFETCH=1 -o src/x8-packw/gen/x8-packw-x16c8-gemm-goi-avx256skx-prfm.c &

### WAsm Relaxed SIMD
### C8 packing
tools/xngen src/x8-packw/kr-wasmdot.c.in -D NR=8  -D KR=8 -D TYPE=int8_t -D IZP=0   -o src/qs8-packw/gen/qs8-packw-x8c8-gemm-goi-wasmrelaxedsimd.c &
tools/xngen src/x8-packw/kr-wasmdot.c.in -D NR=8  -D KR=8 -D TYPE=int8_t -D IZP=128 -o src/qs8-qu8-packw/gen/qs8-qu8-packw-x8c8-gemm-goi-wasmrelaxedsimd.c &

wait
