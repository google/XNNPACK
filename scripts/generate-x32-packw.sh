#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x32-packw/scalar.c.in -D NR=2  -D KUNROLL=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x2-scalar-int-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=4  -D KUNROLL=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x4-scalar-int-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=8  -D KUNROLL=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x8-scalar-int-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=16 -D KUNROLL=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x16-scalar-int-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=2  -D KUNROLL=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x2-scalar-float-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=4  -D KUNROLL=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x4-scalar-float-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=8  -D KUNROLL=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x8-scalar-float-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=16 -D KUNROLL=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x16-scalar-float-x4.c &

################################### ARM NEON ##################################
### NR multiple of 4
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=1 -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8-neon-ld4lane-x4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=1 -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8-neon-ld4lane-prfm-x4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=12 -D SR=1 -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x12-neon-ld4lane-x4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=12 -D SR=1 -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x12-neon-ld4lane-prfm-x4.c &

### SR 4
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=4 -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8s4-neon-ld4lane-x4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=4 -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8s4-neon-ld4lane-prfm-x4.c &

### NR2 micro-kernels
tools/xngen src/x32-packw/NR2-neon.c.in -D NR=2 -D PREFETCH=0 -D KUNROLL=2 -o src/x32-packw/gen/x32-packw-x2-neon-ld2lane-x2.c &
tools/xngen src/x32-packw/NR2-neon.c.in -D NR=2 -D PREFETCH=1 -D KUNROLL=2 -o src/x32-packw/gen/x32-packw-x2-neon-ld2lane-prfm-x2.c &

################################### x86 SSE ###################################
### NR multiple of 4
tools/xngen src/x32-packw/sse2.c.in -D NR=8  -D PREFETCH=0 -D KUNROLL=4 -D AVX=0 -o src/x32-packw/gen/x32-packw-x8-sse2-x4.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=16 -D PREFETCH=0 -D KUNROLL=4 -D AVX=0 -o src/x32-packw/gen/x32-packw-x16-sse2-x4.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=8  -D PREFETCH=0 -D KUNROLL=8 -D AVX=0 -o src/x32-packw/gen/x32-packw-x8-sse2-x8.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=16 -D PREFETCH=0 -D KUNROLL=8 -D AVX=0 -o src/x32-packw/gen/x32-packw-x16-sse2-x8.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=8  -D PREFETCH=1 -D KUNROLL=4 -D AVX=0 -o src/x32-packw/gen/x32-packw-x8-sse2-prfm-x4.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=16 -D PREFETCH=1 -D KUNROLL=4 -D AVX=0 -o src/x32-packw/gen/x32-packw-x16-sse2-prfm-x4.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=8  -D PREFETCH=1 -D KUNROLL=8 -D AVX=0 -o src/x32-packw/gen/x32-packw-x8-sse2-prfm-x8.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=16 -D PREFETCH=1 -D KUNROLL=8 -D AVX=0 -o src/x32-packw/gen/x32-packw-x16-sse2-prfm-x8.c &

### SR 4
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=8  -D SR=4 -D PREFETCH=0 -D KUNROLL=4  -D AVX=0 -o src/x32-packw/gen/x32-packw-x8s4-sse2-x4.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=16 -D SR=4 -D PREFETCH=0 -D KUNROLL=4  -D AVX=0 -o src/x32-packw/gen/x32-packw-x16s4-sse2-x4.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=8  -D SR=4 -D PREFETCH=0 -D KUNROLL=8  -D AVX=0 -o src/x32-packw/gen/x32-packw-x8s4-sse2-x8.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=16 -D SR=4 -D PREFETCH=0 -D KUNROLL=8  -D AVX=0 -o src/x32-packw/gen/x32-packw-x16s4-sse2-x8.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=8  -D SR=4 -D PREFETCH=1 -D KUNROLL=4  -D AVX=0 -o src/x32-packw/gen/x32-packw-x8s4-sse2-prfm-x4.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=16 -D SR=4 -D PREFETCH=1 -D KUNROLL=4  -D AVX=0 -o src/x32-packw/gen/x32-packw-x16s4-sse2-prfm-x4.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=8  -D SR=4 -D PREFETCH=1 -D KUNROLL=8  -D AVX=0 -o src/x32-packw/gen/x32-packw-x8s4-sse2-prfm-x8.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=16 -D SR=4 -D PREFETCH=1 -D KUNROLL=8  -D AVX=0 -o src/x32-packw/gen/x32-packw-x16s4-sse2-prfm-x8.c &

### KR 4
tools/xngen src/x32-packw/c4-sse2.c.in -D NR=2  -D KR=4 -D PREFETCH=0 -o src/x32-packw/gen/x32-packw-x2c4-sse2-x4.c &
tools/xngen src/x32-packw/c4-sse2.c.in -D NR=2  -D KR=4 -D PREFETCH=1 -o src/x32-packw/gen/x32-packw-x2c4-sse2-prfm-x4.c &


################################### x86 AVX ###################################
### NR multiple of 8
tools/xngen src/x32-packw/avx.c.in -D NR=8  -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8-avx-x4.c &
tools/xngen src/x32-packw/avx.c.in -D NR=16 -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x16-avx-x4.c &
tools/xngen src/x32-packw/avx.c.in -D NR=8  -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8-avx-prfm-x4.c &
tools/xngen src/x32-packw/avx.c.in -D NR=16 -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x16-avx-prfm-x4.c &

### SR 4
tools/xngen src/x32-packw/s4-avx.c.in -D NR=8  -D SR=4 -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8s4-avx-x4.c &
tools/xngen src/x32-packw/s4-avx.c.in -D NR=16 -D SR=4 -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x16s4-avx-x4.c &
tools/xngen src/x32-packw/s4-avx.c.in -D NR=8  -D SR=4 -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8s4-avx-prfm-x4.c &
tools/xngen src/x32-packw/s4-avx.c.in -D NR=16 -D SR=4 -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x16s4-avx-prfm-x4.c &

################################### x86 AVX512 ##################################
### NR multiple of 16
tools/xngen src/x32-packw/avx512.c.in -D NR=16 -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x16-avx512f-x4.c &
tools/xngen src/x32-packw/avx512.c.in -D NR=16 -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x16-avx512f-prfm-x4.c &

################################## Wasm SIMD ##################################
### NR multiple of 4
tools/xngen src/x32-packw/wasmsimd.c.in -D NR=8 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8-wasmsimd-x4.c &

### SR 4
tools/xngen src/x32-packw/s4-wasmsimd.c.in -D NR=8 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8s4-wasmsimd-x1.c &

### KR 4
tools/xngen src/x32-packw/c4-wasmsimd.c.in -D NR=2 -D KR=4 -o src/x32-packw/gen/x32-packw-x2c4-wasmsimd-x4.c &

################################## Unit tests #################################
tools/generate-packw-test.py --spec test/x32-packw.yaml --output test/x32-packw.cc &

wait
