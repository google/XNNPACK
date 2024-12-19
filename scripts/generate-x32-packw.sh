#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x32-packw/scalar.c.in -D NR=2  -D KBLOCK=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x2-gemm-goi-scalar-int-u4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=3  -D KBLOCK=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x3-gemm-goi-scalar-int-u4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=4  -D KBLOCK=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x4-gemm-goi-scalar-int-u4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=8  -D KBLOCK=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x8-gemm-goi-scalar-int-u4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=16 -D KBLOCK=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x16-gemm-goi-scalar-int-u4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=2  -D KBLOCK=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x2-gemm-goi-scalar-float-u4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=3  -D KBLOCK=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x3-gemm-goi-scalar-float-u4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=4  -D KBLOCK=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x4-gemm-goi-scalar-float-u4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=8  -D KBLOCK=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x8-gemm-goi-scalar-float-u4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=16 -D KBLOCK=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x16-gemm-goi-scalar-float-u4.c &

### GIO KR=1
tools/xngen src/x32-packw/gio-scalar.c.in -D NR=4  -o src/x32-packw/gen/x32-packw-x4-gemm-gio-scalar.c &
tools/xngen src/x32-packw/gio-scalar.c.in -D NR=8  -o src/x32-packw/gen/x32-packw-x8-gemm-gio-scalar.c &
tools/xngen src/x32-packw/gio-scalar.c.in -D NR=16 -o src/x32-packw/gen/x32-packw-x16-gemm-gio-scalar.c &
tools/xngen src/x32-packw/gio-scalar.c.in -D NR=32 -o src/x32-packw/gen/x32-packw-x32-gemm-gio-scalar.c &

################################### ARM NEON ##################################
### NR multiple of 4
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=1 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-neon-ld4lane-u4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=1 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-neon-ld4lane-u4-prfm.c &
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=1 -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-neon-ld4lane-u8.c &
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=1 -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-neon-ld4lane-u8-prfm.c &
tools/xngen src/x32-packw/neon.c.in -D NR=12 -D SR=1 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x12-gemm-goi-neon-ld4lane-u4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=12 -D SR=1 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x12-gemm-goi-neon-ld4lane-u4-prfm.c &
tools/xngen src/x32-packw/neon.c.in -D NR=12 -D SR=1 -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x12-gemm-goi-neon-ld4lane-u8.c &
tools/xngen src/x32-packw/neon.c.in -D NR=12 -D SR=1 -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x12-gemm-goi-neon-ld4lane-u8-prfm.c &
tools/xngen src/x32-packw/neon.c.in -D NR=16 -D SR=1 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-neon-ld4lane-u4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=16 -D SR=1 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-neon-ld4lane-u4-prfm.c &
tools/xngen src/x32-packw/neon.c.in -D NR=16 -D SR=1 -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-neon-ld4lane-u8.c &
tools/xngen src/x32-packw/neon.c.in -D NR=16 -D SR=1 -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-neon-ld4lane-u8-prfm.c &

### SR 4
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=4 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-neon-ld4lane-u4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=4 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-neon-ld4lane-u4-prfm.c &
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=4 -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-neon-ld4lane-u8.c &
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=4 -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-neon-ld4lane-u8-prfm.c &

### NR2 micro-kernels
tools/xngen src/x32-packw/NR2-neon.c.in -D NR=2 -D PREFETCH=0 -D KBLOCK=2 -o src/x32-packw/gen/x32-packw-x2-gemm-goi-neon-ld2lane-u2.c &
tools/xngen src/x32-packw/NR2-neon.c.in -D NR=2 -D PREFETCH=1 -D KBLOCK=2 -o src/x32-packw/gen/x32-packw-x2-gemm-goi-neon-ld2lane-u2-prfm.c &

################################### x86 SSE ###################################
### NR multiple of 4
tools/xngen src/x32-packw/sse2.c.in -D NR=8  -D PREFETCH=0 -D KBLOCK=4 -D AVX=0 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-sse2-u4.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=16 -D PREFETCH=0 -D KBLOCK=4 -D AVX=0 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-sse2-u4.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=8  -D PREFETCH=0 -D KBLOCK=8 -D AVX=0 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-sse2-u8.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=16 -D PREFETCH=0 -D KBLOCK=8 -D AVX=0 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-sse2-u8.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=8  -D PREFETCH=1 -D KBLOCK=4 -D AVX=0 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-sse2-u4-prfm.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=16 -D PREFETCH=1 -D KBLOCK=4 -D AVX=0 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-sse2-u4-prfm.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=8  -D PREFETCH=1 -D KBLOCK=8 -D AVX=0 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-sse2-u8-prfm.c &
tools/xngen src/x32-packw/sse2.c.in -D NR=16 -D PREFETCH=1 -D KBLOCK=8 -D AVX=0 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-sse2-u8-prfm.c &

### SR 4
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=8  -D SR=4 -D PREFETCH=0 -D KBLOCK=4  -D AVX=0 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-sse2-u4.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=16 -D SR=4 -D PREFETCH=0 -D KBLOCK=4  -D AVX=0 -o src/x32-packw/gen/x32-packw-x16s4-gemm-goi-sse2-u4.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=8  -D SR=4 -D PREFETCH=0 -D KBLOCK=8  -D AVX=0 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-sse2-u8.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=16 -D SR=4 -D PREFETCH=0 -D KBLOCK=8  -D AVX=0 -o src/x32-packw/gen/x32-packw-x16s4-gemm-goi-sse2-u8.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=8  -D SR=4 -D PREFETCH=1 -D KBLOCK=4  -D AVX=0 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-sse2-u4-prfm.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=16 -D SR=4 -D PREFETCH=1 -D KBLOCK=4  -D AVX=0 -o src/x32-packw/gen/x32-packw-x16s4-gemm-goi-sse2-u4-prfm.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=8  -D SR=4 -D PREFETCH=1 -D KBLOCK=8  -D AVX=0 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-sse2-u8-prfm.c &
tools/xngen src/x32-packw/s4-sse2.c.in -D NR=16 -D SR=4 -D PREFETCH=1 -D KBLOCK=8  -D AVX=0 -o src/x32-packw/gen/x32-packw-x16s4-gemm-goi-sse2-u8-prfm.c &

### KR 4
tools/xngen src/x32-packw/c4-sse2.c.in -D NR=2  -D KR=4 -D PREFETCH=0 -o src/x32-packw/gen/x32-packw-x2c4-gemm-goi-sse2-u4.c &
tools/xngen src/x32-packw/c4-sse2.c.in -D NR=2  -D KR=4 -D PREFETCH=1 -o src/x32-packw/gen/x32-packw-x2c4-gemm-goi-sse2-u4-prfm.c &

################################### x86 AVX ###################################
### NR multiple of 8
tools/xngen src/x32-packw/avx.c.in -D NR=8  -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-avx-u4.c &
tools/xngen src/x32-packw/avx.c.in -D NR=16 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-avx-u4.c &
tools/xngen src/x32-packw/avx.c.in -D NR=8  -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-avx-u4-prfm.c &
tools/xngen src/x32-packw/avx.c.in -D NR=16 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-avx-u4-prfm.c &

### SR 4
tools/xngen src/x32-packw/s4-avx.c.in -D NR=8  -D SR=4 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-avx-u4.c &
tools/xngen src/x32-packw/s4-avx.c.in -D NR=16 -D SR=4 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x16s4-gemm-goi-avx-u4.c &
tools/xngen src/x32-packw/s4-avx.c.in -D NR=8  -D SR=4 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-avx-u4-prfm.c &
tools/xngen src/x32-packw/s4-avx.c.in -D NR=16 -D SR=4 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x16s4-gemm-goi-avx-u4-prfm.c &

### GIO NR multiple of 8
tools/xngen src/x32-packw/gio-avx.c.in -D NR=8  -D PREFETCH=0 -D KBLOCK=1 -o src/x32-packw/gen/x32-packw-x8-gemm-gio-avx-u1.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=8  -D PREFETCH=1 -D KBLOCK=1 -o src/x32-packw/gen/x32-packw-x8-gemm-gio-avx-u1-prfm.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=16 -D PREFETCH=0 -D KBLOCK=1 -o src/x32-packw/gen/x32-packw-x16-gemm-gio-avx-u1.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=16 -D PREFETCH=1 -D KBLOCK=1 -o src/x32-packw/gen/x32-packw-x16-gemm-gio-avx-u1-prfm.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=32 -D PREFETCH=0 -D KBLOCK=1 -o src/x32-packw/gen/x32-packw-x32-gemm-gio-avx-u1.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=32 -D PREFETCH=1 -D KBLOCK=1 -o src/x32-packw/gen/x32-packw-x32-gemm-gio-avx-u1-prfm.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=8  -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x8-gemm-gio-avx-u8.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=8  -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x8-gemm-gio-avx-u8-prfm.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=16 -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x16-gemm-gio-avx-u8.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=16 -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x16-gemm-gio-avx-u8-prfm.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=32 -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x32-gemm-gio-avx-u8.c &
tools/xngen src/x32-packw/gio-avx.c.in -D NR=32 -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x32-gemm-gio-avx-u8-prfm.c &

################################### x86 AVX512 ##################################
### NR multiple of 16
tools/xngen src/x32-packw/avx512.c.in -D NR=16 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-avx512f-u4.c &
tools/xngen src/x32-packw/avx512.c.in -D NR=16 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x16-gemm-goi-avx512f-u4-prfm.c &
tools/xngen src/x32-packw/avx512.c.in -D NR=32 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x32-gemm-goi-avx512f-u4.c &
tools/xngen src/x32-packw/avx512.c.in -D NR=32 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x32-gemm-goi-avx512f-u4-prfm.c &

### GIO NR multiple of 16
tools/xngen src/x32-packw/gio-avx512.c.in -D NR=16 -D PREFETCH=0 -D KBLOCK=1 -o src/x32-packw/gen/x32-packw-x16-gemm-gio-avx512f-u1.c &
tools/xngen src/x32-packw/gio-avx512.c.in -D NR=16 -D PREFETCH=1 -D KBLOCK=1 -o src/x32-packw/gen/x32-packw-x16-gemm-gio-avx512f-u1-prfm.c &
tools/xngen src/x32-packw/gio-avx512.c.in -D NR=32 -D PREFETCH=0 -D KBLOCK=1 -o src/x32-packw/gen/x32-packw-x32-gemm-gio-avx512f-u1.c &
tools/xngen src/x32-packw/gio-avx512.c.in -D NR=32 -D PREFETCH=1 -D KBLOCK=1 -o src/x32-packw/gen/x32-packw-x32-gemm-gio-avx512f-u1-prfm.c &
tools/xngen src/x32-packw/gio-avx512.c.in -D NR=16 -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x16-gemm-gio-avx512f-u8.c &
tools/xngen src/x32-packw/gio-avx512.c.in -D NR=16 -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x16-gemm-gio-avx512f-u8-prfm.c &
tools/xngen src/x32-packw/gio-avx512.c.in -D NR=32 -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x32-gemm-gio-avx512f-u8.c &
tools/xngen src/x32-packw/gio-avx512.c.in -D NR=32 -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x32-gemm-gio-avx512f-u8-prfm.c &

################################## Wasm SIMD ##################################
### NR multiple of 4
tools/xngen src/x32-packw/wasmsimd.c.in -D NR=8 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8-gemm-goi-wasmsimd-u4.c &

### SR 4
tools/xngen src/x32-packw/s4-wasmsimd.c.in -D NR=8 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8s4-gemm-goi-wasmsimd-u4.c &

### KR 4
tools/xngen src/x32-packw/c4-wasmsimd.c.in -D NR=2 -D KR=4 -o src/x32-packw/gen/x32-packw-x2c4-gemm-goi-wasmsimd-u4.c &

################################## RISC-V VECTOR ##################################
tools/xngen src/x32-packw/rvv.c.in -D NR=m1 -D KBLOCK=2 -o src/x32-packw/gen/x32-packw-x1v-gemm-goi-rvv-u2.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m1 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x1v-gemm-goi-rvv-u4.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m1 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x1v-gemm-goi-rvv-u8.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m2 -D KBLOCK=2 -o src/x32-packw/gen/x32-packw-x2v-gemm-goi-rvv-u2.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m2 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x2v-gemm-goi-rvv-u4.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m2 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x2v-gemm-goi-rvv-u8.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m4 -D KBLOCK=2 -o src/x32-packw/gen/x32-packw-x4v-gemm-goi-rvv-u2.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m4 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x4v-gemm-goi-rvv-u4.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m4 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x4v-gemm-goi-rvv-u8.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m8 -D KBLOCK=2 -o src/x32-packw/gen/x32-packw-x8v-gemm-goi-rvv-u2.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m8 -D KBLOCK=4 -o src/x32-packw/gen/x32-packw-x8v-gemm-goi-rvv-u4.c &
tools/xngen src/x32-packw/rvv.c.in -D NR=m8 -D KBLOCK=8 -o src/x32-packw/gen/x32-packw-x8v-gemm-goi-rvv-u8.c &

################################## ARM NEON ###################################
tools/xngen src/x32-packw/gio-simd.c.in -D BATCH_TILES=4,8,12,16    -D PREFETCH=0 -D KBLOCK=2 -D ARCH=neon     -o src/x32-packw/gen/x32-packw-gio-neon-u2.c

################################# x86 SSE41 #################################
tools/xngen src/x32-packw/gio-simd.c.in -D BATCH_TILES=4,8,12,16    -D PREFETCH=0 -D KBLOCK=2 -D ARCH=sse41    -o src/x32-packw/gen/x32-packw-gio-sse41-u2.c

################################## WAsm SIMD ##################################
tools/xngen src/x32-packw/gio-simd.c.in -D BATCH_TILES=4,8,12,16    -D PREFETCH=0 -D KBLOCK=2 -D ARCH=wasmsimd -o src/x32-packw/gen/x32-packw-gio-wasmsimd-u2.c

################################## Hexagon HVX #################################
tools/xngen src/x32-packw/gio-simd.c.in -D BATCH_TILES=32,64,96,128 -D PREFETCH=0 -D KBLOCK=2 -D ARCH=hvx      -o src/x32-packw/gen/x32-packw-gio-hvx-u2.c

wait
