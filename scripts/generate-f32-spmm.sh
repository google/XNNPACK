#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Microkernels without unrolling
tools/xngen src/f32-spmm/scalar.c.in -D MR=1 -D NR=1 -D UNROLL=1 -D DATATYPE=F32 -o src/f32-spmm/gen/f32-spmm-1x1-minmax-scalar.c &
tools/xngen src/f32-spmm/scalar.c.in -D MR=2 -D NR=1 -D UNROLL=1 -D DATATYPE=F32 -o src/f32-spmm/gen/f32-spmm-2x1-minmax-scalar.c &
tools/xngen src/f32-spmm/scalar.c.in -D MR=4 -D NR=1 -D UNROLL=1 -D DATATYPE=F32 -o src/f32-spmm/gen/f32-spmm-4x1-minmax-scalar.c &
tools/xngen src/f32-spmm/scalar.c.in -D MR=8 -D NR=1 -D UNROLL=1 -D DATATYPE=F32 -o src/f32-spmm/gen/f32-spmm-8x1-minmax-scalar.c &
tools/xngen src/f32-spmm/scalar.c.in -D MR=8 -D NR=2 -D UNROLL=1 -D DATATYPE=F32 -o src/f32-spmm/gen/f32-spmm-8x2-minmax-scalar.c &
tools/xngen src/f32-spmm/scalar.c.in -D MR=8 -D NR=4 -D UNROLL=1 -D DATATYPE=F32 -o src/f32-spmm/gen/f32-spmm-8x4-minmax-scalar.c &
### Microkernels with software pipelining
tools/xngen src/f32-spmm/scalar-pipelined.c.in -D MR=1 -D NR=1 -o src/f32-spmm/gen/f32-spmm-1x1-minmax-scalar-pipelined.c &
tools/xngen src/f32-spmm/scalar-pipelined.c.in -D MR=2 -D NR=1 -o src/f32-spmm/gen/f32-spmm-2x1-minmax-scalar-pipelined.c &
tools/xngen src/f32-spmm/scalar-pipelined.c.in -D MR=4 -D NR=1 -o src/f32-spmm/gen/f32-spmm-4x1-minmax-scalar-pipelined.c &
tools/xngen src/f32-spmm/scalar-pipelined.c.in -D MR=8 -D NR=1 -o src/f32-spmm/gen/f32-spmm-8x1-minmax-scalar-pipelined.c &

################################### ARM NEON ##################################
### Microkernels without unrolling
tools/xngen src/f32-spmm/neon.c.in -D MR=4  -D NR=1 -D UNROLL=1 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-4x1-minmax-neon.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=8  -D NR=1 -D UNROLL=1 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-8x1-minmax-neon.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=12 -D NR=1 -D UNROLL=1 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-12x1-minmax-neon.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-16x1-minmax-neon.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=32 -D NR=1 -D UNROLL=1 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-32x1-minmax-neon.c &

tools/xngen src/f32-spmm/neon.c.in -D MR=4  -D NR=1 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-4x1-minmax-neonfma.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=8  -D NR=1 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-8x1-minmax-neonfma.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=12 -D NR=1 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-12x1-minmax-neonfma.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-16x1-minmax-neonfma.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=32 -D NR=1 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-32x1-minmax-neonfma.c &

### Microkernels with 2X unrolling
tools/xngen src/f32-spmm/neon.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-4x1-minmax-neon-x2.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-8x1-minmax-neon-x2.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=16 -D NR=1 -D UNROLL=2 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-16x1-minmax-neon-x2.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=32 -D NR=1 -D UNROLL=2 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-32x1-minmax-neon-x2.c &

tools/xngen src/f32-spmm/neon.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-4x1-minmax-neonfma-x2.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-8x1-minmax-neonfma-x2.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=16 -D NR=1 -D UNROLL=2 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-16x1-minmax-neonfma-x2.c &
tools/xngen src/f32-spmm/neon.c.in -D MR=32 -D NR=1 -D UNROLL=2 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-32x1-minmax-neonfma-x2.c &

### Microkernels for blocks of output channels
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=4  -D NR=2 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-4x2-minmax-aarch64-neonfma.c &
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=8  -D NR=2 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-8x2-minmax-aarch64-neonfma.c &
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=12 -D NR=2 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-12x2-minmax-aarch64-neonfma.c &
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=16 -D NR=2 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-16x2-minmax-aarch64-neonfma.c &
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=32 -D NR=2 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-32x2-minmax-aarch64-neonfma.c &
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=4  -D NR=4 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-4x4-minmax-aarch64-neonfma.c &
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=8  -D NR=4 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-8x4-minmax-aarch64-neonfma.c &
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=12 -D NR=4 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-12x4-minmax-aarch64-neonfma.c &
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=16 -D NR=4 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-16x4-minmax-aarch64-neonfma.c &
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=32 -D NR=4 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-32x4-minmax-aarch64-neonfma.c &

### Microkernels with software pipelining
tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=4  -D NR=1 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-4x1-minmax-neon-pipelined.c &
tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=8  -D NR=1 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-8x1-minmax-neon-pipelined.c &
tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=16 -D NR=1 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-16x1-minmax-neon-pipelined.c &
tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=32 -D NR=1 -D FMA=0 -o src/f32-spmm/gen/f32-spmm-32x1-minmax-neon-pipelined.c &

tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=4  -D NR=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-4x1-minmax-neonfma-pipelined.c &
tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=8  -D NR=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-8x1-minmax-neonfma-pipelined.c &
tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=16 -D NR=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-16x1-minmax-neonfma-pipelined.c &
tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=32 -D NR=1 -D FMA=1 -o src/f32-spmm/gen/f32-spmm-32x1-minmax-neonfma-pipelined.c &

################################### x86 SSE ###################################
### Microkernels without unrolling
tools/xngen src/f32-spmm/sse.c.in -D MR=4 -D NR=1 -D UNROLL=1 -o src/f32-spmm/gen/f32-spmm-4x1-minmax-sse.c &
tools/xngen src/f32-spmm/sse.c.in -D MR=8  -D NR=1 -D UNROLL=1 -o src/f32-spmm/gen/f32-spmm-8x1-minmax-sse.c &
tools/xngen src/f32-spmm/sse.c.in -D MR=16 -D NR=1 -D UNROLL=1 -o src/f32-spmm/gen/f32-spmm-16x1-minmax-sse.c &
tools/xngen src/f32-spmm/sse.c.in -D MR=32 -D NR=1 -D UNROLL=1 -o src/f32-spmm/gen/f32-spmm-32x1-minmax-sse.c &

################################### WASM SIMD ###################################
### Microkernels without unrolling.
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmsimd-arm.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmsimd-arm.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmsimd-arm.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmsimd-arm.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmsimd-x86.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmsimd-x86.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmsimd-x86.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmsimd-x86.c &

tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-arm.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-arm.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-arm.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-arm.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-x86.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-x86.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-x86.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-x86.c &

### Microkernels with 2X unrolling
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmsimd-arm-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmsimd-arm-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmsimd-arm-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmsimd-arm-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmsimd-x86-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmsimd-x86-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmsimd-x86-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmsimd-x86-x2.c &

tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-arm-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-arm-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-arm-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-arm-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-x86-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-x86-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-x86-x2.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-x86-x2.c &

### Microkernels with 4X unrolling
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=4 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=4 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=4 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=4 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=4 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=4 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=4 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=4 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmsimd-x86-x4.c &

tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=4 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-arm-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=4 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-arm-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=4 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-arm-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=4 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-arm-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=4  -D NR=1 -D UNROLL=4 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-x86-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=8  -D NR=1 -D UNROLL=4 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-x86-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=16 -D NR=1 -D UNROLL=4 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-x86-x4.c &
tools/xngen src/f32-spmm/wasmsimd.c.in -D MR=32 -D NR=1 -D UNROLL=4 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-x86-x4.c &

### Microkernels with software pipelining
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=4  -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmsimd-arm-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=8  -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmsimd-arm-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmsimd-arm-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=32 -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmsimd-arm-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=4  -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmsimd-x86-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=8  -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmsimd-x86-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmsimd-x86-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=32 -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmsimd-x86-pipelined.c &

tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=4  -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-arm-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=8  -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-arm-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-arm-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=32 -D NR=1 -D UNROLL=1 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-arm-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=4  -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-x86-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=8  -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-x86-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-x86-pipelined.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=32 -D NR=1 -D UNROLL=1 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-x86-pipelined.c &

### Microkernels with software pipelining and 2X unrolling
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmsimd-arm-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmsimd-arm-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=16 -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmsimd-arm-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=32 -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=        -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmsimd-arm-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmsimd-x86-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmsimd-x86-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=16 -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmsimd-x86-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=32 -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=        -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmsimd-x86-pipelined-x2.c &

tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-arm-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-arm-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=16 -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-arm-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=32 -D NR=1 -D UNROLL=2 -D MINMAX=MINMAX  -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-arm-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-4x1-minmax-wasmrelaxedsimd-x86-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-8x1-minmax-wasmrelaxedsimd-x86-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=16 -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-16x1-minmax-wasmrelaxedsimd-x86-pipelined-x2.c &
tools/xngen src/f32-spmm/wasmsimd-pipelined.c.in -D MR=32 -D NR=1 -D UNROLL=2 -D MINMAX=PMINMAX -D ARCH=RELAXED -o src/f32-spmm/gen/f32-spmm-32x1-minmax-wasmrelaxedsimd-x86-pipelined-x2.c &

wait
