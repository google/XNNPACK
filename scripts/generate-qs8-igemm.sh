#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=GEMMLOWP -D VARIANT=      -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/1x2-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=GEMMLOWP -D VARIANT=      -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/2x2-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=GEMMLOWP -D VARIANT=      -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/3x2-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=GEMMLOWP -D VARIANT=      -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/4x2-minmax-gemmlowp-scalar.c

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/1x2-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/2x2-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/3x2-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/4x2-minmax-fp32-scalar-lrint.c

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/1x2-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/2x2-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/3x2-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/4x2-minmax-fp32-scalar-lrint.c

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/1x2-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/2x2-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/3x2-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/4x2-minmax-fp32-scalar-magic.c

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/1x2-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/2x2-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/3x2-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/4x2-minmax-fp32-scalar-magic.c

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=GEMMLOWP -D VARIANT=      -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/1x4-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=GEMMLOWP -D VARIANT=      -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/2x4-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=GEMMLOWP -D VARIANT=      -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/3x4-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=GEMMLOWP -D VARIANT=      -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/4x4-minmax-gemmlowp-scalar.c

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/1x4-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/2x4-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/3x4-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/4x4-minmax-fp32-scalar-lrint.c

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/1x4-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/2x4-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/3x4-minmax-fp32-scalar-lrint.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=LRINT -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/4x4-minmax-fp32-scalar-lrint.c

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/1x4-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/2x4-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/3x4-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=0 -D WASM=0 -o src/qs8-igemm/gen/4x4-minmax-fp32-scalar-magic.c

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/1x4-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/2x4-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/3x4-minmax-fp32-scalar-magic.c
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32     -D VARIANT=MAGIC -D CHANNELWISE=1 -D WASM=0 -o src/qc8-igemm/gen/4x4-minmax-fp32-scalar-magic.c

################################## WAsm SIMD ##################################
### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-wasmsimd-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-wasmsimd-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-wasmsimd-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x4c8-minmax-fp32-wasmsimd-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x4c8-minmax-fp32-wasmsimd-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/3x4c8-minmax-fp32-wasmsimd-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/1x4c8-minmax-fp32-wasmsimd-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/2x4c8-minmax-fp32-wasmsimd-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/3x4c8-minmax-fp32-wasmsimd-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-wasmsimd-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-wasmsimd-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-wasmsimd-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x4c8-minmax-fp32-wasmsimd-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x4c8-minmax-fp32-wasmsimd-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/3x4c8-minmax-fp32-wasmsimd-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/1x4c8-minmax-fp32-wasmsimd-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/2x4c8-minmax-fp32-wasmsimd-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/3x4c8-minmax-fp32-wasmsimd-ld128.c

################################### ARM NEON ##################################
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x8-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x8-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x8-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x8-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/6x8-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x16-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x16-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x16-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x16-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/6x16-minmax-gemmlowp-neon-mlal-lane.c

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x16-minmax-fp32-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x16-minmax-fp32-neon-mlal-lane.c

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-igemm/gen/1x16-minmax-fp32-neon-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-igemm/gen/4x16-minmax-fp32-neon-mlal-lane.c

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-igemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-igemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-igemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-igemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/6x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/6x16-minmax-gemmlowp-neon-mlal-lane-prfm.c

tools/xngen src/qs8-igemm/neon-mull-addw-dup.c.in -D MR=1 -D NR=8 -o src/qs8-igemm/gen/1x8-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-igemm/neon-mull-addw-dup.c.in -D MR=2 -D NR=8 -o src/qs8-igemm/gen/2x8-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-igemm/neon-mull-addw-dup.c.in -D MR=3 -D NR=8 -o src/qs8-igemm/gen/3x8-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-igemm/neon-mull-addw-dup.c.in -D MR=4 -D NR=8 -o src/qs8-igemm/gen/4x8-minmax-gemmlowp-neon-mull-addw-dup.c

tools/xngen src/qs8-igemm/neon-mull-addw-dup.c.in -D MR=1 -D NR=16 -o src/qs8-igemm/gen/1x16-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-igemm/neon-mull-addw-dup.c.in -D MR=2 -D NR=16 -o src/qs8-igemm/gen/2x16-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-igemm/neon-mull-addw-dup.c.in -D MR=3 -D NR=16 -o src/qs8-igemm/gen/3x16-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-igemm/neon-mull-addw-dup.c.in -D MR=4 -D NR=16 -o src/qs8-igemm/gen/4x16-minmax-gemmlowp-neon-mull-addw-dup.c

### C2 micro-kernels
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x16c2-minmax-gemmlowp-neon-mull-padal-dup.c

tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c

tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x8c2-minmax-fp32-neon-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x8c2-minmax-fp32-neon-mlal-padal-dup.c

tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-igemm/gen/1x8c2-minmax-fp32-neon-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-igemm/gen/2x8c2-minmax-fp32-neon-mlal-padal-dup.c

tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-igemm/gen/1x8c2-minmax-fp32-neonv8-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-igemm/gen/2x8c2-minmax-fp32-neonv8-mlal-padal-dup.c

tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-igemm/gen/1x8c2-minmax-fp32-neonv8-mlal-padal-dup.c
tools/xngen src/qs8-igemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-igemm/gen/2x8c2-minmax-fp32-neonv8-mlal-padal-dup.c

### C8 micro-kernels
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x8c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x8c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x16c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x16c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x16c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x16c8-minmax-gemmlowp-neon-mull-padal.c

tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x8c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x8c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x16c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x16c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/3x16c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/4x16c8-minmax-gemmlowp-neon-mlal-padal.c

tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/1x8c8-minmax-fp32-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-igemm/gen/2x8c8-minmax-fp32-neon-mlal-padal.c

tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-igemm/gen/1x8c8-minmax-fp32-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-igemm/gen/2x8c8-minmax-fp32-neon-mlal-padal.c

tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-igemm/gen/1x8c8-minmax-fp32-neonv8-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-igemm/gen/2x8c8-minmax-fp32-neonv8-mlal-padal.c

tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-igemm/gen/1x8c8-minmax-fp32-neonv8-mlal-padal.c
tools/xngen src/qs8-igemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-igemm/gen/2x8c8-minmax-fp32-neonv8-mlal-padal.c

### C16 micro-kernels
tools/xngen src/qs8-igemm/c16-neon-mlal-padal.c.in -D MR=1 -D NR=8  -o src/qs8-igemm/gen/1x8c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c16-neon-mlal-padal.c.in -D MR=2 -D NR=8  -o src/qs8-igemm/gen/2x8c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c16-neon-mlal-padal.c.in -D MR=3 -D NR=8  -o src/qs8-igemm/gen/3x8c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c16-neon-mlal-padal.c.in -D MR=4 -D NR=8  -o src/qs8-igemm/gen/4x8c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c16-neon-mlal-padal.c.in -D MR=1 -D NR=16 -o src/qs8-igemm/gen/1x16c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c16-neon-mlal-padal.c.in -D MR=2 -D NR=16 -o src/qs8-igemm/gen/2x16c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c16-neon-mlal-padal.c.in -D MR=3 -D NR=16 -o src/qs8-igemm/gen/3x16c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-igemm/c16-neon-mlal-padal.c.in -D MR=4 -D NR=16 -o src/qs8-igemm/gen/4x16c16-minmax-gemmlowp-neon-mlal-padal.c

### C4 micro-kernels
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x8c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x8c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/6x8c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/8x8c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x16c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/6x16c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/8x16c4-minmax-gemmlowp-neondot.c

tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/6x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/8x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x16c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/6x16c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/8x16c4-minmax-fp32-neondot.c

tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/1x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/4x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/6x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/8x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/1x16c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/4x16c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/6x16c4-minmax-fp32-neondot.c
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/8x16c4-minmax-fp32-neondot.c


############################### AArch64 assembly ##############################
# Cortex A53 micro-kernel
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in   -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16-minmax-gemmlowp-aarch64-neon-mlal-lane-cortex-a53.S
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in   -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16-minmax-gemmlowp-aarch64-neon-mlal-lane-prfm-cortex-a53.S

tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-cortex-a53.S
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-cortex-a53.S

tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-cortex-a53.S
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-cortex-a53.S

### C4 micro-kernels
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-cortex-a55.S
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-ld64.S
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-ld128.S

tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld64.S
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S

tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld64.S
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S

### C8 / C16 micro-kernels
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-cortex-a53.S
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm-cortex-a53.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-cortex-a53.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm-cortex-a53.S
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal.S
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=0 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=1 -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm.S
tools/xngen src/qs8-igemm/2x8c16-aarch64-neon-mlal-padal.S.in                         -D REQUANTIZATION=GEMMLOWP -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x8c16-minmax-gemmlowp-aarch64-neon-mlal-padal.S

tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-padal-cortex-a53.S
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-padal-prfm-cortex-a53.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-padal-cortex-a53.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-padal-prfm-cortex-a53.S
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-padal.S
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-padal-prfm.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-padal.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-padal-prfm.S
tools/xngen src/qs8-igemm/2x8c16-aarch64-neon-mlal-padal.S.in                         -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-igemm/gen/2x8c16-minmax-fp32-aarch64-neon-mlal-padal.S

tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-padal-cortex-a53.S
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-padal-prfm-cortex-a53.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-padal-cortex-a53.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-padal-prfm-cortex-a53.S
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-padal.S
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-padal-prfm.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-padal.S
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-padal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-padal-prfm.S
tools/xngen src/qs8-igemm/2x8c16-aarch64-neon-mlal-padal.S.in                         -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-igemm/gen/2x8c16-minmax-fp32-aarch64-neon-mlal-padal.S


################################### x86 SSE ###################################
### C2 micro-kernels
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-sse2-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/4x4c2-minmax-fp32-sse2-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/4x4c2-minmax-fp32-sse2-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-ssse3-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-ssse3-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-ssse3-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-ssse3-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-sse41-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/4x4c2-minmax-fp32-sse41-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/4x4c2-minmax-fp32-sse41-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-avx-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/4x4c2-minmax-fp32-avx-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/1x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/2x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/3x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/4x4c2-minmax-fp32-avx-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-xop-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/4x4c2-minmax-fp32-xop-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/1x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/2x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/3x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/4x4c2-minmax-fp32-xop-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-sse2-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/4x4c2-minmax-fp32-sse2-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/4x4c2-minmax-fp32-sse2-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-ssse3-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-ssse3-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-ssse3-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-ssse3-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-sse41-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/4x4c2-minmax-fp32-sse41-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/4x4c2-minmax-fp32-sse41-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-avx-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/4x4c2-minmax-fp32-avx-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/1x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/2x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/3x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/4x4c2-minmax-fp32-avx-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c2-minmax-gemmlowp-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c2-minmax-gemmlowp-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c2-minmax-gemmlowp-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/4x4c2-minmax-gemmlowp-xop-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/4x4c2-minmax-fp32-xop-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/1x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/2x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/3x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/4x4c2-minmax-fp32-xop-ld128.c

### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-sse2-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c8-minmax-fp32-sse2-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/3x4c8-minmax-fp32-sse2-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-ssse3-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-ssse3-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-ssse3-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c8-minmax-fp32-ssse3-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c8-minmax-fp32-ssse3-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c8-minmax-fp32-ssse3-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-sse41-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c8-minmax-fp32-sse41-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/3x4c8-minmax-fp32-sse41-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-avx-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c8-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c8-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c8-minmax-fp32-avx-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/1x4c8-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/2x4c8-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/3x4c8-minmax-fp32-avx-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-xop-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/1x4c8-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/2x4c8-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qs8-igemm/gen/3x4c8-minmax-fp32-xop-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/1x4c8-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/2x4c8-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64  -o src/qc8-igemm/gen/3x4c8-minmax-fp32-xop-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-sse2-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c8-minmax-fp32-sse2-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/3x4c8-minmax-fp32-sse2-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-ssse3-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-ssse3-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-ssse3-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c8-minmax-fp32-ssse3-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c8-minmax-fp32-ssse3-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c8-minmax-fp32-ssse3-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-sse41-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c8-minmax-fp32-sse41-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/3x4c8-minmax-fp32-sse41-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-avx-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c8-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c8-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c8-minmax-fp32-avx-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/1x4c8-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/2x4c8-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/3x4c8-minmax-fp32-avx-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c8-minmax-gemmlowp-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c8-minmax-gemmlowp-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c8-minmax-gemmlowp-xop-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/1x4c8-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/2x4c8-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qs8-igemm/gen/3x4c8-minmax-fp32-xop-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/1x4c8-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/2x4c8-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128 -o src/qc8-igemm/gen/3x4c8-minmax-fp32-xop-ld128.c

################################### x86 AVX2 ##################################
### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-igemm/gen/1x8c8-minmax-gemmlowp-avx2.c
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=2 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-igemm/gen/2x8c8-minmax-gemmlowp-avx2.c
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=3 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-igemm/gen/3x8c8-minmax-gemmlowp-avx2.c

tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -o src/qs8-igemm/gen/1x8c8-minmax-fp32-avx2.c
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=2 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -o src/qs8-igemm/gen/2x8c8-minmax-fp32-avx2.c
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=3 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -o src/qs8-igemm/gen/3x8c8-minmax-fp32-avx2.c

tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -o src/qc8-igemm/gen/1x8c8-minmax-fp32-avx2.c
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=2 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -o src/qc8-igemm/gen/2x8c8-minmax-fp32-avx2.c
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=3 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -o src/qc8-igemm/gen/3x8c8-minmax-fp32-avx2.c

################################## x86 AVX512 #################################
### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=1 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-igemm/gen/1x16c8-minmax-gemmlowp-avx512skx.c
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=2 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-igemm/gen/2x16c8-minmax-gemmlowp-avx512skx.c
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=3 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-igemm/gen/3x16c8-minmax-gemmlowp-avx512skx.c
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=4 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-igemm/gen/4x16c8-minmax-gemmlowp-avx512skx.c

tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=1 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=FP32 -o src/qs8-igemm/gen/1x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=2 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=FP32 -o src/qs8-igemm/gen/2x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=3 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=FP32 -o src/qs8-igemm/gen/3x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=4 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=FP32 -o src/qs8-igemm/gen/4x16c8-minmax-fp32-avx512skx.c

tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=1 -D VARIANT=LD256 -D CHANNELWISE=1 -D REQUANTIZATION=FP32 -o src/qc8-igemm/gen/1x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=2 -D VARIANT=LD256 -D CHANNELWISE=1 -D REQUANTIZATION=FP32 -o src/qc8-igemm/gen/2x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=3 -D VARIANT=LD256 -D CHANNELWISE=1 -D REQUANTIZATION=FP32 -o src/qc8-igemm/gen/3x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=4 -D VARIANT=LD256 -D CHANNELWISE=1 -D REQUANTIZATION=FP32 -o src/qc8-igemm/gen/4x16c8-minmax-fp32-avx512skx.c

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/qs8-igemm-minmax-gemmlowp.yaml --output test/qs8-igemm-minmax-gemmlowp.cc
tools/generate-gemm-test.py --spec test/qs8-igemm-minmax-fp32.yaml --output test/qs8-igemm-minmax-fp32.cc
tools/generate-gemm-test.py --spec test/qc8-igemm-minmax-fp32.yaml --output test/qc8-igemm-minmax-fp32.cc
