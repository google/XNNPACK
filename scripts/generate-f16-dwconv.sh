#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

############################## Portable Scalar ###############################
tools/xngen src/f16-dwconv/unipass.c.in -D ARCH=scalar -D CHANNEL_TILE=1 -D KERNEL_TILE=3  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p1c-minmax-scalar-acc2.c &
tools/xngen src/f16-dwconv/unipass.c.in -D ARCH=scalar -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p1c-minmax-scalar-acc2.c &
tools/xngen src/f16-dwconv/unipass.c.in -D ARCH=scalar -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p1c-minmax-scalar-acc2.c &
tools/xngen src/f16-dwconv/unipass.c.in -D ARCH=scalar -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p2c-minmax-scalar-acc2.c &

tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=scalar -D CHANNEL_TILE=1 -D KERNEL_TILE=3  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-3p1c-minmax-scalar-acc2.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=scalar -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-4p1c-minmax-scalar-acc2.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=scalar -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-9p1c-minmax-scalar-acc2.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=scalar -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-25p2c-minmax-scalar-acc2.c &

############################## WASM Relaxed SIMD ############################
tools/xngen src/f16-dwconv/unipass.c.in -D ARCH=wasmrelaxedsimd -D CHANNEL_TILE=8 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p8c-minmax-wasmrelaxedsimd.c &
tools/xngen src/f16-dwconv/unipass.c.in -D ARCH=wasmrelaxedsimd -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p8c-minmax-wasmrelaxedsimd-acc2.c &

tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=wasmrelaxedsimd -D CHANNEL_TILE=8 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-f32acc-dwconv-9p8c-minmax-wasmrelaxedsimd.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=wasmrelaxedsimd -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-25p8c-minmax-wasmrelaxedsimd-acc2.c &

################################### ARM NEON ##################################
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p8c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p8c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p16c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p16c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p32c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p32c-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p8c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p8c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p16c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p16c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p32c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p32c-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p8c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p8c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p16c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p16c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p32c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p32c-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p8c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p8c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p16c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p16c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p32c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p32c-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=8  -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-f32acc-dwconv-3p8c-minmax-neonfp16.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=8  -D KERNEL_TILE=3  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-3p8c-minmax-neonfp16-acc2.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-f32acc-dwconv-3p16c-minmax-neonfp16.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-3p16c-minmax-neonfp16-acc2.c &

tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=8  -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-f32acc-dwconv-4p8c-minmax-neonfp16.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=8  -D KERNEL_TILE=4  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-4p8c-minmax-neonfp16-acc2.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=16 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-f32acc-dwconv-4p16c-minmax-neonfp16.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=16 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-4p16c-minmax-neonfp16-acc2.c &

tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-f32acc-dwconv-9p8c-minmax-neonfp16.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-9p8c-minmax-neonfp16-acc2.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-f32acc-dwconv-9p16c-minmax-neonfp16.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-9p16c-minmax-neonfp16-acc2.c &

tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-f32acc-dwconv-25p8c-minmax-neonfp16.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-25p8c-minmax-neonfp16-acc2.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-f32acc-dwconv-25p16c-minmax-neonfp16.c &
tools/xngen src/f16-dwconv/unipass-f16-f32acc.c.in -D ARCH=neonfp16 -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-f32acc-dwconv-25p16c-minmax-neonfp16-acc2.c &

################################### x86 FMA3 ##################################
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p8c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p8c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p16c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p16c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p32c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p32c-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p8c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p8c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p16c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p16c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p32c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p32c-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p8c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p8c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p16c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p16c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p32c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p32c-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p8c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p8c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p16c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p16c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p32c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p32c-minmax-fma3-acc2.c &

################################### RISC-V Vector #############################
tools/xngen src/f16-dwconv/unipass-rvvfp16arith.c.in -D CHANNEL_TILE=m4 -D KERNEL_TILE=3 -o src/f16-dwconv/gen/f16-dwconv-3p4vc-minmax-rvvfp16arith.c &
tools/xngen src/f16-dwconv/unipass-rvvfp16arith.c.in -D CHANNEL_TILE=m8 -D KERNEL_TILE=3 -o src/f16-dwconv/gen/f16-dwconv-3p8vc-minmax-rvvfp16arith.c &

tools/xngen src/f16-dwconv/unipass-rvvfp16arith.c.in -D CHANNEL_TILE=m4 -D KERNEL_TILE=4 -o src/f16-dwconv/gen/f16-dwconv-4p4vc-minmax-rvvfp16arith.c &
tools/xngen src/f16-dwconv/unipass-rvvfp16arith.c.in -D CHANNEL_TILE=m8 -D KERNEL_TILE=4 -o src/f16-dwconv/gen/f16-dwconv-4p8vc-minmax-rvvfp16arith.c &

tools/xngen src/f16-dwconv/unipass-rvvfp16arith.c.in -D CHANNEL_TILE=m4 -D KERNEL_TILE=9 -o src/f16-dwconv/gen/f16-dwconv-9p4vc-minmax-rvvfp16arith.c &
tools/xngen src/f16-dwconv/unipass-rvvfp16arith.c.in -D CHANNEL_TILE=m8 -D KERNEL_TILE=9 -o src/f16-dwconv/gen/f16-dwconv-9p8vc-minmax-rvvfp16arith.c &

tools/xngen src/f16-dwconv/unipass-rvvfp16arith.c.in -D CHANNEL_TILE=m4 -D KERNEL_TILE=25 -o src/f16-dwconv/gen/f16-dwconv-25p4vc-minmax-rvvfp16arith.c &
tools/xngen src/f16-dwconv/unipass-rvvfp16arith.c.in -D CHANNEL_TILE=m8 -D KERNEL_TILE=25 -o src/f16-dwconv/gen/f16-dwconv-25p8vc-minmax-rvvfp16arith.c &

################################### x86 AVX512FP16 ###########################
tools/xngen src/f16-dwconv/unipass-avx512fp16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p32c-minmax-avx512fp16.c &
tools/xngen src/f16-dwconv/unipass-avx512fp16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p32c-minmax-avx512fp16.c &
tools/xngen src/f16-dwconv/unipass-avx512fp16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p32c-minmax-avx512fp16.c &
tools/xngen src/f16-dwconv/unipass-avx512fp16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p32c-minmax-avx512fp16-acc2.c &
tools/xngen src/f16-dwconv/unipass-avx512fp16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p32c-minmax-avx512fp16.c &
tools/xngen src/f16-dwconv/unipass-avx512fp16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p32c-minmax-avx512fp16-acc2.c &

wait
