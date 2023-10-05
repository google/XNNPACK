#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=1  -o src/x8-lut/gen/x8-lut-scalar-u1.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=2  -o src/x8-lut/gen/x8-lut-scalar-u2.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=4  -o src/x8-lut/gen/x8-lut-scalar-u4.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=8  -o src/x8-lut/gen/x8-lut-scalar-u8.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=16 -o src/x8-lut/gen/x8-lut-scalar-u16.c &

################################## WAsm SIMD ##################################
tools/xngen src/x8-lut/wasmsimd.c.in -D BATCH_TILE=16 -o src/x8-lut/gen/x8-lut-wasmsimd-u16.c &
tools/xngen src/x8-lut/wasmsimd.c.in -D BATCH_TILE=32 -o src/x8-lut/gen/x8-lut-wasmsimd-u32.c &
tools/xngen src/x8-lut/wasmsimd.c.in -D BATCH_TILE=48 -o src/x8-lut/gen/x8-lut-wasmsimd-u48.c &
tools/xngen src/x8-lut/wasmsimd.c.in -D BATCH_TILE=64 -o src/x8-lut/gen/x8-lut-wasmsimd-u64.c &

############################## WAsm Relaxed SIMD ##############################
tools/xngen src/x8-lut/wasmpshufb.c.in -D BATCH_TILE=16 -o src/x8-lut/gen/x8-lut-wasmpshufb-u16.c &
tools/xngen src/x8-lut/wasmpshufb.c.in -D BATCH_TILE=32 -o src/x8-lut/gen/x8-lut-wasmpshufb-u32.c &
tools/xngen src/x8-lut/wasmpshufb.c.in -D BATCH_TILE=48 -o src/x8-lut/gen/x8-lut-wasmpshufb-u48.c &
tools/xngen src/x8-lut/wasmpshufb.c.in -D BATCH_TILE=64 -o src/x8-lut/gen/x8-lut-wasmpshufb-u64.c &

################################## ARM64 NEON #################################
tools/xngen src/x8-lut/neon-tbx128x4.c.in -D BATCH_TILE=16 -o src/x8-lut/gen/x8-lut-aarch64-neon-tbx128x4-u16.c &
tools/xngen src/x8-lut/neon-tbx128x4.c.in -D BATCH_TILE=32 -o src/x8-lut/gen/x8-lut-aarch64-neon-tbx128x4-u32.c &
tools/xngen src/x8-lut/neon-tbx128x4.c.in -D BATCH_TILE=48 -o src/x8-lut/gen/x8-lut-aarch64-neon-tbx128x4-u48.c &
tools/xngen src/x8-lut/neon-tbx128x4.c.in -D BATCH_TILE=64 -o src/x8-lut/gen/x8-lut-aarch64-neon-tbx128x4-u64.c &

################################### x86 SSE ###################################
tools/xngen src/x8-lut/ssse3.c.in -D AVX=0 -D BATCH_TILE=16 -o src/x8-lut/gen/x8-lut-ssse3-u16.c &
tools/xngen src/x8-lut/ssse3.c.in -D AVX=0 -D BATCH_TILE=32 -o src/x8-lut/gen/x8-lut-ssse3-u32.c &

tools/xngen src/x8-lut/ssse3.c.in -D AVX=1 -D BATCH_TILE=16 -o src/x8-lut/gen/x8-lut-avx-u16.c &
tools/xngen src/x8-lut/ssse3.c.in -D AVX=1 -D BATCH_TILE=32 -o src/x8-lut/gen/x8-lut-avx-u32.c &
tools/xngen src/x8-lut/ssse3.c.in -D AVX=1 -D BATCH_TILE=48 -o src/x8-lut/gen/x8-lut-avx-u48.c &
tools/xngen src/x8-lut/ssse3.c.in -D AVX=1 -D BATCH_TILE=64 -o src/x8-lut/gen/x8-lut-avx-u64.c &

################################### x86 AVX2 ##################################
tools/xngen src/x8-lut/avx2.c.in -D BATCH_TILE=32  -o src/x8-lut/gen/x8-lut-avx2-u32.c &
tools/xngen src/x8-lut/avx2.c.in -D BATCH_TILE=64  -o src/x8-lut/gen/x8-lut-avx2-u64.c &
tools/xngen src/x8-lut/avx2.c.in -D BATCH_TILE=96  -o src/x8-lut/gen/x8-lut-avx2-u96.c &
tools/xngen src/x8-lut/avx2.c.in -D BATCH_TILE=128 -o src/x8-lut/gen/x8-lut-avx2-u128.c &

################################## x86 AVX512 #################################
tools/xngen src/x8-lut/avx512skx-vpshufb.c.in -D BATCH_TILE=64  -o src/x8-lut/gen/x8-lut-avx512skx-vpshufb-u64.c &
tools/xngen src/x8-lut/avx512skx-vpshufb.c.in -D BATCH_TILE=128 -o src/x8-lut/gen/x8-lut-avx512skx-vpshufb-u128.c &
tools/xngen src/x8-lut/avx512skx-vpshufb.c.in -D BATCH_TILE=192 -o src/x8-lut/gen/x8-lut-avx512skx-vpshufb-u192.c &
tools/xngen src/x8-lut/avx512skx-vpshufb.c.in -D BATCH_TILE=256 -o src/x8-lut/gen/x8-lut-avx512skx-vpshufb-u256.c &

tools/xngen src/x8-lut/avx512vbmi-vpermx2b.c.in -D BATCH_TILE=64  -o src/x8-lut/gen/x8-lut-avx512vbmi-vpermx2b-u64.c &
tools/xngen src/x8-lut/avx512vbmi-vpermx2b.c.in -D BATCH_TILE=128 -o src/x8-lut/gen/x8-lut-avx512vbmi-vpermx2b-u128.c &
tools/xngen src/x8-lut/avx512vbmi-vpermx2b.c.in -D BATCH_TILE=192 -o src/x8-lut/gen/x8-lut-avx512vbmi-vpermx2b-u192.c &
tools/xngen src/x8-lut/avx512vbmi-vpermx2b.c.in -D BATCH_TILE=256 -o src/x8-lut/gen/x8-lut-avx512vbmi-vpermx2b-u256.c &

wait
