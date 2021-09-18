#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=1  -o src/x8-lut/gen/lut-scalar-x1.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=2  -o src/x8-lut/gen/lut-scalar-x2.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=4  -o src/x8-lut/gen/lut-scalar-x4.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=8  -o src/x8-lut/gen/lut-scalar-x8.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=16 -o src/x8-lut/gen/lut-scalar-x16.c &

################################## WAsm SIMD ##################################
tools/xngen src/x8-lut/wasmsimd.c.in -D BATCH_TILE=16 -o src/x8-lut/gen/lut-wasmsimd-x16.c &
tools/xngen src/x8-lut/wasmsimd.c.in -D BATCH_TILE=32 -o src/x8-lut/gen/lut-wasmsimd-x32.c &
tools/xngen src/x8-lut/wasmsimd.c.in -D BATCH_TILE=48 -o src/x8-lut/gen/lut-wasmsimd-x48.c &
tools/xngen src/x8-lut/wasmsimd.c.in -D BATCH_TILE=64 -o src/x8-lut/gen/lut-wasmsimd-x64.c &

################################## ARM64 NEON #################################
tools/xngen src/x8-lut/neon-tbx128x4.c.in -D BATCH_TILE=16 -o src/x8-lut/gen/lut-neon-tbx128x4-x16.c &
tools/xngen src/x8-lut/neon-tbx128x4.c.in -D BATCH_TILE=32 -o src/x8-lut/gen/lut-neon-tbx128x4-x32.c &
tools/xngen src/x8-lut/neon-tbx128x4.c.in -D BATCH_TILE=48 -o src/x8-lut/gen/lut-neon-tbx128x4-x48.c &
tools/xngen src/x8-lut/neon-tbx128x4.c.in -D BATCH_TILE=64 -o src/x8-lut/gen/lut-neon-tbx128x4-x64.c &

################################### x86 SSE ###################################
tools/xngen src/x8-lut/ssse3.c.in -D AVX=0 -D BATCH_TILE=16 -o src/x8-lut/gen/lut-ssse3-x16.c &
tools/xngen src/x8-lut/ssse3.c.in -D AVX=0 -D BATCH_TILE=32 -o src/x8-lut/gen/lut-ssse3-x32.c &

tools/xngen src/x8-lut/ssse3.c.in -D AVX=1 -D BATCH_TILE=16 -o src/x8-lut/gen/lut-avx-x16.c &
tools/xngen src/x8-lut/ssse3.c.in -D AVX=1 -D BATCH_TILE=32 -o src/x8-lut/gen/lut-avx-x32.c &
tools/xngen src/x8-lut/ssse3.c.in -D AVX=1 -D BATCH_TILE=48 -o src/x8-lut/gen/lut-avx-x48.c &
tools/xngen src/x8-lut/ssse3.c.in -D AVX=1 -D BATCH_TILE=64 -o src/x8-lut/gen/lut-avx-x64.c &

################################### x86 AVX2 ##################################
tools/xngen src/x8-lut/avx2.c.in -D BATCH_TILE=32  -o src/x8-lut/gen/lut-avx2-x32.c &
tools/xngen src/x8-lut/avx2.c.in -D BATCH_TILE=64  -o src/x8-lut/gen/lut-avx2-x64.c &
tools/xngen src/x8-lut/avx2.c.in -D BATCH_TILE=96  -o src/x8-lut/gen/lut-avx2-x96.c &
tools/xngen src/x8-lut/avx2.c.in -D BATCH_TILE=128 -o src/x8-lut/gen/lut-avx2-x128.c &

################################## x86 AVX512 #################################
tools/xngen src/x8-lut/avx512skx-vpshufb.c.in -D BATCH_TILE=64  -o src/x8-lut/gen/lut-avx512skx-vpshufb-x64.c &
tools/xngen src/x8-lut/avx512skx-vpshufb.c.in -D BATCH_TILE=128 -o src/x8-lut/gen/lut-avx512skx-vpshufb-x128.c &
tools/xngen src/x8-lut/avx512skx-vpshufb.c.in -D BATCH_TILE=192 -o src/x8-lut/gen/lut-avx512skx-vpshufb-x192.c &
tools/xngen src/x8-lut/avx512skx-vpshufb.c.in -D BATCH_TILE=256 -o src/x8-lut/gen/lut-avx512skx-vpshufb-x256.c &

################################## Unit tests #################################
tools/generate-lut-test.py --spec test/x8-lut.yaml --output test/x8-lut.cc &

wait
