#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
for b in 1 2 4 8; do
  tools/xngen src/f32-vrsqrt/scalar-rsqrt.c.in -D BATCH_TILE=${b} -o src/f32-vrsqrt/gen/f32-vrsqrt-scalar-rsqrt-u${b}.c &
done

################################### x86 SSE ###################################
for b in 4 8 12 16 20 24 28 32; do
  tools/xngen src/f32-vrsqrt/sse-rsqrt.c.in -D BATCH_TILE=${b} -o src/f32-vrsqrt/gen/f32-vrsqrt-sse-rsqrt-u${b}.c &
done

################################### x86 AVX ###################################
for b in 8 16 24 32 40 48 56 64; do
  tools/xngen src/f32-vrsqrt/avx-rsqrt.c.in -D BATCH_TILE=${b} -o src/f32-vrsqrt/gen/f32-vrsqrt-avx-rsqrt-u${b}.c &
done

################################### x86 FMA3 ##################################
for b in 8 16 24 32 40 48 56 64; do
  tools/xngen src/f32-vrsqrt/fma3-rsqrt.c.in -D BATCH_TILE=${b} -o src/f32-vrsqrt/gen/f32-vrsqrt-fma3-rsqrt-u${b}.c &
done

################################# x86 AVX512F #################################
for b in 16 32 48 64 80 96 112 128; do
  tools/xngen src/f32-vrsqrt/avx512f-rsqrt.c.in -D BATCH_TILE=${b} -o src/f32-vrsqrt/gen/f32-vrsqrt-avx512f-rsqrt-u${b}.c &
done

wait
