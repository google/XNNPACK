#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=8  TYPE=int8_t  TILE_HEIGHT=1 TILE_WIDTH=2 -o src/x8-transposec/gen/x8-transposec-1x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=8  TYPE=int8_t  TILE_HEIGHT=1 TILE_WIDTH=4 -o src/x8-transposec/gen/x8-transposec-1x4-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=8  TYPE=int8_t  TILE_HEIGHT=2 TILE_WIDTH=1 -o src/x8-transposec/gen/x8-transposec-2x1-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=8  TYPE=int8_t  TILE_HEIGHT=2 TILE_WIDTH=2 -o src/x8-transposec/gen/x8-transposec-2x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=8  TYPE=int8_t  TILE_HEIGHT=2 TILE_WIDTH=4 -o src/x8-transposec/gen/x8-transposec-2x4-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=8  TYPE=int8_t  TILE_HEIGHT=4 TILE_WIDTH=1 -o src/x8-transposec/gen/x8-transposec-4x1-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=8  TYPE=int8_t  TILE_HEIGHT=4 TILE_WIDTH=2 -o src/x8-transposec/gen/x8-transposec-4x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=8  TYPE=int8_t  TILE_HEIGHT=4 TILE_WIDTH=4 -o src/x8-transposec/gen/x8-transposec-4x4-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=16 TYPE=int16_t TILE_HEIGHT=1 TILE_WIDTH=2 -o src/x16-transposec/gen/x16-transposec-1x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=16 TYPE=int16_t TILE_HEIGHT=1 TILE_WIDTH=4 -o src/x16-transposec/gen/x16-transposec-1x4-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=16 TYPE=int16_t TILE_HEIGHT=2 TILE_WIDTH=1 -o src/x16-transposec/gen/x16-transposec-2x1-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=16 TYPE=int16_t TILE_HEIGHT=2 TILE_WIDTH=2 -o src/x16-transposec/gen/x16-transposec-2x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=16 TYPE=int16_t TILE_HEIGHT=2 TILE_WIDTH=4 -o src/x16-transposec/gen/x16-transposec-2x4-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=16 TYPE=int16_t TILE_HEIGHT=4 TILE_WIDTH=1 -o src/x16-transposec/gen/x16-transposec-4x1-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=16 TYPE=int16_t TILE_HEIGHT=4 TILE_WIDTH=2 -o src/x16-transposec/gen/x16-transposec-4x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=16 TYPE=int16_t TILE_HEIGHT=4 TILE_WIDTH=4 -o src/x16-transposec/gen/x16-transposec-4x4-scalar-int.c &
tools/xngen src/x24-transposec/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2                      -o src/x24-transposec/gen/x24-transposec-1x2-scalar.c &
tools/xngen src/x24-transposec/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=4                      -o src/x24-transposec/gen/x24-transposec-1x4-scalar.c &
tools/xngen src/x24-transposec/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1                      -o src/x24-transposec/gen/x24-transposec-2x1-scalar.c &
tools/xngen src/x24-transposec/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2                      -o src/x24-transposec/gen/x24-transposec-2x2-scalar.c &
tools/xngen src/x24-transposec/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=4                      -o src/x24-transposec/gen/x24-transposec-2x4-scalar.c &
tools/xngen src/x24-transposec/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1                      -o src/x24-transposec/gen/x24-transposec-4x1-scalar.c &
tools/xngen src/x24-transposec/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2                      -o src/x24-transposec/gen/x24-transposec-4x2-scalar.c &
tools/xngen src/x24-transposec/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=4                      -o src/x24-transposec/gen/x24-transposec-4x4-scalar.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=int     -o src/x32-transposec/gen/x32-transposec-1x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=1 TILE_WIDTH=4 TYPE=int     -o src/x32-transposec/gen/x32-transposec-1x4-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=int     -o src/x32-transposec/gen/x32-transposec-2x1-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=int     -o src/x32-transposec/gen/x32-transposec-2x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=2 TILE_WIDTH=4 TYPE=int     -o src/x32-transposec/gen/x32-transposec-2x4-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=int     -o src/x32-transposec/gen/x32-transposec-4x1-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=int     -o src/x32-transposec/gen/x32-transposec-4x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=4 TILE_WIDTH=4 TYPE=int     -o src/x32-transposec/gen/x32-transposec-4x4-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=float   -o src/x32-transposec/gen/x32-transposec-1x2-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=1 TILE_WIDTH=4 TYPE=float   -o src/x32-transposec/gen/x32-transposec-1x4-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=float   -o src/x32-transposec/gen/x32-transposec-2x1-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=float   -o src/x32-transposec/gen/x32-transposec-2x2-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=2 TILE_WIDTH=4 TYPE=float   -o src/x32-transposec/gen/x32-transposec-2x4-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=float   -o src/x32-transposec/gen/x32-transposec-4x1-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=float   -o src/x32-transposec/gen/x32-transposec-4x2-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=32 TILE_HEIGHT=4 TILE_WIDTH=4 TYPE=float   -o src/x32-transposec/gen/x32-transposec-4x4-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=64 TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=int64_t -o src/x64-transposec/gen/x64-transposec-1x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=64 TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=int64_t -o src/x64-transposec/gen/x64-transposec-2x1-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=64 TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=int64_t -o src/x64-transposec/gen/x64-transposec-2x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=64 TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=int64_t -o src/x64-transposec/gen/x64-transposec-4x1-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=64 TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=int64_t -o src/x64-transposec/gen/x64-transposec-4x2-scalar-int.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=64 TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=double  -o src/x64-transposec/gen/x64-transposec-1x2-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=64 TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=double  -o src/x64-transposec/gen/x64-transposec-2x1-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=64 TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=double  -o src/x64-transposec/gen/x64-transposec-2x2-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=64 TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=double  -o src/x64-transposec/gen/x64-transposec-4x1-scalar-float.c &
tools/xngen src/x32-transposec/scalar.c.in -D SIZE=64 TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=double  -o src/x64-transposec/gen/x64-transposec-4x2-scalar-float.c &

#################################### SSE2 ###################################
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=8  IN_PTRS=REUSE OUT_PTRS=MOV     -o src/x8-transposec/gen/x8-transposec-16x16-reuse-mov-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=8  IN_PTRS=REUSE OUT_PTRS=SWITCH  -o src/x8-transposec/gen/x8-transposec-16x16-reuse-switch-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=16 IN_PTRS=REUSE OUT_PTRS=MOV     -o src/x16-transposec/gen/x16-transposec-8x8-reuse-mov-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=16 IN_PTRS=REUSE OUT_PTRS=SWITCH  -o src/x16-transposec/gen/x16-transposec-8x8-reuse-switch-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=16 IN_PTRS=REUSE OUT_PTRS=MULTI   -o src/x16-transposec/gen/x16-transposec-8x8-reuse-multi-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=16 IN_PTRS=MULTI OUT_PTRS=SWITCH  -o src/x16-transposec/gen/x16-transposec-8x8-multi-switch-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=16 IN_PTRS=MULTI OUT_PTRS=MOV     -o src/x16-transposec/gen/x16-transposec-8x8-multi-mov-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=32 IN_PTRS=REUSE OUT_PTRS=MOV     -o src/x32-transposec/gen/x32-transposec-4x4-reuse-mov-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=32 IN_PTRS=REUSE OUT_PTRS=SWITCH  -o src/x32-transposec/gen/x32-transposec-4x4-reuse-switch-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=32 IN_PTRS=REUSE OUT_PTRS=MULTI   -o src/x32-transposec/gen/x32-transposec-4x4-reuse-multi-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=32 IN_PTRS=MULTI OUT_PTRS=SWITCH  -o src/x32-transposec/gen/x32-transposec-4x4-multi-switch-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=32 IN_PTRS=MULTI OUT_PTRS=MULTI   -o src/x32-transposec/gen/x32-transposec-4x4-multi-multi-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=32 IN_PTRS=MULTI OUT_PTRS=MOV     -o src/x32-transposec/gen/x32-transposec-4x4-multi-mov-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=64 IN_PTRS=REUSE OUT_PTRS=MOV     -o src/x64-transposec/gen/x64-transposec-2x2-reuse-mov-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=64 IN_PTRS=REUSE OUT_PTRS=SWITCH  -o src/x64-transposec/gen/x64-transposec-2x2-reuse-switch-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=64 IN_PTRS=REUSE OUT_PTRS=MULTI   -o src/x64-transposec/gen/x64-transposec-2x2-reuse-multi-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=64 IN_PTRS=MULTI OUT_PTRS=SWITCH  -o src/x64-transposec/gen/x64-transposec-2x2-multi-switch-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=64 IN_PTRS=MULTI OUT_PTRS=MULTI   -o src/x64-transposec/gen/x64-transposec-2x2-multi-multi-sse2.c &
tools/xngen src/x32-transposec/sse2.c.in -D SIZE=64 IN_PTRS=MULTI OUT_PTRS=MOV     -o src/x64-transposec/gen/x64-transposec-2x2-multi-mov-sse2.c &

#################################### AVX ###################################
tools/xngen src/x32-transposec/avx.c.in -D SIZE=32 ARCH=AVX IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x32-transposec/gen/x32-transposec-8x8-reuse-mov-avx.c &
tools/xngen src/x32-transposec/avx.c.in -D SIZE=32 ARCH=AVX IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x32-transposec/gen/x32-transposec-8x8-reuse-switch-avx.c &
tools/xngen src/x32-transposec/avx.c.in -D SIZE=32 ARCH=AVX IN_PTRS=REUSE OUT_PTRS=MULTI  -o src/x32-transposec/gen/x32-transposec-8x8-reuse-multi-avx.c &
tools/xngen src/x32-transposec/avx.c.in -D SIZE=32 ARCH=AVX IN_PTRS=MULTI OUT_PTRS=SWITCH -o src/x32-transposec/gen/x32-transposec-8x8-multi-switch-avx.c &
tools/xngen src/x32-transposec/avx.c.in -D SIZE=32 ARCH=AVX IN_PTRS=MULTI OUT_PTRS=MOV    -o src/x32-transposec/gen/x32-transposec-8x8-multi-mov-avx.c &
tools/xngen src/x32-transposec/avx.c.in -D SIZE=64 ARCH=AVX IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x64-transposec/gen/x64-transposec-4x4-reuse-mov-avx.c &
tools/xngen src/x32-transposec/avx.c.in -D SIZE=64 ARCH=AVX IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x64-transposec/gen/x64-transposec-4x4-reuse-switch-avx.c &
tools/xngen src/x32-transposec/avx.c.in -D SIZE=64 ARCH=AVX IN_PTRS=REUSE OUT_PTRS=MULTI  -o src/x64-transposec/gen/x64-transposec-4x4-reuse-multi-avx.c &
tools/xngen src/x32-transposec/avx.c.in -D SIZE=64 ARCH=AVX IN_PTRS=MULTI OUT_PTRS=SWITCH -o src/x64-transposec/gen/x64-transposec-4x4-multi-switch-avx.c &
tools/xngen src/x32-transposec/avx.c.in -D SIZE=64 ARCH=AVX IN_PTRS=MULTI OUT_PTRS=MULTI  -o src/x64-transposec/gen/x64-transposec-4x4-multi-multi-avx.c &
tools/xngen src/x32-transposec/avx.c.in -D SIZE=64 ARCH=AVX IN_PTRS=MULTI OUT_PTRS=MOV    -o src/x64-transposec/gen/x64-transposec-4x4-multi-mov-avx.c &

#################################### AVX2 ###################################
tools/xngen src/x32-transposec/avx2.c.in -D SIZE=8  IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x8-transposec/gen/x8-transposec-32x32-reuse-mov-avx2.c &
tools/xngen src/x32-transposec/avx2.c.in -D SIZE=8  IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x8-transposec/gen/x8-transposec-32x32-reuse-switch-avx2.c &
tools/xngen src/x32-transposec/avx2.c.in -D SIZE=16 IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x16-transposec/gen/x16-transposec-16x16-reuse-mov-avx2.c &
tools/xngen src/x32-transposec/avx2.c.in -D SIZE=16 IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x16-transposec/gen/x16-transposec-16x16-reuse-switch-avx2.c &

#################################### ARM NEON ###############################
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=8 VECTOR_SIZE=64  IN_PTRS=MULTI OUT_PTRS=DEC    -o src/x8-transposec/gen/x8-transposec-8x8-multi-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=8 VECTOR_SIZE=64  IN_PTRS=MULTI OUT_PTRS=MOV    -o src/x8-transposec/gen/x8-transposec-8x8-multi-mov-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=8 VECTOR_SIZE=64  IN_PTRS=MULTI OUT_PTRS=SWITCH -o src/x8-transposec/gen/x8-transposec-8x8-multi-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=8 VECTOR_SIZE=64  IN_PTRS=REUSE OUT_PTRS=DEC    -o src/x8-transposec/gen/x8-transposec-8x8-reuse-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=8 VECTOR_SIZE=64  IN_PTRS=REUSE OUT_PTRS=MULTI  -o src/x8-transposec/gen/x8-transposec-8x8-reuse-multi-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=8 VECTOR_SIZE=64  IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x8-transposec/gen/x8-transposec-8x8-reuse-mov-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=8 VECTOR_SIZE=64  IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x8-transposec/gen/x8-transposec-8x8-reuse-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=8 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=DEC    -o src/x8-transposec/gen/x8-transposec-16x16-reuse-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=8 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x8-transposec/gen/x8-transposec-16x16-reuse-mov-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=8 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x8-transposec/gen/x8-transposec-16x16-reuse-switch-zip-neon.c &

tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=64 IN_PTRS=MULTI OUT_PTRS=DEC    -o src/x16-transposec/gen/x16-transposec-4x4-multi-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=64 IN_PTRS=MULTI OUT_PTRS=MOV    -o src/x16-transposec/gen/x16-transposec-4x4-multi-mov-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=64 IN_PTRS=MULTI OUT_PTRS=MULTI  -o src/x16-transposec/gen/x16-transposec-4x4-multi-multi-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=64 IN_PTRS=MULTI OUT_PTRS=SWITCH -o src/x16-transposec/gen/x16-transposec-4x4-multi-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=64 IN_PTRS=REUSE OUT_PTRS=DEC    -o src/x16-transposec/gen/x16-transposec-4x4-reuse-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=64 IN_PTRS=REUSE OUT_PTRS=MULTI  -o src/x16-transposec/gen/x16-transposec-4x4-reuse-multi-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=64 IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x16-transposec/gen/x16-transposec-4x4-reuse-mov-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=64 IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x16-transposec/gen/x16-transposec-4x4-reuse-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=DEC    -o src/x16-transposec/gen/x16-transposec-8x8-reuse-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x16-transposec/gen/x16-transposec-8x8-reuse-mov-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x16-transposec/gen/x16-transposec-8x8-reuse-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=MULTI  -o src/x16-transposec/gen/x16-transposec-8x8-reuse-multi-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=SWITCH -o src/x16-transposec/gen/x16-transposec-8x8-multi-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=DEC    -o src/x16-transposec/gen/x16-transposec-8x8-multi-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=16 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=MOV    -o src/x16-transposec/gen/x16-transposec-8x8-multi-mov-zip-neon.c &

tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=64  IN_PTRS=REUSE OUT_PTRS=DEC    -o src/x32-transposec/gen/x32-transposec-2x2-reuse-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=64  IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x32-transposec/gen/x32-transposec-2x2-reuse-mov-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=64  IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x32-transposec/gen/x32-transposec-2x2-reuse-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=64  IN_PTRS=REUSE OUT_PTRS=MULTI  -o src/x32-transposec/gen/x32-transposec-2x2-reuse-multi-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=64  IN_PTRS=MULTI OUT_PTRS=SWITCH -o src/x32-transposec/gen/x32-transposec-2x2-multi-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=64  IN_PTRS=MULTI OUT_PTRS=MULTI  -o src/x32-transposec/gen/x32-transposec-2x2-multi-multi-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=64  IN_PTRS=MULTI OUT_PTRS=DEC    -o src/x32-transposec/gen/x32-transposec-2x2-multi-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=64  IN_PTRS=MULTI OUT_PTRS=MOV    -o src/x32-transposec/gen/x32-transposec-2x2-multi-mov-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=DEC    -o src/x32-transposec/gen/x32-transposec-4x4-reuse-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x32-transposec/gen/x32-transposec-4x4-reuse-mov-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x32-transposec/gen/x32-transposec-4x4-reuse-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=MULTI  -o src/x32-transposec/gen/x32-transposec-4x4-reuse-multi-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=SWITCH -o src/x32-transposec/gen/x32-transposec-4x4-multi-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=MULTI  -o src/x32-transposec/gen/x32-transposec-4x4-multi-multi-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=DEC    -o src/x32-transposec/gen/x32-transposec-4x4-multi-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=32 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=MOV    -o src/x32-transposec/gen/x32-transposec-4x4-multi-mov-zip-neon.c &

tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=64 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=DEC    -o src/x64-transposec/gen/x64-transposec-2x2-reuse-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=64 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x64-transposec/gen/x64-transposec-2x2-reuse-mov-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=64 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x64-transposec/gen/x64-transposec-2x2-reuse-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=64 VECTOR_SIZE=128 IN_PTRS=REUSE OUT_PTRS=MULTI  -o src/x64-transposec/gen/x64-transposec-2x2-reuse-multi-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=64 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=SWITCH -o src/x64-transposec/gen/x64-transposec-2x2-multi-switch-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=64 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=MULTI  -o src/x64-transposec/gen/x64-transposec-2x2-multi-multi-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=64 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=DEC    -o src/x64-transposec/gen/x64-transposec-2x2-multi-dec-zip-neon.c &
tools/xngen src/x32-transposec/neon-zip.c.in -D SIZE=64 VECTOR_SIZE=128 IN_PTRS=MULTI OUT_PTRS=MOV    -o src/x64-transposec/gen/x64-transposec-2x2-multi-mov-zip-neon.c &

#################################### WAsm SIMD ###############################
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=8  IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x8-transposec/gen/x8-transposec-16x16-reuse-mov-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=8  IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x8-transposec/gen/x8-transposec-16x16-reuse-switch-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=16 IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x16-transposec/gen/x16-transposec-8x8-reuse-mov-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=16 IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x16-transposec/gen/x16-transposec-8x8-reuse-switch-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=16 IN_PTRS=REUSE OUT_PTRS=MULTI  -o src/x16-transposec/gen/x16-transposec-8x8-reuse-multi-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=16 IN_PTRS=MULTI OUT_PTRS=SWITCH -o src/x16-transposec/gen/x16-transposec-8x8-multi-switch-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=16 IN_PTRS=MULTI OUT_PTRS=MOV    -o src/x16-transposec/gen/x16-transposec-8x8-multi-mov-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=32 IN_PTRS=REUSE OUT_PTRS=MOV    -o src/x32-transposec/gen/x32-transposec-4x4-reuse-mov-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=32 IN_PTRS=REUSE OUT_PTRS=SWITCH -o src/x32-transposec/gen/x32-transposec-4x4-reuse-switch-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=32 IN_PTRS=REUSE OUT_PTRS=MULTI  -o src/x32-transposec/gen/x32-transposec-4x4-reuse-multi-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=32 IN_PTRS=MULTI OUT_PTRS=SWITCH -o src/x32-transposec/gen/x32-transposec-4x4-multi-switch-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=32 IN_PTRS=MULTI OUT_PTRS=MULTI  -o src/x32-transposec/gen/x32-transposec-4x4-multi-multi-wasmsimd.c &
tools/xngen src/x32-transposec/wasmsimd.c.in -D SIZE=32 IN_PTRS=MULTI OUT_PTRS=MOV    -o src/x32-transposec/gen/x32-transposec-4x4-multi-mov-wasmsimd.c &

wait
