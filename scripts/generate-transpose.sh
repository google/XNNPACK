#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=int8_t  SIZE=8  -o src/x8-transpose/gen/1x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=4 TYPE=int8_t  SIZE=8  -o src/x8-transpose/gen/1x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=int8_t  SIZE=8  -o src/x8-transpose/gen/2x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=int8_t  SIZE=8  -o src/x8-transpose/gen/2x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=4 TYPE=int8_t  SIZE=8  -o src/x8-transpose/gen/2x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=int8_t  SIZE=8  -o src/x8-transpose/gen/4x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=int8_t  SIZE=8  -o src/x8-transpose/gen/4x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=4 TYPE=int8_t  SIZE=8  -o src/x8-transpose/gen/4x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/1x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=4 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/1x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/2x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/2x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=4 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/2x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/4x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/4x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=4 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/4x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=int     SIZE=32 -o src/x32-transpose/gen/1x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=4 TYPE=int     SIZE=32 -o src/x32-transpose/gen/1x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=int     SIZE=32 -o src/x32-transpose/gen/2x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=int     SIZE=32 -o src/x32-transpose/gen/2x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=4 TYPE=int     SIZE=32 -o src/x32-transpose/gen/2x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=int     SIZE=32 -o src/x32-transpose/gen/4x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=int     SIZE=32 -o src/x32-transpose/gen/4x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=4 TYPE=int     SIZE=32 -o src/x32-transpose/gen/4x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=float   SIZE=32 -o src/x32-transpose/gen/1x2-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=4 TYPE=float   SIZE=32 -o src/x32-transpose/gen/1x4-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=float   SIZE=32 -o src/x32-transpose/gen/2x1-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=float   SIZE=32 -o src/x32-transpose/gen/2x2-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=4 TYPE=float   SIZE=32 -o src/x32-transpose/gen/2x4-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=float   SIZE=32 -o src/x32-transpose/gen/4x1-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=float   SIZE=32 -o src/x32-transpose/gen/4x2-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=4 TYPE=float   SIZE=32 -o src/x32-transpose/gen/4x4-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=int64_t SIZE=64 -o src/x64-transpose/gen/1x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=int64_t SIZE=64 -o src/x64-transpose/gen/2x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=int64_t SIZE=64 -o src/x64-transpose/gen/2x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=int64_t SIZE=64 -o src/x64-transpose/gen/4x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=int64_t SIZE=64 -o src/x64-transpose/gen/4x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=double  SIZE=64 -o src/x64-transpose/gen/1x2-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=double  SIZE=64 -o src/x64-transpose/gen/2x1-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=double  SIZE=64 -o src/x64-transpose/gen/2x2-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=double  SIZE=64 -o src/x64-transpose/gen/4x1-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=double  SIZE=64 -o src/x64-transpose/gen/4x2-scalar-float.c &

#################################### SSE2 ###################################
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=MOV    SIZE=8  -o src/x8-transpose/gen/16x16-sse2-reuse-mov.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=SWITCH SIZE=8  -o src/x8-transpose/gen/16x16-sse2-reuse-switch.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=MOV    SIZE=16 -o src/x16-transpose/gen/8x8-sse2-reuse-mov.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=SWITCH SIZE=16 -o src/x16-transpose/gen/8x8-sse2-reuse-switch.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=MULTI  SIZE=16 -o src/x16-transpose/gen/8x8-sse2-reuse-multi.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=SWITCH SIZE=16 -o src/x16-transpose/gen/8x8-sse2-multi-switch.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=MOV    SIZE=16 -o src/x16-transpose/gen/8x8-sse2-multi-mov.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=MOV    SIZE=32 -o src/x32-transpose/gen/4x4-sse2-reuse-mov.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=SWITCH SIZE=32 -o src/x32-transpose/gen/4x4-sse2-reuse-switch.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=MULTI  SIZE=32 -o src/x32-transpose/gen/4x4-sse2-reuse-multi.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=SWITCH SIZE=32 -o src/x32-transpose/gen/4x4-sse2-multi-switch.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=MULTI  SIZE=32 -o src/x32-transpose/gen/4x4-sse2-multi-multi.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=MOV    SIZE=32 -o src/x32-transpose/gen/4x4-sse2-multi-mov.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=MOV    SIZE=64 -o src/x64-transpose/gen/2x2-sse2-reuse-mov.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=SWITCH SIZE=64 -o src/x64-transpose/gen/2x2-sse2-reuse-switch.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=MULTI  SIZE=64 -o src/x64-transpose/gen/2x2-sse2-reuse-multi.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=SWITCH SIZE=64 -o src/x64-transpose/gen/2x2-sse2-multi-switch.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=MULTI  SIZE=64 -o src/x64-transpose/gen/2x2-sse2-multi-multi.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=MOV    SIZE=64 -o src/x64-transpose/gen/2x2-sse2-multi-mov.c &

#################################### ARM NEON ###############################
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=DEC     SIZE=8  -o src/x8-transpose/gen/16x16-neon-reuse-dec-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=MOV     SIZE=8  -o src/x8-transpose/gen/16x16-neon-reuse-mov-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=SWITCH  SIZE=8  -o src/x8-transpose/gen/16x16-neon-reuse-switch-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=DEC     SIZE=16 -o src/x16-transpose/gen/8x8-neon-reuse-dec-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=MOV     SIZE=16 -o src/x16-transpose/gen/8x8-neon-reuse-mov-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=SWITCH  SIZE=16 -o src/x16-transpose/gen/8x8-neon-reuse-switch-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=MULTI   SIZE=16 -o src/x16-transpose/gen/8x8-neon-reuse-multi-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=MULTI OUT_PTRS=SWITCH  SIZE=16 -o src/x16-transpose/gen/8x8-neon-multi-switch-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=MULTI OUT_PTRS=DEC     SIZE=16 -o src/x16-transpose/gen/8x8-neon-multi-dec-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=MULTI OUT_PTRS=MOV     SIZE=16 -o src/x16-transpose/gen/8x8-neon-multi-mov-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=DEC     SIZE=32 -o src/x32-transpose/gen/4x4-neon-reuse-dec-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=MOV     SIZE=32 -o src/x32-transpose/gen/4x4-neon-reuse-mov-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=SWITCH  SIZE=32 -o src/x32-transpose/gen/4x4-neon-reuse-switch-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=REUSE OUT_PTRS=MULTI   SIZE=32 -o src/x32-transpose/gen/4x4-neon-reuse-multi-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=MULTI OUT_PTRS=SWITCH  SIZE=32 -o src/x32-transpose/gen/4x4-neon-multi-switch-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=MULTI OUT_PTRS=MULTI   SIZE=32 -o src/x32-transpose/gen/4x4-neon-multi-multi-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=MULTI OUT_PTRS=DEC     SIZE=32 -o src/x32-transpose/gen/4x4-neon-multi-dec-zip.c &
tools/xngen src/x32-transpose/neon-zip.c.in -D IN_PTRS=MULTI OUT_PTRS=MOV     SIZE=32 -o src/x32-transpose/gen/4x4-neon-multi-mov-zip.c &

################################## Unit tests #################################
tools/generate-transpose-test.py --spec test/x8-transpose.yaml  --output=test/x8-transpose.cc &
tools/generate-transpose-test.py --spec test/x16-transpose.yaml --output=test/x16-transpose.cc &
tools/generate-transpose-test.py --spec test/x32-transpose.yaml --output=test/x32-transpose.cc &
tools/generate-transpose-test.py --spec test/x64-transpose.yaml --output=test/x64-transpose.cc &

wait
