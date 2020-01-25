#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=4  -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x4.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x8.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x8-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x12.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x12-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x12-acc3.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x16.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x16-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x16-acc4.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x20.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x20-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=5 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-p5-x20-acc5.c

tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=4  -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x4.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x8.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x8-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x12.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x12-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x12-acc3.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x16.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x16-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x16-acc4.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x20.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x20-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=5 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/neon-lut64-p2-x20-acc5.c

tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=4  -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x4.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x8.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x8-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x12.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x12-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x12-acc3.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x16.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x16-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x16-acc4.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x20.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x20-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=5 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-p5-x20-acc5.c

tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=4  -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x4.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x8.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x8-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x12.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x12-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x12-acc3.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x16.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x16-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x16-acc4.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x20.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x20-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=5 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/neonfma-lut64-p2-x20-acc5.c

################################### x86 SSE2 ##################################
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=4  -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x4.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x8.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x8-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x12.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x12-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x12-acc3.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x16.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x16-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x16-acc4.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x20.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x20-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/sse2-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=5 -o src/f32-raddstoreexpminusmax/gen/sse2-p5-x20-acc5.c

################################### x86 AVX2 ##################################
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=64 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x64.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=64 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x64-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=64 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x64-acc4.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=72 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x72.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=72 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x72-acc3.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=80 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x80.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=80 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x80-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=80 -D ACCUMULATORS=5 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x80-acc5.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=96 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x96.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=96 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x96-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=96 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x96-acc3.c
tools/xngen src/f32-raddstoreexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=96 -D ACCUMULATORS=6 -o src/f32-raddstoreexpminusmax/gen/avx2-p5-x96-acc6.c

################################# x86 AVX512F #################################
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=128 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x128.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=128 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x128-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=128 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x128-acc4.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=144 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x144.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=144 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x144-acc3.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=160 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x160.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=160 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x160-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=160 -D ACCUMULATORS=5 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x160-acc5.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=192 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=192 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=192 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192-acc3.c
tools/xngen src/f32-raddstoreexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=192 -D ACCUMULATORS=6 -o src/f32-raddstoreexpminusmax/gen/avx512f-p5-scalef-x192-acc6.c

#################################### PSIMD ####################################
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=4  -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x4.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x8.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=8  -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x8-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x12.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x12-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=12 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x12-acc3.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x16.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x16-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=16 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x16-acc4.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x20.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x20-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/psimd-p5.c.in -D ELEMENTS_TILE=20 -D ACCUMULATORS=5 -o src/f32-raddstoreexpminusmax/gen/psimd-p5-x20-acc5.c

################################### Scalar ####################################
tools/xngen src/f32-raddstoreexpminusmax/scalar-p5.c.in -D ELEMENTS_TILE=1 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/scalar-p5-x1.c
tools/xngen src/f32-raddstoreexpminusmax/scalar-p5.c.in -D ELEMENTS_TILE=2 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/scalar-p5-x2.c
tools/xngen src/f32-raddstoreexpminusmax/scalar-p5.c.in -D ELEMENTS_TILE=2 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/scalar-p5-x2-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/scalar-p5.c.in -D ELEMENTS_TILE=4 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/scalar-p5-x4.c
tools/xngen src/f32-raddstoreexpminusmax/scalar-p5.c.in -D ELEMENTS_TILE=4 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/scalar-p5-x4-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/scalar-p5.c.in -D ELEMENTS_TILE=4 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/scalar-p5-x4-acc4.c

tools/xngen src/f32-raddstoreexpminusmax/scalar-lut64-p2.c.in -D ELEMENTS_TILE=1 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x1.c
tools/xngen src/f32-raddstoreexpminusmax/scalar-lut64-p2.c.in -D ELEMENTS_TILE=2 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x2.c
tools/xngen src/f32-raddstoreexpminusmax/scalar-lut64-p2.c.in -D ELEMENTS_TILE=2 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x2-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/scalar-lut64-p2.c.in -D ELEMENTS_TILE=4 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x4.c
tools/xngen src/f32-raddstoreexpminusmax/scalar-lut64-p2.c.in -D ELEMENTS_TILE=4 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x4-acc2.c
tools/xngen src/f32-raddstoreexpminusmax/scalar-lut64-p2.c.in -D ELEMENTS_TILE=4 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/scalar-lut64-p2-x4-acc4.c

################################## Unit tests #################################
tools/generate-raddstoreexpminusmax-test.py --spec test/f32-raddstoreexpminusmax.yaml --output test/f32-raddstoreexpminusmax.cc
