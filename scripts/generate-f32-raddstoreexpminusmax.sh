#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u4.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u8.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u8-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u12.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u12-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u12-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u16.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u16-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u16-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u20.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u20-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=5 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-p5-u20-acc5.c &

tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u4.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u8.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u8-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=12 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u12.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=12 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u12-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u12-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u16.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u16-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u16-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=20 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u20.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=20 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u20-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=20 -D ACCUMULATORS=5 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neon-rr2-lut64-p2-u20-acc5.c &

tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u4.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u8.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u8-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u12.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u12-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u12-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u16.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u16-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u16-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u20.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u20-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=5 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-p5-u20-acc5.c &

tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u4.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u8.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u8-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=12 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u12.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=12 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u12-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u12-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u16.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u16-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u16-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=20 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u20.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=20 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u20-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in -D BATCH_TILE=20 -D ACCUMULATORS=5 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-neonfma-rr1-lut64-p2-u20-acc5.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-raddstoreexpminusmax/rvv-rr2-p6.c.in -D LMUL=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-rvv-rr2-p6-u2v.c &
tools/xngen src/f32-raddstoreexpminusmax/rvv-rr2-p6.c.in -D LMUL=4 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-rvv-rr2-p6-u4v.c &

################################### x86 SSE2 ##################################
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u4.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u8.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u8-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u12.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u12-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u12-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u16.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u16-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u16-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u20.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u20-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/sse2-rr2-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=5 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-sse2-rr2-p5-u20-acc5.c &

################################### x86 AVX2 ##################################
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=32 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u32.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u32-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u32-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=64 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u64.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u64-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u64-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=72 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u72.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=72 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u72-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=80 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u80.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=80 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u80-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=80 -D ACCUMULATORS=5 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u80-acc5.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u96.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u96-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u96-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/avx2-rr1-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=6 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u96-acc6.c &

################################# x86 AVX512F #################################
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=64  -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u64.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=64  -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u64-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=64  -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u64-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=128 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u128.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=128 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u128-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=128 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u128-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=144 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u144.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=144 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u144-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=160 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u160.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=160 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u160-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=160 -D ACCUMULATORS=5 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u160-acc5.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=192 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u192.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=192 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u192-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=192 -D ACCUMULATORS=3 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u192-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=192 -D ACCUMULATORS=6 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx512f-rr1-p5-scalef-u192-acc6.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u4.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u8.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u8-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u12.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u12-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u12-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u16.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u16-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u16-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u20.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u20-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=5 -D FMA=0 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmsimd-rr2-p5-u20-acc5.c &

tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u4.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u8.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u8-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u12.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u12-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u12-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u16.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u16-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u16-acc4.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u20.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u20-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/wasmsimd-rr2-p5.c.in -D BATCH_TILE=20 -D ACCUMULATORS=5 -D FMA=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u20-acc5.c &

################################## Hexagon HVX ##################################
tools/xngen src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in -D BATCH_TILE=32 -D ACCUMULATORS=1 -D -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u32.c &
tools/xngen src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in -D BATCH_TILE=64 -D ACCUMULATORS=1 -D -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u64.c &
tools/xngen src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -D -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u64-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=1 -D -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u96.c &
tools/xngen src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=2 -D -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u96-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=3 -D -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u96-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in -D BATCH_TILE=128 -D ACCUMULATORS=1 -D -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u128.c &
tools/xngen src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in -D BATCH_TILE=128 -D ACCUMULATORS=2 -D -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u128-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in -D BATCH_TILE=128 -D ACCUMULATORS=3 -D -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u128-acc3.c &
tools/xngen src/f32-raddstoreexpminusmax/hvx-rr2-p5.c.in -D BATCH_TILE=128 -D ACCUMULATORS=4 -D -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-hvx-rr2-p5-u128-acc4.c &

################################### Scalar ####################################
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-p5.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-p5-u1.c &
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-p5.c.in -D BATCH_TILE=2 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-p5-u2.c &
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-p5.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-p5-u2-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-p5.c.in -D BATCH_TILE=4 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-p5-u4.c &
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-p5.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-p5-u4-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-p5.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-p5-u4-acc4.c &

tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-lut64-p2.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-lut64-p2-u1.c &
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-lut64-p2.c.in -D BATCH_TILE=2 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-lut64-p2-u2.c &
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-lut64-p2.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-lut64-p2-u2-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-lut64-p2.c.in -D BATCH_TILE=4 -D ACCUMULATORS=1 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-lut64-p2-u4.c &
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-lut64-p2.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-lut64-p2-u4-acc2.c &
tools/xngen src/f32-raddstoreexpminusmax/scalar-rr2-lut64-p2.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -o src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-scalar-rr2-lut64-p2-u4-acc4.c &

wait
