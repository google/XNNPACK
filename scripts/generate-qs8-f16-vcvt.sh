#!/bin/bash
################################## ARM NEON ###################################
tools/xngen src/qs8-f16-vcvt/neon.c.in -D BATCH_TILE=8  -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-neonfp16arith-u8.c &
tools/xngen src/qs8-f16-vcvt/neon.c.in -D BATCH_TILE=16 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-neonfp16arith-u16.c &
tools/xngen src/qs8-f16-vcvt/neon.c.in -D BATCH_TILE=24 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-neonfp16arith-u24.c &
tools/xngen src/qs8-f16-vcvt/neon.c.in -D BATCH_TILE=32 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-neonfp16arith-u32.c &

################################# x86 256-bit #################################
tools/xngen src/qs8-f16-vcvt/avx2.c.in -D BATCH_TILE=16 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-avx2-u16.c &
tools/xngen src/qs8-f16-vcvt/avx2.c.in -D BATCH_TILE=24 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-avx2-u24.c &
tools/xngen src/qs8-f16-vcvt/avx2.c.in -D BATCH_TILE=32 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-avx2-u32.c &
tools/xngen src/qs8-f16-vcvt/avx2.c.in -D BATCH_TILE=64 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-avx2-u64.c &
