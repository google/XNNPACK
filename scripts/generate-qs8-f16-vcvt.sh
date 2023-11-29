#!/bin/bash
################################## ARM NEON ###################################
tools/xngen src/qs8-f16-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-neonfp16arith-u8.c &
tools/xngen src/qs8-f16-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-neonfp16arith-u16.c &
tools/xngen src/qs8-f16-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-neonfp16arith-u24.c &
tools/xngen src/qs8-f16-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f16-vcvt/gen/qs8-f16-vcvt-neonfp16arith-u32.c &

