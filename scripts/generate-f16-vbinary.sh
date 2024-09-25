#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=ADD     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vadd-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=ADD     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vadd-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=DIV     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vdiv-aarch64-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=DIV     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vdiv-aarch64-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MAX     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vmax-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MAX     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vmax-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MIN     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vmin-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MIN     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vmin-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MUL     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vmul-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MUL     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vmul-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vsqrdiff-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vsqrdiff-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SUB     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vsub-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SUB     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vsub-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=PRELU   -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vprelu-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=PRELU   -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vprelu-neonfp16arith-u16.c &

tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=ADD      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vaddc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=ADD      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vaddc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=DIV      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vdivc-aarch64-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=DIV      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vdivc-aarch64-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RDIV     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vrdivc-aarch64-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RDIV     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vrdivc-aarch64-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MAX      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vmaxc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MAX      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vmaxc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MIN      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vminc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MIN      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vminc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MUL      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vmulc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MUL      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vmulc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SQRDIFF  -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vsqrdiffc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SQRDIFF  -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vsqrdiffc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SUB      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vsubc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SUB      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vsubc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RSUB     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vrsubc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RSUB     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vrsubc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=PRELU    -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vpreluc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=PRELU    -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vpreluc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RPRELU   -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vrpreluc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RPRELU   -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vrpreluc-neonfp16arith-u16.c &

################################### ARM FP16 ##################################
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vadd-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vadd-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vadd-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vdiv-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vdiv-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vdiv-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vmax-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vmin-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vmin-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vmin-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vmul-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vmul-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vmul-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vsub-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vsub-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vsub-fp16arith-u4.c &

tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vaddc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vaddc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vaddc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vdivc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vdivc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vdivc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RDIV    -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vrdivc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RDIV    -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vrdivc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RDIV    -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vrdivc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vmaxc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vmaxc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vmaxc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vminc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vminc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vminc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vmulc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vmulc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vmulc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vsubc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vsubc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vsubc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RSUB    -D BATCH_TILE=1 -o src/f16-vbinary/gen/f16-vrsubc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RSUB    -D BATCH_TILE=2 -o src/f16-vbinary/gen/f16-vrsubc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RSUB    -D BATCH_TILE=4 -o src/f16-vbinary/gen/f16-vrsubc-fp16arith-u4.c &

################################### x86 AVX512 FP16 ##################################
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=ADD      -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vadd-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=ADD      -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vadd-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=DIV      -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vdiv-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=DIV      -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vdiv-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=MAX      -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vmax-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=MAX      -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vmax-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=MIN      -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vmin-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=MIN      -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vmin-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=MUL      -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vmul-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=MUL      -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vmul-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=SQRDIFF  -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vsqrdiff-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=SQRDIFF  -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vsqrdiff-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=SUB      -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vsub-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=SUB      -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vsub-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=PRELU    -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vprelu-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vop-avx512fp16.c.in -D OP=PRELU    -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vprelu-avx512fp16-u64.c &

tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=ADD     -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vaddc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=ADD     -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vaddc-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=DIV     -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vdivc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=DIV     -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vdivc-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=RDIV    -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vrdivc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=RDIV    -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vrdivc-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=MAX     -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vmaxc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=MAX     -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vmaxc-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=MIN     -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vminc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=MIN     -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vminc-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=MUL     -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vmulc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=MUL     -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vmulc-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=SQRDIFF -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vsqrdiffc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=SQRDIFF -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vsqrdiffc-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=SUB     -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vsubc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=SUB     -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vsubc-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=RSUB    -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vrsubc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=RSUB    -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vrsubc-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=PRELU   -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vpreluc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=PRELU   -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vpreluc-avx512fp16-u64.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=RPRELU  -D BATCH_TILE=32  -o src/f16-vbinary/gen/f16-vrpreluc-avx512fp16-u32.c &
tools/xngen src/f16-vbinary/vopc-avx512fp16.c.in -D OP=RPRELU  -D BATCH_TILE=64 -o src/f16-vbinary/gen/f16-vrpreluc-avx512fp16-u64.c &

################################### x86 F16C ##################################
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=ADD      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vadd-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=ADD      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vadd-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=DIV      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vdiv-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=DIV      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vdiv-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MAX      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MAX      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vmax-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MIN      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vmin-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MIN      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vmin-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MUL      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vmul-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MUL      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vmul-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SQRDIFF  -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vsqrdiff-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SQRDIFF  -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vsqrdiff-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SUB      -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vsub-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SUB      -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vsub-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=PRELU    -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vprelu-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=PRELU    -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vprelu-f16c-u16.c &

tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=ADD     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vaddc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=ADD     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vaddc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=DIV     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vdivc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=DIV     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vdivc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RDIV    -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vrdivc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RDIV    -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vrdivc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MAX     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vmaxc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MAX     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vmaxc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MIN     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vminc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MIN     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vminc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MUL     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vmulc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MUL     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vmulc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vsqrdiffc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vsqrdiffc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SUB     -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vsubc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SUB     -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vsubc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RSUB    -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vrsubc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RSUB    -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vrsubc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=PRELU   -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vpreluc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=PRELU   -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vpreluc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RPRELU  -D BATCH_TILE=8  -o src/f16-vbinary/gen/f16-vrpreluc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RPRELU  -D BATCH_TILE=16 -o src/f16-vbinary/gen/f16-vrpreluc-f16c-u16.c &

wait
