#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=ADD     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=DIV     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vdiv-aarch64-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=DIV     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vdiv-aarch64-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MAX     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MIN     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MUL     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=PRELU   -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SQRDIFF -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SUB     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-neon-u16.c &

tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=ADD     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=DIV     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vdivc-aarch64-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=DIV     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vdivc-aarch64-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RDIV    -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrdivc-aarch64-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RDIV    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrdivc-aarch64-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MAX     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MIN     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MUL     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=PRELU   -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RPRELU  -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RPRELU  -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SQRDIFF -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SUB     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RSUB    -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RSUB    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-neon-u16.c &

################################# ARM NEON-BF16 ################################
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=ADD     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MAX     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MIN     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MUL     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=PRELU   -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SQRDIFF -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SUB     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-neonbf16-u16.c &

tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=ADD     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MAX     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MIN     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MUL     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=PRELU   -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=RPRELU  -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=RPRELU  -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SQRDIFF -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SUB     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=RSUB    -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=RSUB    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-neonbf16-u16.c &

################################## x86 AVX512SKX ##############################
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=ADD     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=DIV     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vdiv-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=DIV     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vdiv-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MAX     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MIN     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MUL     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=PRELU   -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=SQRDIFF -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=SUB     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-avx512skx-u32.c &

tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=ADD     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=DIV     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vdivc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=DIV     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vdivc-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=RDIV    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrdivc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=RDIV    -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrdivc-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MAX     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MIN     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=MUL     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=PRELU   -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=RPRELU  -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=RPRELU  -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=SQRDIFF -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=SUB     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-avx512skx-u32.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=RSUB    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-avx512skx-u16.c &
tools/xngen src/bf16-vbinary/avx512skx.c.in -D OP=RSUB    -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-avx512skx-u32.c &

################################## x86 AVX512BF16 #############################
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=ADD     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=DIV     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vdiv-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=DIV     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vdiv-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MAX     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MIN     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MUL     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=PRELU   -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=SQRDIFF -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=SUB     -D BATCH_TILE=32 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-avx512bf16-u32.c &

tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=ADD     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=DIV     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vdivc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=DIV     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vdivc-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=RDIV    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrdivc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=RDIV    -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrdivc-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MAX     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MIN     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=MUL     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=PRELU   -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=RPRELU  -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=RPRELU  -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=SQRDIFF -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=SUB     -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-avx512bf16-u32.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=RSUB    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-avx512bf16-u16.c &
tools/xngen src/bf16-vbinary/avx512bf16.c.in -D OP=RSUB    -D BATCH_TILE=32 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-avx512bf16-u32.c &

wait
