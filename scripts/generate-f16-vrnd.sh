#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDNE -D BATCH_TILE=8  -o src/f16-vrnd/gen/f16-vrndne-neonfp16arith-u8.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDNE -D BATCH_TILE=16 -o src/f16-vrnd/gen/f16-vrndne-neonfp16arith-u16.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDZ  -D BATCH_TILE=8  -o src/f16-vrnd/gen/f16-vrndz-neonfp16arith-u8.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDZ  -D BATCH_TILE=16 -o src/f16-vrnd/gen/f16-vrndz-neonfp16arith-u16.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDU  -D BATCH_TILE=8  -o src/f16-vrnd/gen/f16-vrndu-neonfp16arith-u8.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDU  -D BATCH_TILE=16 -o src/f16-vrnd/gen/f16-vrndu-neonfp16arith-u16.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDD  -D BATCH_TILE=8  -o src/f16-vrnd/gen/f16-vrndd-neonfp16arith-u8.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDD  -D BATCH_TILE=16 -o src/f16-vrnd/gen/f16-vrndd-neonfp16arith-u16.c &

################################# x86 F16C #################################
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDNE -D BATCH_TILE=8  -o src/f16-vrnd/gen/f16-vrndne-f16c-u8.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDNE -D BATCH_TILE=16 -o src/f16-vrnd/gen/f16-vrndne-f16c-u16.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDZ  -D BATCH_TILE=8  -o src/f16-vrnd/gen/f16-vrndz-f16c-u8.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDZ  -D BATCH_TILE=16 -o src/f16-vrnd/gen/f16-vrndz-f16c-u16.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDU  -D BATCH_TILE=8  -o src/f16-vrnd/gen/f16-vrndu-f16c-u8.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDU  -D BATCH_TILE=16 -o src/f16-vrnd/gen/f16-vrndu-f16c-u16.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDD  -D BATCH_TILE=8  -o src/f16-vrnd/gen/f16-vrndd-f16c-u8.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDD  -D BATCH_TILE=16 -o src/f16-vrnd/gen/f16-vrndd-f16c-u16.c &

wait
