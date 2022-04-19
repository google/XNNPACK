#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDNE -D BATCH_TILE=8  -o src/f16-vrnd/gen/vrndne-neonfp16arith-x8.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDNE -D BATCH_TILE=16 -o src/f16-vrnd/gen/vrndne-neonfp16arith-x16.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDZ  -D BATCH_TILE=8  -o src/f16-vrnd/gen/vrndz-neonfp16arith-x8.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDZ  -D BATCH_TILE=16 -o src/f16-vrnd/gen/vrndz-neonfp16arith-x16.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDU  -D BATCH_TILE=8  -o src/f16-vrnd/gen/vrndu-neonfp16arith-x8.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDU  -D BATCH_TILE=16 -o src/f16-vrnd/gen/vrndu-neonfp16arith-x16.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDD  -D BATCH_TILE=8  -o src/f16-vrnd/gen/vrndd-neonfp16arith-x8.c &
tools/xngen src/f16-vrnd/neonfp16arith.c.in -D OP=RNDD  -D BATCH_TILE=16 -o src/f16-vrnd/gen/vrndd-neonfp16arith-x16.c &

################################# x86 F16C #################################
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDNE -D BATCH_TILE=8  -o src/f16-vrnd/gen/vrndne-f16c-x8.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDNE -D BATCH_TILE=16 -o src/f16-vrnd/gen/vrndne-f16c-x16.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDZ  -D BATCH_TILE=8  -o src/f16-vrnd/gen/vrndz-f16c-x8.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDZ  -D BATCH_TILE=16 -o src/f16-vrnd/gen/vrndz-f16c-x16.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDU  -D BATCH_TILE=8  -o src/f16-vrnd/gen/vrndu-f16c-x8.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDU  -D BATCH_TILE=16 -o src/f16-vrnd/gen/vrndu-f16c-x16.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDD  -D BATCH_TILE=8  -o src/f16-vrnd/gen/vrndd-f16c-x8.c &
tools/xngen src/f16-vrnd/f16c.c.in -D OP=RNDD  -D BATCH_TILE=16 -o src/f16-vrnd/gen/vrndd-f16c-x16.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f16-vrndne.yaml --output test/f16-vrndne.cc &
tools/generate-vunary-test.py --spec test/f16-vrndz.yaml  --output test/f16-vrndz.cc &
tools/generate-vunary-test.py --spec test/f16-vrndu.yaml  --output test/f16-vrndu.cc &
tools/generate-vunary-test.py --spec test/f16-vrndd.yaml  --output test/f16-vrndd.cc &

wait
