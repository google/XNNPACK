#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vclamp/neonfp16arith.c.in -D BATCH_TILE=8  -o src/f16-vclamp/gen/vclamp-neonfp16arith-x8.c &
tools/xngen src/f16-vclamp/neonfp16arith.c.in -D BATCH_TILE=16 -o src/f16-vclamp/gen/vclamp-neonfp16arith-x16.c &

################################### x86 F16C ##################################
tools/xngen src/f16-vclamp/f16c.c.in -D BATCH_TILE=8  -o src/f16-vclamp/gen/vclamp-f16c-x8.c &
tools/xngen src/f16-vclamp/f16c.c.in -D BATCH_TILE=16 -o src/f16-vclamp/gen/vclamp-f16c-x16.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f16-vclamp.yaml --output test/f16-vclamp.cc &

wait
