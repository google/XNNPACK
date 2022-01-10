#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vmulcaddc/neonfp16arith.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f16-vmulcaddc/gen/c8-minmax-neonfp16arith-2x.c &
tools/xngen src/f16-vmulcaddc/neonfp16arith.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f16-vmulcaddc/gen/c16-minmax-neonfp16arith-2x.c &

################################### x86 FMA3 ##################################
tools/xngen src/f16-vmulcaddc/fma3.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f16-vmulcaddc/gen/c8-minmax-fma3-2x.c &
tools/xngen src/f16-vmulcaddc/fma3.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f16-vmulcaddc/gen/c16-minmax-fma3-2x.c &

################################## Unit tests #################################
tools/generate-vmulcaddc-test.py --spec test/f16-vmulcaddc-minmax.yaml --output test/f16-vmulcaddc-minmax.cc &

wait
