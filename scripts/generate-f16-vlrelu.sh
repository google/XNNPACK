#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vlrelu/neonfp16arith.c.in -D BATCH_TILE=8  -o src/f16-vlrelu/gen/vlrelu-neonfp16arith-x8.c &
tools/xngen src/f16-vlrelu/neonfp16arith.c.in -D BATCH_TILE=16 -o src/f16-vlrelu/gen/vlrelu-neonfp16arith-x16.c &

################################### x86 F16C ##################################
tools/xngen src/f16-vlrelu/f16c.c.in -D BATCH_TILE=8  -o src/f16-vlrelu/gen/vlrelu-f16c-x8.c &
tools/xngen src/f16-vlrelu/f16c.c.in -D BATCH_TILE=16 -o src/f16-vlrelu/gen/vlrelu-f16c-x16.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f16-vlrelu.yaml --output test/f16-vlrelu.cc &

wait
