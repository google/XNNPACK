#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vlrelu/neonfp16arith.c.in -D BATCH_TILE=8  -o src/f16-vlrelu/gen/f16-vlrelu-neonfp16arith-u8.c &
tools/xngen src/f16-vlrelu/neonfp16arith.c.in -D BATCH_TILE=16 -o src/f16-vlrelu/gen/f16-vlrelu-neonfp16arith-u16.c &

################################### x86 F16C ##################################
tools/xngen src/f16-vlrelu/f16c.c.in -D BATCH_TILE=8  -o src/f16-vlrelu/gen/f16-vlrelu-f16c-u8.c &
tools/xngen src/f16-vlrelu/f16c.c.in -D BATCH_TILE=16 -o src/f16-vlrelu/gen/f16-vlrelu-f16c-u16.c &

wait
