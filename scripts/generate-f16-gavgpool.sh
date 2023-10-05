#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f16-gavgpool/unipass-neonfp16arith.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -o src/f16-gavgpool/gen/f16-gavgpool-7x-minmax-neonfp16arith-c8.c &
tools/xngen src/f16-gavgpool/unipass-neonfp16arith.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -o src/f16-gavgpool/gen/f16-gavgpool-7x-minmax-neonfp16arith-c16.c &
tools/xngen src/f16-gavgpool/unipass-neonfp16arith.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -o src/f16-gavgpool/gen/f16-gavgpool-7x-minmax-neonfp16arith-c24.c &
tools/xngen src/f16-gavgpool/unipass-neonfp16arith.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -o src/f16-gavgpool/gen/f16-gavgpool-7x-minmax-neonfp16arith-c32.c &

tools/xngen src/f16-gavgpool/multipass-neonfp16arith.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -o src/f16-gavgpool/gen/f16-gavgpool-7p7x-minmax-neonfp16arith-c8.c &
tools/xngen src/f16-gavgpool/multipass-neonfp16arith.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -o src/f16-gavgpool/gen/f16-gavgpool-7p7x-minmax-neonfp16arith-c16.c &
tools/xngen src/f16-gavgpool/multipass-neonfp16arith.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -o src/f16-gavgpool/gen/f16-gavgpool-7p7x-minmax-neonfp16arith-c24.c &
tools/xngen src/f16-gavgpool/multipass-neonfp16arith.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -o src/f16-gavgpool/gen/f16-gavgpool-7p7x-minmax-neonfp16arith-c32.c &

################################### x86 F16C ###################################
tools/xngen src/f16-gavgpool/unipass-f16c.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -o src/f16-gavgpool/gen/f16-gavgpool-7x-minmax-f16c-c8.c &
tools/xngen src/f16-gavgpool/unipass-f16c.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -o src/f16-gavgpool/gen/f16-gavgpool-7x-minmax-f16c-c16.c &
tools/xngen src/f16-gavgpool/unipass-f16c.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -o src/f16-gavgpool/gen/f16-gavgpool-7x-minmax-f16c-c24.c &
tools/xngen src/f16-gavgpool/unipass-f16c.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -o src/f16-gavgpool/gen/f16-gavgpool-7x-minmax-f16c-c32.c &

tools/xngen src/f16-gavgpool/multipass-f16c.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -o src/f16-gavgpool/gen/f16-gavgpool-7p7x-minmax-f16c-c8.c &
tools/xngen src/f16-gavgpool/multipass-f16c.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -o src/f16-gavgpool/gen/f16-gavgpool-7p7x-minmax-f16c-c16.c &
tools/xngen src/f16-gavgpool/multipass-f16c.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -o src/f16-gavgpool/gen/f16-gavgpool-7p7x-minmax-f16c-c24.c &
tools/xngen src/f16-gavgpool/multipass-f16c.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -o src/f16-gavgpool/gen/f16-gavgpool-7p7x-minmax-f16c-c32.c &

wait
