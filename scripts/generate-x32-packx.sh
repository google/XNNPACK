#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


################################### ARM NEON ##################################
### NR multiple of 4
tools/xngen src/x32-packx/neon.c.in -D MR=4 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packx/gen/x32-packx-4x-neon-st4-u4.c &
tools/xngen src/x32-packx/neon.c.in -D MR=4 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packx/gen/x32-packx-4x-neon-st4-u4-prfm.c &
tools/xngen src/x32-packx/neon.c.in -D MR=8 -D PREFETCH=0 -D KBLOCK=4 -o src/x32-packx/gen/x32-packx-8x-neon-st4-u4.c &
tools/xngen src/x32-packx/neon.c.in -D MR=8 -D PREFETCH=1 -D KBLOCK=4 -o src/x32-packx/gen/x32-packx-8x-neon-st4-u4-prfm.c &
tools/xngen src/x32-packx/neon.c.in -D MR=4 -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packx/gen/x32-packx-4x-neon-st4-u8.c &
tools/xngen src/x32-packx/neon.c.in -D MR=4 -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packx/gen/x32-packx-4x-neon-st4-u8-prfm.c &
tools/xngen src/x32-packx/neon.c.in -D MR=8 -D PREFETCH=0 -D KBLOCK=8 -o src/x32-packx/gen/x32-packx-8x-neon-st4-u8.c &
tools/xngen src/x32-packx/neon.c.in -D MR=8 -D PREFETCH=1 -D KBLOCK=8 -o src/x32-packx/gen/x32-packx-8x-neon-st4-u8-prfm.c &

wait
