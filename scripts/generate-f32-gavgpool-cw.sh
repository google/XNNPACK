#!/bin/sh
# Copyright 2024 Imagination Technologies, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################ RISC-V Vector ################################
tools/xngen src/f32-gavgpool-cw/rvv.c.in -D LMUL=1 -o src/f32-gavgpool-cw/gen/f32-gavgpool-cw-rvv-u1v.c &
tools/xngen src/f32-gavgpool-cw/rvv.c.in -D LMUL=2 -o src/f32-gavgpool-cw/gen/f32-gavgpool-cw-rvv-u2v.c &
tools/xngen src/f32-gavgpool-cw/rvv.c.in -D LMUL=4 -o src/f32-gavgpool-cw/gen/f32-gavgpool-cw-rvv-u4v.c &
tools/xngen src/f32-gavgpool-cw/rvv.c.in -D LMUL=8 -o src/f32-gavgpool-cw/gen/f32-gavgpool-cw-rvv-u8v.c

wait
