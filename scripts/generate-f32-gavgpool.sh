#!/bin/sh
# Copyright 2024 Imagination Technologies, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################ RISC-V Vector ################################
tools/xngen src/f32-gavgpool/rvv_7x.c.in -D LMUL=1 -o src/f32-gavgpool/gen/f32-gavgpool-7x-minmax-rvv-c1v.c &
tools/xngen src/f32-gavgpool/rvv_7x.c.in -D LMUL=2 -o src/f32-gavgpool/gen/f32-gavgpool-7x-minmax-rvv-c2v.c &
tools/xngen src/f32-gavgpool/rvv_7x.c.in -D LMUL=4 -o src/f32-gavgpool/gen/f32-gavgpool-7x-minmax-rvv-c4v.c &

tools/xngen src/f32-gavgpool/rvv_7p7x.c.in -D LMUL=1 -o src/f32-gavgpool/gen/f32-gavgpool-7p7x-minmax-rvv-c1v.c &
tools/xngen src/f32-gavgpool/rvv_7p7x.c.in -D LMUL=2 -o src/f32-gavgpool/gen/f32-gavgpool-7p7x-minmax-rvv-c2v.c &
tools/xngen src/f32-gavgpool/rvv_7p7x.c.in -D LMUL=4 -o src/f32-gavgpool/gen/f32-gavgpool-7p7x-minmax-rvv-c4v.c &

wait
