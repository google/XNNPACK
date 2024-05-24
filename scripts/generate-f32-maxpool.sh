#!/bin/sh
# Copyright 2024 Imagination Technologies, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################ RISC-V Vector ################################
tools/xngen src/f32-maxpool/rvv.c.in -D LMUL=1 -o src/f32-maxpool/gen/f32-maxpool-9p8x-minmax-rvv-c1v.c &
tools/xngen src/f32-maxpool/rvv.c.in -D LMUL=2 -o src/f32-maxpool/gen/f32-maxpool-9p8x-minmax-rvv-c2v.c &

wait
