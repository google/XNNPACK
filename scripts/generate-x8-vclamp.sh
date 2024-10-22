#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################ RISC-V Vector ################################
tools/xngen src/s8-vclamp/rvv.c.in -D LMUL=1 -D DATATYPE=S8 -o src/s8-vclamp/gen/s8-vclamp-rvv-u1v.c &
tools/xngen src/s8-vclamp/rvv.c.in -D LMUL=2 -D DATATYPE=S8 -o src/s8-vclamp/gen/s8-vclamp-rvv-u2v.c &
tools/xngen src/s8-vclamp/rvv.c.in -D LMUL=4 -D DATATYPE=S8 -o src/s8-vclamp/gen/s8-vclamp-rvv-u4v.c &
tools/xngen src/s8-vclamp/rvv.c.in -D LMUL=8 -D DATATYPE=S8 -o src/s8-vclamp/gen/s8-vclamp-rvv-u8v.c &

tools/xngen src/s8-vclamp/rvv.c.in -D LMUL=1 -D DATATYPE=U8 -o src/u8-vclamp/gen/u8-vclamp-rvv-u1v.c &
tools/xngen src/s8-vclamp/rvv.c.in -D LMUL=2 -D DATATYPE=U8 -o src/u8-vclamp/gen/u8-vclamp-rvv-u2v.c &
tools/xngen src/s8-vclamp/rvv.c.in -D LMUL=4 -D DATATYPE=U8 -o src/u8-vclamp/gen/u8-vclamp-rvv-u4v.c &
tools/xngen src/s8-vclamp/rvv.c.in -D LMUL=8 -D DATATYPE=U8 -o src/u8-vclamp/gen/u8-vclamp-rvv-u8v.c &

wait
