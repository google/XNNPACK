#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x2-minmax-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4-minmax-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8-minmax-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x2-minmax-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4-minmax-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8-minmax-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4-minmax-scalar.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x2-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-1x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-2x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-3x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-4x2-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x2-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-1x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-2x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-3x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-4x2-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x2-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-1x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-2x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-3x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-4x2-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-1x2-minmax-rndnu-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-2x2-minmax-rndnu-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-3x2-minmax-rndnu-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-4x2-minmax-rndnu-scalar.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-1x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-2x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-3x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-4x4-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-1x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-2x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-3x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-4x4-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-1x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-2x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-3x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-4x4-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-1x4-minmax-rndnu-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-2x4-minmax-rndnu-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-3x4-minmax-rndnu-scalar.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-igemm/gen/qu8-igemm-4x4-minmax-rndnu-scalar.c &

##################################### WAsm ####################################
tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x2-minmax-wasm.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4-minmax-wasm.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8-minmax-wasm.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x2-minmax-wasm.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4-minmax-wasm.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8-minmax-wasm.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4-minmax-wasm.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x2-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-igemm/gen/qu8-igemm-1x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-igemm/gen/qu8-igemm-2x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-igemm/gen/qu8-igemm-3x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-igemm/gen/qu8-igemm-4x2-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-igemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-igemm/gen/qu8-igemm-1x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-igemm/gen/qu8-igemm-2x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-igemm/gen/qu8-igemm-3x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-igemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-igemm/gen/qu8-igemm-4x4-minmax-fp32-wasm-fmagic.c &

################################## WAsm SIMD ##################################
### C2 micro-kernels
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64    -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c2-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64    -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c2-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64    -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c2-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64    -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c2-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=4 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=4 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c2-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c2-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c2-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=4 -D VARIANT=LD128 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c2-minmax-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=4 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-wasmsimd-dot16x2.c.in   -D MR=4 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64  -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &

### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c8-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c8-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c8-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c8-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128 -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c8-minmax-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &

############################## WAsm Relaxed SIMD ##############################
### C16 micro-kernels
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c16-minmax-wasmsdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c16-minmax-wasmsdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c16-minmax-wasmsdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -D SDOT=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c16-minmax-wasmsdot.c &

tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c16-minmax-fp32-wasmsdot.c &

tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=1 -D NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=2 -D NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=3 -D NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=4 -D NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c16-minmax-fp32-wasmsdot.c &

tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c16-minmax-fp32-wasmusdot.c &

tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=1 -D NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=2 -D NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=3 -D NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c16-minmax-fp32-wasmusdot.c &
tools/xngen src/qs8-igemm/MRx4c16-wasmdot.c.in -D MR=4 -D NR=8 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -D SDOT=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c16-minmax-fp32-wasmusdot.c &

################################## ARMv6 SIMD #################################
tools/xngen src/qs8-igemm/c4-armsimd32.c.in -D MR=1 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-igemm/c4-armsimd32.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x2c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-igemm/c4-armsimd32.c.in -D MR=2 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-igemm/c4-armsimd32.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x2c4-minmax-fp32-armsimd32.c &

tools/xngen src/qs8-igemm/c4-armsimd32.c.in -D MR=1 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-1x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-igemm/c4-armsimd32.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-1x2c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-igemm/c4-armsimd32.c.in -D MR=2 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-2x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-igemm/c4-armsimd32.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-2x2c4-minmax-fp32-armsimd32.c &

################################### ARM NEON ##################################
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x16-minmax-neon-mlal-lane.c &

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x16-minmax-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x16-minmax-fp32-neon-mlal-lane.c &

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x16-minmax-fp32-neonv8-mlal-lane.c &

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x16-minmax-fp32-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x16-minmax-fp32-neonv8-mlal-lane-prfm.c &

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-1x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-2x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-3x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-6x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-1x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-2x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-3x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-6x16-minmax-rndnu-neon-mlal-lane.c &

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-1x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-4x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-1x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-4x16-minmax-fp32-neon-mlal-lane.c &

tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-igemm/gen/qu8-igemm-1x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-igemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-igemm/gen/qu8-igemm-4x16-minmax-fp32-neonv8-mlal-lane.c &

### C2 micro-kernels
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c2-minmax-fp32-neon-mlal-dup.c &
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c2-minmax-fp32-neon-mlal-dup.c &

tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c2-minmax-fp32-neonv8-mlal-dup.c &
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c2-minmax-fp32-neonv8-mlal-dup.c &

### C2 LD1R micro-kernels
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c2-minmax-fp32-neon-mlal-ld1r.c &
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c2-minmax-fp32-neon-mlal-ld1r.c &

tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c2-minmax-fp32-neonv8-mlal-ld1r.c &
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c2-minmax-fp32-neonv8-mlal-ld1r.c &

### C2 LD2R micro-kernels
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c2-minmax-fp32-neon-mlal-ld2r.c &
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c2-minmax-fp32-neon-mlal-ld2r.c &

tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c2-minmax-fp32-neonv8-mlal-ld2r.c &
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c2-minmax-fp32-neonv8-mlal-ld2r.c &

### C2 LD4R micro-kernels
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c2-minmax-fp32-neon-mlal-ld4r.c &
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c2-minmax-fp32-neon-mlal-ld4r.c &

tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD4R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c2-minmax-fp32-neonv8-mlal-ld4r.c &
tools/xngen src/qs8-igemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD4R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c2-minmax-fp32-neonv8-mlal-ld4r.c &

### C2S4 micro-kernels
tools/xngen src/qs8-igemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -D ARMV8=0 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c2s4-minmax-neonfp16arith-mlal.c &
tools/xngen src/qs8-igemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -D ARMV8=0 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x8c2s4-minmax-neonfp16arith-mlal.c &
tools/xngen src/qs8-igemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c2s4-minmax-neon-mlal.c &
tools/xngen src/qs8-igemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -D ARMV8=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8c2s4-minmax-neon-mlal.c &

tools/xngen src/qs8-igemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c2s4-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-igemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c2s4-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-igemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c2s4-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-igemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c2s4-minmax-fp32-neonv8-mlal.c &

### C4 micro-kernels
tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c4-minmax-fp32-neon-mlal-dup.c &
tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c4-minmax-fp32-neon-mlal-dup.c &

tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c4-minmax-fp32-neonv8-mlal-dup.c &
tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c4-minmax-fp32-neonv8-mlal-dup.c &

### C4 LD1R micro-kernels
tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c4-minmax-fp32-neon-mlal-ld1r.c &
tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c4-minmax-fp32-neon-mlal-ld1r.c &

tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c4-minmax-fp32-neonv8-mlal-ld1r.c &
tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c4-minmax-fp32-neonv8-mlal-ld1r.c &

### C4 LD2R micro-kernels
tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c4-minmax-fp32-neon-mlal-ld2r.c &
tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c4-minmax-fp32-neon-mlal-ld2r.c &

tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c4-minmax-fp32-neonv8-mlal-ld2r.c &
tools/xngen src/qs8-igemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c4-minmax-fp32-neonv8-mlal-ld2r.c &

### C4S2 micro-kernels
tools/xngen src/qs8-igemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c4s2-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-igemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c4s2-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-igemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c4s2-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-igemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c4s2-minmax-fp32-neonv8-mlal.c &

### C8 micro-kernels
tools/xngen src/qs8-igemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-igemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-igemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-igemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-neonv8-mlal.c &

### C16 micro-kernels
### C4 micro-kernels
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c4-minmax-fp32-neondot.c &

tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=2  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-6x8c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x8c4-minmax-neondotfp16arith.c &

tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=2  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-6x16c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x16c4-minmax-neondotfp16arith.c &

tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=32 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x32c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=2  -D NR=32 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x32c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=32 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x32c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=32 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-6x32c4-minmax-neondotfp16arith.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=32 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x32c4-minmax-neondotfp16arith.c &

tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=2  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x8c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x8c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=2  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x16c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x16c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=1  -D NR=32 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x32c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=2  -D NR=32 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x32c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=4  -D NR=32 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x32c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=6  -D NR=32 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x32c4-minmax-neondot.c &
tools/xngen src/qs8-igemm/c4-neondot.c.in -D MR=8  -D NR=32 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x32c4-minmax-neondot.c &

### C8 micro-kernels
tools/xngen src/qs8-igemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-neondot-ld64.c &
tools/xngen src/qs8-igemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=0 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-neondot-ld64.c &
tools/xngen src/qs8-igemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-aarch64-neondot-ld128.c &
tools/xngen src/qs8-igemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=1 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-aarch64-neondot-ld128.c &

tools/xngen src/qs8-igemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-neondot-ld64.c &
tools/xngen src/qs8-igemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-fp32-neondot-ld64.c &
tools/xngen src/qs8-igemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-aarch64-neondot-ld128.c &
tools/xngen src/qs8-igemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-fp32-aarch64-neondot-ld128.c &

############################### AArch32 assembly ##############################
### Cortex-A53 lane micro-kernels
tools/xngen src/qs8-igemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-igemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &
tools/xngen src/qs8-igemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35.S &
tools/xngen src/qs8-igemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35-prfm.S &

tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-ld64-prfm.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-ld64.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-ld64-prfm.S &

tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35-prfm.S &

### QU8 micro-kernels
tools/xngen src/qs8-igemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-igemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &

tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64-prfm.S &

tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-igemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-igemm/gen/qu8-igemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &

### C4 micro-kernels
tools/xngen src/qs8-igemm/4x8c4-aarch32-neondot-ld64.S.in                           -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c4-minmax-fp32-asm-aarch32-neondot-ld64.S &
tools/xngen src/qs8-igemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x8c4-minmax-asm-aarch32-neondotfp16arith-cortex-a55.S &
tools/xngen src/qs8-igemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8c4-minmax-asm-aarch32-neondot-cortex-a55.S &
tools/xngen src/qs8-igemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c4-minmax-fp32-asm-aarch32-neondot-cortex-a55.S &

############################### AArch64 assembly ##############################
### Cortex-A53 lane micro-kernels
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64-prfm.S &

### QU8 micro-kernels
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64-prfm.S &

tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a75.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75.S &
tools/xngen src/qs8-igemm/4x16-aarch64-neon-mlal-lane-cortex-a75.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-igemm/gen/qu8-igemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75-prfm.S &

### C4 micro-kernels
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-cortex-a55.S &
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-cortex-a55.S &
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S &
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S &

tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c4-minmax-fp32-asm-aarch64-neondot-cortex-a55.S &
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S &
tools/xngen src/qs8-igemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld128.S &

### C8 / C16 micro-kernels
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S &
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S &
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal.S &
tools/xngen src/qs8-igemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal.S &
tools/xngen src/qs8-igemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-igemm/2x8c16-aarch64-neon-mlal.S.in                         -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c16-minmax-fp32-asm-aarch64-neon-mlal.S &

################################### x86 SSE ###################################
### C2 micro-kernels
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-1x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-2x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-3x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-4x4c2-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-1x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-2x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-3x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-4x4c2-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-1x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-2x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-3x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-4x4c2-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-1x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-2x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-3x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-4x4c2-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-1x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-2x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-3x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-4x4c2-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-1x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-2x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-3x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-4x4c2-minmax-fp32-avx-ld128.c &

### c2s4 micro-kernels
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2s4-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-1x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-2x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-3x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-4x4c2s4-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2s4-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-1x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-2x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-3x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-4x4c2s4-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2s4-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-1x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-2x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-3x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-4x4c2s4-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2s4-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-1x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-2x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-3x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-4x4c2s4-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2s4-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-1x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-2x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-3x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-4x4c2s4-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x4c2s4-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-1x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-2x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-3x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-4x4c2s4-minmax-fp32-avx-ld128.c &

### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c8-minmax-sse2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c8-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-1x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-2x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-3x4c8-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c8-minmax-sse41-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c8-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-1x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-2x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-3x4c8-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD64  -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c8-minmax-avx-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c8-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-1x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-2x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD64  -o src/qu8-igemm/gen/qu8-igemm-3x4c8-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c8-minmax-sse2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c8-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-1x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-2x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-3x4c8-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c8-minmax-sse41-ld128.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c8-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-1x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-2x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-3x4c8-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D DATATYPE=QD8 -D REQUANTIZATION=     -D VARIANT=LD128 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x4c8-minmax-avx-ld128.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x4c8-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-1x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-2x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-igemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32 -D VARIANT=LD128 -o src/qu8-igemm/gen/qu8-igemm-3x4c8-minmax-fp32-avx-ld128.c &

### NEON I8MM C8 micro-kernels
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x16c8-minmax-neoni8mm.c &

tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c8-minmax-neoni8mm.c &

tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-igemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c8-minmax-fp32-neoni8mm.c &

################################### x86 AVX2 ##################################
### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x8c8-minmax-avx2.c &

tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8c8-minmax-avx2.c &

tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c8-minmax-fp32-avx2.c &

tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-1x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-2x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-3x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-4x8c8-minmax-fp32-avx2.c &

################################## x86 AVX256SKX #################################
### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-5x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-7x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x8c8-minmax-avx256skx.c &

tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x8c8-minmax-avx256skx.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x8c8-minmax-avx256skx.c &

tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-avx256skx.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-avx256skx.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c8-minmax-fp32-avx256skx.c &
tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c8-minmax-fp32-avx256skx.c &

tools/xngen src/qs8-igemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QU8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-1x8c8-minmax-fp32-avx256skx.c &

################################## x86 AVX512 #################################
### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c8-minmax-avx512skx.c &

tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c8-minmax-fp32-avx512skx.c &

tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-1x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-5x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-7x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-8x16c8-minmax-fp32-avx512skx.c &

tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c8-minmax-avx512skx-prfm.c &

tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c8-minmax-fp32-avx512skx-prfm.c &

tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-1x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-5x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-7x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-igemm/gen/qu8-igemm-8x16c8-minmax-fp32-avx512skx-prfm.c &

################################## x86 AVX512 VNNI #################################
### C4 micro-kernels
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-9x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-10x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-12x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-14x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=28 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-28x16c4-minmax-avx512vnni.c &

tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=1   -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=5   -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=7   -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=8   -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=9   -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-9x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=10  -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-10x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=12  -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-12x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=14  -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-14x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=28  -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-28x16c4-minmax-avx512vnni.c &

tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=1  -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=5  -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=7  -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=8  -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=9  -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-9x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=10 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-10x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=12 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-12x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=14 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-14x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=28 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-28x16c4-minmax-avx512vnni-prfm.c &

tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=1   -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=5   -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=7   -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=8   -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=9   -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-9x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=10  -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-10x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=12  -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-12x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=14  -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-14x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c4-avx512vnni.c.in -D MR=28  -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-28x16c4-minmax-avx512vnni-prfm.c &

### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-9x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-10x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-12x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-14x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-9x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-10x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-12x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-14x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-9x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-10x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-12x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-14x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=1  -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=5  -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=7  -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=8  -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=9  -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-9x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=10 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-10x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=12 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-12x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx16c8-avx512vnni.c.in -D MR=14 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-14x16c8-minmax-avx512vnni-prfm.c &
################################## x86 AVX512VL VNNI #################################
### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-9x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-10x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-12x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-14x8c8-minmax-fp32-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-9x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-10x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-12x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-14x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-5x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-7x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-9x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-10x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-12x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-14x8c8-minmax-avx256vnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-9x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-10x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-12x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-14x8c8-minmax-fp32-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-9x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-10x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-12x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-14x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1  -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5  -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-5x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7  -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-7x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8  -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=9  -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-9x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=10 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-10x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=12 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-12x8c8-minmax-avx256vnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=14 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-14x8c8-minmax-avx256vnni-prfm.c &
################################## x86 AVXVNNI #################################
### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x8c8-minmax-fp32-avxvnni.c &

tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x8c8-minmax-avxvnni.c &

tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-3x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-5x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-6x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-7x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x8c8-minmax-avxvnni.c &

tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-6x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-8x8c8-minmax-fp32-avxvnni-prfm.c &

tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-5x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-6x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-8x8c8-minmax-avxvnni-prfm.c &

tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-3x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-5x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-6x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-7x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-igemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-8x8c8-minmax-avxvnni-prfm.c &

################################## x86 AVX512 AMX #################################
### C4 micro-kernels
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=16 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=16 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x16c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=16 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x16c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=16 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x16c4-minmax-fp32-avx512amx-prfm.c &

tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=32 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x32c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=32 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x32c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=32 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x32c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=32 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x32c4-minmax-fp32-avx512amx-prfm.c &

tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=64 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x64c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=64 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x64c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=64 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x64c4-minmax-fp32-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=64 -D DATATYPE=QC8 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x64c4-minmax-fp32-avx512amx-prfm.c &

tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=16 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=16 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x16c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=16 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x16c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=16 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x16c4-minmax-avx512amx-prfm.c &

tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=32 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x32c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=32 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x32c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=32 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x32c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=32 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x32c4-minmax-avx512amx-prfm.c &

tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=64 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=64 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=64 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=64 -D DATATYPE=QD8_F32 -D REQUANTIZATION= -o src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x64c4-minmax-avx512amx-prfm.c &

tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=1  -D NR=64 -D DATATYPE=QD8_F16 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=7  -D NR=64 -D DATATYPE=QD8_F16 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-7x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=0 -D MR=16 -D NR=64 -D DATATYPE=QD8_F16 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-16x64c4-minmax-avx512amx.c &
tools/xngen src/qs8-igemm/c4-avx512amx.c.in -D GFNI=0 -D PREFETCH=1 -D MR=16 -D NR=64 -D DATATYPE=QD8_F16 -D REQUANTIZATION= -o src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-16x64c4-minmax-avx512amx-prfm.c &

wait
