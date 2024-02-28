#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4-minmax-scalar.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=1 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x1-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4-minmax-scalar.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x2-minmax-fp32-scalar-fmagic.c &

# tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-1x2-minmax-fp32-scalar-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-2x2-minmax-fp32-scalar-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-3x2-minmax-fp32-scalar-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-4x2-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x2-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x2-minmax-fp32-scalar-imagic.c &

# tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-1x2-minmax-fp32-scalar-imagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-2x2-minmax-fp32-scalar-imagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-3x2-minmax-fp32-scalar-imagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-4x2-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x2-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x2-minmax-fp32-scalar-lrintf.c &

# tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-1x2-minmax-fp32-scalar-lrintf.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-2x2-minmax-fp32-scalar-lrintf.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-3x2-minmax-fp32-scalar-lrintf.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-4x2-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x2-minmax-fp32-scalar-lrintf.c &

# tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-1x2-minmax-rndnu-scalar.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-2x2-minmax-rndnu-scalar.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-3x2-minmax-rndnu-scalar.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-4x2-minmax-rndnu-scalar.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x2-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x2-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x2-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x2-minmax-rndnu-scalar.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4-minmax-fp32-scalar-fmagic.c &

# tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-1x4-minmax-fp32-scalar-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-2x4-minmax-fp32-scalar-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-3x4-minmax-fp32-scalar-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-4x4-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x4-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4-minmax-fp32-scalar-imagic.c &

# tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-1x4-minmax-fp32-scalar-imagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-2x4-minmax-fp32-scalar-imagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-3x4-minmax-fp32-scalar-imagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-4x4-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x4-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4-minmax-fp32-scalar-lrintf.c &

# tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-1x4-minmax-fp32-scalar-lrintf.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-2x4-minmax-fp32-scalar-lrintf.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-3x4-minmax-fp32-scalar-lrintf.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-4x4-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x4-minmax-fp32-scalar-lrintf.c &

# tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-1x4-minmax-rndnu-scalar.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-2x4-minmax-rndnu-scalar.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-3x4-minmax-rndnu-scalar.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/qs8-gemm-4x4-minmax-rndnu-scalar.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-1x4-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-2x4-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-3x4-minmax-rndnu-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/qu8-gemm-4x4-minmax-rndnu-scalar.c &

##################################### WAsm ####################################
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x2-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x2-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QD8 -D WASM=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4-minmax-wasm.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=1 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x2-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=1 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=1 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=1 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x2-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=1 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=1 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8-minmax-wasm.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QC4 -D WASM=1 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4-minmax-wasm.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x2-minmax-fp32-wasm-fmagic.c &

# tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/qs8-gemm-1x2-minmax-fp32-wasm-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/qs8-gemm-2x2-minmax-fp32-wasm-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/qs8-gemm-3x2-minmax-fp32-wasm-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/qs8-gemm-4x2-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/qu8-gemm-1x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/qu8-gemm-2x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/qu8-gemm-3x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/qu8-gemm-4x2-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4-minmax-fp32-wasm-fmagic.c &

# tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/qs8-gemm-1x4-minmax-fp32-wasm-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/qs8-gemm-2x4-minmax-fp32-wasm-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/qs8-gemm-3x4-minmax-fp32-wasm-fmagic.c &
# tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/qs8-gemm-4x4-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/qu8-gemm-1x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/qu8-gemm-2x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/qu8-gemm-3x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/qu8-gemm-4x4-minmax-fp32-wasm-fmagic.c &

################################## ARMv6 SIMD #################################
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=1 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x2c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=2 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x2c4-minmax-fp32-armsimd32.c &

# tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=1 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x1c4-minmax-fp32-armsimd32.c &
# tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x2c4-minmax-fp32-armsimd32.c &
# tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=2 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x1c4-minmax-fp32-armsimd32.c &
# tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x2c4-minmax-fp32-armsimd32.c &

tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=1 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x2c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=2 -D NR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x1c4-minmax-fp32-armsimd32.c &
tools/xngen src/qs8-gemm/c4-armsimd32.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x2c4-minmax-fp32-armsimd32.c &

################################## WAsm SIMD ##################################
### C2 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c2-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c2-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c2-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c2-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c2-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c2-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c2-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c2-minmax-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x4c2-xw-minmax-fp32-wasmsimd-dot16x2.c &
# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x4c2-xw-minmax-fp32-wasmsimd-dot16x2.c &
# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x4c2-xw-minmax-fp32-wasmsimd-dot16x2.c &
# tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x4c2-xw-minmax-fp32-wasmsimd-dot16x2.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64   -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64   -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64   -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64   -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c2s4-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128  -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c2s4-minmax-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=         -D DATATYPE=QC4 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=         -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x4c8-xw-minmax-fp32-wasmsimd-dot16x2.c &
# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x4c8-xw-minmax-fp32-wasmsimd-dot16x2.c &
# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x4c8-xw-minmax-fp32-wasmsimd-dot16x2.c &
# tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x4c8-xw-minmax-fp32-wasmsimd-dot16x2.c &

############################## WAsm Relaxed SIMD ##############################
### C16 micro-kernels
tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=3 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c16-minmax-fp32-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c16-minmax-fp32-wasmsdot.c &

tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=1 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c16-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=2 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c16-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=3 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c16-minmax-wasmsdot.c &
tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=4 -D REQUANTIZATION=     -D DATATYPE=QD8 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c16-minmax-wasmsdot.c &

# tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x4c16-minmax-fp32-wasmsdot.c &
# tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=2 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x4c16-minmax-fp32-wasmsdot.c &
# tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=3 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x4c16-minmax-fp32-wasmsdot.c &
# tools/xngen src/qs8-gemm/MRx4c16-wasmsdot.c.in -D MR=4 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x4c16-minmax-fp32-wasmsdot.c &

################################### ARM NEON ##################################
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16-minmax-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16-minmax-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x16-minmax-neonfp16arith-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x16-minmax-neonfp16arith-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x16-minmax-neonfp16arith-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F16 -D ARMV8=0 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x16-minmax-neonfp16arith-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16-minmax-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16-minmax-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=         -D DATATYPE=QC4_F32 -D ARMV8=0 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16-minmax-neon-mlal-lane-prfm.c &

# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8-minmax-rndnu-neon-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8-minmax-rndnu-neon-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x8-minmax-rndnu-neon-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-neon-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-6x8-minmax-rndnu-neon-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x16-minmax-rndnu-neon-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x16-minmax-rndnu-neon-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x16-minmax-rndnu-neon-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-rndnu-neon-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-6x16-minmax-rndnu-neon-mlal-lane.c &

# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8-minmax-rndnu-neon-mlal-lane-prfm.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8-minmax-rndnu-neon-mlal-lane-prfm.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x8-minmax-rndnu-neon-mlal-lane-prfm.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-neon-mlal-lane-prfm.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-6x8-minmax-rndnu-neon-mlal-lane-prfm.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x16-minmax-rndnu-neon-mlal-lane-prfm.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x16-minmax-rndnu-neon-mlal-lane-prfm.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x16-minmax-rndnu-neon-mlal-lane-prfm.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-rndnu-neon-mlal-lane-prfm.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-6x16-minmax-rndnu-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16-minmax-fp32-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16-minmax-fp32-neonv8-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16-minmax-fp32-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16-minmax-fp32-neonv8-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-2x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-3x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-6x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-2x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-3x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-6x16-minmax-rndnu-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=      -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16-minmax-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=      -D DATATYPE=QD8 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16-minmax-neon-mlal-lane.c &

# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x16-minmax-fp32-neon-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-fp32-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-fp32-neon-mlal-lane.c &

# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gemm/gen/qs8-gemm-1x16-minmax-fp32-neonv8-mlal-lane.c &
# tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-fp32-neonv8-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gemm/gen/qu8-gemm-1x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-fp32-neonv8-mlal-lane.c &

# tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8-minmax-rndnu-neon-mull-addw-dup.c &
# tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8-minmax-rndnu-neon-mull-addw-dup.c &
# tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x8-minmax-rndnu-neon-mull-addw-dup.c &
# tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-neon-mull-addw-dup.c &
# tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x16-minmax-rndnu-neon-mull-addw-dup.c &
# tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x16-minmax-rndnu-neon-mull-addw-dup.c &
# tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x16-minmax-rndnu-neon-mull-addw-dup.c &
# tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-rndnu-neon-mull-addw-dup.c &

### C2 micro-kernels
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-3x8c2-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-4x8c2-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x16c2-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x16c2-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-3x16c2-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-4x16c2-minmax-rndnu-neon-mull-dup.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-3x8c2-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-4x8c2-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x16c2-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x16c2-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-3x16c2-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-4x16c2-minmax-rndnu-neon-mlal-dup.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-fp32-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-fp32-neon-mlal-dup.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neon-mlal-dup.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-fp32-neonv8-mlal-dup.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-fp32-neonv8-mlal-dup.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neonv8-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neonv8-mlal-dup.c &

### C2 LD1R micro-kernels
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-3x8c2-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-4x8c2-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x16c2-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x16c2-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-3x16c2-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-4x16c2-minmax-rndnu-neon-mull-ld1r.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-3x8c2-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-4x8c2-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x16c2-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x16c2-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-3x16c2-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-4x16c2-minmax-rndnu-neon-mlal-ld1r.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-fp32-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-fp32-neon-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neon-mlal-ld1r.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-fp32-neonv8-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-fp32-neonv8-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neonv8-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neonv8-mlal-ld1r.c &

### C2 LD2R micro-kernels
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-3x8c2-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-4x8c2-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x16c2-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x16c2-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-3x16c2-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-4x16c2-minmax-rndnu-neon-mull-ld2r.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-3x8c2-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-4x8c2-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x16c2-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x16c2-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-3x16c2-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-4x16c2-minmax-rndnu-neon-mlal-ld2r.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-fp32-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-fp32-neon-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neon-mlal-ld2r.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-fp32-neonv8-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-fp32-neonv8-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neonv8-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neonv8-mlal-ld2r.c &

### C2 LD4R micro-kernels
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-rndnu-neon-mull-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-rndnu-neon-mull-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-3x8c2-minmax-rndnu-neon-mull-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-4x8c2-minmax-rndnu-neon-mull-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-1x16c2-minmax-rndnu-neon-mull-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-2x16c2-minmax-rndnu-neon-mull-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-3x16c2-minmax-rndnu-neon-mull-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-4x16c2-minmax-rndnu-neon-mull-ld4r.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-rndnu-neon-mlal-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-rndnu-neon-mlal-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-3x8c2-minmax-rndnu-neon-mlal-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-4x8c2-minmax-rndnu-neon-mlal-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-1x16c2-minmax-rndnu-neon-mlal-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-2x16c2-minmax-rndnu-neon-mlal-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-3x16c2-minmax-rndnu-neon-mlal-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-4x16c2-minmax-rndnu-neon-mlal-ld4r.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-fp32-neon-mlal-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-fp32-neon-mlal-ld4r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD4R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neon-mlal-ld4r.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-1x8c2-minmax-fp32-neonv8-mlal-ld4r.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=LD4R -o src/qs8-gemm/gen/qs8-gemm-2x8c2-minmax-fp32-neonv8-mlal-ld4r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD4R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2-minmax-fp32-neonv8-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD4R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2-minmax-fp32-neonv8-mlal-ld4r.c &

### C2S4 micro-kernels
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c2s4-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8c2s4-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x8c2s4-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8c2s4-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x16c2s4-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x16c2s4-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x16c2s4-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x16c2s4-minmax-rndnu-neon-mull.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c2s4-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8c2s4-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x8c2s4-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8c2s4-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x16c2s4-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x16c2s4-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x16c2s4-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x16c2s4-minmax-rndnu-neon-mlal.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -D ARMV8=0 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c2s4-minmax-neonfp16arith.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -D ARMV8=0 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c2s4-minmax-neonfp16arith.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c2s4-minmax-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -D ARMV8=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c2s4-minmax-neon-mlal.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c2s4-minmax-fp32-neon-mlal.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8c2s4-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2s4-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2s4-minmax-fp32-neon-mlal.c &

# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gemm/gen/qs8-gemm-1x8c2s4-minmax-fp32-neonv8-mlal.c &
# tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gemm/gen/qs8-gemm-2x8c2s4-minmax-fp32-neonv8-mlal.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c2s4-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c2s4-minmax-fp32-neonv8-mlal.c &

### C4 micro-kernels
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-3x8c4-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-4x8c4-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x16c4-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-3x16c4-minmax-rndnu-neon-mull-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-neon-mull-dup.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-3x8c4-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-4x8c4-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x16c4-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-3x16c4-minmax-rndnu-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-neon-mlal-dup.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-fp32-neon-mlal-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-fp32-neon-mlal-dup.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neon-mlal-dup.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-fp32-neonv8-mlal-dup.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-fp32-neonv8-mlal-dup.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neonv8-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=DUP  -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neonv8-mlal-dup.c &

### C4 LD1R micro-kernels
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-3x8c4-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-4x8c4-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x16c4-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-3x16c4-minmax-rndnu-neon-mull-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-neon-mull-ld1r.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-3x8c4-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-4x8c4-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x16c4-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-3x16c4-minmax-rndnu-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-neon-mlal-ld1r.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-fp32-neon-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-fp32-neon-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neon-mlal-ld1r.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-fp32-neonv8-mlal-ld1r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-fp32-neonv8-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neonv8-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD1R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neonv8-mlal-ld1r.c &

### C4 LD2R micro-kernels
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-3x8c4-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-4x8c4-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x16c4-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-3x16c4-minmax-rndnu-neon-mull-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-neon-mull-ld2r.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-3x8c4-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-4x8c4-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x16c4-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-3x16c4-minmax-rndnu-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-neon-mlal-ld2r.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-fp32-neon-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-fp32-neon-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neon-mlal-ld2r.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-fp32-neonv8-mlal-ld2r.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-gemm/gen/qs8-gemm-2x8c4-minmax-fp32-neonv8-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neonv8-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -D DUP=LD2R -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4-minmax-fp32-neonv8-mlal-ld2r.c &

### C4S2 micro-kernels
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c4s2-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8c4s2-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x8c4s2-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8c4s2-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x16c4s2-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x16c4s2-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x16c4s2-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x16c4s2-minmax-rndnu-neon-mull.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c4s2-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8c4s2-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x8c4s2-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8c4s2-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x16c4s2-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x16c4s2-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x16c4s2-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x16c4s2-minmax-rndnu-neon-mlal.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c4s2-minmax-fp32-neon-mlal.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8c4s2-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4s2-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4s2-minmax-fp32-neon-mlal.c &

# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gemm/gen/qs8-gemm-1x8c4s2-minmax-fp32-neonv8-mlal.c &
# tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gemm/gen/qs8-gemm-2x8c4s2-minmax-fp32-neonv8-mlal.c &

tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4s2-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c4s2-minmax-fp32-neonv8-mlal.c &

### C8 micro-kernels
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x8c8-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8c8-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x16c8-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x16c8-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x16c8-minmax-rndnu-neon-mull.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x16c8-minmax-rndnu-neon-mull.c &

# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x8c8-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8c8-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x16c8-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x16c8-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-3x16c8-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x16c8-minmax-rndnu-neon-mlal.c &

# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-fp32-neon-mlal.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-neon-mlal.c &

# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-fp32-neonv8-mlal.c &
# tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-fp32-neonv8-mlal.c &

tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-neonv8-mlal.c &

### C16 micro-kernels
# tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/qs8-gemm-1x8c16-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/qs8-gemm-2x8c16-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/qs8-gemm-3x8c16-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/qs8-gemm-4x8c16-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/qs8-gemm-1x16c16-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/qs8-gemm-2x16c16-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/qs8-gemm-3x16c16-minmax-rndnu-neon-mlal.c &
# tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/qs8-gemm-4x16c16-minmax-rndnu-neon-mlal.c &

### C4 micro-kernels
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c4-minmax-neondotfp16arith.c

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x16c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x16c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x16c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x16c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x16c4-minmax-neondotfp16arith.c

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c4-minmax-neondot.c

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c4-minmax-neondot.c

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c4-minmax-neondot.c

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c4-minmax-neondotfp16arith.c

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x16c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x16c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x16c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x16c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x16c4-minmax-neondotfp16arith.c
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x16c4-minmax-neondotfp16arith.c

# tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-fp32-neondot.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c4-minmax-fp32-neondot.c &

# tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c4-minmax-rndnu-neondot.c &
# tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x8c4-minmax-rndnu-neondot.c &
# tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-6x8c4-minmax-rndnu-neondot.c &
# tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-8x8c4-minmax-rndnu-neondot.c &
# tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-rndnu-neondot.c &
# tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-neondot.c &
# tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-6x16c4-minmax-rndnu-neondot.c &
# tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-8x16c4-minmax-rndnu-neondot.c &

tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=2  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=3  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=5  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-5x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-6x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-8x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=2  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=3  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=5  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-5x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-6x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-8x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=1  -D NR=32 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x32c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=2  -D NR=32 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x32c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=3  -D NR=32 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x32c4-minmax-rndnu-neondot.c &

tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x16c4-minmax-fp32-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=2  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x16c4-minmax-fp32-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16c4-minmax-fp32-neondot.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-neondot-ld64.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=0 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-neondot-ld64.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-neondot-ld64.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-neondot-ld64.c &
# tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D LD128=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-fp32-neondot-ld64.c &
# tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D LD128=0 -o src/qs8-gemm/gen/qs8-gemm-1x16c8-minmax-fp32-neondot-ld64.c &
# tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D LD128=0 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-rndnu-neondot-ld64.c &
# tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D LD128=0 -o src/qs8-gemm/gen/qs8-gemm-1x16c8-minmax-rndnu-neondot-ld64.c &

tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-aarch64-neondot-ld128.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=      -D DATATYPE=QD8 -D LD128=1 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-aarch64-neondot-ld128.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-aarch64-neondot-ld128.c &
tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D LD128=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-aarch64-neondot-ld128.c &
# tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D LD128=1 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-fp32-aarch64-neondot-ld128.c &
# tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D LD128=1 -o src/qs8-gemm/gen/qs8-gemm-1x16c8-minmax-fp32-aarch64-neondot-ld128.c &
# tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D LD128=1 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-rndnu-aarch64-neondot-ld128.c &
# tools/xngen src/qs8-gemm/c8-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D LD128=1 -o src/qs8-gemm/gen/qs8-gemm-1x16c8-minmax-rndnu-aarch64-neondot-ld128.c &

############################### AArch32 assembly ##############################
### Cortex-A53 lane micro-kernels
# tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64.S &
# tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64-prfm.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-ld64-prfm.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-ld64-prfm.S &

# tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53.S &
# tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a53-prfm.S &

# tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S &
# tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35-prfm.S &

# tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S &
# tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/qs8-gemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &
tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &
tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35.S &
tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8-minmax-fp32-asm-aarch32-neonv8-mlal-lane-cortex-a35-prfm.S &

### QU8 micro-kernels
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-ld64-prfm.S &

tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-4x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &

tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/1x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/qu8-gemm-1x8-minmax-rndnu-asm-aarch32-neon-mlal-lane-cortex-a7-prfm.S &

### C4 micro-kernels
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=      -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c4-minmax-asm-aarch32-neondotfp16arith-cortex-a55.S &
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=      -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c4-minmax-asm-aarch32-neondot-cortex-a55.S &
# tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-ld64.S.in                           -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x8c4-minmax-rndnu-asm-aarch32-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-ld64.S.in                           -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c4-minmax-fp32-asm-aarch32-neondot-ld64.S &
# tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x8c4-minmax-rndnu-asm-aarch32-neondot-cortex-a55.S &
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c4-minmax-fp32-asm-aarch32-neondot-cortex-a55.S &

############################### AArch64 assembly ##############################
### Cortex-A53 lane micro-kernels
# tools/xngen src/qs8-gemm/4x8-aarch64-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64.S &
# tools/xngen src/qs8-gemm/4x8-aarch64-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x8-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64-prfm.S &

# tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53.S &
# tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S &

# tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53.S &
# tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S &

# tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64.S &
# tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64-prfm.S &

# tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64.S &
# tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64-prfm.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16-minmax-fp32-asm-aarch64-neon-mlal-lane-ld64-prfm.S &

### QU8 micro-kernels
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a53-prfm.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-ld64-prfm.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a75.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a75.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16-minmax-rndnu-asm-aarch64-neon-mlal-lane-cortex-a75-prfm.S &

### C4 micro-kernels
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=         -D DATATYPE=QD8_F16     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondotfp16arith-cortex-a55.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=         -D DATATYPE=QD8_F32     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-cortex-a55.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=         -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=         -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld128.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=         -D DATATYPE=QD8     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-asm-aarch64-neondot-ld64.S &

# tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-rndnu-asm-aarch64-neondot-ld32.S &
# tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-rndnu-asm-aarch64-neondot-ld64.S &
# tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-asm-aarch64-neondot-cortex-a55.S &
# tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-asm-aarch64-neondot-ld32.S &
# tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-asm-aarch64-neondot-ld64.S &
# tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-rndnu-asm-aarch64-neondot-ld128.S &

# tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-fp32-asm-aarch64-neondot-ld32.S &
# tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S &
# tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-cortex-a55.S &
# tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld32.S &
# tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S &
# tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld128.S &

tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-asm-aarch64-neondot-ld32.S &
tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-cortex-a55.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld32.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld128.S &

### C4 QU8 micro-kernels
tools/xngen src/qu8-gemm/4x8c4-aarch64-neondot-cortex-a55.S.in  -D REQUANTIZATION=RNDNU -o src/qu8-gemm/gen/qu8-gemm-4x8c4-minmax-rndnu-asm-aarch64-neondot-cortex-a55.S &
tools/xngen src/qu8-gemm/4x8c4-aarch64-neondot-ld128.S.in       -D REQUANTIZATION=RNDNU -o src/qu8-gemm/gen/qu8-gemm-4x8c4-minmax-rndnu-asm-aarch64-neondot-ld128.S &
tools/xngen src/qu8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=RNDNU -o src/qu8-gemm/gen/qu8-gemm-4x16c4-minmax-rndnu-asm-aarch64-neondot-cortex-a55.S &
tools/xngen src/qu8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=RNDNU -o src/qu8-gemm/gen/qu8-gemm-4x16c4-minmax-rndnu-asm-aarch64-neondot-ld128.S &

tools/xngen src/qu8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-cortex-a55.S &
tools/xngen src/qu8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-4x16c4-minmax-fp32-asm-aarch64-neondot-ld128.S &

### C8 / C16 micro-kernels
# tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-rndnu-asm-aarch64-neon-mlal-cortex-a53.S &
# tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-rndnu-asm-aarch64-neon-mlal-cortex-a53-prfm.S &
# tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-rndnu-asm-aarch64-neon-mlal-cortex-a53.S &
# tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-rndnu-asm-aarch64-neon-mlal-cortex-a53-prfm.S &
# tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-rndnu-asm-aarch64-neon-mlal.S &
# tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-rndnu-asm-aarch64-neon-mlal-prfm.S &
# tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-rndnu-asm-aarch64-neon-mlal.S &
# tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-rndnu-asm-aarch64-neon-mlal-prfm.S &
# tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mull.S.in                          -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-rndnu-asm-aarch64-neon-mull.S &
# tools/xngen src/qs8-gemm/2x8c16-aarch64-neon-mlal.S.in                         -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c16-minmax-rndnu-asm-aarch64-neon-mlal.S &

# tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S &
# tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S &
# tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S &
# tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S &
# tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal.S &
# tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S &
# tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal.S &
# tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S &
# tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mull.S.in                          -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mull.S &
# tools/xngen src/qs8-gemm/2x8c16-aarch64-neon-mlal.S.in                         -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c16-minmax-fp32-asm-aarch64-neon-mlal.S &

tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-cortex-a53-prfm.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mull.S.in                          -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-asm-aarch64-neon-mull.S &
tools/xngen src/qs8-gemm/2x8c16-aarch64-neon-mlal.S.in                         -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c16-minmax-fp32-asm-aarch64-neon-mlal.S &

### NEON I8MM C8 micro-kernels
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F16 -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x32c8-minmax-neoni8mm.c &

tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F16 -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x32c8-minmax-neoni8mm.c &

tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QC4_F32 -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x32c8-minmax-neoni8mm.c &

tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x32c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x32-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x32-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=5 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x32-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x32-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=7 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x32-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=32 -D REQUANTIZATION= -D DATATYPE=QD8_F32 -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x32-minmax-neoni8mm.c &

tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QC8 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-neoni8mm.c &

# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x16c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x16c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x8c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x16c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x8c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-6x8c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-6x16c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-8x8c8-minmax-fp32-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-8x16c8-minmax-fp32-neoni8mm.c &

# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x8c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-1x16c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x8c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-2x16c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x8c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-3x16c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x8c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-4x16c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-6x8c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-6x16c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-8x8c8-minmax-rndnu-neoni8mm.c &
# tools/xngen src/qs8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -o src/qs8-gemm/gen/qs8-gemm-8x16c8-minmax-rndnu-neoni8mm.c &

tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-6x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-6x16c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-8x8c8-minmax-fp32-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-8x16c8-minmax-fp32-neoni8mm.c &

tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x8c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-1x16c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x8c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-2x16c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x8c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-3x16c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x8c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-4x16c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-6x8c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=6 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-6x16c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-8x8c8-minmax-rndnu-neoni8mm.c &
tools/xngen src/qu8-gemm/c8-neoni8mm.c.in -D MR=8 -D NR=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-gemm/gen/qu8-gemm-8x16c8-minmax-rndnu-neoni8mm.c &

################################### x86 SSE ###################################
### C2 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-sse2-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c2-minmax-fp32-sse2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c2-minmax-fp32-sse2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c2-minmax-fp32-sse2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-4x4c2-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-sse41-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c2-minmax-fp32-sse41-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c2-minmax-fp32-sse41-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c2-minmax-fp32-sse41-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-4x4c2-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-avx-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c2-minmax-fp32-avx-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c2-minmax-fp32-avx-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c2-minmax-fp32-avx-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-4x4c2-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-xop-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c2-minmax-fp32-xop-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c2-minmax-fp32-xop-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c2-minmax-fp32-xop-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-4x4c2-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-sse2-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c2-minmax-fp32-sse2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c2-minmax-fp32-sse2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c2-minmax-fp32-sse2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-4x4c2-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-sse41-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c2-minmax-fp32-sse41-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c2-minmax-fp32-sse41-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c2-minmax-fp32-sse41-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-4x4c2-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-avx-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c2-minmax-fp32-avx-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c2-minmax-fp32-avx-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c2-minmax-fp32-avx-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-4x4c2-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2-minmax-fp32-xop-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c2-minmax-fp32-xop-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c2-minmax-fp32-xop-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c2-minmax-fp32-xop-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-4x4c2-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2-minmax-fp32-xop-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c2-xw-minmax-fp32-sse2.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c2-xw-minmax-fp32-sse2.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c2-xw-minmax-fp32-sse2.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-4x4c2-xw-minmax-fp32-sse2.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c2-xw-minmax-fp32-sse41.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c2-xw-minmax-fp32-sse41.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c2-xw-minmax-fp32-sse41.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-4x4c2-xw-minmax-fp32-sse41.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c2-xw-minmax-fp32-avx.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c2-xw-minmax-fp32-avx.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c2-xw-minmax-fp32-avx.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-4x4c2-xw-minmax-fp32-avx.c &

# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c2-xw-minmax-fp32-xop.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c2-xw-minmax-fp32-xop.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c2-xw-minmax-fp32-xop.c &
# tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-4x4c2-xw-minmax-fp32-xop.c &

################################### x86 SSE ###################################
### C2S4 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-sse2-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-minmax-fp32-sse2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-minmax-fp32-sse2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-minmax-fp32-sse2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-sse41-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-minmax-fp32-sse41-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-minmax-fp32-sse41-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-minmax-fp32-sse41-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-avx-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-minmax-fp32-avx-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-minmax-fp32-avx-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-minmax-fp32-avx-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-xop-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-minmax-fp32-xop-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-minmax-fp32-xop-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-minmax-fp32-xop-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-sse2-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-minmax-fp32-sse2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-minmax-fp32-sse2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-minmax-fp32-sse2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-sse41-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-minmax-fp32-sse41-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-minmax-fp32-sse41-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-minmax-fp32-sse41-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-avx-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-minmax-fp32-avx-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-minmax-fp32-avx-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-minmax-fp32-avx-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x4c2s4-minmax-fp32-xop-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-minmax-fp32-xop-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-minmax-fp32-xop-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-minmax-fp32-xop-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-4x4c2s4-minmax-fp32-xop-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-xw-minmax-fp32-sse2.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-xw-minmax-fp32-sse2.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-xw-minmax-fp32-sse2.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-xw-minmax-fp32-sse2.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-xw-minmax-fp32-sse41.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-xw-minmax-fp32-sse41.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-xw-minmax-fp32-sse41.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-xw-minmax-fp32-sse41.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-xw-minmax-fp32-avx.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-xw-minmax-fp32-avx.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-xw-minmax-fp32-avx.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-xw-minmax-fp32-avx.c &

# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c2s4-xw-minmax-fp32-xop.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c2s4-xw-minmax-fp32-xop.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c2s4-xw-minmax-fp32-xop.c &
# tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-4x4c2s4-xw-minmax-fp32-xop.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-sse2-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-sse2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-sse2-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-sse2-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-ssse3-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-ssse3-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-ssse3-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-sse41-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-sse41-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-sse41-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-avx-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-avx-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-avx-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD64     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-xop-ld64.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-xop-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-xop-ld64.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-sse2-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-sse2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-sse2-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-sse2-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-ssse3-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-ssse3-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-ssse3-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-sse41-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-sse41-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-sse41-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-avx-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-avx-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-avx-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x4c8-minmax-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x4c8-minmax-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x4c8-minmax-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QD8 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x4c8-minmax-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC4 -D REQUANTIZATION=         -D VARIANT=LD128    -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x4c8-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x4c8-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x4c8-minmax-fp32-xop-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-1x4c8-minmax-fp32-xop-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-2x4c8-minmax-fp32-xop-ld128.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/qs8-gemm-3x4c8-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-1x4c8-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-2x4c8-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/qu8-gemm-3x4c8-minmax-fp32-xop-ld128.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c8-xw-minmax-fp32-sse2.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c8-xw-minmax-fp32-sse2.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c8-xw-minmax-fp32-sse2.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c8-xw-minmax-fp32-ssse3.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c8-xw-minmax-fp32-ssse3.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c8-xw-minmax-fp32-ssse3.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c8-xw-minmax-fp32-sse41.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c8-xw-minmax-fp32-sse41.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c8-xw-minmax-fp32-sse41.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c8-xw-minmax-fp32-avx.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c8-xw-minmax-fp32-avx.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c8-xw-minmax-fp32-avx.c &

# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-1x4c8-xw-minmax-fp32-xop.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-2x4c8-xw-minmax-fp32-xop.c &
# tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/qs8-gemm-3x4c8-xw-minmax-fp32-xop.c &


################################### x86 AVX2 ##################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-1x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-2x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-3x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QU8     -D AVX=2 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-4x8c8-minmax-fp32-avx2.c &

################################### x86 AVX512VL ##################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=6 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avx512skx.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=6 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avx512skx.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx512skx.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx512skx.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avx512skx.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D DATATYPE=QU8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-1x8c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D DATATYPE=QU8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-2x8c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D DATATYPE=QU8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-3x8c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=4 -D DATATYPE=QU8     -D AVX=10 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-4x8c8-minmax-fp32-avx512skx.c &


################################## x86 AVX512 #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=6 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QC8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-avx512skx.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-1x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-2x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-3x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-4x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-5x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=6 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-6x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-7x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QU8 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-8x16c8-minmax-fp32-avx512skx.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=6 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QD8 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-avx512skx.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=6 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QC4 -D PREFETCH=0 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512skx.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=6 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QC8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-avx512skx-prfm.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-1x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-2x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-3x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-4x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-5x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=6 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-6x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-7x16c8-minmax-fp32-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QU8 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/qu8-gemm-8x16c8-minmax-fp32-avx512skx-prfm.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=6 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QD8 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-avx512skx-prfm.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=5 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=6 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=7 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512skx-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=8 -D DATATYPE=QC4 -D PREFETCH=1 -D REQUANTIZATION=     -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512skx-prfm.c &

################################## x86 AVX512VNNI #################################
### C4 micro-kernels
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=2 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=3 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=6 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c4-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c4-minmax-fp32-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=2 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=3 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=6 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c4-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=2 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=3 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=6 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c4-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=2 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=3 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=6 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c4-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c4-minmax-fp32-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=2 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=3 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=6 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c4-minmax-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=2 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=3 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=6 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c4-minmax-avx512vnni-prfm.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=2 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=3 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=4 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=6 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=2 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=3 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=4 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=6 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=2 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=3 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=4 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=6 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=2 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=3 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=4 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=6 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8 -D DATATYPE=QC8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x16c8-minmax-fp32-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=2 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=3 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=4 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=6 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8 -D DATATYPE=QD8 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x16c8-minmax-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=2 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=3 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=4 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=6 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8 -D DATATYPE=QC4 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512vnni-prfm.c &

################################## x86 AVX512VNNI GFNI #################################
### C4 micro-kernels
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=2 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=3 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=6 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c4-minmax-avx512vnnigfni.c &

tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=1 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=2 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=3 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=4 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=5 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=6 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=7 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c4-avx512vnni.c.in -D MR=8 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c4-minmax-avx512vnnigfni-prfm.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=2 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=3 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=4 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=6 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512vnnigfni.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=1 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=2 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=3 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=4 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=5 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=6 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=7 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512vnni.c.in -D MR=8 -D DATATYPE=QC4 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x16c8-minmax-avx512vnnigfni-prfm.c &

################################## x86 AVX512VL VNNI EVEX #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x8c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x8c8-minmax-fp32-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx512vnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx512vnni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x8c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x8c8-minmax-fp32-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC8     -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avx512vnni-prfm.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx512vnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx512vnni-prfm.c &

################################## x86 AVX512VL VNNI GFNI EVEX #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx512vnnigfni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx512vnnigfni-prfm.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx512vnnigfni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx512vnnigfni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx512vnnigfni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=10 -D GFNI=1 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx512vnnigfni-prfm.c &

################################## x86 AVXVNNI #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x8c8-minmax-fp32-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-avxvnni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avxvnni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avxvnni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avxvnni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avxvnni.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=0 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avxvnni.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-6x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x8c8-minmax-fp32-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC8     -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION=FP32 -o src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-8x8c8-minmax-fp32-avxvnni-prfm.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c8-minmax-avxvnni-prfm.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F32 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avxvnni-prfm.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-5x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-6x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QD8_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-8x8c8-minmax-avxvnni-prfm.c &

tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=1 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=2 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=3 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=4 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=5 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=6 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=7 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avxvnni-prfm.c &
tools/xngen src/qs8-gemm/MRx8c8-avxvnni.c.in -D MR=8 -D DATATYPE=QC4_F16 -D AVX=2 -D GFNI=0 -D PREFETCH=1 -D REQUANTIZATION= -o src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avxvnni-prfm.c &

### C4 micro-kernels
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=1 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c4-minmax-avxvnni-u2-acc2.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=2 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c4-minmax-avxvnni-u2-acc2.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=3 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c4-minmax-avxvnni-u2-acc2.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=4 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c4-minmax-avxvnni-u2-acc2.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=5 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c4-minmax-avxvnni-u2-acc2.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=6 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c4-minmax-avxvnni-u2-acc2.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=7 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c4-minmax-avxvnni-u2-acc2.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=2 -D ACCUMULATORS=2 -D MR=8 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c4-minmax-avxvnni-u2-acc2.c

tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=1 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c4-minmax-avxvnni-u4-acc4.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=2 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c4-minmax-avxvnni-u4-acc4.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=3 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c4-minmax-avxvnni-u4-acc4.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=4 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c4-minmax-avxvnni-u4-acc4.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=5 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-5x8c4-minmax-avxvnni-u4-acc4.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=6 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-6x8c4-minmax-avxvnni-u4-acc4.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=7 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x8c4-minmax-avxvnni-u4-acc4.c
tools/xngen src/qs8-gemm/MRx8c4-avxvnni.c.in -D UNROLL=4 -D ACCUMULATORS=4 -D MR=8 -D DATATYPE=QD8 -D REQUANTIZATION= -o src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-8x8c4-minmax-avxvnni-u4-acc4.c

wait
