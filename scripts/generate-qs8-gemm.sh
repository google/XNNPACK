#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/1x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/2x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/3x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/4x2-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/1x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/2x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/3x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/4x2-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/1x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/2x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/3x2-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/4x2-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/1x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/2x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/3x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/4x2-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/1x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/2x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/3x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/4x2-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/1x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/2x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/3x2-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/4x2-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/1x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/2x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/3x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/4x2-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/1x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/2x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/3x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/4x2-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/1x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/2x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/3x2-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/4x2-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/1x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/2x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/3x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/4x4-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/1x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/2x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/3x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/4x4-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/1x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/2x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/3x4-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/4x4-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/1x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/2x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/3x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/4x4-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/1x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/2x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/3x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/4x4-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/1x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/2x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/3x4-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/4x4-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/1x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/2x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/3x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-gemm/gen/4x4-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/1x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/2x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/3x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gemm/gen/4x4-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/1x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/2x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/3x4-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gemm/gen/4x4-minmax-fp32-scalar-lrintf.c &

##################################### WAsm ####################################
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-gemm/gen/1x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-gemm/gen/2x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-gemm/gen/3x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-gemm/gen/4x2-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/1x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/2x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/3x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/4x2-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/1x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/2x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/3x2-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/4x2-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-gemm/gen/1x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-gemm/gen/2x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-gemm/gen/3x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-gemm/gen/4x4-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/1x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/2x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/3x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-gemm/gen/4x4-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/1x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/2x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/3x4-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-gemm/gen/4x4-minmax-fp32-wasm-fmagic.c &

################################## WAsm SIMD ##################################
### C2 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/1x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/2x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/3x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/1x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/2x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/3x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x4c2-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/1x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/2x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/3x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/1x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/2x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/3x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x4c2-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c2-xw-minmax-fp32-wasmsimd-dot16x2.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c2-xw-minmax-fp32-wasmsimd-dot16x2.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c2-xw-minmax-fp32-wasmsimd-dot16x2.c &
tools/xngen src/qs8-gemm/MRx4c2-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x4c2-xw-minmax-fp32-wasmsimd-dot16x2.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64   -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/1x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/2x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/3x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x4c2s4-minmax-fp32-wasmsimd-dot16x2-ld128.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x4c8-minmax-fp32-wasmsimd-dot16x2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x4c8-minmax-fp32-wasmsimd-dot16x2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=1 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-wasmsimd-dot16x2.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=2 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-wasmsimd-dot16x2.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=3 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-wasmsimd-dot16x2.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-dot16x2.c.in -D MR=4 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x4c8-xw-minmax-fp32-wasmsimd-dot16x2.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-mul16-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-mul16-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-mul16-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-mul16-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-mul16-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-mul16-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul32.c.in -D MR=1 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-mul32-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul32.c.in -D MR=2 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-mul32-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul32.c.in -D MR=3 -D VARIANT=LD64     -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-mul32-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-mul16-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-mul16-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -o src/qc8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-mul16-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-mul16-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-mul16-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-mul16-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul32.c.in -D MR=1 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/1x4c8-minmax-fp32-wasmsimd-mul32-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul32.c.in -D MR=2 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/2x4c8-minmax-fp32-wasmsimd-mul32-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul32.c.in -D MR=3 -D VARIANT=LD128    -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -o src/qu8-gemm/gen/3x4c8-minmax-fp32-wasmsimd-mul32-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=1 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=2 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd-mul16.c.in -D MR=3 -D VARIANT=EXTENDED -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -o src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-wasmsimd-mul16.c &

################################### ARM NEON ##################################
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/1x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/2x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/3x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/6x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/1x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/2x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/3x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/6x16-minmax-rndnu-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/1x8-minmax-rndnu-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/2x8-minmax-rndnu-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/3x8-minmax-rndnu-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x8-minmax-rndnu-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/6x8-minmax-rndnu-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/1x16-minmax-rndnu-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/2x16-minmax-rndnu-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/3x16-minmax-rndnu-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x16-minmax-rndnu-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/6x16-minmax-rndnu-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/1x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/2x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/3x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/4x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/6x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/1x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/2x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/3x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/4x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/6x16-minmax-fp32-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/1x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/2x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/3x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/4x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/6x8-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/2x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/3x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/6x16-minmax-fp32-neonv8-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/1x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/2x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/3x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/4x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/6x8-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/1x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/2x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/3x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/4x16-minmax-fp32-neon-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/6x16-minmax-fp32-neon-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/1x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/2x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/3x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/4x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/6x8-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/1x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/2x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/3x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/4x16-minmax-fp32-neonv8-mlal-lane-prfm.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/6x16-minmax-fp32-neonv8-mlal-lane-prfm.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/1x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/2x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/3x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/4x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/6x8-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/1x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/2x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/3x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/4x16-minmax-rndnu-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/6x16-minmax-rndnu-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/1x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x16-minmax-fp32-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/1x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/4x8-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/1x16-minmax-fp32-neon-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/4x16-minmax-fp32-neon-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gemm/gen/1x16-minmax-fp32-neonv8-mlal-lane.c &
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gemm/gen/4x16-minmax-fp32-neonv8-mlal-lane.c &

tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8-minmax-rndnu-neon-mull-addw-dup.c &
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8-minmax-rndnu-neon-mull-addw-dup.c &
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/3x8-minmax-rndnu-neon-mull-addw-dup.c &
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x8-minmax-rndnu-neon-mull-addw-dup.c &
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x16-minmax-rndnu-neon-mull-addw-dup.c &
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x16-minmax-rndnu-neon-mull-addw-dup.c &
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/3x16-minmax-rndnu-neon-mull-addw-dup.c &
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x16-minmax-rndnu-neon-mull-addw-dup.c &

### C2 micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mull-dup.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mlal-dup.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-dup.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=DUP  -o src/qc8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=DUP  -o src/qc8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-dup.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=DUP  -o src/qs8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=DUP  -o src/qs8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-dup.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=DUP  -o src/qc8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-dup.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=DUP  -o src/qc8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-dup.c &

### C2 LD1R micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mull-ld1r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=LD1R -o src/qc8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=LD1R -o src/qc8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=LD1R -o src/qs8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=LD1R -o src/qs8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=LD1R -o src/qc8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=LD1R -o src/qc8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld1r.c &

### C2 LD2R micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mull-ld2r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=LD2R -o src/qc8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=LD2R -o src/qc8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=LD2R -o src/qs8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=LD2R -o src/qs8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=LD2R -o src/qc8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=LD2R -o src/qc8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld2r.c &

### C2 LD4R micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mull-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mull-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mull-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mull-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mull-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mull-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mull-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mull-ld4r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/1x8c2-minmax-rndnu-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/2x8c2-minmax-rndnu-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/3x8c2-minmax-rndnu-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/4x8c2-minmax-rndnu-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/1x16c2-minmax-rndnu-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/2x16c2-minmax-rndnu-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/3x16c2-minmax-rndnu-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/4x16c2-minmax-rndnu-neon-mlal-ld4r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD4R -o src/qs8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld4r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=LD4R -o src/qc8-gemm/gen/1x8c2-minmax-fp32-neon-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=LD4R -o src/qc8-gemm/gen/2x8c2-minmax-fp32-neon-mlal-ld4r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=LD4R -o src/qs8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=LD4R -o src/qs8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld4r.c &

tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=LD4R -o src/qc8-gemm/gen/1x8c2-minmax-fp32-neonv8-mlal-ld4r.c &
tools/xngen src/qs8-gemm/c2-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=LD4R -o src/qc8-gemm/gen/2x8c2-minmax-fp32-neonv8-mlal-ld4r.c &

### C2S4 micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x8c2s4-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x8c2s4-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x8c2s4-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x8c2s4-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x16c2s4-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x16c2s4-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x16c2s4-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x16c2s4-minmax-rndnu-neon-mull.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x8c2s4-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x8c2s4-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x8c2s4-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x8c2s4-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x16c2s4-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x16c2s4-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x16c2s4-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x16c2s4-minmax-rndnu-neon-mlal.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x8c2s4-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x8c2s4-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-gemm/gen/1x8c2s4-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-gemm/gen/2x8c2s4-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-gemm/gen/1x8c2s4-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-gemm/gen/2x8c2s4-minmax-fp32-neonv8-mlal.c &

tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-gemm/gen/1x8c2s4-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c2-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-gemm/gen/2x8c2s4-minmax-fp32-neonv8-mlal.c &

### C4 micro-kernels
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mull-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mull-dup.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mlal-dup.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=DUP  -o src/qs8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-dup.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=DUP  -o src/qc8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=DUP  -o src/qc8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-dup.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=DUP  -o src/qs8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=DUP  -o src/qs8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-dup.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=DUP  -o src/qc8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-dup.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=DUP  -o src/qc8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-dup.c &

### C4 LD1R micro-kernels
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mull-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mull-ld1r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD1R -o src/qs8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=LD1R -o src/qc8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=LD1R -o src/qc8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=LD1R -o src/qs8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=LD1R -o src/qs8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld1r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=LD1R -o src/qc8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld1r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=LD1R -o src/qc8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld1r.c &

### C4 LD2R micro-kernels
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mull-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mull-ld2r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/1x8c4-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/2x8c4-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/3x8c4-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/4x8c4-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/1x16c4-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/2x16c4-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/3x16c4-minmax-rndnu-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-neon-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -D DUP=LD2R -o src/qs8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=LD2R -o src/qc8-gemm/gen/1x8c4-minmax-fp32-neon-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -D DUP=LD2R -o src/qc8-gemm/gen/2x8c4-minmax-fp32-neon-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=LD2R -o src/qs8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -D DUP=LD2R -o src/qs8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld2r.c &

tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=LD2R -o src/qc8-gemm/gen/1x8c4-minmax-fp32-neonv8-mlal-ld2r.c &
tools/xngen src/qs8-gemm/c4-neon-mull-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -D DUP=LD2R -o src/qc8-gemm/gen/2x8c4-minmax-fp32-neonv8-mlal-ld2r.c &

### C4S2 micro-kernels
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x8c4s2-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x8c4s2-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x8c4s2-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x8c4s2-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x16c4s2-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x16c4s2-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x16c4s2-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x16c4s2-minmax-rndnu-neon-mull.c &

tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x8c4s2-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x8c4s2-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x8c4s2-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x8c4s2-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x16c4s2-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x16c4s2-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x16c4s2-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x16c4s2-minmax-rndnu-neon-mlal.c &

tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x8c4s2-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x8c4s2-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-gemm/gen/1x8c4s2-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-gemm/gen/2x8c4s2-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-gemm/gen/1x8c4s2-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-gemm/gen/2x8c4s2-minmax-fp32-neonv8-mlal.c &

tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-gemm/gen/1x8c4s2-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c4-neon-mull-shuffle.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-gemm/gen/2x8c4s2-minmax-fp32-neonv8-mlal.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x8c8-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x8c8-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=3 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x8c8-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=4 -D NR=8  -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x8c8-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x16c8-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x16c8-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=3 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x16c8-minmax-rndnu-neon-mull.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=4 -D NR=16 -D MLA=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x16c8-minmax-rndnu-neon-mull.c &

tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x8c8-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x8c8-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=3 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x8c8-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=4 -D NR=8  -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x8c8-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x16c8-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x16c8-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=3 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/3x16c8-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=4 -D NR=16 -D MLA=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/4x16c8-minmax-rndnu-neon-mlal.c &

tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/1x8c8-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=0 -o src/qs8-gemm/gen/2x8c8-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-gemm/gen/1x8c8-minmax-fp32-neon-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=0 -o src/qc8-gemm/gen/2x8c8-minmax-fp32-neon-mlal.c &

tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-gemm/gen/1x8c8-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D ARMV8=1 -o src/qs8-gemm/gen/2x8c8-minmax-fp32-neonv8-mlal.c &

tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=1 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-gemm/gen/1x8c8-minmax-fp32-neonv8-mlal.c &
tools/xngen src/qs8-gemm/c8-neon-mull.c.in -D MR=2 -D NR=8  -D MLA=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D ARMV8=1 -o src/qc8-gemm/gen/2x8c8-minmax-fp32-neonv8-mlal.c &

### C16 micro-kernels
tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=1 -D NR=8  -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/1x8c16-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=2 -D NR=8  -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/2x8c16-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=3 -D NR=8  -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/3x8c16-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=4 -D NR=8  -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/4x8c16-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=1 -D NR=16 -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/1x16c16-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=2 -D NR=16 -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/2x16c16-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=3 -D NR=16 -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/3x16c16-minmax-rndnu-neon-mlal.c &
tools/xngen src/qs8-gemm/c16-neon-mlal.c.in -D MR=4 -D NR=16 -D REQUANTIZATION=RNDNU -o src/qs8-gemm/gen/4x16c16-minmax-rndnu-neon-mlal.c &

### C4 micro-kernels
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8c4-minmax-fp32-neondot.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -o src/qc8-gemm/gen/1x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -o src/qc8-gemm/gen/4x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -o src/qc8-gemm/gen/6x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -o src/qc8-gemm/gen/8x8c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -o src/qc8-gemm/gen/1x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -o src/qc8-gemm/gen/4x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -o src/qc8-gemm/gen/6x16c4-minmax-fp32-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -o src/qc8-gemm/gen/8x16c4-minmax-fp32-neondot.c &

tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/6x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/8x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/6x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qs8-gemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qs8-gemm/gen/8x16c4-minmax-rndnu-neondot.c &

tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/1x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=2  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/2x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=3  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/3x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/4x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=5  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/5x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/6x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/8x8c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/1x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=2  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/2x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=3  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/3x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/4x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=5  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/5x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/6x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/8x16c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=1  -D NR=32 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/1x32c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=2  -D NR=32 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/2x32c4-minmax-rndnu-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=3  -D NR=32 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -o src/qu8-gemm/gen/3x32c4-minmax-rndnu-neondot.c &

tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qu8-gemm/gen/1x16c4-minmax-fp32-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=2  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qu8-gemm/gen/2x16c4-minmax-fp32-neondot.c &
tools/xngen src/qu8-gemm/c4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qu8-gemm/gen/4x16c4-minmax-fp32-neondot.c &

############################### AArch32 assembly ##############################
### Cortex-A53 lane micro-kernels
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-prfm-ld64.S &

tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neonv8-mlal-lane-prfm-cortex-a53.S &

tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a7.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-gemm/gen/4x8-minmax-fp32-aarch32-neon-mlal-lane-prfm-cortex-a7.S &

### QU8 micro-kernels
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-ld64.S &

tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a53.S.in  -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a53.S &

tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-cortex-a7.S &
tools/xngen src/qs8-gemm/4x8-aarch32-neon-mlal-lane-cortex-a7.S.in   -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-cortex-a7.S &

### C4 micro-kernels
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-ld64.S.in                           -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x8c4-minmax-rndnu-aarch32-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-ld64.S.in                           -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x8c4-minmax-fp32-aarch32-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x8c4-minmax-rndnu-aarch32-neondot-cortex-a55.S &
tools/xngen src/qs8-gemm/4x8c4-aarch32-neondot-cortex-a55.S.in                     -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x8c4-minmax-fp32-aarch32-neondot-cortex-a55.S &

############################### AArch64 assembly ##############################
### Cortex-A53 lane micro-kernels
tools/xngen src/qs8-gemm/4x8-aarch64-neon-mlal-lane-ld64.S.in        -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x8-minmax-rndnu-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x8-aarch64-neon-mlal-lane-ld64.S.in        -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x8-minmax-rndnu-aarch64-neon-mlal-lane-prfm-ld64.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-cortex-a53.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-cortex-a53.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-cortex-a53.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-ld64.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=0 -D DATATYPE=QS8 -o src/qs8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-ld64.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=FP32  -D CHANNELWISE=1 -D DATATYPE=QC8 -o src/qc8-gemm/gen/4x16-minmax-fp32-aarch64-neon-mlal-lane-prfm-ld64.S &

### QU8 micro-kernels
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-cortex-a53.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-cortex-a53.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-ld64.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-ld64.S.in       -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-ld64.S &

tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a75.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-cortex-a75.S &
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a75.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU -D CHANNELWISE=0 -D DATATYPE=QU8 -o src/qu8-gemm/gen/4x16-minmax-rndnu-aarch64-neon-mlal-lane-prfm-cortex-a75.S &

### C4 micro-kernels
tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x16c4-minmax-rndnu-aarch64-neondot-ld32.S &
tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x16c4-minmax-rndnu-aarch64-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-cortex-a55.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld32.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld128.S &

tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x16c4-minmax-fp32-aarch64-neondot-ld32.S &
tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x16c4-minmax-fp32-aarch64-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld32.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=FP32     -D CHANNELWISE=0 -o src/qs8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S &

tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-gemm/gen/1x16c4-minmax-fp32-aarch64-neondot-ld32.S &
tools/xngen src/qs8-gemm/1x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-gemm/gen/1x16c4-minmax-fp32-aarch64-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld32.S.in       -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld32.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld64.S.in       -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld64.S &
tools/xngen src/qs8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=FP32     -D CHANNELWISE=1 -o src/qc8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S &

### C4 QU8 micro-kernels
tools/xngen src/qu8-gemm/4x8c4-aarch64-neondot-cortex-a55.S.in  -D REQUANTIZATION=RNDNU -o src/qu8-gemm/gen/4x8c4-minmax-rndnu-aarch64-neondot-cortex-a55.S &
tools/xngen src/qu8-gemm/4x8c4-aarch64-neondot-ld128.S.in        -D REQUANTIZATION=RNDNU -o src/qu8-gemm/gen/4x8c4-minmax-rndnu-aarch64-neondot-ld128.S &
tools/xngen src/qu8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=RNDNU -o src/qu8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-cortex-a55.S &
tools/xngen src/qu8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=RNDNU -o src/qu8-gemm/gen/4x16c4-minmax-rndnu-aarch64-neondot-ld128.S &

tools/xngen src/qu8-gemm/4x16c4-aarch64-neondot-cortex-a55.S.in -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-cortex-a55.S &
tools/xngen src/qu8-gemm/4x16c4-aarch64-neondot-ld128.S.in      -D REQUANTIZATION=FP32 -o src/qu8-gemm/gen/4x16c4-minmax-fp32-aarch64-neondot-ld128.S &

### C8 / C16 micro-kernels
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal-prfm-cortex-a53.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal-prfm-cortex-a53.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8c8-minmax-rndnu-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mull.S.in                          -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c8-minmax-rndnu-aarch64-neon-mull.S &
tools/xngen src/qs8-gemm/2x8c16-aarch64-neon-mlal.S.in                         -D REQUANTIZATION=RNDNU    -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c16-minmax-rndnu-aarch64-neon-mlal.S &

tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32 -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32 -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32 -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32 -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32 -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32 -D CHANNELWISE=0 -o src/qs8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32 -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32 -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mull.S.in                          -D REQUANTIZATION=FP32 -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mull.S &
tools/xngen src/qs8-gemm/2x8c16-aarch64-neon-mlal.S.in                         -D REQUANTIZATION=FP32 -D CHANNELWISE=0 -o src/qs8-gemm/gen/2x8c16-minmax-fp32-aarch64-neon-mlal.S &

tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32 -D CHANNELWISE=1 -o src/qc8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32 -D CHANNELWISE=1 -o src/qc8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=0 -D REQUANTIZATION=FP32 -D CHANNELWISE=1 -o src/qc8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-cortex-a53.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-cortex-a53.S.in -D PREFETCH=1 -D REQUANTIZATION=FP32 -D CHANNELWISE=1 -o src/qc8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm-cortex-a53.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32 -D CHANNELWISE=1 -o src/qc8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal.S &
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32 -D CHANNELWISE=1 -o src/qc8-gemm/gen/1x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=0 -D REQUANTIZATION=FP32 -D CHANNELWISE=1 -o src/qc8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal.S.in            -D PREFETCH=1 -D REQUANTIZATION=FP32 -D CHANNELWISE=1 -o src/qc8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mlal-prfm.S &
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mull.S.in                          -D REQUANTIZATION=FP32 -D CHANNELWISE=1 -o src/qc8-gemm/gen/2x8c8-minmax-fp32-aarch64-neon-mull.S &
tools/xngen src/qs8-gemm/2x8c16-aarch64-neon-mlal.S.in                         -D REQUANTIZATION=FP32 -D CHANNELWISE=1 -o src/qc8-gemm/gen/2x8c16-minmax-fp32-aarch64-neon-mlal.S &

################################### x86 SSE ###################################
### C2 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c2-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/4x4c2-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c2-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/4x4c2-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c2-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/4x4c2-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c2-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/4x4c2-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c2-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/4x4c2-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c2-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/4x4c2-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c2-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/4x4c2-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c2-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/4x4c2-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-fp32-sse2.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2-xw-minmax-fp32-sse2.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2-xw-minmax-fp32-sse2.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-fp32-sse2.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-fp32-sse41.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2-xw-minmax-fp32-sse41.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2-xw-minmax-fp32-sse41.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-fp32-sse41.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-fp32-avx.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2-xw-minmax-fp32-avx.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2-xw-minmax-fp32-avx.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-fp32-avx.c &

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-fp32-xop.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2-xw-minmax-fp32-xop.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2-xw-minmax-fp32-xop.c &
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-fp32-xop.c &

################################### x86 SSE ###################################
### C2S4 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c2s4-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/4x4c2s4-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c2s4-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/4x4c2s4-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c2s4-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/4x4c2s4-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c2s4-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/4x4c2s4-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2s4-xw-minmax-fp32-sse2.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2s4-xw-minmax-fp32-sse2.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2s4-xw-minmax-fp32-sse2.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2s4-xw-minmax-fp32-sse2.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2s4-xw-minmax-fp32-sse41.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2s4-xw-minmax-fp32-sse41.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2s4-xw-minmax-fp32-sse41.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2s4-xw-minmax-fp32-sse41.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2s4-xw-minmax-fp32-avx.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2s4-xw-minmax-fp32-avx.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2s4-xw-minmax-fp32-avx.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2s4-xw-minmax-fp32-avx.c &

tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2s4-xw-minmax-fp32-xop.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2s4-xw-minmax-fp32-xop.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2s4-xw-minmax-fp32-xop.c &
tools/xngen src/qs8-gemm/MRx4c2s4-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2s4-xw-minmax-fp32-xop.c &

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c8-minmax-fp32-sse2-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-fp32-ssse3-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-fp32-ssse3-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-fp32-ssse3-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c8-minmax-fp32-sse41-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c8-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c8-minmax-fp32-avx-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c8-minmax-fp32-avx-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c8-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qu8-gemm/gen/3x4c8-minmax-fp32-xop-ld64.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c8-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c8-minmax-fp32-sse2-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c8-minmax-fp32-sse2-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-fp32-ssse3-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-fp32-ssse3-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-fp32-ssse3-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c8-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c8-minmax-fp32-sse41-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c8-minmax-fp32-sse41-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c8-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c8-minmax-fp32-avx-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c8-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c8-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c8-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/1x4c8-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/2x4c8-minmax-fp32-xop-ld128.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qu8-gemm/gen/3x4c8-minmax-fp32-xop-ld128.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-sse2.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-sse2.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-sse2.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-ssse3.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-ssse3.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-ssse3.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-sse41.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-sse41.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-sse41.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-avx.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-avx.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-avx.c &

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-fp32-xop.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-fp32-xop.c &
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-fp32-xop.c &

################################### x86 AVX2 ##################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=LD128    -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/1x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=LD128    -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/2x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=LD128    -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/3x8c8-minmax-fp32-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=LD128    -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/1x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=LD128    -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/2x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=LD128    -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/3x8c8-minmax-fp32-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=LD128    -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-gemm/gen/1x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=LD128    -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-gemm/gen/2x8c8-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=LD128    -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-gemm/gen/3x8c8-minmax-fp32-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=EXTENDED -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/1x8c8-xw-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=EXTENDED -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/2x8c8-xw-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=EXTENDED -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/3x8c8-xw-minmax-fp32-avx2.c &

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=EXTENDED -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/1x8c8-xw-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=EXTENDED -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/2x8c8-xw-minmax-fp32-avx2.c &
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=EXTENDED -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/3x8c8-xw-minmax-fp32-avx2.c &

################################## x86 AVX512 #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D VARIANT=LD256 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D VARIANT=LD256 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/2x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D VARIANT=LD256 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/3x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D VARIANT=LD256 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D VARIANT=LD256 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D VARIANT=LD256 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/2x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D VARIANT=LD256 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/3x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D VARIANT=LD256 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c &

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D VARIANT=LD256 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D VARIANT=LD256 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-gemm/gen/2x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D VARIANT=LD256 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-gemm/gen/3x16c8-minmax-fp32-avx512skx.c &
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D VARIANT=LD256 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c &

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/qc8-gemm-minmax-fp32.yaml --output test/qc8-gemm-minmax-fp32.cc --output test/qc8-gemm-minmax-fp32-2.cc --output test/qc8-gemm-minmax-fp32-3.cc &
tools/generate-gemm-test.py --spec test/qs8-gemm-minmax-fp32.yaml --output test/qs8-gemm-minmax-fp32.cc --output test/qs8-gemm-minmax-fp32-2.cc &
tools/generate-gemm-test.py --spec test/qu8-gemm-minmax-fp32.yaml --output test/qu8-gemm-minmax-fp32.cc --output test/qu8-gemm-minmax-fp32-2.cc &

tools/generate-gemm-test.py --spec test/qs8-gemm-minmax-rndnu.yaml --output test/qs8-gemm-minmax-rndnu.cc --output test/qs8-gemm-minmax-rndnu-2.cc --output test/qs8-gemm-minmax-rndnu-3.cc --output test/qs8-gemm-minmax-rndnu-4.cc --output test/qs8-gemm-minmax-rndnu-5.cc &
tools/generate-gemm-test.py --spec test/qu8-gemm-minmax-rndnu.yaml --output test/qu8-gemm-minmax-rndnu.cc --output test/qu8-gemm-minmax-rndnu-2.cc &

wait
