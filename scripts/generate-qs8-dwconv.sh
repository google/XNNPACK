#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ###################################
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up1x9-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up2x9-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up4x9-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up1x9-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up2x9-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up4x9-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up1x9-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up2x9-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up4x9-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up1x9-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up2x9-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up4x9-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up1x9-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up2x9-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up4x9-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up1x9-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up2x9-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up4x9-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up1x9-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up2x9-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up4x9-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up1x9-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up2x9-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up4x9-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up1x9-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up2x9-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up4x9-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up1x25-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up2x25-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up4x25-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up1x25-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up2x25-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up4x25-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up1x25-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up2x25-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up4x25-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up1x25-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up2x25-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up4x25-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up1x25-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up2x25-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up4x25-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up1x25-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up2x25-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up4x25-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up1x25-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up2x25-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qc8-dwconv/gen/up4x25-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up1x25-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up2x25-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/up4x25-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up1x25-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up2x25-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/up4x25-minmax-fp32-scalar-lrintf.c &

#################################### WAsm ####################################
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-dwconv/gen/up1x9-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-dwconv/gen/up2x9-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-dwconv/gen/up4x9-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/up1x9-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/up2x9-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/up4x9-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/up1x9-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/up2x9-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/up4x9-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-dwconv/gen/up1x25-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-dwconv/gen/up2x25-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qc8-dwconv/gen/up4x25-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/up1x25-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/up2x25-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/up4x25-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/up1x25-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/up2x25-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/up4x25-minmax-fp32-wasm-fmagic.c &

################################## ARM NEON ##################################
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qc8-dwconv/gen/up8x9-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qc8-dwconv/gen/up8x9-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qc8-dwconv/gen/up8x9-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qc8-dwconv/gen/up8x9-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/up8x9-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/up8x9-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qc8-dwconv/gen/up8x25-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qc8-dwconv/gen/up8x25-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qc8-dwconv/gen/up8x25-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qc8-dwconv/gen/up8x25-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/up8x25-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/up8x25-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-dwconv/gen/up8x9-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-dwconv/gen/up24x9-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-dwconv/gen/up32x9-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up8x9-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up16x9-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up24x9-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up32x9-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up8x9-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up16x9-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up24x9-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up32x9-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-dwconv/gen/up8x9-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-dwconv/gen/up24x9-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-dwconv/gen/up32x9-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/up8x9-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/up16x9-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/up24x9-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/up32x9-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/up8x9-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/up16x9-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/up24x9-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/up32x9-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up8x9-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up16x9-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up24x9-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up32x9-minmax-rndnu-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up8x9-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up16x9-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up24x9-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up32x9-minmax-rndnu-neon-mul16.c &

tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in  -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up8x9-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in  -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up16x9-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in  -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up24x9-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in  -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up32x9-minmax-rndnu-neon-mul8.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-dwconv/gen/up8x25-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-dwconv/gen/up24x25-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qc8-dwconv/gen/up32x25-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up8x25-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up16x25-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up24x25-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up32x25-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up8x25-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up16x25-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up24x25-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up32x25-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-dwconv/gen/up8x25-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-dwconv/gen/up24x25-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qc8-dwconv/gen/up32x25-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/up8x25-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/up16x25-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/up24x25-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/up32x25-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/up8x25-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/up16x25-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/up24x25-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/up32x25-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up8x25-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up16x25-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up24x25-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/up32x25-minmax-rndnu-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up8x25-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up16x25-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up24x25-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up32x25-minmax-rndnu-neon-mul16.c &

tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up8x25-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up16x25-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up24x25-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/up32x25-minmax-rndnu-neon-mul8.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qc8-dwconv/gen/up8x9-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qc8-dwconv/gen/up24x9-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qc8-dwconv/gen/up8x9-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qc8-dwconv/gen/up16x9-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qc8-dwconv/gen/up24x9-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/up8x9-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/up16x9-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/up24x9-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/up8x9-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/up16x9-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/up24x9-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/up8x9-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/up16x9-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/up24x9-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qc8-dwconv/gen/up8x25-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qc8-dwconv/gen/up24x25-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qc8-dwconv/gen/up8x25-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qc8-dwconv/gen/up16x25-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qc8-dwconv/gen/up24x25-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/up8x25-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/up16x25-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/up24x25-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/up8x25-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/up16x25-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/up24x25-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/up8x25-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/up16x25-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/up24x25-minmax-fp32-wasmsimd-mul16.c &

################################### x86 SSE ###################################
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x9-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x9-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x9-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x9-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x25-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x25-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x25-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x25-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x9-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x9-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x9-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x9-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x25-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x25-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x25-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x25-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x9-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-avx-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-avx-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x9-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-avx-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-avx-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x9-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x9-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-xop-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-xop-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-xop-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-xop-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x25-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-avx-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-avx-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x25-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-avx-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-avx-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x25-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x25-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-xop-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-xop-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-xop-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-xop-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x9-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x9-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x9-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x9-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x25-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x25-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x25-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x25-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x9-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x9-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x9-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x9-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x25-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x25-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x25-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x25-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x9-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x9-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x9-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x9-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x25-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x25-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x25-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x25-minmax-fp32-xop-mul32.c &

################################### x86 AVX2 ##################################
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-vpmovsx.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-vpmovsx.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-vpmovsx.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-vpmovsx.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up32x9-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up32x25-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x9-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x9-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up32x9-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x9-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x9-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up32x9-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x9-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x9-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up32x9-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up8x25-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up24x25-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up32x25-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up8x25-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up24x25-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up32x25-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up8x25-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x25-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up32x25-minmax-fp32-avx2-mul32.c &

################################## x86 AVX512 #################################
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x9-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up32x9-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x9-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up32x9-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x9-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up32x9-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up16x25-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qc8-dwconv/gen/up32x25-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up16x25-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/up32x25-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up16x25-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/up32x25-minmax-fp32-avx512skx-mul32.c &

################################## Unit tests #################################
tools/generate-dwconv-test.py --spec test/qc8-dwconv-minmax-fp32.yaml --output test/qc8-dwconv-minmax-fp32.cc &
tools/generate-dwconv-test.py --spec test/qs8-dwconv-minmax-fp32.yaml --output test/qs8-dwconv-minmax-fp32.cc &
tools/generate-dwconv-test.py --spec test/qu8-dwconv-minmax-fp32.yaml --output test/qu8-dwconv-minmax-fp32.cc &

tools/generate-dwconv-test.py --spec test/qs8-dwconv-minmax-rndnu.yaml --output test/qs8-dwconv-minmax-rndnu.cc &
tools/generate-dwconv-test.py --spec test/qu8-dwconv-minmax-rndnu.yaml --output test/qu8-dwconv-minmax-rndnu.cc &

wait
