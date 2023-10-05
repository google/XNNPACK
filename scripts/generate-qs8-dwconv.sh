#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ###################################
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=3  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p1c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p2c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p2c-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-4p2c-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p1c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p2c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p4c-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p1c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p2c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p4c-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p1c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p2c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p4c-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p1c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p2c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p4c-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p1c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p2c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p4c-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p1c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p2c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p4c-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p1c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p2c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p4c-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p1c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p2c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p4c-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p1c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p2c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p4c-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p1c-minmax-rndnu-scalar.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p2c-minmax-rndnu-scalar.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p4c-minmax-rndnu-scalar.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p1c-minmax-rndnu-scalar.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p2c-minmax-rndnu-scalar.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D VARIANT=      -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p4c-minmax-rndnu-scalar.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p1c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p2c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p4c-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p1c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p2c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p4c-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p1c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p2c-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p4c-minmax-fp32-scalar-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p1c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p2c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p4c-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p1c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p2c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p4c-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p1c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p2c-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p4c-minmax-fp32-scalar-imagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p1c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p2c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p4c-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p1c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p2c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p4c-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p1c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p2c-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p4c-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l1c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l2c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l4c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l1c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l2c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l4c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l1c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l2c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l4c1s1r-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l1c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l2c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l4c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l1c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l2c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l4c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l1c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l2c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l4c1s1r-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l1c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l2c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l4c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l1c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l2c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l4c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l1c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l2c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l4c1s1r-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l1c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l2c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l4c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l1c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l2c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l4c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l1c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l2c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l4c1s1r-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l1c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l2c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l4c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l1c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l2c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l4c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l1c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l2c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l4c1s1r-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l1c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l2c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l4c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l1c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l2c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l4c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l1c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l2c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l4c1s1r-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l1c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l2c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l4c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l1c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l2c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l4c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l1c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l2c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l4c1s1r-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l1c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l2c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l4c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l1c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l2c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l4c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l1c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l2c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l4c1s1r-minmax-fp32-scalar-lrintf.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l1c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l2c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l4c1s1r-minmax-fp32-scalar-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l1c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l2c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=IMAGIC -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l4c1s1r-minmax-fp32-scalar-imagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l1c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l2c1s1r-minmax-fp32-scalar-lrintf.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=LRINTF -D DATATYPE=QC8 -D WASM=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l4c1s1r-minmax-fp32-scalar-lrintf.c &

#################################### WAsm ####################################
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p2c-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p1c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p2c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p4c-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-9p1c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-9p2c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-9p4c-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-9p1c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-9p2c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-9p4c-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p1c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p2c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p4c-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-25p1c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-25p2c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-25p4c-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-25p1c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-25p2c-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-25p4c-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l1c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l2c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l4c1s1r-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l1c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l2c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l4c1s1r-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l1c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l2c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=1 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l4c1s1r-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l1c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l2c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l4c1s1r-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l1c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l2c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l4c1s1r-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l1c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l2c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=1 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l4c1s1r-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l1c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l2c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l4c1s1r-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l1c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l2c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l4c1s1r-minmax-fp32-wasm-fmagic.c &

tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l1c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=2 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l2c1s1r-minmax-fp32-wasm-fmagic.c &
tools/xngen src/qs8-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D VARIANT=FMAGIC -D DATATYPE=QC8 -D WASM=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l4c1s1r-minmax-fp32-wasm-fmagic.c &

################################## ARM NEON ##################################
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p8c-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p16c-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p16c-minmax-fp32-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p8c-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p16c-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p16c-minmax-fp32-neonv8-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-4p8c-minmax-fp32-neon-mla8-ld64.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-neon-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-neonv8-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-neonv8-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-rndnu-neon-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-rndnu-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-neon-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-neonv8-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-neonv8-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-rndnu-neon-mul8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-rndnu-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p32c-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-9p32c-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-rndnu-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p32c-minmax-rndnu-neon-mul16.c &

tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in  -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in  -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in  -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p32c-minmax-rndnu-neon-mul8.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p32c-minmax-fp32-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32     -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-25p32c-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-rndnu-neon-mul16.c &

tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU    -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p32c-minmax-rndnu-neon-mul16.c &

tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/unipass-neon-mul8.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D REQUANTIZATION=RNDNU     -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p32c-minmax-rndnu-neon-mul8.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-rndnu-neon-mul8-ld128.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-rndnu-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-rndnu-neon-mul8-ld128.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-rndnu-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-rndnu-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-rndnu-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-rndnu-neon-mul8-ld128.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-rndnu-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-neon-mul8-ld128.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-neon-mul8-ld128.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-neon-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-neon-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-neon-mul8-ld128.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-neon-mla8-ld128.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-neonv8-mul8-ld128.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-neonv8-mla8-ld128.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-neonv8-mul8-ld128.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-neonv8-mla8-ld128.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-neonv8-mul8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD64  -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-neonv8-mla8-ld64.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=0 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-neonv8-mul8-ld128.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D MLA=1 -D LOAD_VARIANT=LD128 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-neonv8-mla8-ld128.c &

tools/xngen src/qu8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c8s8r-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c8s8r-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l32c8s8r-minmax-rndnu-neon-mul8.c &

tools/xngen src/qu8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c8s8r-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c8s8r-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l32c8s8r-minmax-rndnu-neon-mul8.c &

tools/xngen src/qu8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c8s8r-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c8s8r-minmax-rndnu-neon-mul8.c &
tools/xngen src/qu8-dwconv/multipass-neon-mul8.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l32c8s8r-minmax-rndnu-neon-mul8.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c8s8r-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c8s8r-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c8s8r-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l32c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l32c8s8r-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l32c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l32c8s8r-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l32c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l32c8s8r-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c8s8r-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c8s8r-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c8s8r-minmax-fp32-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-neonv8-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ARMV8=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c8s8r-minmax-fp32-neonv8-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c8s8r-minmax-rndnu-neon-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c8s8r-minmax-rndnu-neon-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c8s8r-minmax-rndnu-neon-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l32c8s8r-minmax-rndnu-neon-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l32c8s8r-minmax-rndnu-neon-mul16.c &

tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c8s8r-minmax-rndnu-neon-mul16.c &
tools/xngen src/qs8-dwconv/multipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l32c8s8r-minmax-rndnu-neon-mul16.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p16c-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-wasmsimd-mul16.c &


tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=0 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QS8 -D ADD16=1 -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c8s8r-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c8s8r-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QU8 -D ADD16=0 -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c8s8r-minmax-fp32-wasmsimd-mul16.c &

tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &

tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=0 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-wasmsimd-mul16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D REQUANTIZATION=FP32  -D DATATYPE=QC8 -D ADD16=1 -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-wasmsimd-mul16-add16.c &

################################### x86 SSE ###################################
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p8c-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-sse2-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-sse2-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-sse2-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-sse2-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p8c-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-sse41-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-sse41-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-sse41-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-sse41-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p16c-minmax-fp32-avx-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p16c-minmax-fp32-xop-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-avx-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-avx-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-xop-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-xop-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-xop-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-xop-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-avx-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-avx-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-avx-mul16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-avx-mul16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-xop-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-xop-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-xop-mul16-add16.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-xop-mul16-add16.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/unipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-fp32-sse2-mul16-add16.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-fp32-sse41-mul16-add16.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c8s8r-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c8s8r-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-sse2-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-sse2-mul16.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-sse2-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=2 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-sse2-mul16-add16.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-sse41-mul16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=0 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-sse41-mul16.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-sse41-mul16-add16.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D ADD16=1 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-sse41-mul16-add16.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c4s4r-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c4s4r-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c4s4r-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c4s4r-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c4s4r-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c4s4r-minmax-fp32-xop-mul32.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c4s4r-minmax-fp32-sse41-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=0 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c4s4r-minmax-fp32-sse41-mul32.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c4s4r-minmax-fp32-avx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=0 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c4s4r-minmax-fp32-avx-mul32.c &

tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c4s4r-minmax-fp32-xop-mul32.c &
tools/xngen src/qs8-dwconv/multipass-sse-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D SSE=4 -D AVX=1 -D XOP=1 -D         -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c4s4r-minmax-fp32-xop-mul32.c &

################################### x86 AVX2 ##################################
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-avx2-mul16-vpmovsx.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-avx2-mul16-vpmovsx.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-avx2-mul16-vpmovsx.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-avx2-mul16-vpmovsx.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-avx2-mul16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-avx2-mul16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-avx2-mul16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-avx2-mul16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p16c-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p32c-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p32c-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpmovsx.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D ADD16=0 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul16-vpunpck.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D ADD16=1 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c8s8r-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l32c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l32c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l32c8s8r-minmax-fp32-avx2-mul32.c &

tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-avx2-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c8s8r-minmax-fp32-avx2-mul32.c &

################################## x86 AVX512 #################################
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p32c-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-9p32c-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-25p32c-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QS8 -D REQUANTIZATION=FP32     -o src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c16s1r-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-5f5m5l32c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-6f6m7l32c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QU8 -D REQUANTIZATION=FP32     -o src/qu8-dwconv/gen/qu8-dwconv-8f8m9l32c16s1r-minmax-fp32-avx512skx-mul32.c &

tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32  -D FIRST_PASS_TILE=5  -D MIDDLE_PASS_TILE=5  -D LAST_PASS_TILE=5 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32  -D FIRST_PASS_TILE=6  -D MIDDLE_PASS_TILE=6  -D LAST_PASS_TILE=7 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c16s1r-minmax-fp32-avx512skx-mul32.c &
tools/xngen src/qs8-dwconv/multipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32  -D FIRST_PASS_TILE=8  -D MIDDLE_PASS_TILE=8  -D LAST_PASS_TILE=9 -D DATATYPE=QC8 -D REQUANTIZATION=FP32     -o src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c16s1r-minmax-fp32-avx512skx-mul32.c &

wait
