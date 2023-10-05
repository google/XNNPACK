#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ####################################
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-scalar-imagic-c1.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-scalar-imagic-c2.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-scalar-imagic-c4.c &

tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-scalar-fmagic-c1.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-scalar-fmagic-c2.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-scalar-fmagic-c4.c &

tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-scalar-lrintf-c1.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-scalar-lrintf-c2.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-scalar-lrintf-c4.c &

tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-scalar-imagic-c1.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-scalar-imagic-c2.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-scalar-imagic-c4.c &

tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-scalar-fmagic-c1.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-scalar-fmagic-c2.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-scalar-fmagic-c4.c &

tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-scalar-lrintf-c1.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-scalar-lrintf-c2.c &
tools/xngen src/qs8-gavgpool/unipass-scalar.c.in -D ROW_TILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-scalar-lrintf-c4.c &

tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-scalar-imagic-c1.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-scalar-imagic-c2.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-scalar-imagic-c4.c &

tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-scalar-fmagic-c1.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-scalar-fmagic-c2.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-scalar-fmagic-c4.c &

tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-scalar-lrintf-c1.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-scalar-lrintf-c2.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QS8 -D WASM=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-scalar-lrintf-c4.c &

tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-scalar-imagic-c1.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-scalar-imagic-c2.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-scalar-imagic-c4.c &

tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-scalar-fmagic-c1.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-scalar-fmagic-c2.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-scalar-fmagic-c4.c &

tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=1 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-scalar-lrintf-c1.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=2 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-scalar-lrintf-c2.c &
tools/xngen src/qs8-gavgpool/multipass-scalar.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=4 -D REQUANTIZATION=FP32 -D VARIANT=LRINTF -D DATATYPE=QU8 -D WASM=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-scalar-lrintf-c4.c &

################################## ARM NEON ###################################
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-neon-c8.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-neon-c16.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-neon-c24.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-neon-c32.c &

tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-neonv8-c8.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-neonv8-c16.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-neonv8-c24.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-neonv8-c32.c &

tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-rndnu-neon-c8.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-rndnu-neon-c16.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-rndnu-neon-c24.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-rndnu-neon-c32.c &

tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-neon-c8.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-neon-c16.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-neon-c24.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-neon-c32.c &

tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-neonv8-c8.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-neonv8-c16.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-neonv8-c24.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-neonv8-c32.c &

tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-rndnu-neon-c8.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-rndnu-neon-c16.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-rndnu-neon-c24.c &
tools/xngen src/qs8-gavgpool/unipass-neon.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-rndnu-neon-c32.c &

tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-neon-c8.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-neon-c16.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-neon-c24.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-neon-c32.c &

tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-neonv8-c8.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-neonv8-c16.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-neonv8-c24.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -D ARMV8=1 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-neonv8-c32.c &

tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-rndnu-neon-c8.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-rndnu-neon-c16.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-rndnu-neon-c24.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=RNDNU -D DATATYPE=QS8 -D ARMV8=0 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-rndnu-neon-c32.c &

tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-neon-c8.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-neon-c16.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-neon-c24.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-neon-c32.c &

tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-neonv8-c8.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-neonv8-c16.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-neonv8-c24.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -D ARMV8=1 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-neonv8-c32.c &

tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-rndnu-neon-c8.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-rndnu-neon-c16.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-rndnu-neon-c24.c &
tools/xngen src/qs8-gavgpool/multipass-neon.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=RNDNU -D DATATYPE=QU8 -D ARMV8=0 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-rndnu-neon-c32.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-wasmsimd-c8.c &
tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-wasmsimd-c16.c &
tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-wasmsimd-c24.c &
tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-wasmsimd-c32.c &

tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-wasmsimd-c8.c &
tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-wasmsimd-c16.c &
tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-wasmsimd-c24.c &
tools/xngen src/qs8-gavgpool/unipass-wasmsimd.c.in -D ROW_TILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-wasmsimd-c32.c &

tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-wasmsimd-c8.c &
tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-wasmsimd-c16.c &
tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-wasmsimd-c24.c &
tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-wasmsimd-c32.c &

tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-wasmsimd-c8.c &
tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-wasmsimd-c16.c &
tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-wasmsimd-c24.c &
tools/xngen src/qs8-gavgpool/multipass-wasmsimd.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=32 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-wasmsimd-c32.c &

################################### x86 SSE ###################################
tools/xngen src/qs8-gavgpool/unipass-sse2.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-sse2-c8.c &
tools/xngen src/qs8-gavgpool/unipass-sse2.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-sse2-c16.c &
tools/xngen src/qs8-gavgpool/unipass-sse2.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-sse2-c24.c &

tools/xngen src/qs8-gavgpool/unipass-sse4.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-sse41-c8.c &
tools/xngen src/qs8-gavgpool/unipass-sse4.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-sse41-c16.c &
tools/xngen src/qs8-gavgpool/unipass-sse4.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7x-minmax-fp32-sse41-c24.c &

tools/xngen src/qs8-gavgpool/unipass-sse2.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-sse2-c8.c &
tools/xngen src/qs8-gavgpool/unipass-sse2.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-sse2-c16.c &
tools/xngen src/qs8-gavgpool/unipass-sse2.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-sse2-c24.c &

tools/xngen src/qs8-gavgpool/unipass-sse4.c.in -D ROW_TILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-sse41-c8.c &
tools/xngen src/qs8-gavgpool/unipass-sse4.c.in -D ROW_TILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-sse41-c16.c &
tools/xngen src/qs8-gavgpool/unipass-sse4.c.in -D ROW_TILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7x-minmax-fp32-sse41-c24.c &

tools/xngen src/qs8-gavgpool/multipass-sse2.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-sse2-c8.c &
tools/xngen src/qs8-gavgpool/multipass-sse2.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-sse2-c16.c &
tools/xngen src/qs8-gavgpool/multipass-sse2.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-sse2-c24.c &

tools/xngen src/qs8-gavgpool/multipass-sse4.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-sse41-c8.c &
tools/xngen src/qs8-gavgpool/multipass-sse4.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-sse41-c16.c &
tools/xngen src/qs8-gavgpool/multipass-sse4.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-gavgpool/gen/qs8-gavgpool-7p7x-minmax-fp32-sse41-c24.c &

tools/xngen src/qs8-gavgpool/multipass-sse2.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-sse2-c8.c &
tools/xngen src/qs8-gavgpool/multipass-sse2.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-sse2-c16.c &
tools/xngen src/qs8-gavgpool/multipass-sse2.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-sse2-c24.c &

tools/xngen src/qs8-gavgpool/multipass-sse4.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-sse41-c8.c &
tools/xngen src/qs8-gavgpool/multipass-sse4.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-sse41-c16.c &
tools/xngen src/qs8-gavgpool/multipass-sse4.c.in -D ROW_TILE=7 -D ROW_SUBTILE=7 -D CHANNEL_TILE=24 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-gavgpool/gen/qu8-gavgpool-7p7x-minmax-fp32-sse41-c24.c &

wait
