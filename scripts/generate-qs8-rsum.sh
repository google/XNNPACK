#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-rsum/scalar.c.in -D CHANNEL_TILE=1 -D ACCUMULATORS=1 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D WASM=0 -o src/qs8-rsum/gen/qs8-rdsum-minmax-fp32-scalar-imagic-u1-acc1.c &
tools/xngen src/qs8-rsum/scalar.c.in -D CHANNEL_TILE=2 -D ACCUMULATORS=1 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D WASM=0 -o src/qs8-rsum/gen/qs8-rdsum-minmax-fp32-scalar-imagic-u2-acc1.c &
tools/xngen src/qs8-rsum/scalar.c.in -D CHANNEL_TILE=4 -D ACCUMULATORS=1 -D REQUANTIZATION=FP32 -D VARIANT=IMAGIC -D WASM=0 -o src/qs8-rsum/gen/qs8-rdsum-minmax-fp32-scalar-imagic-u4-acc1.c &
