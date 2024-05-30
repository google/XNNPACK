#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-rdsum/scalar.c.in -D ACCUMULATORS=7 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-minmax-fp32-scalar-u1-acc1.c &

################################## ARM NEON ###################################
tools/xngen src/qs8-rdsum/neon.c.in -D CHANNELS=16  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-neon-c16.c &
tools/xngen src/qs8-rdsum/neon.c.in -D CHANNELS=32  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-neon-c32.c &
tools/xngen src/qs8-rdsum/neon.c.in -D CHANNELS=64  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-neon-c64.c &

wait
