#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-rdsum/scalar.c.in -D ACCUMULATORS=7 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-minmax-fp32-scalar-u1-acc1.c &

wait
