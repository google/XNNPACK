#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-rdsum/scalar.c.in -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-scalar.c &

wait
