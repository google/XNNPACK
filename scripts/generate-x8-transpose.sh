#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### SSE2 ###################################
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=DEC SIZE=8 -o src/x8-transpose/gen/16x16-reuse-dec-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=SWITCH SIZE=8 -o src/x8-transpose/gen/16x16-reuse-switch-sse2.c &

################################## Unit tests #################################
tools/generate-transpose-test.py --spec test/x8-transpose.yaml --output=test/x8-transpose.cc &

wait
