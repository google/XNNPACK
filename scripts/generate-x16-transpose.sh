#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### SSE2 ###################################
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=DEC SIZE=16 -o src/x16-transpose/gen/8x8-reuse-dec-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=SWITCH SIZE=16 -o src/x16-transpose/gen/8x8-reuse-switch-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=MULTI SIZE=16 -o src/x16-transpose/gen/8x8-reuse-multi-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=SWITCH SIZE=16 -o src/x16-transpose/gen/8x8-multi-switch-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=DEC SIZE=16 -o src/x16-transpose/gen/8x8-multi-dec-sse2.c &

################################## Unit tests #################################
tools/generate-transpose-test.py --spec test/x16-transpose.yaml --output=test/x16-transpose.cc &

wait
