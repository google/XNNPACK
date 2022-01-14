#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### SSE2 ###################################
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=DEC SIZE=64 -o src/x64-transpose/gen/2x2-reuse-dec-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=SWITCH SIZE=64 -o src/x64-transpose/gen/2x2-reuse-switch-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=REUSE OUT_PTRS=MULTI SIZE=64 -o src/x64-transpose/gen/2x2-reuse-multi-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=SWITCH SIZE=64 -o src/x64-transpose/gen/2x2-multi-switch-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=MULTI SIZE=64 -o src/x64-transpose/gen/2x2-multi-multi-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D IN_PTRS=MULTI OUT_PTRS=DEC SIZE=64 -o src/x64-transpose/gen/2x2-multi-dec-sse2.c &

################################## Unit tests #################################
tools/generate-transpose-test.py --spec test/x64-transpose.yaml --output=test/x64-transpose.cc &

wait
