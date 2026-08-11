#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

tools/xngen src/bf16-vunary/scalar.c.in \
  -o src/bf16-vunary/gen/bf16-vunary-scalar.c &
scalar_pid=$!
tools/xngen src/bf16-vunary/neon-f32acc.c.in \
  -o src/bf16-vunary/gen/bf16-vunary-neon.c &
neon_pid=$!
tools/xngen src/bf16-vunary/avx512skx.c.in \
  -o src/bf16-vunary/gen/bf16-vunary-avx512skx.c &
avx512skx_pid=$!

status=0
wait "${scalar_pid}" || status=1
wait "${neon_pid}" || status=1
wait "${avx512skx_pid}" || status=1
exit "${status}"
