# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for sse2fma
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_SSE2FMA_MICROKERNEL_SRCS
  src/f32-vapproxgelu/gen/f32-vapproxgelu-sse2fma-rational-12-10-div.c
  src/f32-vcos/gen/f32-vcos-sse2fma-rational-5-4-div.c
  src/f32-vexp/gen/f32-vexp-sse2fma-rational-3-2-div.c
  src/f32-vgelu/gen/f32-vgelu-sse2fma-rational-12-10-div.c
  src/f32-vhswish/gen/f32-vhswish-sse2fma.c
  src/f32-vlog/gen/f32-vlog-sse2fma-rational-3-3-div.c
  src/f32-vsin/gen/f32-vsin-sse2fma-rational-5-4-div.c
  src/f32-vtanh/gen/f32-vtanh-sse2fma-rational-9-8-div.c)

SET(NON_PROD_SSE2FMA_MICROKERNEL_SRCS)

SET(ALL_SSE2FMA_MICROKERNEL_SRCS ${PROD_SSE2FMA_MICROKERNEL_SRCS} + ${NON_PROD_SSE2FMA_MICROKERNEL_SRCS})
