# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for neonfp16
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_NEONFP16_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/f16-f32-vcvt-neonfp16-u16.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-neonfp16-u16.c)

SET(NON_PROD_NEONFP16_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/f16-f32-vcvt-neonfp16-u8.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-neonfp16-u8.c)

SET(ALL_NEONFP16_MICROKERNEL_SRCS ${PROD_NEONFP16_MICROKERNEL_SRCS} + ${NON_PROD_NEONFP16_MICROKERNEL_SRCS})
