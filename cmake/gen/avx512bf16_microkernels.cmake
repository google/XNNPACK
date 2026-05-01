# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for avx512bf16
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_AVX512BF16_MICROKERNEL_SRCS
  src/f32-bf16-vcvt/gen/f32-bf16-vcvt-avx512bf16-u16.c)

SET(NON_PROD_AVX512BF16_MICROKERNEL_SRCS
  src/f32-bf16-vcvt/gen/f32-bf16-vcvt-avx512bf16-u32.c)

SET(ALL_AVX512BF16_MICROKERNEL_SRCS ${PROD_AVX512BF16_MICROKERNEL_SRCS} ${NON_PROD_AVX512BF16_MICROKERNEL_SRCS})
