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
  src/bf16-f32-igemm/gen/bf16-f32-igemm-1x32c2-minmax-avx512bf16-broadcast.c
  src/bf16-f32-igemm/gen/bf16-f32-igemm-7x32c2-minmax-avx512bf16-broadcast.c
  src/bf16-vbinary/gen/bf16-vadd-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vaddc-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vdiv-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vdivc-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vmax-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vmaxc-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vmin-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vminc-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vmul-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vmulc-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vprelu-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vpreluc-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vrdivc-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vrpreluc-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vrsubc-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vsqrdiff-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vsqrdiffc-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vsub-avx512bf16-u32.c
  src/bf16-vbinary/gen/bf16-vsubc-avx512bf16-u32.c
  src/f32-bf16-vcvt/gen/f32-bf16-vcvt-avx512bf16-u16.c)

SET(NON_PROD_AVX512BF16_MICROKERNEL_SRCS
  src/bf16-f32-igemm/gen/bf16-f32-igemm-1x16c2-minmax-avx512bf16-broadcast.c
  src/bf16-f32-igemm/gen/bf16-f32-igemm-4x16c2-minmax-avx512bf16-broadcast.c
  src/bf16-f32-igemm/gen/bf16-f32-igemm-4x32c2-minmax-avx512bf16-broadcast.c
  src/bf16-f32-igemm/gen/bf16-f32-igemm-5x16c2-minmax-avx512bf16-broadcast.c
  src/bf16-f32-igemm/gen/bf16-f32-igemm-5x32c2-minmax-avx512bf16-broadcast.c
  src/bf16-f32-igemm/gen/bf16-f32-igemm-6x16c2-minmax-avx512bf16-broadcast.c
  src/bf16-f32-igemm/gen/bf16-f32-igemm-6x32c2-minmax-avx512bf16-broadcast.c
  src/bf16-f32-igemm/gen/bf16-f32-igemm-7x16c2-minmax-avx512bf16-broadcast.c
  src/bf16-f32-igemm/gen/bf16-f32-igemm-8x16c2-minmax-avx512bf16-broadcast.c
  src/bf16-f32-igemm/gen/bf16-f32-igemm-8x32c2-minmax-avx512bf16-broadcast.c
  src/bf16-vbinary/gen/bf16-vadd-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vaddc-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vdiv-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vdivc-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vmax-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vmaxc-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vmin-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vminc-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vmul-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vmulc-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vprelu-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vpreluc-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vrdivc-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vrpreluc-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vrsubc-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vsqrdiff-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vsqrdiffc-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vsub-avx512bf16-u16.c
  src/bf16-vbinary/gen/bf16-vsubc-avx512bf16-u16.c
  src/f32-bf16-vcvt/gen/f32-bf16-vcvt-avx512bf16-u32.c)

SET(ALL_AVX512BF16_MICROKERNEL_SRCS ${PROD_AVX512BF16_MICROKERNEL_SRCS} ${NON_PROD_AVX512BF16_MICROKERNEL_SRCS})
