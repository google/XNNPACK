# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for sve2
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_SVE2_MICROKERNEL_SRCS
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-sve2-lane-ld128.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-8x8-minmax-sve2-lane-ld128.c)

SET(NON_PROD_SVE2_MICROKERNEL_SRCS
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x16-minmax-sve2-lane-ld128.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-sve2-lane-ld128.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x16-minmax-sve2-lane-ld128.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-sve2-lane-ld128.c)

SET(ALL_SVE2_MICROKERNEL_SRCS ${PROD_SVE2_MICROKERNEL_SRCS} ${NON_PROD_SVE2_MICROKERNEL_SRCS})
