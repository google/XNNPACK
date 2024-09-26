# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for neondot_aarch64
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_NEONDOT_AARCH64_MICROKERNEL_SRCS
  src/qp8-f32-qb4w-gemm/qp8-f32-qb4w-gemm-minmax-1x4c16s2-aarch64-neondot.c
  src/qp8-f32-qb4w-gemm/qp8-f32-qb4w-gemm-minmax-1x8c16s2-aarch64-neondot.c
  src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-minmax-1x8c16s2-aarch64-neondot.c)

SET(NON_PROD_NEONDOT_AARCH64_MICROKERNEL_SRCS
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-aarch64-neondot-ld128.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c8-minmax-aarch64-neondot-ld128.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-aarch64-neondot-ld128.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c8-minmax-aarch64-neondot-ld128.c
  src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-minmax-1x4c16s2-aarch64-neondot.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-aarch64-neondot-ld128.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c8-minmax-fp32-aarch64-neondot-ld128.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-aarch64-neondot-ld128.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c8-minmax-fp32-aarch64-neondot-ld128.c)

SET(ALL_NEONDOT_AARCH64_MICROKERNEL_SRCS ${PROD_NEONDOT_AARCH64_MICROKERNEL_SRCS} + ${NON_PROD_NEONDOT_AARCH64_MICROKERNEL_SRCS})
