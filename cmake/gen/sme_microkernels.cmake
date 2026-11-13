# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for sme
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_SME_MICROKERNEL_SRCS
  src/x8-packq/x8-packq-f32qp8-sme.c)

SET(NON_PROD_SME_MICROKERNEL_SRCS
  src/pf32-qc4w-gemm/pf32-qc4w-gemm-1x32-minmax-sme.c
  src/pf32-qc4w-gemm/pf32-qc4w-gemm-32x32-minmax-sme.c
  src/pf32-qc8w-gemm/pf32-qc8w-gemm-1x32-minmax-sme.c
  src/pf32-qc8w-gemm/pf32-qc8w-gemm-32x32-minmax-sme.c
  src/x32-pack-lh/x32-packlh-sme.c)

SET(ALL_SME_MICROKERNEL_SRCS ${PROD_SME_MICROKERNEL_SRCS} ${NON_PROD_SME_MICROKERNEL_SRCS})
