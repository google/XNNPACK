# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for avxvnniint8
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_AVXVNNIINT8_MICROKERNEL_SRCS
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avxvnniint8-prfm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-5x8c8-minmax-fp32-avxvnniint8-prfm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-avxvnniint8-prfm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-5x8c8-minmax-fp32-avxvnniint8-prfm.c)

SET(NON_PROD_AVXVNNIINT8_MICROKERNEL_SRCS)

SET(ALL_AVXVNNIINT8_MICROKERNEL_SRCS ${PROD_AVXVNNIINT8_MICROKERNEL_SRCS} + ${NON_PROD_AVXVNNIINT8_MICROKERNEL_SRCS})
