# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for hmx
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_HMX_MICROKERNEL_SRCS
  src/f16-gemm/gen/f16-gemm-32x64c2-minmax-hmx-pingpong.c)

SET(NON_PROD_HMX_MICROKERNEL_SRCS
  src/f16-gemm/gen/f16-gemm-32x64c2-minmax-hmx.c)

SET(ALL_HMX_MICROKERNEL_SRCS ${PROD_HMX_MICROKERNEL_SRCS} ${NON_PROD_HMX_MICROKERNEL_SRCS})
