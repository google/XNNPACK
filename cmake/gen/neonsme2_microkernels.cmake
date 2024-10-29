# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for neonsme2
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_NEONSME2_MICROKERNEL_SRCS
  src/pf32-gemm/pf32-gemm-32x32-minmax-neonsme2.c
  src/x32-pack-lh/x32-packlh-neonsme2.c)

SET(NON_PROD_NEONSME2_MICROKERNEL_SRCS)

SET(ALL_NEONSME2_MICROKERNEL_SRCS ${PROD_NEONSME2_MICROKERNEL_SRCS} + ${NON_PROD_NEONSME2_MICROKERNEL_SRCS})
