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
  src/pf16-gemm/pf16-gemm-32x32-minmax-neonsme2.c
  src/pf32-gemm/pf32-gemm-1x32-minmax-neonsme2.c
  src/pf32-gemm/pf32-gemm-32x32-minmax-neonsme2.c
  src/pqs8-qc8w-gemm/pqs8-qc8w-gemm-1x32c4-minmax-neonsme2.c
  src/pqs8-qc8w-gemm/pqs8-qc8w-gemm-32x32c4-minmax-neonsme2.c
  src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-minmax-1x128c4-neonsme2.c
  src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-minmax-32x128c4-neonsme2.c
  src/x8-pack-lh/x8--packlh-neonsme2.c
  src/x16-pack-lh/x16-packlh-neonsme2.c
  src/x32-pack-lh/x32-packlh-neonsme2.c)

SET(NON_PROD_NEONSME2_MICROKERNEL_SRCS)

SET(ALL_NEONSME2_MICROKERNEL_SRCS ${PROD_NEONSME2_MICROKERNEL_SRCS} + ${NON_PROD_NEONSME2_MICROKERNEL_SRCS})
