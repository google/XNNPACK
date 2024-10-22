# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for rvvfp16arith
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_RVVFP16ARITH_MICROKERNEL_SRCS)

SET(NON_PROD_RVVFP16ARITH_MICROKERNEL_SRCS
  src/f16-vclamp/gen/f16-vclamp-rvvfp16arith-u1v.c
  src/f16-vclamp/gen/f16-vclamp-rvvfp16arith-u2v.c
  src/f16-vclamp/gen/f16-vclamp-rvvfp16arith-u4v.c
  src/f16-vclamp/gen/f16-vclamp-rvvfp16arith-u8v.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-rvvfp16arith-u1v.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-rvvfp16arith-u2v.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-rvvfp16arith-u4v.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-rvvfp16arith-u8v.c)

SET(ALL_RVVFP16ARITH_MICROKERNEL_SRCS ${PROD_RVVFP16ARITH_MICROKERNEL_SRCS} + ${NON_PROD_RVVFP16ARITH_MICROKERNEL_SRCS})
