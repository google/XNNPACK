# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for rvvfp16arith
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_RVVFP16ARITH_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/f16-f32-vcvt-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vadd-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vaddc-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vdiv-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vdivc-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vmax-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vmaxc-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vmin-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vminc-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vmul-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vmulc-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vprelu-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vpreluc-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vrdivc-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vrpreluc-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vrsubc-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vsqrdiff-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vsqrdiffc-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vsub-rvvfp16arith-u8v.c
  src/f16-vbinary/gen/f16-vsubc-rvvfp16arith-u8v.c
  src/f16-vclamp/gen/f16-vclamp-rvvfp16arith-u8v.c
  src/f16-vunary/gen/f16-vabs-rvvfp16arith-u8v.c
  src/f16-vunary/gen/f16-vneg-rvvfp16arith-u8v.c
  src/f16-vunary/gen/f16-vsqr-rvvfp16arith-u8v.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-rvvfp16arith-u8v.c)

SET(NON_PROD_RVVFP16ARITH_MICROKERNEL_SRCS
  src/f16-f32-vcvt/gen/f16-f32-vcvt-rvvfp16arith-u1v.c
  src/f16-f32-vcvt/gen/f16-f32-vcvt-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vadd-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vadd-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vadd-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vaddc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vaddc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vaddc-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vdiv-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vdiv-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vdiv-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vdivc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vdivc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vdivc-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vmax-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vmax-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vmax-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vmaxc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vmaxc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vmaxc-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vmin-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vmin-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vmin-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vminc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vminc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vminc-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vmul-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vmul-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vmul-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vmulc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vmulc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vmulc-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vprelu-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vprelu-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vprelu-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vpreluc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vpreluc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vpreluc-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vrdivc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vrdivc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vrdivc-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vrpreluc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vrpreluc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vrpreluc-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vrsubc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vrsubc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vrsubc-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vsqrdiff-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vsqrdiff-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vsqrdiff-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vsqrdiffc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vsqrdiffc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vsqrdiffc-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vsub-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vsub-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vsub-rvvfp16arith-u4v.c
  src/f16-vbinary/gen/f16-vsubc-rvvfp16arith-u1v.c
  src/f16-vbinary/gen/f16-vsubc-rvvfp16arith-u2v.c
  src/f16-vbinary/gen/f16-vsubc-rvvfp16arith-u4v.c
  src/f16-vclamp/gen/f16-vclamp-rvvfp16arith-u1v.c
  src/f16-vclamp/gen/f16-vclamp-rvvfp16arith-u2v.c
  src/f16-vclamp/gen/f16-vclamp-rvvfp16arith-u4v.c
  src/f16-vunary/gen/f16-vabs-rvvfp16arith-u1v.c
  src/f16-vunary/gen/f16-vabs-rvvfp16arith-u2v.c
  src/f16-vunary/gen/f16-vabs-rvvfp16arith-u4v.c
  src/f16-vunary/gen/f16-vneg-rvvfp16arith-u1v.c
  src/f16-vunary/gen/f16-vneg-rvvfp16arith-u2v.c
  src/f16-vunary/gen/f16-vneg-rvvfp16arith-u4v.c
  src/f16-vunary/gen/f16-vsqr-rvvfp16arith-u1v.c
  src/f16-vunary/gen/f16-vsqr-rvvfp16arith-u2v.c
  src/f16-vunary/gen/f16-vsqr-rvvfp16arith-u4v.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-rvvfp16arith-u1v.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-rvvfp16arith-u2v.c
  src/f32-f16-vcvt/gen/f32-f16-vcvt-rvvfp16arith-u4v.c)

SET(ALL_RVVFP16ARITH_MICROKERNEL_SRCS ${PROD_RVVFP16ARITH_MICROKERNEL_SRCS} + ${NON_PROD_RVVFP16ARITH_MICROKERNEL_SRCS})
