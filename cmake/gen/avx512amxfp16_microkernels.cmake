# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for avx512amxfp16
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_AVX512AMXFP16_MICROKERNEL_SRCS
  src/f16-gemm/gen/f16-gemm-1x64c2-minmax-avx512amxfp16.c
  src/f16-gemm/gen/f16-gemm-16x64c2-minmax-avx512amxfp16.c
  src/f16-igemm/gen/f16-igemm-1x64c2-minmax-avx512amxfp16.c
  src/f16-igemm/gen/f16-igemm-16x64c2-minmax-avx512amxfp16.c)

SET(NON_PROD_AVX512AMXFP16_MICROKERNEL_SRCS
  src/f16-gemm/gen/f16-gemm-1x16c2-minmax-avx512amxfp16.c
  src/f16-gemm/gen/f16-gemm-1x32c2-minmax-avx512amxfp16.c
  src/f16-gemm/gen/f16-gemm-16x16c2-minmax-avx512amxfp16-prfm.c
  src/f16-gemm/gen/f16-gemm-16x16c2-minmax-avx512amxfp16.c
  src/f16-gemm/gen/f16-gemm-16x32c2-minmax-avx512amxfp16-prfm.c
  src/f16-gemm/gen/f16-gemm-16x32c2-minmax-avx512amxfp16.c
  src/f16-gemm/gen/f16-gemm-16x64c2-minmax-avx512amxfp16-prfm.c
  src/f16-igemm/gen/f16-igemm-1x16c2-minmax-avx512amxfp16.c
  src/f16-igemm/gen/f16-igemm-1x32c2-minmax-avx512amxfp16.c
  src/f16-igemm/gen/f16-igemm-16x16c2-minmax-avx512amxfp16-prfm.c
  src/f16-igemm/gen/f16-igemm-16x16c2-minmax-avx512amxfp16.c
  src/f16-igemm/gen/f16-igemm-16x32c2-minmax-avx512amxfp16-prfm.c
  src/f16-igemm/gen/f16-igemm-16x32c2-minmax-avx512amxfp16.c
  src/f16-igemm/gen/f16-igemm-16x64c2-minmax-avx512amxfp16-prfm.c)

SET(ALL_AVX512AMXFP16_MICROKERNEL_SRCS ${PROD_AVX512AMXFP16_MICROKERNEL_SRCS} ${NON_PROD_AVX512AMXFP16_MICROKERNEL_SRCS})
