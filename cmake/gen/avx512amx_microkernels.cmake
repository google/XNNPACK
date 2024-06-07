# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for avx512amx
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(ALL_AVX512AMX_MICROKERNEL_SRCS
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x64c4-minmax-avx512amx.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-7x64c4-minmax-avx512amx.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-16x64c4-minmax-avx512amx-prfm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-16x64c4-minmax-avx512amx.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x64c4-minmax-avx512amx.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-7x64c4-minmax-avx512amx.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-16x64c4-minmax-avx512amx-prfm.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-16x64c4-minmax-avx512amx.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x16c4-minmax-avx512amx.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x32c4-minmax-avx512amx.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x64c4-minmax-avx512amx.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x16c4-minmax-avx512amx.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x32c4-minmax-avx512amx.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x64c4-minmax-avx512amx.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x16c4-minmax-avx512amx-prfm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x16c4-minmax-avx512amx.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x32c4-minmax-avx512amx-prfm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x32c4-minmax-avx512amx.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x64c4-minmax-avx512amx-prfm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-16x64c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x16c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x32c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x64c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x16c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x32c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-7x64c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x16c4-minmax-avx512amx-prfm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x16c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x32c4-minmax-avx512amx-prfm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x32c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x64c4-minmax-avx512amx-prfm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x64c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x16c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x32c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x64c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x16c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x32c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-7x64c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x16c4-minmax-avx512amx-prfm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x16c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x32c4-minmax-avx512amx-prfm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x32c4-minmax-avx512amx.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x64c4-minmax-avx512amx-prfm.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x64c4-minmax-avx512amx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x16c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x32c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x64c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x16c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x32c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-7x64c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x16c4-minmax-fp32-avx512amx-prfm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x16c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x32c4-minmax-fp32-avx512amx-prfm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x32c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x64c4-minmax-fp32-avx512amx-prfm.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-16x64c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x16c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x32c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x64c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x16c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x32c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-7x64c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x16c4-minmax-fp32-avx512amx-prfm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x16c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x32c4-minmax-fp32-avx512amx-prfm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x32c4-minmax-fp32-avx512amx.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x64c4-minmax-fp32-avx512amx-prfm.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-16x64c4-minmax-fp32-avx512amx.c)
