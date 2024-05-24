# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for wasmsimd32
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(WASMSIMD32_ASM_MICROKERNEL_SRCS)

SET(WASMSIMD32_JIT_MICROKERNEL_SRCS
  src/f32-gemm/MRx8-wasmsimd32-x86-loadsplat.cc
  src/f32-gemm/MRx8-wasmsimd32-x86-splat.cc
  src/f32-gemm/MRx8s4-wasmsimd32-x86.cc
  src/f32-igemm/MRx8-wasmsimd32-x86-loadsplat.cc
  src/f32-igemm/MRx8-wasmsimd32-x86-splat.cc
  src/f32-igemm/MRx8s4-wasmsimd32-x86.cc)
