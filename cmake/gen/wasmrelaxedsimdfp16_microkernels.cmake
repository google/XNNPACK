# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for wasmrelaxedsimdfp16
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_WASMRELAXEDSIMDFP16_MICROKERNEL_SRCS
  src/f16-avgpool/gen/f16-avgpool-9p-minmax-wasmrelaxedsimdfp16-u8.c
  src/f16-dwconv/gen/f16-dwconv-9p8c-minmax-wasmrelaxedsimd.c
  src/f16-dwconv/gen/f16-dwconv-25p8c-minmax-wasmrelaxedsimd-acc2.c
  src/f16-rminmax/gen/f16-rmax-wasmrelaxedsimdfp16-u32-acc2.c
  src/f16-vapproxgelu/gen/f16-vapproxgelu-wasmrelaxedsimd-rational-6-4-div.c
  src/f16-vcos/gen/f16-vcos-wasmrelaxedsimd-poly-3.c
  src/f16-vexp/gen/f16-vexp-wasmrelaxedsimd-poly-3.c
  src/f16-vlog/gen/f16-vlog-wasmrelaxedsimd-rational-1-3-div.c
  src/f16-vsin/gen/f16-vsin-wasmrelaxedsimd-poly-3.c
  src/f16-vsqrt/gen/f16-vsqrt-wasmrelaxedsimd-sqrt.c
  src/f16-vtanh/gen/f16-vtanh-wasmrelaxedsimd-rational-5-4-div.c)

SET(NON_PROD_WASMRELAXEDSIMDFP16_MICROKERNEL_SRCS
  src/f16-gemm/gen/f16-gemm-1x8-minmax-wasmrelaxedsimd-splat.c
  src/f16-gemm/gen/f16-gemm-1x16-minmax-wasmrelaxedsimd-splat.c
  src/f16-gemm/gen/f16-gemm-4x8-minmax-wasmrelaxedsimd-splat.c
  src/f16-gemm/gen/f16-gemm-4x16-minmax-wasmrelaxedsimd-splat.c
  src/f16-gemm/gen/f16-gemm-6x8-minmax-wasmrelaxedsimd-splat.c
  src/f16-gemm/gen/f16-gemm-6x16-minmax-wasmrelaxedsimd-splat.c
  src/f16-gemm/gen/f16-gemm-8x8-minmax-wasmrelaxedsimd-splat.c
  src/f16-gemm/gen/f16-gemm-8x16-minmax-wasmrelaxedsimd-splat.c
  src/f16-igemm/gen/f16-igemm-1x8-minmax-wasmrelaxedsimd-splat.c
  src/f16-igemm/gen/f16-igemm-1x16-minmax-wasmrelaxedsimd-splat.c
  src/f16-igemm/gen/f16-igemm-4x8-minmax-wasmrelaxedsimd-splat.c
  src/f16-igemm/gen/f16-igemm-4x16-minmax-wasmrelaxedsimd-splat.c
  src/f16-igemm/gen/f16-igemm-6x8-minmax-wasmrelaxedsimd-splat.c
  src/f16-igemm/gen/f16-igemm-6x16-minmax-wasmrelaxedsimd-splat.c
  src/f16-igemm/gen/f16-igemm-8x8-minmax-wasmrelaxedsimd-splat.c
  src/f16-igemm/gen/f16-igemm-8x16-minmax-wasmrelaxedsimd-splat.c
  src/f16-rminmax/gen/f16-rmax-wasmrelaxedsimdfp16-u8.c
  src/f16-rminmax/gen/f16-rmax-wasmrelaxedsimdfp16-u16-acc2.c
  src/f16-rminmax/gen/f16-rmax-wasmrelaxedsimdfp16-u24-acc3.c
  src/f16-rminmax/gen/f16-rmax-wasmrelaxedsimdfp16-u32-acc4.c
  src/f16-vcos/gen/f16-vcos-wasmrelaxedsimd-rational-3-2-div.c
  src/f16-vlog/gen/f16-vlog-wasmrelaxedsimd-rational-1-3-nr.c
  src/f16-vsin/gen/f16-vsin-wasmrelaxedsimd-rational-3-2-div.c
  src/f16-vtanh/gen/f16-vtanh-wasmrelaxedsimd-rational-5-4-nr.c)

SET(ALL_WASMRELAXEDSIMDFP16_MICROKERNEL_SRCS ${PROD_WASMRELAXEDSIMDFP16_MICROKERNEL_SRCS} ${NON_PROD_WASMRELAXEDSIMDFP16_MICROKERNEL_SRCS})
