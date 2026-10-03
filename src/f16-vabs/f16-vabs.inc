// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f16_vabs_ukernel__neon_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f16_vabs_ukernel__neon_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vabs_ukernel__neonfp16arith_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vabs_ukernel__neonfp16arith_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_SSE2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_sse2, xnn_f16_vabs_ukernel__sse2_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_sse2, xnn_f16_vabs_ukernel__sse2_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_SSE2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f16_vabs_ukernel__avx2_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f16_vabs_ukernel__avx2_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_vabs_ukernel__avx512skx_u32, 32, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_vabs_ukernel__avx512skx_u64, 64, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f16_vabs_ukernel__rvv_u1v, 1, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f16_vabs_ukernel__rvv_u2v, 2, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f16_vabs_ukernel__rvv_u4v, 4, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f16_vabs_ukernel__rvv_u8v, 8, true, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

#if XNN_ENABLE_RISCV_FP16_VECTOR && (XNN_ARCH_RISCV)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vabs_ukernel__rvvfp16arith_u1v, 1, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vabs_ukernel__rvvfp16arith_u2v, 2, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vabs_ukernel__rvvfp16arith_u4v, 4, true, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vabs_ukernel__rvvfp16arith_u8v, 8, true, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_RISCV_FP16_VECTOR && (XNN_ARCH_RISCV)

#if XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
XNN_UKERNEL(xnn_arch_hvx, xnn_f16_vabs_ukernel__hvx_u64, 64, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_hvx, xnn_f16_vabs_ukernel__hvx_u128, 128, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_HEXAGON && XNN_ENABLE_HVX

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f16_vabs_ukernel__wasmsimd_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vabs_ukernel__wasmsimd_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_UKERNEL(xnn_arch_none, xnn_f16_vabs_ukernel__scalar_u1, 1, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vabs_ukernel__scalar_u2, 2, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_f16_vabs_ukernel__scalar_u4, 4, false, xnn_float16, struct xnn_f16_default_params, NULL)
