// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// clang-format off

#ifndef XNN_QUANTIZED
#define XNN_QUANTIZED(T) T
#define XNN_DEFINED_QUANTIZED
#endif


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_vlrelu_ukernel__neon_u8, 8, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_vlrelu_ukernel__neon_u16, 16, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_vlrelu_ukernel__neon_u32, 32, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qs8_vlrelu_ukernel__rvv_u1v, 1, true, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qs8_vlrelu_ukernel__rvv_u2v, 2, true, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__sse2_u16, 16, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__sse2_u32, 32, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_ssse3, xnn_qs8_vlrelu_ukernel__ssse3_u16, 16, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_ssse3, xnn_qs8_vlrelu_ukernel__ssse3_u32, 32, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qs8_vlrelu_ukernel__sse41_u8, 8, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qs8_vlrelu_ukernel__sse41_u16, 16, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qs8_vlrelu_ukernel__sse41_u32, 32, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qs8_vlrelu_ukernel__avx_u8, 8, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qs8_vlrelu_ukernel__avx_u16, 16, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qs8_vlrelu_ukernel__avx_u32, 32, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_vlrelu_ukernel__avx2_u16, 16, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_vlrelu_ukernel__avx2_u32, 32, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_vlrelu_ukernel__avx2_u64, 64, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16, 16, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32, 32, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8, 8, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16, 16, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32, 32, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16, 16, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32, 32, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8, 8, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16, 16, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32, 32, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM
XNN_UKERNEL(xnn_arch_arm_v6, xnn_qs8_vlrelu_ukernel__armsimd32_u4, 4, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(xnn_arch_arm_v6, xnn_qs8_vlrelu_ukernel__armsimd32_u8, 8, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
#endif  // XNN_ARCH_ARM

XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__scalar_select_u1, 1, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__scalar_select_u2, 2, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__scalar_select_u4, 4, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__scalar_andxor_u1, 1, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__scalar_andxor_u2, 2, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)
XNN_UKERNEL(0, xnn_qs8_vlrelu_ukernel__scalar_andxor_u4, 4, false, XNN_QUANTIZED(int8_t), struct xnn_qs8_lrelu_params, xnn_init_qs8_lrelu_scalar_params)

#ifdef XNN_DEFINED_QUANTIZED
#undef XNN_DEFINED_QUANTIZED
#undef XNN_QUANTIZED
#endif

