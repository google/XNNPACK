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
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_f32_vcvt_ukernel__neon_u8, 8, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_f32_vcvt_ukernel__neon_u16, 16, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_f32_vcvt_ukernel__neon_u24, 24, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_f32_vcvt_ukernel__neon_u32, 32, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__sse2_u8, 8, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__sse2_u16, 16, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__sse2_u24, 24, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__sse2_u32, 32, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qs8_f32_vcvt_ukernel__sse41_u8, 8, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qs8_f32_vcvt_ukernel__sse41_u16, 16, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qs8_f32_vcvt_ukernel__sse41_u24, 24, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qs8_f32_vcvt_ukernel__sse41_u32, 32, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qs8_f32_vcvt_ukernel__avx_u8, 8, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qs8_f32_vcvt_ukernel__avx_u16, 16, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qs8_f32_vcvt_ukernel__avx_u24, 24, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_qs8_f32_vcvt_ukernel__avx_u32, 32, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_f32_vcvt_ukernel__avx2_u8, 8, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_f32_vcvt_ukernel__avx2_u16, 16, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_f32_vcvt_ukernel__avx2_u24, 24, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_f32_vcvt_ukernel__avx2_u32, 32, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_qs8_f32_vcvt_ukernel__avx512skx_u16, 16, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_qs8_f32_vcvt_ukernel__avx512skx_u32, 32, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_qs8_f32_vcvt_ukernel__avx512skx_u48, 48, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_qs8_f32_vcvt_ukernel__avx512skx_u64, 64, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__wasmsimd_u8, 8, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__wasmsimd_u16, 16, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__wasmsimd_u24, 24, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__wasmsimd_u32, 32, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__rvv_u1v, 1, true, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__rvv_u2v, 2, true, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__scalar_u1, 1, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__scalar_u2, 2, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__scalar_u3, 3, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_f32_vcvt_ukernel__scalar_u4, 4, false, XNN_QUANTIZED(int8_t), float, struct xnn_qs8_f32_cvt_params, xnn_init_qs8_f32_cvt_scalar_params)


#ifdef XNN_DEFINED_QUANTIZED
#undef XNN_DEFINED_QUANTIZED
#undef XNN_QUANTIZED
#endif

