// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

#ifndef XNN_QUANTIZED
#define XNN_QUANTIZED(T) T
#define XNN_DEFINED_QUANTIZED
#endif


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qu8_vlrelu_ukernel__neon_u8, 8, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qu8_vlrelu_ukernel__neon_u16, 16, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_qu8_vlrelu_ukernel__neon_u32, 32, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_qu8_vlrelu_ukernel__rvv_u1v, 1, true, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_qu8_vlrelu_ukernel__rvv_u2v, 2, true, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
#endif  // XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__sse2_u16, 16, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__sse2_u32, 32, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_ssse3, xnn_qu8_vlrelu_ukernel__ssse3_u16, 16, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_ssse3, xnn_qu8_vlrelu_ukernel__ssse3_u32, 32, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qu8_vlrelu_ukernel__sse41_u8, 8, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qu8_vlrelu_ukernel__sse41_u16, 16, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse4_1, xnn_qu8_vlrelu_ukernel__sse41_u32, 32, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qu8_vlrelu_ukernel__avx_u8, 8, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qu8_vlrelu_ukernel__avx_u16, 16, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx, xnn_qu8_vlrelu_ukernel__avx_u32, 32, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qu8_vlrelu_ukernel__avx2_u16, 16, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qu8_vlrelu_ukernel__avx2_u32, 32, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_qu8_vlrelu_ukernel__avx2_u64, 64, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__wasmsimd_arm_u16, 16, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__wasmsimd_arm_u32, 32, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__wasmsimd_x86_u8, 8, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__wasmsimd_x86_u16, 16, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__wasmsimd_x86_u32, 32, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16, 16, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32, 32, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8, 8, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16, 16, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32, 32, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_v6, xnn_qu8_vlrelu_ukernel__armsimd32_u4, 4, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_v6, xnn_qu8_vlrelu_ukernel__armsimd32_u8, 8, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
#endif  // XNN_ARCH_ARM

XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__scalar_select_u1, 1, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__scalar_select_u2, 2, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__scalar_select_u4, 4, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__scalar_andxor_u1, 1, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__scalar_andxor_u2, 2, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_qu8_vlrelu_ukernel__scalar_andxor_u4, 4, false, XNN_QUANTIZED(uint8_t), struct xnn_qu8_lrelu_params, xnn_init_qu8_lrelu_scalar_params)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

#ifdef XNN_DEFINED_QUANTIZED
#undef XNN_DEFINED_QUANTIZED
#undef XNN_QUANTIZED
#endif

