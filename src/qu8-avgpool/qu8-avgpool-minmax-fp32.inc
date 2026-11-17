// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qu8_avgpool_minmax_fp32_ukernel_9p__neon_u16, 16, 9, uint8_t, struct xnn_qu8_avgpool_minmax_params, xnn_init_qu8_avgpool_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_SSE41 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_qu8_avgpool_minmax_fp32_ukernel_9p__sse41_u16, 16, 9, uint8_t, struct xnn_qu8_avgpool_minmax_params, xnn_init_qu8_avgpool_minmax_scalar_params)
#endif  // XNN_ENABLE_SSE41 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_avgpool_minmax_fp32_ukernel_9p__avx2_u32, 32, 9, uint8_t, struct xnn_qu8_avgpool_minmax_params, xnn_init_qu8_avgpool_minmax_scalar_params)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_qu8_avgpool_minmax_fp32_ukernel_9p__avx512skx_u32, 32, 9, uint8_t, struct xnn_qu8_avgpool_minmax_params, xnn_init_qu8_avgpool_minmax_scalar_params)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_qu8_avgpool_minmax_fp32_ukernel_9p__wasmsimd_u16, 16, 9, uint8_t, struct xnn_qu8_avgpool_minmax_params, xnn_init_qu8_avgpool_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
XNN_UKERNEL(xnn_arch_hvx, xnn_qu8_avgpool_minmax_fp32_ukernel_9p__hvx_u32, 32, 9, uint8_t, struct xnn_qu8_avgpool_minmax_params, xnn_init_qu8_avgpool_minmax_scalar_params)
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qu8_avgpool_minmax_fp32_ukernel_9p__rvv_u2v, (2*xnn_init_hardware_config()->vlenb/sizeof(uint8_t)), 9, uint8_t, struct xnn_qu8_avgpool_minmax_params, xnn_init_qu8_avgpool_minmax_scalar_params)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

XNN_UKERNEL(xnn_arch_none, xnn_qu8_avgpool_minmax_fp32_ukernel_9p__scalar_fmagic_u1, 1, 9, uint8_t, struct xnn_qu8_avgpool_minmax_params, xnn_init_qu8_avgpool_minmax_scalar_params)
