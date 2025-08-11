// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_rdmax_ukernel_2p2x__neon_u32, 2, 32, false, float, float, void*, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_f32_rdmax_ukernel_2p2x__sse2_u32, 2, 32, false, float, float, void*, NULL)
XNN_UKERNEL(xnn_arch_x86_avx, xnn_f32_rdmax_ukernel_2p2x__avx_u32, 2, 32, false, float, float, void*, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_rdmax_ukernel_2p2x__avx512f_u32, 2, 32, false, float, float, void*, NULL)
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f32_rdmax_ukernel_2p2x__wasmsimd_u32, 2, 32, false, float, float, void*, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
XNN_UKERNEL(xnn_arch_hvx, xnn_f32_rdmax_ukernel_2p2x__hvx_u32, 2, 32, false, float, float, void*, NULL)
#endif  // XNN_ARCH_HEXAGON && XNN_ENABLE_HVX

XNN_UKERNEL(xnn_arch_none, xnn_f32_rdmax_ukernel_2p2x__scalar_u2, 2, 2, false, float, float, void*, NULL)
