// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

XNN_UKERNEL(0, xnn_s8_rdmax_ukernel_2p2x__scalar_c2, 2, 2, false, int8_t, int8_t, void*, NULL)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_s8_rdmax_ukernel_2p2x__neon_c32, 2, 32, false, int8_t, int8_t, void*, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_x86_sse4_1, xnn_s8_rdmax_ukernel_2p2x__sse41_c32, 2, 32, false, int8_t, int8_t, void*, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(0, xnn_s8_rdmax_ukernel_2p2x__wasmsimd_c32, 2, 32, false, int8_t, int8_t, void*, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
XNN_UKERNEL(0, xnn_s8_rdmax_ukernel_2p2x__hvx_c128, 2, 128, false, int8_t, int8_t, void*, NULL)
#endif  // XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
