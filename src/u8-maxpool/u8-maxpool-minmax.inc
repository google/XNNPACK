// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_u8_maxpool_minmax_ukernel_9p__neon_u16, 16, 9, uint8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_SSE2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_sse2, xnn_u8_maxpool_minmax_ukernel_9p__sse2_u16, 16, 9, uint8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
#endif  // XNN_ENABLE_SSE2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_u8_maxpool_minmax_ukernel_9p__wasmsimd_u16, 16, 9, uint8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_UKERNEL(xnn_arch_none, xnn_u8_maxpool_minmax_ukernel_9p__scalar_u1, 1, 9, uint8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)


