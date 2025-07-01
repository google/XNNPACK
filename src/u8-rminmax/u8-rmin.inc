// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__scalar_u1, 1, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__scalar_u2_acc2, 2, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__scalar_u3_acc3, 3, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__scalar_u4_acc2, 4, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__scalar_u4_acc4, 4, false, uint8_t, uint8_t, void*, NULL)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_u8_rmin_ukernel__neon_u16, 16, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_u8_rmin_ukernel__neon_u32_acc2, 32, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_u8_rmin_ukernel__neon_u48_acc3, 48, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_u8_rmin_ukernel__neon_u64_acc2, 64, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_u8_rmin_ukernel__neon_u64_acc4, 64, false, uint8_t, uint8_t, void*, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__sse2_u16, 16, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__sse2_u32_acc2, 32, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__sse2_u48_acc3, 48, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__sse2_u64_acc2, 64, false, uint8_t, uint8_t, void*, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__sse2_u64_acc4, 64, false, uint8_t, uint8_t, void*, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_u8_rmin_ukernel__wasmsimd_u32_acc2, 32, false, uint8_t, uint8_t, void*, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
XNN_UKERNEL(xnn_arch_hvx, xnn_u8_rmin_ukernel__hvx_u256_acc2, 256, false, uint8_t, uint8_t, void*, NULL)
#endif  // XNN_ARCH_HEXAGON && XNN_ENABLE_HVX

