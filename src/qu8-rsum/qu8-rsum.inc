// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__scalar_u1, 1, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__scalar_u2, 2, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__scalar_u4, 4, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qu8_rsum_ukernel__neon_u16, 16, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qu8_rsum_ukernel__neon_u32_acc2, 32, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qu8_rsum_ukernel__neon_u64_acc2, 64, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qu8_rsum_ukernel__neon_u64_acc4, 64, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__sse2_u16, 16, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__sse2_u32_acc2, 32, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__sse2_u64_acc2, 64, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__sse2_u64_acc4, 64, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_rsum_ukernel__avx2_u32, 32, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_rsum_ukernel__avx2_u64_acc2, 64, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_rsum_ukernel__avx2_u128_acc2, 128, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_rsum_ukernel__avx2_u128_acc4, 128, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__wasmsimd_u8, 8, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__wasmsimd_u16_acc2, 16, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__wasmsimd_u32_acc2, 32, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_rsum_ukernel__wasmsimd_u32_acc4, 32, false, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qu8_rsum_ukernel__rvv_u1v, 1, true, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qu8_rsum_ukernel__rvv_u2v, 2, true, uint8_t, uint32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)

