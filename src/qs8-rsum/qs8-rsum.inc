// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__scalar_u1, 1, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__scalar_u2, 2, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__scalar_u4, 4, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_rsum_ukernel__neon_u16, 16, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_rsum_ukernel__neon_u32_acc2, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_rsum_ukernel__neon_u64_acc2, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_qs8_rsum_ukernel__neon_u64_acc4, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_dot, xnn_qs8_rsum_ukernel__neondot_u16, 16, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_dot, xnn_qs8_rsum_ukernel__neondot_u32_acc2, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_dot, xnn_qs8_rsum_ukernel__neondot_u64_acc2, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_dot, xnn_qs8_rsum_ukernel__neondot_u64_acc4, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_x86_ssse3, xnn_qs8_rsum_ukernel__ssse3_u16, 16, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_ssse3, xnn_qs8_rsum_ukernel__ssse3_u32_acc2, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_ssse3, xnn_qs8_rsum_ukernel__ssse3_u64_acc2, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_ssse3, xnn_qs8_rsum_ukernel__ssse3_u64_acc4, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_rsum_ukernel__avx2_u32, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_rsum_ukernel__avx2_u64_acc2, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_rsum_ukernel__avx2_u128_acc2, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qs8_rsum_ukernel__avx2_u128_acc4, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_rsum_ukernel__avxvnni_u32, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_rsum_ukernel__avxvnni_u64_acc2, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_rsum_ukernel__avxvnni_u128_acc2, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avxvnni, xnn_qs8_rsum_ukernel__avxvnni_u128_acc4, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ENABLE_AVXVNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_rsum_ukernel__avx256vnni_u32, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_rsum_ukernel__avx256vnni_u64_acc2, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_rsum_ukernel__avx256vnni_u128_acc2, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx256vnni, xnn_qs8_rsum_ukernel__avx256vnni_u128_acc4, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ENABLE_AVX256VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx256skx, xnn_qs8_rsum_ukernel__avx256skx_u32, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx256skx, xnn_qs8_rsum_ukernel__avx256skx_u64_acc2, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx256skx, xnn_qs8_rsum_ukernel__avx256skx_u128_acc2, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx256skx, xnn_qs8_rsum_ukernel__avx256skx_u128_acc4, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_qs8_rsum_ukernel__avx512skx_u64, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_qs8_rsum_ukernel__avx512skx_u128_acc2, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_qs8_rsum_ukernel__avx512skx_u256_acc2, 256, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_qs8_rsum_ukernel__avx512skx_u256_acc4, 256, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512vnni, xnn_qs8_rsum_ukernel__avx512vnni_u64, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512vnni, xnn_qs8_rsum_ukernel__avx512vnni_u128_acc2, 128, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512vnni, xnn_qs8_rsum_ukernel__avx512vnni_u256_acc2, 256, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512vnni, xnn_qs8_rsum_ukernel__avx512vnni_u256_acc4, 256, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ENABLE_AVX512VNNI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__wasmrelaxedsimd_u16, 16, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__wasmrelaxedsimd_u32_acc2, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__wasmrelaxedsimd_u64_acc2, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__wasmrelaxedsimd_u64_acc4, 64, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__wasmsimd_u8, 8, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__wasmsimd_u16_acc2, 16, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__wasmsimd_u32_acc2, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_qs8_rsum_ukernel__wasmsimd_u32_acc4, 32, false, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qs8_rsum_ukernel__rvv_u1v, 1, true, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_qs8_rsum_ukernel__rvv_u2v, 2, true, int8_t, int32_t, struct xnn_qs8_rsum_params, NULL)
#endif  // XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)

