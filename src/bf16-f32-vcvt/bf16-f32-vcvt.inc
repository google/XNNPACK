// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_f32_vcvt_ukernel__neon_u8, 8, false, xnn_bfloat16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_f32_vcvt_ukernel__neon_u16, 16, false, xnn_bfloat16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_f32_vcvt_ukernel__neon_u24, 24, false, xnn_bfloat16, float, void, NULL)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_bf16_f32_vcvt_ukernel__neon_u32, 32, false, xnn_bfloat16, float, void, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_f32_vcvt_ukernel__avx512skx_u16, 16, false, xnn_bfloat16, float, void, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_f32_vcvt_ukernel__avx512skx_u32, 32, false, xnn_bfloat16, float, void, NULL)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

XNN_UKERNEL(xnn_arch_none, xnn_bf16_f32_vcvt_ukernel__scalar_u1, 1, false, xnn_bfloat16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_bf16_f32_vcvt_ukernel__scalar_u2, 2, false, xnn_bfloat16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_bf16_f32_vcvt_ukernel__scalar_u3, 3, false, xnn_bfloat16, float, void, NULL)
XNN_UKERNEL(xnn_arch_none, xnn_bf16_f32_vcvt_ukernel__scalar_u4, 4, false, xnn_bfloat16, float, void, NULL)
