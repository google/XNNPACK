// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_rminmax_ukernel__neonfp16arith_u8, 8, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_rminmax_ukernel__neonfp16arith_u16_acc2, 16, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_rminmax_ukernel__neonfp16arith_u24_acc3, 24, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_rminmax_ukernel__neonfp16arith_u32_acc2, 32, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_rminmax_ukernel__neonfp16arith_u32_acc4, 32, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_rminmax_ukernel__avx512fp16_u32, 32, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_rminmax_ukernel__avx512fp16_u64_acc2, 64, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_rminmax_ukernel__avx512fp16_u96_acc3, 96, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_rminmax_ukernel__avx512fp16_u128_acc2, 128, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512fp16, xnn_f16_rminmax_ukernel__avx512fp16_u128_acc4, 128, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
#endif  // XNN_ENABLE_AVX512FP16 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_rminmax_ukernel__avx512skx_u16, 16, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_rminmax_ukernel__avx512skx_u32_acc2, 32, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_rminmax_ukernel__avx512skx_u48_acc3, 48, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_rminmax_ukernel__avx512skx_u64_acc2, 64, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_f16_rminmax_ukernel__avx512skx_u64_acc4, 64, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

XNN_UKERNEL(xnn_arch_none, xnn_f16_rminmax_ukernel__scalar_u1, 1, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f16_rminmax_ukernel__scalar_u2_acc2, 2, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f16_rminmax_ukernel__scalar_u3_acc3, 3, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f16_rminmax_ukernel__scalar_u4_acc2, 4, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_none, xnn_f16_rminmax_ukernel__scalar_u4_acc4, 4, false, xnn_float16, xnn_float16, struct xnn_f16_default_params, ((struct xnn_init_f16_default_params_fn) NULL))
