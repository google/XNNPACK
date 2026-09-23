// clang-format off
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// Arguments are:
// XNN_UKERNEL(arch, name, c_block, pipelined, cr, kr, datatype, weights_type,params_type, init_fn)
XNN_UKERNEL(0, xnn_bf16_f32_dwconv_minmax_ukernel_3p2c__scalar, 2, false, 2, 3, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(0, xnn_bf16_f32_dwconv_minmax_ukernel_4p2c__scalar, 2, false, 2, 4, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(0, xnn_bf16_f32_dwconv_minmax_ukernel_9p2c__scalar, 2, false, 2, 9, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(0, xnn_bf16_f32_dwconv_minmax_ukernel_25p2c__scalar_acc2, 2, false, 2, 25, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_bf16_f32_dwconv_minmax_ukernel_3p8c__neonfma, 8, false, 8, 3, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_bf16_f32_dwconv_minmax_ukernel_3p16c__neonfma, 16, false, 16, 3, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_bf16_f32_dwconv_minmax_ukernel_4p8c__neonfma, 8, false, 8, 4, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_bf16_f32_dwconv_minmax_ukernel_4p16c__neonfma, 16, false, 16, 4, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_bf16_f32_dwconv_minmax_ukernel_9p8c__neonfma, 8, false, 8, 9, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_bf16_f32_dwconv_minmax_ukernel_9p16c__neonfma, 16, false, 16, 9, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_bf16_f32_dwconv_minmax_ukernel_25p8c__neonfma_acc2, 8, false, 8, 25, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_bf16_f32_dwconv_minmax_ukernel_25p16c__neonfma_acc2, 16, false, 16, 25, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#if XNN_ENABLE_ARM_BF16
XNN_UKERNEL(xnn_arch_arm_neon_bf16, xnn_bf16_f32_dwconv_minmax_ukernel_3p8c__neonbf16, 8, false, 8, 3, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_bf16, xnn_bf16_f32_dwconv_minmax_ukernel_3p16c__neonbf16, 16, false, 16, 3, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_bf16, xnn_bf16_f32_dwconv_minmax_ukernel_4p8c__neonbf16, 8, false, 8, 4, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_bf16, xnn_bf16_f32_dwconv_minmax_ukernel_4p16c__neonbf16, 16, false, 16, 4, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_bf16, xnn_bf16_f32_dwconv_minmax_ukernel_9p8c__neonbf16, 8, false, 8, 9, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_bf16, xnn_bf16_f32_dwconv_minmax_ukernel_9p16c__neonbf16, 16, false, 16, 9, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_bf16, xnn_bf16_f32_dwconv_minmax_ukernel_25p8c__neonbf16_acc2, 8, false, 8, 25, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_bf16, xnn_bf16_f32_dwconv_minmax_ukernel_25p16c__neonbf16_acc2, 16, false, 16, 25, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_BF16
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_bf16_f32_dwconv_minmax_ukernel_3p8c__avx2, 8, false, 8, 3, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_bf16_f32_dwconv_minmax_ukernel_3p16c__avx2, 16, false, 16, 3, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_bf16_f32_dwconv_minmax_ukernel_4p8c__avx2, 8, false, 8, 4, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_bf16_f32_dwconv_minmax_ukernel_4p16c__avx2, 16, false, 16, 4, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_bf16_f32_dwconv_minmax_ukernel_9p8c__avx2, 8, false, 8, 9, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_bf16_f32_dwconv_minmax_ukernel_9p16c__avx2, 16, false, 16, 9, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_bf16_f32_dwconv_minmax_ukernel_25p8c__avx2_acc2, 8, false, 8, 25, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_bf16_f32_dwconv_minmax_ukernel_25p16c__avx2_acc2, 16, false, 16, 25, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_f32_dwconv_minmax_ukernel_3p16c__avx512skx, 16, false, 16, 3, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_f32_dwconv_minmax_ukernel_3p32c__avx512skx, 32, false, 32, 3, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_f32_dwconv_minmax_ukernel_4p16c__avx512skx, 16, false, 16, 4, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_f32_dwconv_minmax_ukernel_4p32c__avx512skx, 32, false, 32, 4, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_f32_dwconv_minmax_ukernel_9p16c__avx512skx, 16, false, 16, 9, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_f32_dwconv_minmax_ukernel_9p32c__avx512skx, 32, false, 32, 9, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_f32_dwconv_minmax_ukernel_25p16c__avx512skx_acc2, 16, false, 16, 25, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_avx512skx, xnn_bf16_f32_dwconv_minmax_ukernel_25p32c__avx512skx_acc2, 32, false, 32, 25, xnn_bfloat16, void, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
