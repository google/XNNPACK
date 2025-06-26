// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u32, 64, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u32_acc2, 32,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u32_acc4, 32,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u64, 64, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc2, 64,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc4, 64,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u72, 72,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u72_acc3, 72,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u80, 80,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc2, 80,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc5, 80,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u96, 96,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc2, 96,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc3, 96,  float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc6, 96,  float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u64, 64, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u64_acc2, 64, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u64_acc4, 64, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128, 128, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc2, 128, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc4, 128, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144, 144, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144_acc3, 144, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160, 160, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc2, 160, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc5, 160, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192, 192, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc2, 192, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc3, 192, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc6, 192, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
