// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vscaleextexp_ukernel__avx2_p5_u8, 8, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vscaleextexp_ukernel__avx2_p5_u16, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vscaleextexp_ukernel__avx2_p5_u24, 24, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_f32_vscaleextexp_ukernel__avx2_p5_u32, 32, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_u16, 16, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_u32, 32, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_u48, 48, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL(xnn_arch_x86_avx512f, xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_u64, 64, float, struct xnn_f32_default_params, NULL)
#endif //XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
