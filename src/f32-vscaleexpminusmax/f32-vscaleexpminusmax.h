// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, element_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif
#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, element_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u8, 8, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u16, 16, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u24, 24, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u32, 32, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u40, 40, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u48, 48, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u56, 56, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u64, 64, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u72, 72, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u80, 80, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u88, 88, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u96, 96, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
#endif

#if XNN_ENABLE_AVX512F && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u16, 16, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u32, 32, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u48, 48, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u64, 64, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u80, 80, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u96, 96, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u112, 112, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u128, 128, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u144, 144, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u160, 160, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u176, 176, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512f, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u192, 192, float, struct xnn_f32_default_params, ((xnn_f32_vscaleexpminusmax_ukernel_fn) NULL))
#endif

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif
#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
