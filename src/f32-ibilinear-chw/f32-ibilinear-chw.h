// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, pixel_tile, channel_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, pixel_tile, channel_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, pixel_tile, channel_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, pixel_tile, channel_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_ibilinear_chw_ukernel__scalar_p1, 1, 1, float, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_ibilinear_chw_ukernel__scalar_p2, 2, 1, float, void, nullptr) 
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_ibilinear_chw_ukernel__scalar_p4, 4, 1, float, void, nullptr)  

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_ibilinear_chw_ukernel__neon_p4, 4, 1, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_ibilinear_chw_ukernel__neon_p8, 8, 1, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_ibilinear_chw_ukernel__neon_p16, 16, 1, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_ibilinear_chw_ukernel__neonfma_p4, 4, 1, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_ibilinear_chw_ukernel__neonfma_p8, 8, 1, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fma, xnn_f32_ibilinear_chw_ukernel__neonfma_p8, 16, 1, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_ibilinear_chw_ukernel__sse_p4, 4, 1, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_ibilinear_chw_ukernel__sse_p8, 8, 1, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_ibilinear_chw_ukernel__wasmsimd_p4, 4, 1, float, struct xnn_f32_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_ibilinear_chw_ukernel__wasmsimd_p8, 8, 1, float, struct xnn_f32_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
