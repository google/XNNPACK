// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, pixel_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, channel_tile, pixel_tile, datatype, params_type, init_params)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, pixel_tile, datatype, params_type, init_params) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, pixel_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_ibilinear_ukernel__scalar_c1, 1, 1, int8_t, void, nullptr)
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_ibilinear_ukernel__scalar_c2, 2, 1, int8_t, void, nullptr) 
XNN_UKERNEL_WITH_PARAMS(0, xnn_s8_ibilinear_ukernel__scalar_c4, 4, 1, int8_t, void, nullptr)  

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_s8_ibilinear_ukernel__neon_c8, 8, 1, int8_t, struct xnn_s8_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_s8_ibilinear_ukernel__neon_c16, 16, 1, int8_t, struct xnn_s8_default_params, NULL)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_X86, xnn_s8_ibilinear_ukernel__sse2_c8, 8, 1, int8_t, struct xnn_s8_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_X86, xnn_s8_ibilinear_ukernel__sse2_c16, 16, 1, int8_t, struct xnn_s8_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_X86, xnn_s8_ibilinear_ukernel__sse41_c8, 8, 1, int8_t, struct xnn_s8_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_X86, xnn_s8_ibilinear_ukernel__sse41_c16, 16, 1, int8_t, struct xnn_s8_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c8, 8, 1, int8_t, struct xnn_s8_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c16, 16, 1, int8_t, struct xnn_s8_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c8, 8, 1, int8_t, struct xnn_s8_default_params, NULL)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_s8_ibilinear_ukernel__wasmsimd_mul32_c16, 16, 1, int8_t, struct xnn_s8_default_params, NULL)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
