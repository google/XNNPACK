// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include "xnnpack/microparams-init.h"
#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile, channel_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, row_tile, channel_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, row_tile, channel_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, row_tile, channel_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_X86, xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, 2, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_X86, xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, 2, 8, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, 2, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, 2, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, 2, 8, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_ARM, xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, 2, 8, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, 2, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, 2, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, 2, 8, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, 2, 8, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, 2, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, 2, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, 2, 8, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMRELAXEDSIMD, xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, 2, 8, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASM, xnn_f32_vmulcaddc_minmax_ukernel_c1__wasm_2x, 2, 1, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASM, xnn_f32_vmulcaddc_minmax_ukernel_c2__wasm_2x, 2, 2, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASM, xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x, 2, 4, float, union xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
