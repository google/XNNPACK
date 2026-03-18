// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vmulcaddc_minmax_ukernel_c4__neon_2x, 2, 4, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vmulcaddc_minmax_ukernel_c4__neonfma_2x, 2, 4, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_vmulcaddc_minmax_ukernel_c8__neon_2x, 2, 8, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fma, xnn_f32_vmulcaddc_minmax_ukernel_c8__neonfma_2x, 2, 8, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_SSE && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
XNN_UKERNEL(xnn_arch_x86_sse, xnn_f32_vmulcaddc_minmax_ukernel_c4__sse_2x, 2, 4, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_x86_sse, xnn_f32_vmulcaddc_minmax_ukernel_c8__sse_2x, 2, 8, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ENABLE_SSE && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_arm_2x, 2, 4, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_wasm_is_x86, xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmsimd_x86_2x, 2, 4, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_arm_2x, 2, 8, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_wasm_is_x86, xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmsimd_x86_2x, 2, 8, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif //XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_wasm_fma, xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_fma_2x, 2, 4, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vmulcaddc_minmax_ukernel_c4__wasmrelaxedsimd_2x, 2, 4, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_wasm_fma, xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_fma_2x, 2, 8, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
 XNN_UKERNEL(xnn_arch_none, xnn_f32_vmulcaddc_minmax_ukernel_c8__wasmrelaxedsimd_2x, 2, 8, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

XNN_UKERNEL(xnn_arch_none, xnn_f32_vmulcaddc_minmax_ukernel_c1__scalar_2x, 2, 1, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vmulcaddc_minmax_ukernel_c2__scalar_2x, 2, 2, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_f32_vmulcaddc_minmax_ukernel_c4__scalar_2x, 2, 4, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
