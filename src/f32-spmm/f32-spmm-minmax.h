// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype, params_type, init_params)  \
  XNN_UKERNEL(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype)   \
  XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

// SCALAR
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_spmm_minmax_ukernel_2x1__scalar_pipelined, 2, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_spmm_minmax_ukernel_4x1__scalar, 4, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_spmm_minmax_ukernel_8x4__scalar, 8, 4, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_spmm_minmax_ukernel_1x1__scalar_pipelined, 1, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_spmm_minmax_ukernel_2x1__scalar, 2, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_spmm_minmax_ukernel_8x1__scalar, 8, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_spmm_minmax_ukernel_8x1__scalar_pipelined, 8, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_spmm_minmax_ukernel_1x1__scalar, 1, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_spmm_minmax_ukernel_4x1__scalar_pipelined, 4, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_spmm_minmax_ukernel_8x2__scalar, 8, 2, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_4x1__neon_pipelined, 4, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_4x1__neon_x2, 4, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_8x1__neonfma, 8, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_12x1__neonfma, 12, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_16x1__neonfma_pipelined, 16, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_32x1__neon, 32, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_4x1__neonfma, 4, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_4x1__neonfma_pipelined, 4, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_4x1__neonfma_x2, 4, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_8x1__neon_x2, 8, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_8x1__neonfma_pipelined, 8, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_8x1__neonfma_x2, 8, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_12x1__neon, 12, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_16x1__neon, 16, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_16x1__neon_pipelined, 16, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_16x1__neonfma_x2, 16, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_32x1__neonfma_pipelined, 32, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_32x1__neonfma_x2, 32, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_4x1__neon, 4, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_8x1__neon, 8, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_16x1__neon_x2, 16, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_32x1__neon_x2, 32, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_8x1__neon_pipelined, 8, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_16x1__neonfma, 16, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_32x1__neon_pipelined, 32, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_32x1__neonfma, 32, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_8x2__aarch64_neonfma, 8, 2, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_12x2__aarch64_neonfma, 12, 2, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_16x4__aarch64_neonfma, 16, 4, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_32x4__aarch64_neonfma, 32, 4, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_8x4__aarch64_neonfma, 8, 4, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_12x4__aarch64_neonfma, 12, 4, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_16x2__aarch64_neonfma, 16, 2, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_32x2__aarch64_neonfma, 32, 2, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_spmm_minmax_ukernel_4x2__aarch64_neonfma, 4, 2, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse, xnn_f32_spmm_minmax_ukernel_4x1__sse, 4, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse, xnn_f32_spmm_minmax_ukernel_16x1__sse, 16, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse, xnn_f32_spmm_minmax_ukernel_32x1__sse, 32, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_sse, xnn_f32_spmm_minmax_ukernel_8x1__sse, 8, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_x2, 8, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_pipelined_x2, 8, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x2, 8, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_x4, 16, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_pipelined_x2, 16, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_x4, 16, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_x4, 32, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x4, 32, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_pipelined, 4, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_pipelined_x2, 4, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_x4, 8, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_pipelined, 8, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x4, 8, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm, 16, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_x2, 16, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86, 16, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_pipelined, 16, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_x2, 16, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm, 32, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_pipelined, 32, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_pipelined_x2, 32, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_x2, 32, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86, 32, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x2, 32, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm, 4, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_pipelined_x2, 4, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_x4, 4, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86, 4, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_pipelined, 4, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_x4, 4, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_pipelined, 8, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_pipelined, 16, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_pipelined, 32, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_pipelined_x2, 32, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_x2, 4, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_x2, 4, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm, 8, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_pipelined_x2, 8, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86, 8, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_pipelined_x2, 16, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_pipelined, 4, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_x4, 4, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_pipelined_x2, 4, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_x4, 4, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_pipelined, 16, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_x4, 32, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_pipelined_x2, 32, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_x4, 32, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_x2, 4, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_x2, 4, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_pipelined, 8, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_pipelined_x2, 8, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_pipelined_x2, 16, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm, 32, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_x2, 32, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86, 32, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_pipelined, 32, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_x2, 32, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm, 4, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86, 4, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_pipelined_x2, 8, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_x2, 8, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_pipelined, 8, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_x2, 8, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_x2, 16, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_pipelined_x2, 16, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_x2, 16, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_pipelined, 32, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_pipelined_x2, 4, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_pipelined, 4, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm, 8, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_x4, 8, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86, 8, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_x4, 8, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm, 16, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_pipelined, 16, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_x4, 16, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86, 16, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_x4, 16, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(XNN_ARCH_WASMSIMD, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_pipelined_x2, 32, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_32x1__hvx, 32, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_32x1__hvx_pipelined, 32, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_32x1__hvx_x2, 32, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_64x1__hvx, 64, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_64x1__hvx_pipelined_x4, 64, 1, true, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_128x1__hvx_pipelined_x4, 128, 1, true, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_32x1__hvx_x4, 32, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_64x1__hvx_pipelined_x2, 64, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_128x1__hvx_pipelined_x2, 128, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_32x1__hvx_pipelined_x4, 32, 1, true, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_64x1__hvx_x4, 64, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_128x1__hvx, 128, 1, false, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_128x1__hvx_x2, 128, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_32x1__hvx_pipelined_x2, 32, 1, true, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_64x1__hvx_pipelined, 64, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_64x1__hvx_x2, 64, 1, false, 2, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_128x1__hvx_pipelined, 128, 1, true, 1, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_hexagon, xnn_f32_spmm_minmax_ukernel_128x1__hvx_x4, 128, 1, false, 4, float, struct xnn_f32_default_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
