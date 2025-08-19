// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, channel_tile, primary_tile, datatype, params_type, init_params

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(xnn_arch_none, xnn_f32_maxpool_minmax_ukernel_9p__sse2_u4, 4, 9, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL(xnn_arch_arm_neon, xnn_f32_maxpool_minmax_ukernel_9p__neon_u4, 4, 9, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL(xnn_arch_none, xnn_f32_maxpool_minmax_ukernel_9p__wasmsimd_u4, 4, 9, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f32_maxpool_minmax_ukernel_9p__rvv_u1v, (1*xnn_init_hardware_config()->vlenb/sizeof(float)), 9, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_riscv_vector, xnn_f32_maxpool_minmax_ukernel_9p__rvv_u2v, (2*xnn_init_hardware_config()->vlenb/sizeof(float)), 9, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR

#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
XNN_UKERNEL(xnn_arch_none, xnn_f32_maxpool_minmax_ukernel_9p__hvx_u32, 32, 9, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON

XNN_UKERNEL(xnn_arch_none, xnn_f32_maxpool_minmax_ukernel_9p__scalar_u1, 1, 9, float, struct xnn_f32_minmax_params, xnn_init_f32_minmax_scalar_params)


