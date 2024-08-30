// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

// arch_flags, ukernel, channel_tile, channel_scaled_tile, primary_tile, incremental_tile, qmin, qmax, params_type, init_params

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4, 4, 4, 9, 8, -16384, 16384, xnn_init_f32_minmax_params_fn, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_f32_maxpool_minmax_ukernel_9p8x__neon_c4, 4, 4, 9, 8, -16384, 16384, xnn_init_f32_minmax_params_fn, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_arm_c4, 4, 4, 9, 8, -16384, 16384, xnn_init_f32_minmax_params_fn, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_maxpool_minmax_ukernel_9p8x__wasmsimd_x86_c4, 4, 4, 9, 8, -16384, 16384, xnn_init_f32_minmax_params_fn, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_maxpool_minmax_ukernel_9p8x__wasm_c1, 1, 1, 9, 8, -16384, 16384, xnn_init_f32_minmax_params_fn, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_c1v, 1, (1*xnn_init_hardware_config()->vlenb/sizeof(float)), 9, 8, -16384, 16384, xnn_init_f32_minmax_params_fn, xnn_init_f32_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_c2v, 2, (2*xnn_init_hardware_config()->vlenb/sizeof(float)), 9, 8, -16384, 16384, xnn_init_f32_minmax_params_fn, xnn_init_f32_minmax_scalar_params)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

XNN_UKERNEL_WITH_PARAMS(0, xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1, 1, 1, 9, 8, -16384, 16384, xnn_init_f32_minmax_params_fn, xnn_init_f32_minmax_scalar_params)


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

