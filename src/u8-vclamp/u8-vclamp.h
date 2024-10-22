// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon, xnn_u8_vclamp_ukernel__neon_u64, 64, false, uint8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(0, xnn_u8_vclamp_ukernel__sse2_u64, 64, false, uint8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx2, xnn_u8_vclamp_ukernel__avx2_u128, 128, false, uint8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_avx512skx, xnn_u8_vclamp_ukernel__avx512skx_u256, 256, false, uint8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_u8_vclamp_ukernel__rvv_u1v, 1, true, int8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_u8_vclamp_ukernel__rvv_u2v, 2, true, int8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_u8_vclamp_ukernel__rvv_u4v, 4, true, int8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector, xnn_u8_vclamp_ukernel__rvv_u8v, 8, true, int8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_UKERNEL_WITH_PARAMS(0, xnn_u8_vclamp_ukernel__wasmsimd_u64, 64, false, uint8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_UKERNEL_WITH_PARAMS(0, xnn_u8_vclamp_ukernel__scalar_u4, 4, false, uint8_t, struct xnn_u8_minmax_params, xnn_init_u8_minmax_scalar_params)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
