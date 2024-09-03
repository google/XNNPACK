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


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_vclamp_ukernel__neonfp16arith_u8, 8, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_vclamp_ukernel__neonfp16arith_u16, 16, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_RISCV_FP16_VECTOR && (XNN_ARCH_RISCV)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vclamp_ukernel__rvvfp16arith_u1v, 1, true, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vclamp_ukernel__rvvfp16arith_u2v, 2, true, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vclamp_ukernel__rvvfp16arith_u4v, 4, true, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_riscv_vector_fp16_arith, xnn_f16_vclamp_ukernel__rvvfp16arith_u8v, 8, true, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_RISCV_FP16_VECTOR && (XNN_ARCH_RISCV)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_f16c, xnn_f16_vclamp_ukernel__f16c_u8, 8, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_x86_f16c, xnn_f16_vclamp_ukernel__f16c_u16, 16, false, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
