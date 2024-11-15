// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype, params_type, init_params)  \
  XNN_UKERNEL(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype, params_type)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype, params_type)   \
  XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, 8, 1, false, 1, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, 8, 1, true, 1, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_x2, 8, 1, false, 2, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, 16, 1, false, 1, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, 16, 1, true, 1, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_x2, 16, 1, false, 2, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, 24, 1, false, 1, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, 24, 1, true, 1, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_x2, 24, 1, false, 2, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, 32, 1, false, 1, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, 32, 1, true, 1, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL_WITH_PARAMS(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_x2, 32, 1, false, 2, xnn_float16, union xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
