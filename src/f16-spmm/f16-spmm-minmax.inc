// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith, 8, 1, 1, false, false, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_pipelined, 8, 1, 1, false, true, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_u2, 8, 1, 2, false, false, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith, 16, 1, 1, false, false, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_pipelined, 16, 1, 1, false, true, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_u2, 16, 1, 2, false, false, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith, 24, 1, 1, false, false, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_pipelined, 24, 1, 1, false, true, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_u2, 24, 1, 2, false, false, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith, 32, 1, 1, false, false, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_pipelined, 32, 1, 1, false, true, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_u2, 32, 1, 2, false, false, xnn_float16, struct xnn_f16_minmax_params, xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

