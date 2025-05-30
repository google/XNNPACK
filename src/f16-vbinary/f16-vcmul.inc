// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vcmul_ukernel__neonfp16arith_u8, 8, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vcmul_ukernel__neonfp16arith_u16, 16, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vcmul_ukernel__neonfp16arith_u32, 32, false, xnn_float16, struct xnn_f16_default_params, ((xnn_init_f16_default_params_fn) NULL))
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

