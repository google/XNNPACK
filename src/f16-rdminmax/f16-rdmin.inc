// clang-format off
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_rdmin_ukernel_2p2x__neonfp16arith_u32, 2, 32, false, xnn_float16, xnn_float16, void*, NULL)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

XNN_UKERNEL(xnn_arch_none, xnn_f16_rdmin_ukernel_2p2x__scalar_u2, 2, 2, false, xnn_float16, xnn_float16, void*, NULL)
