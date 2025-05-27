// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vneg_ukernel__neonfp16arith_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(xnn_arch_arm_neon_fp16_arith, xnn_f16_vneg_ukernel__neonfp16arith_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_UKERNEL(0, xnn_f16_vneg_ukernel__sse2_u8, 8, false, xnn_float16, struct xnn_f16_default_params, NULL)
XNN_UKERNEL(0, xnn_f16_vneg_ukernel__sse2_u16, 16, false, xnn_float16, struct xnn_f16_default_params, NULL)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

