// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

XNN_TRANSPOSE_UKERNEL(0, xnn_x24_transposec_ukernel__1x2_scalar, 24, void, 1, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x24_transposec_ukernel__1x4_scalar, 24, void, 1, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x24_transposec_ukernel__2x1_scalar, 24, void, 2, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x24_transposec_ukernel__2x2_scalar, 24, void, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x24_transposec_ukernel__2x4_scalar, 24, void, 2, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x24_transposec_ukernel__4x1_scalar, 24, void, 4, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x24_transposec_ukernel__4x2_scalar, 24, void, 4, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x24_transposec_ukernel__4x4_scalar, 24, void, 4, 4)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_ssse3, xnn_x24_transposec_ukernel__4x4_ssse3, 24, void, 4, 4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x24_transposec_ukernel__2x2_neon_tbl64, 24, void, 2, 2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, 24, void, 4, 4)
#endif  // XNN_ARCH_ARM64


