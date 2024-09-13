// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__1x2_scalar_float, 64, uint64_t, 1, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__1x2_scalar_int, 64, uint64_t, 1, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__2x1_scalar_float, 64, uint64_t, 2, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__2x1_scalar_int, 64, uint64_t, 2, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__2x2_scalar_float, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__2x2_scalar_int, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__4x1_scalar_float, 64, uint64_t, 4, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__4x1_scalar_int, 64, uint64_t, 4, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__4x2_scalar_float, 64, uint64_t, 4, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__4x2_scalar_int, 64, uint64_t, 4, 2)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__2x2_multi_mov_sse2, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__2x2_multi_multi_sse2, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__2x2_multi_switch_sse2, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__2x2_reuse_mov_sse2, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__2x2_reuse_multi_sse2, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x64_transposec_ukernel__2x2_reuse_switch_sse2, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x64_transposec_ukernel__4x4_multi_mov_avx, 64, uint64_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x64_transposec_ukernel__4x4_multi_multi_avx, 64, uint64_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x64_transposec_ukernel__4x4_multi_switch_avx, 64, uint64_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x64_transposec_ukernel__4x4_reuse_mov_avx, 64, uint64_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x64_transposec_ukernel__4x4_reuse_multi_avx, 64, uint64_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x64_transposec_ukernel__4x4_reuse_switch_avx, 64, uint64_t, 4, 4)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x64_transposec_ukernel__2x2_multi_dec_zip_neon, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x64_transposec_ukernel__2x2_multi_mov_zip_neon, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x64_transposec_ukernel__2x2_multi_multi_zip_neon, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x64_transposec_ukernel__2x2_multi_switch_zip_neon, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x64_transposec_ukernel__2x2_reuse_dec_zip_neon, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x64_transposec_ukernel__2x2_reuse_mov_zip_neon, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x64_transposec_ukernel__2x2_reuse_multi_zip_neon, 64, uint64_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x64_transposec_ukernel__2x2_reuse_switch_zip_neon, 64, uint64_t, 2, 2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


