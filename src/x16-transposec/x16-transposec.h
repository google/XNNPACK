// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__1x2_scalar_int, 16, uint16_t, 1, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__1x4_scalar_int, 16, uint16_t, 1, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__2x1_scalar_int, 16, uint16_t, 2, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__2x2_scalar_int, 16, uint16_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__2x4_scalar_int, 16, uint16_t, 2, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__4x1_scalar_int, 16, uint16_t, 4, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__4x2_scalar_int, 16, uint16_t, 4, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__4x4_scalar_int, 16, uint16_t, 4, 4)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__4x8_sse2, 16, uint16_t, 4, 8)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__8x8_multi_mov_sse2, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__8x8_multi_switch_sse2, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx2, xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2, 16, uint16_t, 16, 16)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx2, xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2, 16, uint16_t, 16, 16)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(0, xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd, 16, uint16_t, 8, 8)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon, 16, uint16_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon, 16, uint16_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon, 16, uint16_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon, 16, uint16_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon, 16, uint16_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon, 16, uint16_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon, 16, uint16_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon, 16, uint16_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon, 16, uint16_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon, 16, uint16_t, 8, 8)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


