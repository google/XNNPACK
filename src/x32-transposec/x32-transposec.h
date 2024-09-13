// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__1x2_scalar_float, 32, uint32_t, 1, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__1x2_scalar_int, 32, uint32_t, 1, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__1x4_scalar_float, 32, uint32_t, 1, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__1x4_scalar_int, 32, uint32_t, 1, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__2x1_scalar_float, 32, uint32_t, 2, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__2x1_scalar_int, 32, uint32_t, 2, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__2x2_scalar_float, 32, uint32_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__2x2_scalar_int, 32, uint32_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__2x4_scalar_float, 32, uint32_t, 2, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__2x4_scalar_int, 32, uint32_t, 2, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x1_scalar_float, 32, uint32_t, 4, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x1_scalar_int, 32, uint32_t, 4, 1)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x2_scalar_float, 32, uint32_t, 4, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x2_scalar_int, 32, uint32_t, 4, 2)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_scalar_float, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_scalar_int, 32, uint32_t, 4, 4)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_multi_mov_sse2, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_multi_multi_sse2, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_multi_switch_sse2, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_reuse_mov_sse2, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_reuse_multi_sse2, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_reuse_switch_sse2, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_sse, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x32_transposec_ukernel__8x8_multi_mov_avx, 32, uint32_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x32_transposec_ukernel__8x8_multi_switch_avx, 32, uint32_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x32_transposec_ukernel__8x8_reuse_mov_avx, 32, uint32_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x32_transposec_ukernel__8x8_reuse_multi_avx, 32, uint32_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_x86_avx, xnn_x32_transposec_ukernel__8x8_reuse_switch_avx, 32, uint32_t, 8, 8)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_multi_mov_wasmsimd, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_multi_multi_wasmsimd, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_multi_switch_wasmsimd, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_reuse_mov_wasmsimd, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_reuse_multi_wasmsimd, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(0, xnn_x32_transposec_ukernel__4x4_reuse_switch_wasmsimd, 32, uint32_t, 4, 4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)
XNN_TRANSPOSE_UKERNEL(xnn_arch_riscv_vector, xnn_x32_transposec_ukernel__4x4_rvv, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_riscv_vector, xnn_x32_transposec_ukernel__8x8_rvv, 32, uint32_t, 8, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_riscv_vector, xnn_x32_transposec_ukernel__16x8_rvv, 32, uint32_t, 16, 8)
XNN_TRANSPOSE_UKERNEL(xnn_arch_riscv_vector, xnn_x32_transposec_ukernel__32x8_rvv, 32, uint32_t, 32, 8)
#endif  // XNN_ENABLE_RISCV_VECTOR && (XNN_ARCH_RISCV)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__2x2_multi_dec_zip_neon, 32, uint32_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__2x2_multi_mov_zip_neon, 32, uint32_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__2x2_multi_multi_zip_neon, 32, uint32_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__2x2_multi_switch_zip_neon, 32, uint32_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__2x2_reuse_dec_zip_neon, 32, uint32_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__2x2_reuse_mov_zip_neon, 32, uint32_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__2x2_reuse_multi_zip_neon, 32, uint32_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__2x2_reuse_switch_zip_neon, 32, uint32_t, 2, 2)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__4x4_multi_dec_zip_neon, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__4x4_multi_mov_zip_neon, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__4x4_multi_multi_zip_neon, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__4x4_multi_switch_zip_neon, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__4x4_reuse_dec_zip_neon, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__4x4_reuse_mov_zip_neon, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__4x4_reuse_multi_zip_neon, 32, uint32_t, 4, 4)
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__4x4_reuse_switch_zip_neon, 32, uint32_t, 4, 4)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
XNN_TRANSPOSE_UKERNEL(xnn_arch_arm_neon, xnn_x32_transposec_ukernel__4x4_aarch64_neon_tbl128, 32, uint32_t, 4, 4)
#endif  // XNN_ARCH_ARM64

