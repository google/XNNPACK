// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>

#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams.h>


#ifdef __cplusplus
extern "C" {
#endif

struct xnn_hardware_config {
  char _; // Dummy member variable to comply with the C standard
#if XNN_ARCH_ARM
  bool use_arm_v6;
  bool use_arm_vfpv2;
  bool use_arm_vfpv3;
  bool use_arm_neon;
  bool use_arm_neon_fp16;
  bool use_arm_neon_fma;
  bool use_arm_neon_v8;
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool use_arm_fp16_arith;
  bool use_arm_neon_fp16_arith;
  bool use_arm_neon_bf16;
  bool use_arm_neon_dot;
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  bool use_x86_ssse3;
  bool use_x86_sse4_1;
  bool use_x86_avx;
  bool use_x86_f16c;
  bool use_x86_fma3;
  bool use_x86_xop;
  bool use_x86_avx2;
  bool use_x86_avx512f;
  bool use_x86_avx512vbmi;
  bool use_x86_avx512skx;
#endif
#if XNN_ARCH_RISCV
  bool use_rvv;
#endif
#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  bool is_x86;
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_WASMRELAXEDSIMD
  bool use_wasm_pshufb;
#endif  // XNN_ARCH_WASMRELAXEDSIMD
};

XNN_INTERNAL const struct xnn_hardware_config* xnn_init_hardware_config();


struct xnn_x8_lut_config {
  xnn_x8_lut_ukernel_fn microkernel;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  size_t tile_size;
};

XNN_INTERNAL const struct xnn_x8_lut_config* xnn_init_x8_lut_config();


struct xnn_transpose_subconfig {
  union {
    xnn_transposec_ukernel_fn const_size_ukernel;
    xnn_transposev_ukernel_fn variable_size_ukernel;
  };
  union {
    xnn_init_x8_transpose_params_fn x8;
    xnn_init_x16_transpose_params_fn x16;
    xnn_init_x24_transpose_params_fn x24;
    xnn_init_x32_transpose_params_fn x32;
    xnn_init_x64_transpose_params_fn x64;
  } init;
  // Maximum number of elements to process per ukernel call.
  size_t tile_size;
};

struct xnn_transpose_config {
  struct xnn_transpose_subconfig x8;
  struct xnn_transpose_subconfig x16;
  struct xnn_transpose_subconfig x24;
  struct xnn_transpose_subconfig x32;
  struct xnn_transpose_subconfig xx;
  xnn_vunary_ukernel_fn copy;
};

XNN_INTERNAL const struct xnn_transpose_config* xnn_init_transpose_config();

struct xnn_binary_elementwise_subconfig {
  xnn_vbinary_ukernel_fn op_ukernel;
  xnn_vbinary_ukernel_fn opc_ukernel;
  xnn_vbinary_ukernel_fn ropc_ukernel;
};

struct xnn_binary_elementwise_config {
  struct xnn_binary_elementwise_subconfig minmax;
  struct xnn_binary_elementwise_subconfig linear;
  union {
    xnn_init_f16_minmax_params_fn f16_minmax;
    xnn_init_f32_default_params_fn f32_default;
    xnn_init_f32_minmax_params_fn f32_minmax;
    xnn_init_qs8_add_minmax_params_fn qs8_add;
    xnn_init_qs8_mul_minmax_params_fn qs8_mul;
    xnn_init_qu8_add_minmax_params_fn qu8_add;
    xnn_init_qu8_mul_minmax_params_fn qu8_mul;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  size_t element_tile;
};

XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vadd_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vdiv_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vmax_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vmin_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vmul_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vsub_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vsqrdiff_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vadd_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vdiv_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vmax_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vmin_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vmul_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vsub_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vsqrdiff_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_qs8_vadd_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_qs8_vmul_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_qu8_vadd_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_qu8_vmul_config();

#ifdef __cplusplus
}  // extern "C"
#endif
