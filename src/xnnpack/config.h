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
  bool use_arm_neon_udot;  // Allow udot for armv7 to be disabled.
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  bool use_arm_fp16_arith;
  bool use_arm_neon_fp16_arith;
  bool use_arm_neon_bf16;
  bool use_arm_neon_dot;
  bool use_arm_neon_i8mm;
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
  bool use_x86_avx512vnni;
  bool use_x86_avx512vnnigfni;
  bool use_x86_avx512amx;
  bool use_x86_avxvnni;
#endif
#if XNN_ARCH_RISCV
  bool use_riscv_vector;
  bool use_riscv_vector_fp16_arith;
  // vlenb CSR (VLEN/8). 0 if vector extension is unsupported.
  uint32_t vlenb;
#endif
#if XNN_ARCH_PPC64
  bool use_vsx;
  bool use_vsx3;
  bool use_mma;
#endif
#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  bool is_x86;
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_WASMRELAXEDSIMD
  bool use_wasm_blendvps;
  bool use_wasm_pshufb;
  bool use_wasm_sdot;
  bool use_wasm_fma;
#endif  // XNN_ARCH_WASMRELAXEDSIMD
};

XNN_INTERNAL const struct xnn_hardware_config* xnn_init_hardware_config();

static inline bool xnn_is_f16_compatible_config(const struct xnn_hardware_config hardware_config[XNN_MIN_ELEMENTS(1)]) {
  #if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR) || (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
    return hardware_config->use_arm_neon_fp16_arith;
  #elif (XNN_ARCH_X86 || XNN_ARCH_X86_64) && !XNN_PLATFORM_MOBILE
    return hardware_config->use_x86_avx2;
  #else
    return false;
  #endif
}

static inline bool xnn_is_f16_chw_compatible_config(const struct xnn_hardware_config hardware_config[XNN_MIN_ELEMENTS(1)]) {
  #if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR) || (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
    return hardware_config->use_arm_neon_fp16_arith;
  #else
    return false;
  #endif
}

static inline bool xnn_is_chw_compatible_config(const struct xnn_hardware_config hardware_config[XNN_MIN_ELEMENTS(1)]) {
  #if (XNN_ARCH_X86 || XNN_ARCH_X86_64)
    // Sparse microkernels on x86 currently target only SSE, and on processors
    // with AVX ISA dense inference is expected to be faster than sparse.
    return (!hardware_config->use_x86_avx);
  #else
    return true;
  #endif
}

static inline bool xnn_is_f16_supported_natively(const struct xnn_hardware_config hardware_config[XNN_MIN_ELEMENTS(1)]) {
  #if (XNN_ARCH_ARM && XNN_ENABLE_ARM_FP16_VECTOR && XNN_ENABLE_ARM_FP16_SCALAR) || (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR)
    return hardware_config->use_arm_neon_fp16_arith;
  #else
    return false;
  #endif
}

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
  struct xnn_transpose_subconfig x64;
  struct xnn_transpose_subconfig xx;
  xnn_vunary_ukernel_fn copy;
};

XNN_INTERNAL const struct xnn_transpose_config* xnn_init_transpose_config();

struct xnn_cmul_config {
  xnn_vbinary_ukernel_fn ukernel;
  union {
    xnn_init_f32_default_params_fn f32_default;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  size_t element_tile;
};

XNN_INTERNAL const struct xnn_cmul_config* xnn_init_f16_cmul_config();
XNN_INTERNAL const struct xnn_cmul_config* xnn_init_f32_cmul_config();

struct xnn_binary_elementwise_subconfig {
  xnn_vbinary_ukernel_fn op_ukernel;
  xnn_vbinary_ukernel_fn opc_ukernel;
  xnn_vbinary_ukernel_fn ropc_ukernel;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  size_t element_tile;
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

struct xnn_unary_elementwise_config {
  xnn_vunary_ukernel_fn ukernel;
  union {
    xnn_init_f16_f32_cvt_params_fn f16_f32_cvt;
    xnn_init_f16_qs8_cvt_params_fn f16_qs8_cvt;
    xnn_init_f16_abs_params_fn f16_abs;
    xnn_init_f16_elu_params_fn f16_elu;
    xnn_init_f16_hswish_params_fn f16_hswish;
    xnn_init_f16_lrelu_params_fn f16_lrelu;
    xnn_init_f16_neg_params_fn f16_neg;
    xnn_init_f16_minmax_params_fn f16_minmax;
    xnn_init_f16_sigmoid_params_fn f16_sigmoid;
    xnn_init_f16_sqrt_params_fn f16_sqrt;
    xnn_init_f16_tanh_params_fn f16_tanh;
    xnn_init_f32_abs_params_fn f32_abs;
    xnn_init_f32_default_params_fn f32_default;
    xnn_init_f32_elu_params_fn f32_elu;
    xnn_init_f32_f16_cvt_params_fn f32_f16_cvt;
    xnn_init_f32_hswish_params_fn f32_hswish;
    xnn_init_f32_lrelu_params_fn f32_lrelu;
    xnn_init_f32_minmax_params_fn f32_minmax;
    xnn_init_f32_neg_params_fn f32_neg;
    xnn_init_f32_qs8_cvt_params_fn f32_qs8_cvt;
    xnn_init_f32_qu8_cvt_params_fn f32_qu8_cvt;
    xnn_init_f32_rnd_params_fn f32_rnd;
    xnn_init_f32_rsqrt_params_fn f32_rsqrt;
    xnn_init_f32_sigmoid_params_fn f32_sigmoid;
    xnn_init_f32_sqrt_params_fn f32_sqrt;
    xnn_init_f32_tanh_params_fn f32_tanh;
    xnn_init_qs8_cvt_params_fn qs8_cvt;
    xnn_init_qs8_f16_cvt_params_fn qs8_f16_cvt;
    xnn_init_qs8_f32_cvt_params_fn qs8_f32_cvt;
    xnn_init_qs8_hswish_params_fn qs8_hswish;
    xnn_init_qs8_lrelu_params_fn qs8_lrelu;
    xnn_init_qs16_qs8_cvt_params_fn qs16_qs8_cvt;
    xnn_init_qu8_cvt_params_fn qu8_cvt;
    xnn_init_qu8_f32_cvt_params_fn qu8_f32_cvt;
    xnn_init_qu8_hswish_params_fn qu8_hswish;
    xnn_init_qu8_lrelu_params_fn qu8_lrelu;
    xnn_init_s8_minmax_params_fn s8_minmax;
    xnn_init_u8_minmax_params_fn u8_minmax;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of
  // elements in each call.
  uint8_t element_tile;
};

XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_abs_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_clamp_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_elu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_hswish_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_lrelu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_neg_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_relu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_rndd_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_rndne_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_rndu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_rndz_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_sigmoid_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_sqr_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_sqrt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_tanh_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_to_f32_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_to_qs8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_abs_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_clamp_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_elu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_hswish_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_lrelu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_neg_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_relu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_rndd_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_rndne_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_rndu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_rndz_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config*
xnn_init_f32_rsqrt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_sigmoid_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_sqr_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_sqrt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_tanh_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_to_f16_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_to_qs8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_to_qu8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs8_hswish_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs8_lrelu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs8_to_f16_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs8_to_f32_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs16_to_qs8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qu8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qu8_hswish_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qu8_lrelu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qu8_to_f32_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_s8_clamp_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_u8_clamp_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_xx_copy_config();

struct xnn_reduce_config {
  xnn_reduce_ukernel_fn ukernel;
  union {
    xnn_init_f16_f32acc_scale_params_fn f16_f32acc_scale;
    xnn_init_f16_default_params_fn f16_default;
    xnn_init_f32_default_params_fn f32_default;
    xnn_init_f32_scale_params_fn f32_scale;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of
  // elements in each call.
  size_t element_tile;
};
XNN_INTERNAL const struct xnn_reduce_config* xnn_init_f16_f32acc_rsum_config();
XNN_INTERNAL const struct xnn_reduce_config* xnn_init_f16_rminmax_config();
XNN_INTERNAL const struct xnn_reduce_config* xnn_init_f32_rminmax_config();
XNN_INTERNAL const struct xnn_reduce_config* xnn_init_f32_rsum_config();

struct xnn_xx_fill_config {
  xnn_fill_ukernel_fn ukernel;
  // Number of rows of inputs processed in one tile.
  // For best efficiency, micro-kernel must process a multiple of this number of rows in each call.
  uint8_t row_tile;
};
XNN_INTERNAL const struct xnn_xx_fill_config* xnn_init_xx_fill_config();

struct xnn_xx_pad_config {
  xnn_pad_ukernel_fn ukernel;
  // Number of rows of inputs processed in one tile.
  // For best efficiency, micro-kernel must process a multiple of this number of rows in each call.
  uint8_t row_tile;
};
XNN_INTERNAL const struct xnn_xx_pad_config* xnn_init_xx_pad_config();

struct xnn_avgpool_config {
  xnn_avgpool_unipass_ukernel_fn unipass;
  xnn_avgpool_multipass_ukernel_fn multipass;
  union {
    xnn_init_f16_scaleminmax_params_fn f16;
    xnn_init_f32_scaleminmax_params_fn f32;
    xnn_init_qu8_avgpool_minmax_params_fn qu8;
  } init;
  // Number of rows in a primary tile.
  // Unipass micro-kernel must be called with this number of rows, or fewer.
  // Multipass micro-kernel must be called with more than this number of rows.
  uint8_t primary_tile;
  // Number of rows in an incremental tile.
  // For best efficiency, multipass micro-kernel must process the number of rows in the primary tile plus a multiple
  // of this number of rows in each call. This number has no meaning for the unipass micro-kernel.
  uint8_t incremental_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint16_t channel_tile;
};
XNN_INTERNAL const struct xnn_avgpool_config* xnn_init_f16_avgpool_config();
XNN_INTERNAL const struct xnn_avgpool_config* xnn_init_f32_avgpool_config();
XNN_INTERNAL const struct xnn_avgpool_config* xnn_init_qu8_avgpool_config();

struct xnn_pavgpool_config {
  xnn_pavgpool_unipass_ukernel_fn unipass;
  xnn_pavgpool_multipass_ukernel_fn multipass;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of rows in a primary tile.
  // Unipass micro-kernel must be called with this number of rows, or fewer.
  // Multipass micro-kernel must be called with more than this number of rows.
  uint8_t primary_tile;
  // Number of rows in an incremental tile.
  // For best efficiency, multipass micro-kernel must process the number of rows in the primary tile plus a multiple
  // of this number of rows in each call. This number has no meaning for the unipass micro-kernel.
  uint8_t incremental_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint16_t channel_tile;
};
XNN_INTERNAL const struct xnn_pavgpool_config* xnn_init_f16_pavgpool_config();
XNN_INTERNAL const struct xnn_pavgpool_config* xnn_init_f32_pavgpool_config();

struct xnn_gavgpool_config {
  xnn_gavgpool_unipass_ukernel_fn unipass;
  xnn_gavgpool_multipass_ukernel_fn multipass;
  union {
    xnn_init_f16_scaleminmax_params_fn f16;
    xnn_init_f32_scaleminmax_params_fn f32;
    xnn_init_qs8_avgpool_minmax_params_fn qs8;
    xnn_init_qu8_avgpool_minmax_params_fn qu8;
  } init;
  union {
    xnn_update_f16_scaleminmax_params_fn f16;
    xnn_update_f32_scaleminmax_params_fn f32;
    xnn_update_qs8_avgpool_minmax_params_fn qs8;
    xnn_update_qu8_avgpool_minmax_params_fn qu8;
  } update;
  // Number of rows in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of rows in each call.
  uint16_t row_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint16_t channel_tile;
};
XNN_INTERNAL const struct xnn_gavgpool_config* xnn_init_f16_gavgpool_config();
XNN_INTERNAL const struct xnn_gavgpool_config* xnn_init_f32_gavgpool_config();
XNN_INTERNAL const struct xnn_gavgpool_config* xnn_init_qs8_gavgpool_config();
XNN_INTERNAL const struct xnn_gavgpool_config* xnn_init_qu8_gavgpool_config();

struct xnn_gavgpool_cw_config {
  xnn_gavgpool_cw_ukernel_fn ukernel;
  union {
    xnn_init_f16_gavgpool_neon_params_fn f16;
    xnn_init_f32_gavgpool_params_fn f32;
  } init;
  union {
    xnn_update_f16_gavgpool_neonfp16arith_params_fn f16;
    xnn_update_f32_gavgpool_params_fn f32;
  } update;

  // Number of input pixels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of pixels in each call.
  uint8_t pixel_tile;
  // Channel tile is always 1.
};
XNN_INTERNAL const struct xnn_gavgpool_cw_config* xnn_init_f16_gavgpool_cw_config();
XNN_INTERNAL const struct xnn_gavgpool_cw_config* xnn_init_f32_gavgpool_cw_config();

union xnn_dwconv_ukernel {
  xnn_dwconv_unipass_ukernel_fn unipass;
  xnn_dwconv_multipass_ukernel_fn multipass;
};

struct xnn_dwconv_config {
  union xnn_dwconv_ukernel minmax;
  union xnn_dwconv_ukernel linear;
  union {
    xnn_init_qs8_conv_minmax_params_fn qs8;
    xnn_init_qs8_qc8w_conv_minmax_params_fn qs8_qc8w;
    xnn_init_qu8_conv_minmax_params_fn qu8;
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of channels in a tile.
  uint8_t channel_tile;
  // Number of channels in a subtile. This must be less-than-equal channel_tile. After processing channel_tile, the
  // remainder is processed in tiles of channel_subtile.
  uint8_t channel_subtile;
  // How much to round channels by to get more optimal tiling.
  uint8_t channel_round;
  // Number of elements in the tile. For multipass, this is the tile size for first pass.
  uint8_t primary_tile;
  // Tile size for middle pass. Middle pass can be run multiple times. Will be zero for unipass, non-zero and not
  // greater than last_tile for multipass.
  uint8_t middle_tile;
  // Tile size for last pass. Will be zero for unipass, non-zero for multipass.
  uint8_t last_tile;
};

#define XNN_MAX_F16_DWCONV_UKERNELS 4
#define XNN_MAX_F32_DWCONV_UKERNELS 4
#define XNN_MAX_QC8_DWCONV_UKERNELS 3
#define XNN_MAX_QS8_DWCONV_UKERNELS 2
#define XNN_MAX_QU8_DWCONV_UKERNELS 2

XNN_INTERNAL struct xnn_dwconv_config* xnn_init_f16_dwconv_config();
XNN_INTERNAL struct xnn_dwconv_config* xnn_init_f32_dwconv_config();
XNN_INTERNAL struct xnn_dwconv_config* xnn_init_qs8_qc8w_dwconv_config();
XNN_INTERNAL struct xnn_dwconv_config* xnn_init_qs8_dwconv_config();
XNN_INTERNAL struct xnn_dwconv_config* xnn_init_qu8_dwconv_config();

struct xnn_ibilinear_config {
  xnn_ibilinear_ukernel_fn ukernel;
  // Number of output pixels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of pixels in each call.
  uint8_t pixel_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
};

// Bilinear interpolation (2D).
XNN_INTERNAL const struct xnn_ibilinear_config* xnn_init_f16_ibilinear_config();
XNN_INTERNAL const struct xnn_ibilinear_config* xnn_init_f32_ibilinear_config();
XNN_INTERNAL const struct xnn_ibilinear_config* xnn_init_s8_ibilinear_config();
XNN_INTERNAL const struct xnn_ibilinear_config* xnn_init_u8_ibilinear_config();

struct xnn_ibilinear_chw_config {
  xnn_ibilinear_chw_ukernel_fn ukernel;
  // Number of output pixels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of pixels in each call.
  uint8_t pixel_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
};

// Bilinear interpolation (2D) in CHW layout.
XNN_INTERNAL const struct xnn_ibilinear_chw_config* xnn_init_f16_ibilinear_chw_config();
XNN_INTERNAL const struct xnn_ibilinear_chw_config* xnn_init_f32_ibilinear_chw_config();

struct xnn_prelu_config {
  xnn_prelu_ukernel_fn ukernel;
  // Number of rows in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of rows in each call.
  uint16_t row_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint16_t channel_tile;
};

XNN_INTERNAL const struct xnn_prelu_config* xnn_init_f16_prelu_config();
XNN_INTERNAL const struct xnn_prelu_config* xnn_init_f32_prelu_config();

struct xnn_generated_code_chunk {
  size_t offset;
  size_t offset_end;
};

struct xnn_hmp_dqgemm_ukernel {
  xnn_dqgemm_ukernel_fn function[XNN_MAX_UARCH_TYPES];
#if XNN_PLATFORM_JIT
  struct xnn_generated_code_chunk generated_code_chunk[XNN_MAX_UARCH_TYPES];
#endif  // XNN_PLATFORM_JIT
};

struct xnn_hmp_gemm_ukernel {
  xnn_gemm_ukernel_fn function[XNN_MAX_UARCH_TYPES];
#if XNN_PLATFORM_JIT
  struct xnn_generated_code_chunk generated_code_chunk[XNN_MAX_UARCH_TYPES];
#endif  // XNN_PLATFORM_JIT
};

static inline struct xnn_hmp_dqgemm_ukernel xnn_init_hmp_dqgemm_ukernel(
    xnn_dqgemm_ukernel_fn function) {
  struct xnn_hmp_dqgemm_ukernel ukernel = {{ function }};
#if XNN_PLATFORM_JIT
  ukernel.generated_code_chunk[0].offset = SIZE_MAX;
  ukernel.generated_code_chunk[0].offset_end = SIZE_MAX;
#endif  // XNN_PLATFORM_JIT
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
#if XNN_PLATFORM_JIT
    ukernel.generated_code_chunk[i].offset = SIZE_MAX;
    ukernel.generated_code_chunk[i].offset_end = SIZE_MAX;
#endif  // XNN_PLATFORM_JIT
  }
  return ukernel;
}

struct xnn_hmp_dqigemm_ukernel {
  xnn_dqigemm_ukernel_fn function[XNN_MAX_UARCH_TYPES];
#if XNN_PLATFORM_JIT
  struct xnn_generated_code_chunk generated_code_chunk[XNN_MAX_UARCH_TYPES];
#endif  // XNN_PLATFORM_JIT
};

static inline struct xnn_hmp_dqigemm_ukernel xnn_init_hmp_dqigemm_ukernel(
    xnn_dqigemm_ukernel_fn function) {
  struct xnn_hmp_dqigemm_ukernel ukernel = {{function}};
#if XNN_PLATFORM_JIT
  ukernel.generated_code_chunk[0].offset = SIZE_MAX;
  ukernel.generated_code_chunk[0].offset_end = SIZE_MAX;
#endif  // XNN_PLATFORM_JIT
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
#if XNN_PLATFORM_JIT
    ukernel.generated_code_chunk[i].offset = SIZE_MAX;
    ukernel.generated_code_chunk[i].offset_end = SIZE_MAX;
#endif  // XNN_PLATFORM_JIT
  }
  return ukernel;
}

static inline struct xnn_hmp_gemm_ukernel xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn function) {
  struct xnn_hmp_gemm_ukernel ukernel = {{ function }};
#if XNN_PLATFORM_JIT
  ukernel.generated_code_chunk[0].offset = SIZE_MAX;
  ukernel.generated_code_chunk[0].offset_end = SIZE_MAX;
#endif  // XNN_PLATFORM_JIT
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
#if XNN_PLATFORM_JIT
    ukernel.generated_code_chunk[i].offset = SIZE_MAX;
    ukernel.generated_code_chunk[i].offset_end = SIZE_MAX;
#endif  // XNN_PLATFORM_JIT
  }
  return ukernel;
}

static inline bool xnn_is_hmp_gemm_ukernel(struct xnn_hmp_gemm_ukernel ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_fn = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_fn ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

struct xnn_hmp_igemm_ukernel {
  xnn_igemm_ukernel_fn function[XNN_MAX_UARCH_TYPES];
#if XNN_PLATFORM_JIT
  struct xnn_generated_code_chunk generated_code_chunk[XNN_MAX_UARCH_TYPES];
#endif  // XNN_PLATFORM_JIT
};

static inline struct xnn_hmp_igemm_ukernel xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn function) {
  struct xnn_hmp_igemm_ukernel ukernel = {{ function }};
#if XNN_PLATFORM_JIT
  ukernel.generated_code_chunk[0].offset = SIZE_MAX;
  ukernel.generated_code_chunk[0].offset_end = SIZE_MAX;
#endif  // XNN_PLATFORM_JIT
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
#if XNN_PLATFORM_JIT
    ukernel.generated_code_chunk[i].offset = SIZE_MAX;
    ukernel.generated_code_chunk[i].offset_end = SIZE_MAX;
#endif  // XNN_PLATFORM_JIT
  }
  return ukernel;
}

static inline bool xnn_is_hmp_igemm_ukernel(struct xnn_hmp_igemm_ukernel ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_fn = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_fn ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

// Largest GEMM/IGEMM MR used in init.c is 7 (x86 AVX512).
// Largest GEMM/IGEMM MR is 8 in e2e benchmarks.
#define XNN_MAX_MR 8

struct gemm_fused_ukernels {
  union {
    struct xnn_hmp_gemm_ukernel gemm[XNN_MAX_MR];
    struct xnn_hmp_dqgemm_ukernel dqgemm[XNN_MAX_MR];
  };
  union {
    struct xnn_hmp_igemm_ukernel igemm[XNN_MAX_MR];
    struct xnn_hmp_dqigemm_ukernel dqigemm[XNN_MAX_MR];
  };
};

#if XNN_PLATFORM_JIT
struct xnn_hmp_gemm_codegen {
  xnn_jit_gemm_code_generator_fn function[XNN_MAX_UARCH_TYPES];
};

static inline struct xnn_hmp_gemm_codegen xnn_init_hmp_gemm_codegen(xnn_jit_gemm_code_generator_fn function) {
  struct xnn_hmp_gemm_codegen ukernel = {{ function }};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline bool xnn_is_hmp_gemm_codegen(struct xnn_hmp_gemm_codegen ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_fn = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_fn ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

struct xnn_hmp_igemm_codegen {
  xnn_jit_igemm_code_generator_fn function[XNN_MAX_UARCH_TYPES];
};

static inline struct xnn_hmp_igemm_codegen xnn_init_hmp_igemm_codegen(xnn_jit_igemm_code_generator_fn function) {
  struct xnn_hmp_igemm_codegen ukernel = {{ function }};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline bool xnn_is_hmp_igemm_codegen(struct xnn_hmp_igemm_codegen ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_fn = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_fn ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

struct gemm_codegens {
  struct xnn_hmp_gemm_codegen gemm[XNN_MAX_MR];
  struct xnn_hmp_igemm_codegen igemm[XNN_MAX_MR];
};
#endif  // XNN_PLATFORM_JIT

struct xnn_gemm_config {
  struct gemm_fused_ukernels minmax;
  struct gemm_fused_ukernels relu;
  struct gemm_fused_ukernels linear;
#if XNN_PLATFORM_JIT
  struct gemm_codegens generator;
#endif  // XNN_PLATFORM_JIT
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
    xnn_init_f16_qc4w_minmax_params_fn f16_qc4w;
    xnn_init_f32_qc4w_minmax_params_fn f32_qc4w;
    xnn_init_qs8_conv_minmax_params_fn qs8;
    xnn_init_qs8_qc8w_conv_minmax_params_fn qs8_qc8w;
    xnn_init_qu8_conv_minmax_params_fn qu8;
  } init;
  xnn_packw_gemm_gio_ukernel_fn pack_gemm_gio;
  xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi;
  xnn_pack_conv_goki_w_fn pack_igemm_goki;
  xnn_pack_conv_kgo_w_fn pack_igemm_kgo;
  xnn_pack_deconv_goki_w_fn pack_deconv_goki;
  uint8_t mr;
  uint8_t nr;
  uint8_t log2_kr;
  uint8_t log2_sr;
  uint8_t planes;  // number of 4 bit planes (1 for legacy, 2 for unzip)
};

XNN_INTERNAL struct xnn_gemm_config* xnn_init_f16_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_f32_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_f32_gemm_nr2_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_f32_qc8w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_f32_qc4w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qd8_f16_qc4w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qd8_f16_qc8w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qd8_f32_qc4w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qd8_f32_qc8w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qs8_qc8w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qu8_gemm_config();

struct xnn_maxpool_config {
  xnn_maxpool_ukernel_fn ukernel;
  union {
    xnn_init_s8_minmax_params_fn s8;
    xnn_init_u8_minmax_params_fn u8;
    xnn_init_f32_minmax_params_fn f32;
    xnn_init_f16_minmax_params_fn f16;
  } init;
  // Number of elements in a tile for the first pass.
  uint8_t first_pass_tile_size;
  // Number of elements in a tile for the remainder pass. If the pooling size is less than or equals to
  // first_pass_tile_size, remainder passes are not run. We run as many remainder passes as required to cover the entire
  // pooling window.
  uint8_t remainder_pass_tile_size;
};

XNN_INTERNAL const struct xnn_maxpool_config* xnn_init_f16_maxpool_config();
XNN_INTERNAL const struct xnn_maxpool_config* xnn_init_f32_maxpool_config();
XNN_INTERNAL const struct xnn_maxpool_config* xnn_init_s8_maxpool_config();
XNN_INTERNAL const struct xnn_maxpool_config* xnn_init_u8_maxpool_config();

struct xnn_zip_config {
  xnn_zipc_ukernel_fn x2;
  xnn_zipc_ukernel_fn x3;
  xnn_zipc_ukernel_fn x4;
  xnn_zipv_ukernel_fn xm;
};

XNN_INTERNAL const struct xnn_zip_config* xnn_init_x8_zip_config();
XNN_INTERNAL const struct xnn_zip_config* xnn_init_x32_zip_config();

struct xnn_rmax_config {
  xnn_rmax_ukernel_fn ukernel;
  union {
    xnn_init_f32_default_params_fn f32;
    xnn_init_f16_default_params_fn f16;
  } init;
};

XNN_INTERNAL const struct xnn_rmax_config* xnn_init_f16_rmax_config();
XNN_INTERNAL const struct xnn_rmax_config* xnn_init_f32_rmax_config();
XNN_INTERNAL const struct xnn_rmax_config* xnn_init_u8_rmax_config();

struct xnn_spmm_config {
  xnn_spmm_ukernel_fn ukernel;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of M-dimension elements in a tile.
  // Corresponds to a block of pixels in 1x1 Convolution and a block of batch size in Fully Connected operator.
  uint8_t mr;
  // Number of N-dimension elements in a tile.
  // Corresponds to a block of output channels/features in 1x1 Convolution and Fully Connected operator.
  uint8_t nr;
};


// Sparse Matrix-Dense Matrix Multiplication (NR=1 block).
XNN_INTERNAL const struct xnn_spmm_config* xnn_init_f16_spmm_config();
XNN_INTERNAL const struct xnn_spmm_config* xnn_init_f32_spmm_config();
// Sparse Matrix-Dense Matrix Multiplication (NR=2 block).
XNN_INTERNAL const struct xnn_spmm_config* xnn_init_f32_spmm2_config();
// Sparse Matrix-Dense Matrix Multiplication (NR=4 block).
XNN_INTERNAL const struct xnn_spmm_config* xnn_init_f32_spmm4_config();

struct xnn_dwconv2d_chw_parameters {
  xnn_dwconv2d_chw_ukernel_fn ukernel;
  union {
    xnn_init_f16_chw_params_fn f16;
    xnn_init_f32_chw_params_fn f32;
  } init;
  union {
    xnn_update_f16_chw_params_fn f16;
    xnn_update_f32_chw_params_fn f32;
  } update;
  // Number of output width pixels in a tile.
  uint8_t output_width_tile;
  // Number of output height pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint8_t output_height_tile;
};

struct xnn_dwconv2d_chw_config {
  // Direct 3x3 stride-1 Convolution with padding 1 on left and right in CHW layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_3x3;
  // Direct 3x3 stride-2 Convolution with padding 1 on left and right in CHW layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_3x3s2;
  // Direct 5x5 stride-1 Convolution with padding 2 on left and right in CHW layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_5x5;
  // Direct 5x5 stride-2 Convolution with padding 2 on left and right in CHW layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_5x5s2;
};

XNN_INTERNAL const struct xnn_dwconv2d_chw_config* xnn_init_f16_dwconv2d_chw_config();
XNN_INTERNAL const struct xnn_dwconv2d_chw_config* xnn_init_f32_dwconv2d_chw_config();

struct xnn_conv_hwc2chw_config {
  xnn_conv_hwc2chw_ukernel_fn ukernel_with_symm_padding;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of output channels in a tile.
  // This parameter must be passed as is to weight packing function.
  uint8_t output_channel_tile;
  // Number of output height pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint8_t output_height_tile;
  // Number of output width pixels in a tile.
  uint8_t output_width_tile;
};

// Direct 3x3 stride-2 Convolution with 3 input channels and HWC->CHW layout conversion.
XNN_INTERNAL const struct xnn_conv_hwc2chw_config* xnn_init_f16_conv_hwc2chw_3x3c3s2_config();
XNN_INTERNAL const struct xnn_conv_hwc2chw_config* xnn_init_f32_conv_hwc2chw_3x3c3s2_config();

struct xnn_vmulcaddc_config {
  xnn_vmulcaddc_ukernel_fn ukernel;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
  // Number of rows of inputs processed in one tile.
  // For best efficiency, micro-kernel must process a multiple of this number of rows in each call.
  uint8_t row_tile;
};

XNN_INTERNAL const struct xnn_vmulcaddc_config* xnn_init_f16_vmulcaddc_config();
XNN_INTERNAL const struct xnn_vmulcaddc_config* xnn_init_f32_vmulcaddc_config();

struct xnn_raddstoreexpminusmax_config {
  xnn_raddstoreexpminusmax_ukernel_fn ukernel;
  union {
    xnn_init_f16_expminus_params_fn f16;
    xnn_init_f32_expminus_params_fn f32;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  uint8_t element_tile;
};

XNN_INTERNAL const struct xnn_raddstoreexpminusmax_config* xnn_init_f16_raddstoreexpminusmax_config();
XNN_INTERNAL const struct xnn_raddstoreexpminusmax_config* xnn_init_f32_raddstoreexpminusmax_config();

struct xnn_argmaxpool_config {
  union {
    xnn_argmaxpool_unipass_ukernel_fn up;
    xnn_argmaxpool_multipass_ukernel_fn mp;
  };
  // // Number of elements in a tile for the first pass.
  uint8_t first_pass_tile_size;
  // Number of elements in a tile for the remainder pass. If the pooling size is less than or equals to
  // first_pass_tile_size, remainder passes are not run. We run as many remainder passes as required to cover the entire
  // pooling window.
  uint8_t remainder_pass_tile_size;
};

#define XNN_MAX_F32_ARGMAXPOOL_UKERNELS 3

XNN_INTERNAL const struct xnn_argmaxpool_config* xnn_init_f32_argmaxpool_config();

struct xnn_lut32norm_config {
  xnn_u8_lut32norm_ukernel_fn lut32norm;
};

XNN_INTERNAL const struct xnn_lut32norm_config* xnn_init_u8_lut32norm_config();

struct xnn_unpool_config {
  xnn_unpool_ukernel_fn unpool;
};

XNN_INTERNAL const struct xnn_unpool_config* xnn_init_x32_unpool_config();

#ifdef __cplusplus
}  // extern "C"
#endif
