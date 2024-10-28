// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/microparams-init.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"
#include "xnnpack/requantization.h"
#include "xnnpack/unaligned.h"

size_t xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_scalar.output_min = output_min;
  params->fp32_scalar.output_max = output_max;
  params->fp32_scalar.output_zero_point = output_zero_point;
  return sizeof(params->fp32_scalar);
}

#if XNN_ARCH_ARM
size_t xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_armsimd32.magic_bias = 12582912.0f;
  params->fp32_armsimd32.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_armsimd32.output_min = (uint32_t) (uint8_t) output_min * UINT32_C(0x01010101);
  params->fp32_armsimd32.output_max = (uint32_t) (uint8_t) output_max * UINT32_C(0x01010101);
  return sizeof(params->fp32_armsimd32);
}
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_neon.output_min = output_min;
  params->fp32_neon.output_max = output_max;
  return sizeof(params->fp32_neon);
}

size_t xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_neonv8.output_zero_point = (int16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
  return sizeof(params->fp32_neonv8);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

size_t xnn_init_qs8_conv_minmax_fp32_scalar_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar.scale = scale;
  params->fp32_scalar.output_min = output_min;
  params->fp32_scalar.output_max = output_max;
  params->fp32_scalar.output_zero_point = output_zero_point;
  return sizeof(params->fp32_scalar);
}

size_t xnn_init_qs8_conv_minmax_rndnu_scalar_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  // Compute requantization parameters.
  const uint32_t scale_bits = float_as_uint32(scale);

  const int32_t multiplier =  ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  assert(multiplier >= INT32_C(0x00800000));
  assert(multiplier <= INT32_C(0x00FFFFFF));

  // Shift is in [16, 55] range.
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 56);

  const int64_t rounding = INT64_C(1) << (shift - 1);
  params->rndnu_scalar.multiplier = multiplier;
  params->rndnu_scalar.shift = shift;
  params->rndnu_scalar.rounding = rounding;
  params->rndnu_scalar.output_min = output_min;
  params->rndnu_scalar.output_max = output_max;
  params->rndnu_scalar.output_zero_point = (int32_t) output_zero_point;
  return sizeof(params->rndnu_scalar);
}

size_t xnn_init_qu8_conv_minmax_rndnu16_scalar_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);
  struct ExpMul f32 = parse_f32(scale);

  int exp = f32.exp;
  int left_pre_shift = exp + 1;
  // multiplier_q15 is in the range [2^14, 2^15 - 1]
  int16_t multiplier_q15 = math_min_s32((1 << 15) - 1, math_asr_s32_rounding(f32.multiplier_q24, 9));

  params->rndnu16_scalar.kernel_zero_point = kernel_zero_point;
  params->rndnu16_scalar.multiplier = multiplier_q15;
  params->rndnu16_scalar.left_pre_shift = left_pre_shift;
  params->rndnu16_scalar.output_min = output_min;
  params->rndnu16_scalar.output_max = output_max;
  params->rndnu16_scalar.output_zero_point = (int16_t) output_zero_point;
  return sizeof(params->rndnu16_scalar);
}

#if XNN_ARCH_ARM
size_t xnn_init_qs8_conv_minmax_fp32_armsimd32_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_armsimd32.scale = scale;
  params->fp32_armsimd32.magic_bias = 12582912.0f;
  params->fp32_armsimd32.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_armsimd32.output_min = (uint32_t) (uint8_t) output_min * UINT32_C(0x01010101);
  params->fp32_armsimd32.output_max = (uint32_t) (uint8_t) output_max * UINT32_C(0x01010101);
  return sizeof(params->fp32_armsimd32);
}
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_conv_minmax_fp32_neon_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neon.scale = scale;
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_neon.output_min = output_min;
  params->fp32_neon.output_max = output_max;
  return sizeof(params->fp32_neon);
}

size_t xnn_init_qs8_conv_minmax_fp32_neonv8_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neonv8.scale = scale;
  params->fp32_neonv8.output_zero_point = (int16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
  return sizeof(params->fp32_neonv8);
}

size_t xnn_init_qs8_conv_minmax_rndnu_neon_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  // Compute requantization parameters.
  const uint32_t scale_bits = float_as_uint32(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [-8, 31] range.
  const int32_t shift = 127 + 31 - 32 - (scale_bits >> 23);
  assert(shift >= -8);
  assert(shift < 32);

  // Split shift into pre_shift + post_shift, post_shift in [1, 31] range.
  const int32_t post_shift = math_max_s32(shift, 1);
  const int32_t pre_shift = shift - post_shift;

  params->rndnu_neon.right_pre_shift = -pre_shift;
  params->rndnu_neon.multiplier = multiplier;
  params->rndnu_neon.right_post_shift = -post_shift;
  params->rndnu_neon.output_zero_point = (int16_t) output_zero_point;
  params->rndnu_neon.output_min = output_min;
  params->rndnu_neon.output_max = output_max;
  return sizeof(params->rndnu_neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

size_t xnn_init_qu8_conv_minmax_fp32_scalar_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar.kernel_zero_point = (int32_t) kernel_zero_point;
  params->fp32_scalar.scale = scale;
  params->fp32_scalar.output_min = output_min;
  params->fp32_scalar.output_max = output_max;
  params->fp32_scalar.output_zero_point = output_zero_point;
  return sizeof(params->fp32_scalar);
}

size_t xnn_init_qu8_conv_minmax_rndnu_scalar_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  // Compute requantization parameters.
  const uint32_t scale_bits = float_as_uint32(scale);

  // Multiplier is in [0x00800000, 0x007FFFFF] range.
  const int32_t multiplier = ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  assert(multiplier >= INT32_C(0x00800000));
  assert(multiplier <= INT32_C(0x00FFFFFF));

  // Shift is in [16, 55] range.
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 56);

  const int64_t rounding = INT64_C(1) << (shift - 1);

  params->rndnu_scalar.kernel_zero_point = (int32_t) kernel_zero_point;
  params->rndnu_scalar.multiplier = multiplier;
  params->rndnu_scalar.rounding = rounding;
  params->rndnu_scalar.shift = shift;
  params->rndnu_scalar.output_min = output_min;
  params->rndnu_scalar.output_max = output_max;
  params->rndnu_scalar.output_zero_point = (int32_t) output_zero_point;
  return sizeof(params->rndnu_scalar);
}

#if XNN_ARCH_ARM
size_t xnn_init_qu8_conv_minmax_fp32_armsimd32_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const int32_t minus_kernel_zero_point = -(int32_t) kernel_zero_point;
  params->fp32_armsimd32.scale = scale;
  params->fp32_armsimd32.magic_bias = 12582912.0f;
  params->fp32_armsimd32.minus_kernel_zero_point = (uint32_t) (uint16_t) minus_kernel_zero_point * UINT32_C(0x00010001);
  params->fp32_armsimd32.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_armsimd32.output_min = (uint32_t) output_min * UINT32_C(0x01010101);
  params->fp32_armsimd32.output_max = (uint32_t) output_max * UINT32_C(0x01010101);
  return sizeof(params->fp32_armsimd32);
}
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qu8_conv_minmax_fp32_neon_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neon.kernel_zero_point[0] = kernel_zero_point;
  params->fp32_neon.kernel_zero_point[1] = kernel_zero_point;
  params->fp32_neon.kernel_zero_point[2] = kernel_zero_point;
  params->fp32_neon.kernel_zero_point[3] = kernel_zero_point;
  params->fp32_neon.scale = scale;
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_neon.output_min = output_min;
  params->fp32_neon.output_max = output_max;
  return sizeof(params->fp32_neon);
}

size_t xnn_init_qu8_conv_minmax_fp32_neonv8_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neonv8.kernel_zero_point[0] = kernel_zero_point;
  params->fp32_neonv8.kernel_zero_point[1] = kernel_zero_point;
  params->fp32_neonv8.kernel_zero_point[2] = kernel_zero_point;
  params->fp32_neonv8.kernel_zero_point[3] = kernel_zero_point;
  params->fp32_neonv8.scale = scale;
  params->fp32_neonv8.output_zero_point = (int16_t) (uint16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
  return sizeof(params->fp32_neonv8);
}

size_t xnn_init_qu8_conv_minmax_rndnu_neon_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  // Compute requantization parameters.
  const uint32_t scale_bits = float_as_uint32(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [-8, 31] range.
  const int32_t shift = 127 + 31 - 32 - (scale_bits >> 23);
  assert(shift >= -8);
  assert(shift < 32);

  // Split shift into pre_shift + post_shift, post_shift in [1, 31] range.
  const int32_t post_shift = math_max_s32(shift, 1);
  const int32_t pre_shift = shift - post_shift;

  params->rndnu_neon.kernel_zero_point[0] = kernel_zero_point;
  params->rndnu_neon.kernel_zero_point[1] = kernel_zero_point;
  params->rndnu_neon.kernel_zero_point[2] = kernel_zero_point;
  params->rndnu_neon.kernel_zero_point[3] = kernel_zero_point;
  params->rndnu_neon.right_pre_shift = -pre_shift;
  params->rndnu_neon.multiplier = multiplier;
  params->rndnu_neon.right_post_shift = -post_shift;
  params->rndnu_neon.output_zero_point = (int16_t) (uint16_t) output_zero_point;
  params->rndnu_neon.output_min = output_min;
  params->rndnu_neon.output_max = output_max;
  return sizeof(params->rndnu_neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

void xnn_init_qs8_qc8w_scale_fp32_params(
  size_t channels,
  size_t channels_tile,
  size_t channels_subtile,
  size_t stride,
  size_t substride,
  size_t stride_offset,
  const float scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w)
{
  const size_t tiled_channels = round_down_po2(channels, channels_tile);
  size_t tile_start = 0;
  for (; tile_start < tiled_channels; tile_start += channels_tile) {
    const size_t tile_size = channels_tile;
    for (size_t tile_offset = 0; tile_offset < tile_size; tile_offset++) {
      unaligned_indexed_store_f32(packed_w, tile_offset, scale[tile_start + tile_offset]);
    }
    packed_w = (void*) ((uintptr_t) packed_w + stride);
  }

  packed_w = (void*) ((uintptr_t) packed_w - stride_offset);

  for (; tile_start < channels; tile_start += channels_subtile) {
    const size_t tile_size = min(channels - tile_start, channels_subtile);
    for (size_t tile_offset = 0; tile_offset < tile_size; tile_offset++) {
      unaligned_indexed_store_f32(packed_w, tile_offset, scale[tile_start + tile_offset]);
    }
    packed_w = (void*) ((uintptr_t) packed_w + substride);
  }
}

void xnn_init_qs8_to_qs8_qc8w_scale_fp32_params(
  size_t channels,
  size_t channels_tile,
  size_t channels_subtile,
  size_t stride,
  size_t substride,
  size_t stride_offset,
  const float scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w)
{
  const size_t tiled_channels = round_down_po2(channels, channels_tile);
  size_t tile_start = 0;
  for (; tile_start < tiled_channels; tile_start += channels_tile) {
    const size_t tile_size = channels_tile;
    for (size_t tile_offset = 0; tile_offset < tile_size; tile_offset++) {
      unaligned_indexed_store_f32(packed_w, tile_offset, *scale);
    }
    packed_w = (void*) ((uintptr_t) packed_w + stride);
  }

  packed_w = (void*) ((uintptr_t) packed_w - stride_offset);

  for (; tile_start < channels; tile_start += channels_subtile) {
    const size_t tile_size = min(channels - tile_start, channels_subtile);
    for (size_t tile_offset = 0; tile_offset < tile_size; tile_offset++) {
      unaligned_indexed_store_f32(packed_w, tile_offset, *scale);
    }
    packed_w = (void*) ((uintptr_t) packed_w + substride);
  }
}

void xnn_init_blockwise_scale_fp32_params(
  size_t channels,
  size_t channels_tile,
  size_t channels_subtile,
  size_t stride,
  size_t substride,
  size_t num_blocks,
  size_t block_stride,
  size_t stride_offset,
  const float scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w)
{
  void* packed_w_saved = packed_w;
  for (size_t block_start = 0; block_start < num_blocks; block_start++) {
    packed_w = (void*)((uintptr_t) packed_w_saved + block_start * block_stride);
    const size_t tiled_channels = round_down_po2(channels, channels_tile);
    size_t tile_start = 0;
    for (; tile_start < tiled_channels; tile_start += channels_tile) {
      const size_t tile_size = channels_tile;
      for (size_t tile_offset = 0; tile_offset < tile_size; tile_offset++) {
        size_t scale_index = (tile_start + tile_offset) * num_blocks + block_start;
        // 1/16 because the weight are << 4 in the innermost loop to save a shift
        unaligned_indexed_store_f32(packed_w, tile_offset, scale[scale_index] / 16.0f);
      }
      packed_w = (void*) ((uintptr_t) packed_w + stride);
    }

    packed_w = (void*) ((uintptr_t) packed_w - stride_offset);

    for (; tile_start < channels; tile_start += channels_subtile) {
      const size_t tile_size = min(channels - tile_start, channels_subtile);
      for (size_t tile_offset = 0; tile_offset < tile_size; tile_offset++) {
        size_t scale_index = (tile_start + tile_offset) * num_blocks + block_start;
        // 1/16 because the weight are << 4 in the innermost loop to save a shift
        unaligned_indexed_store_f32(packed_w, tile_offset, scale[scale_index] / 16.0f);
      }
      packed_w = (void*) ((uintptr_t) packed_w + substride);
    }
  }
}

void xnn_init_blockwise_scale_bf16_params(
  size_t channels,
  size_t channels_tile,
  size_t channels_subtile,
  size_t stride,
  size_t substride,
  size_t num_blocks,
  size_t block_stride,
  size_t stride_offset,
  const xnn_bfloat16 scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w)
{
  void* packed_w_saved = packed_w;
  for (size_t block_start = 0; block_start < num_blocks; block_start++) {
    packed_w = (void*)((uintptr_t) packed_w_saved + block_start * block_stride);
    const size_t tiled_channels = round_down_po2(channels, channels_tile);
    size_t tile_start = 0;
    for (; tile_start < tiled_channels; tile_start += channels_tile) {
      const size_t tile_size = channels_tile;
      for (size_t tile_offset = 0; tile_offset < tile_size; tile_offset++) {
        size_t scale_index = (tile_start + tile_offset) * num_blocks + block_start;
        // 1/16 because the weight are << 4 in the innermost loop to save a shift
        float scale_16 = math_cvt_bf16_fp32(xnn_bfloat16_to_float(scale[scale_index]) / 16.0f);
        unaligned_indexed_store_u16(packed_w, tile_offset, scale_16);
      }
      packed_w = (void*) ((uintptr_t) packed_w + stride);
    }

    packed_w = (void*) ((uintptr_t) packed_w - stride_offset);

    for (; tile_start < channels; tile_start += channels_subtile) {
      const size_t tile_size = min(channels - tile_start, channels_subtile);
      for (size_t tile_offset = 0; tile_offset < tile_size; tile_offset++) {
        size_t scale_index = (tile_start + tile_offset) * num_blocks + block_start;
        // 1/16 because the weight are << 4 in the innermost loop to save a shift
        float scale_16 = math_cvt_bf16_fp32(xnn_bfloat16_to_float(scale[scale_index]) / 16.0f);
        unaligned_indexed_store_u16(packed_w, tile_offset, scale_16);
      }
      packed_w = (void*) ((uintptr_t) packed_w + substride);
    }
  }
}

size_t xnn_init_qu8_avgpool_minmax_fp32_scalar_params(
  struct xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar.init_bias = init_bias;
  params->fp32_scalar.scale = scale;
  params->fp32_scalar.output_zero_point = output_zero_point;
  params->fp32_scalar.output_min = output_min;
  params->fp32_scalar.output_max = output_max;
  return sizeof(params->fp32_scalar);
}

void xnn_update_qu8_avgpool_minmax_fp32_scalar_params(
  struct xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar.init_bias = init_bias;
  params->fp32_scalar.scale = scale;
}

size_t xnn_init_f16_scale_scalar_params(
  struct xnn_f16_scale_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale)
{
  params->scalar.scale = scale;
  return sizeof(params[0]);
}

size_t xnn_init_f16_f32acc_scale_scalar_params(
  struct xnn_f16_f32acc_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  params->scalar.scale = scale;
  return sizeof(params[0]);
}

size_t xnn_init_f32_scale_scalar_params(
  struct xnn_f32_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  params->scalar.scale = scale;
  return sizeof(params[0]);
}

void xnn_update_f32_scaleminmax_scalar_params(
  struct xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  params->scalar.scale = scale;
}

size_t xnn_init_f16_scaleminmax_scalar_params(
  struct xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale,
  xnn_float16 min,
  xnn_float16 max)
{
  params->scalar.scale = scale;
  params->scalar.min = min;
  params->scalar.max = max;
  return sizeof(params->scalar);
}

void xnn_update_f16_scaleminmax_scalar_params(
  struct xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale)
{
  params->scalar.scale = scale;
}


size_t xnn_init_f32_scaleminmax_scalar_params(
  struct xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max)
{
  params->scalar.scale = scale;
  params->scalar.min = min;
  params->scalar.max = max;
  return sizeof(params->scalar);
}

size_t xnn_init_bf16_minmax_scalar_params(
  struct xnn_bf16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_bfloat16 output_min,
  xnn_bfloat16 output_max)
{
  params->scalar.min = xnn_bfloat16_to_float(output_min);
  params->scalar.max = xnn_bfloat16_to_float(output_max);
  return sizeof(params->scalar);
}

size_t xnn_init_f16_minmax_scalar_params(
  union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 min,
  xnn_float16 max)
{
  params->scalar.min = min;
  params->scalar.max = max;
  return sizeof(params->scalar);
}

size_t xnn_init_f16_qc4w_minmax_scalar_params(
  struct xnn_f16_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 output_min,
  xnn_float16 output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->scalar.min = output_min;
  params->scalar.max = output_max;
  return sizeof(params->scalar);
}

size_t xnn_init_f16_qb4w_minmax_scalar_params(
  struct xnn_f16_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 output_min,
  xnn_float16 output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  params->scalar.min = output_min;
  params->scalar.max = output_max;
  params->scalar.blocksize = blocksize;
  return sizeof(params->scalar);
}

size_t xnn_init_f32_minmax_scalar_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  params->scalar.min = output_min;
  params->scalar.max = output_max;
  return sizeof(params->scalar);
}

size_t xnn_init_f32_qc4w_minmax_scalar_params(
  struct xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->scalar.min = output_min;
  params->scalar.max = output_max;
  params->scalar.kernel_zero_point = (int32_t) kernel_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_f32_qb4w_minmax_scalar_params(
  struct xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  params->scalar.min = output_min;
  params->scalar.max = output_max;
  params->scalar.blocksize = blocksize;
  return sizeof(params->scalar);
}

size_t xnn_init_f16_elu_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->f16_elu.scalar.prescale = xnn_float16_from_float(1.0f);
  params->f16_elu.scalar.alpha = xnn_float16_from_float(op_params->elu.alpha);
  params->f16_elu.scalar.beta = xnn_float16_from_float(1.0f);
  return sizeof(params->f16_elu);
}


size_t xnn_init_f32_elu_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->f32_elu.scalar.prescale = 1.0f;
  params->f32_elu.scalar.alpha = op_params->elu.alpha;
  params->f32_elu.scalar.beta = 1.0f;
  return sizeof(params->f32_elu);
}


size_t xnn_init_f16_lrelu_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->f16_lrelu.scalar.slope =
      xnn_float16_from_float(op_params->leaky_relu.negative_slope);
  return sizeof(params->f16_lrelu);
}

size_t xnn_init_f32_lrelu_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->f32_lrelu.scalar.slope = op_params->leaky_relu.negative_slope;
  return sizeof(params->f32_lrelu);
}

size_t xnn_init_qs8_lrelu_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  const float negative_slope = op_params->leaky_relu.negative_slope;
  const float input_scale = input_quantization->scale;
  const float output_scale = output_quantization->scale;
  const float positive_scale = input_scale / output_scale;
  const float negative_scale = positive_scale * negative_slope;

  assert(positive_scale >= 0x1.0p-8f);
  assert(positive_scale <= 0x1.0p+7f);
  assert(negative_scale <= 0x1.0p+7f);
  assert(negative_scale >= -0x1.FFFC00p+6f);
  assert(fabsf(negative_scale) >= 0x1.0p-8f);

  const long positive_multiplier = lrintf(256.0f * positive_scale);
  assert(positive_multiplier >= 1L);
  assert(positive_multiplier <= 32768L);
  const long negative_multiplier = lrintf(256.0f * negative_scale);
  assert(negative_multiplier <= 32768L);
  assert(negative_multiplier >= -32767L);
  assert(negative_multiplier != 0L);
  params->qs8_lrelu.scalar.input_zero_point = input_quantization->zero_point;
  params->qs8_lrelu.scalar.positive_multiplier = positive_multiplier;
  params->qs8_lrelu.scalar.negative_multiplier = negative_multiplier;
  params->qs8_lrelu.scalar.output_zero_point = output_quantization->zero_point;
  return sizeof(params->qs8_lrelu);
}

size_t xnn_init_qu8_lrelu_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  const float negative_slope = op_params->leaky_relu.negative_slope;
  const float input_scale = input_quantization->scale;
  const float output_scale = output_quantization->scale;
  const float positive_scale = input_scale / output_scale;
  const float negative_scale = positive_scale * negative_slope;

  assert(positive_scale >= 0x1.0p-8f);
  assert(positive_scale <= 0x1.0p+7f);
  assert(negative_scale <= 0x1.0p+7f);
  assert(negative_scale >= -0x1.FFFC00p+6f);
  assert(fabsf(negative_scale) >= 0x1.0p-8f);

  const long positive_multiplier = lrintf(256.0f * positive_scale);
  assert(positive_multiplier >= 1L);
  assert(positive_multiplier <= 32768L);
  const long negative_multiplier = lrintf(256.0f * negative_scale);
  assert(negative_multiplier <= 32768L);
  assert(negative_multiplier >= -32767L);
  assert(negative_multiplier != 0L);
  params->qu8_lrelu.scalar.input_zero_point = input_quantization->zero_point;
  params->qu8_lrelu.scalar.positive_multiplier = positive_multiplier;
  params->qu8_lrelu.scalar.negative_multiplier = negative_multiplier;
  params->qu8_lrelu.scalar.output_zero_point = output_quantization->zero_point;
  return sizeof(params->qu8_lrelu);
}

size_t xnn_init_qu8_clamp_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  assert(input_quantization->scale == output_quantization->scale);
  assert(input_quantization->zero_point == output_quantization->zero_point);
  params->u8_minmax.scalar.min = xnn_qu8_quantize(op_params->clamp.min, output_quantization->scale, output_quantization->zero_point);
  params->u8_minmax.scalar.max = xnn_qu8_quantize(op_params->clamp.max, output_quantization->scale, output_quantization->zero_point);
  return sizeof(params->u8_minmax);
}

size_t xnn_init_qs8_clamp_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  assert(input_quantization->scale == output_quantization->scale);
  assert(input_quantization->zero_point == output_quantization->zero_point);
  params->s8_minmax.scalar.min = xnn_qs8_quantize(op_params->clamp.min, output_quantization->scale, output_quantization->zero_point);
  params->s8_minmax.scalar.max = xnn_qs8_quantize(op_params->clamp.max, output_quantization->scale, output_quantization->zero_point);
  return sizeof(params->s8_minmax);
}

size_t xnn_init_f16_clamp_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->f16_minmax.scalar.min = xnn_float16_from_float(op_params->clamp.min);
  params->f16_minmax.scalar.max = xnn_float16_from_float(op_params->clamp.max);
  return sizeof(params->f16_minmax);
}

size_t xnn_init_f32_clamp_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->f32_minmax.scalar.min = op_params->clamp.min;
  params->f32_minmax.scalar.max = op_params->clamp.max;
  return sizeof(params->f32_minmax);
}

size_t xnn_init_s8_minmax_scalar_params(
  struct xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max)
{
  assert(output_min < output_max);

  params->scalar.min = (int32_t) output_min;
  params->scalar.max = (int32_t) output_max;
  return sizeof(params->scalar);
}

size_t xnn_init_u8_minmax_scalar_params(
  struct xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  params->scalar.min = (uint32_t) output_min;
  params->scalar.max = (uint32_t) output_max;
  return sizeof(params->scalar);
}

size_t xnn_init_f16_minmax_binary_params(
    union xnn_f16_minmax_params uparams[XNN_MIN_ELEMENTS(1)],
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization) {
  uparams->scalar.min = xnn_float16_from_float(-INFINITY);
  uparams->scalar.max = xnn_float16_from_float(+INFINITY);
  return sizeof(uparams->scalar);
}

size_t xnn_init_f32_minmax_binary_params(
    union xnn_f32_minmax_params uparams[XNN_MIN_ELEMENTS(1)],
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization) {
  uparams->scalar.min = -INFINITY;
  uparams->scalar.max = +INFINITY;
  return sizeof(uparams->scalar);
}

size_t xnn_init_qu8_add_minmax_scalar_params(
    struct xnn_qu8_add_minmax_params uparams[XNN_MIN_ELEMENTS(1)],
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization) {
  assert(a_quantization);
  assert(b_quantization);
  assert(output_quantization);
  const float a_output_scale = a_quantization->scale / output_quantization->scale;
  const float b_output_scale = b_quantization->scale / output_quantization->scale;
  const float abs_a_output_scale = fabsf(a_output_scale);
  const float abs_b_output_scale = fabsf(b_output_scale);
  assert(abs_a_output_scale >= 0x1.0p-10f);
  assert(abs_b_output_scale >= 0x1.0p-10f);
  assert(abs_a_output_scale < 0x1.0p+8f);
  assert(abs_b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_abs_output_scale = math_max_f32(abs_a_output_scale, abs_b_output_scale);
  assert(max_abs_output_scale >= 0x1.0p-10f);
  assert(max_abs_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = float_as_uint32(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(uint32_as_float(float_as_uint32(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(uint32_as_float(float_as_uint32(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  uparams->scalar.bias = rounding -
                         a_multiplier * (int32_t)(uint32_t)a_quantization->zero_point -
                         b_multiplier * (int32_t)(uint32_t)b_quantization->zero_point;
  uparams->scalar.a_zero_point = a_quantization->zero_point;
  uparams->scalar.b_zero_point = b_quantization->zero_point;
  uparams->scalar.a_multiplier = a_multiplier;
  uparams->scalar.b_multiplier = b_multiplier;
  uparams->scalar.shift = shift;
  uparams->scalar.output_min = 0;
  uparams->scalar.output_max = UINT8_MAX;
  uparams->scalar.output_zero_point = (int32_t)(uint32_t)output_quantization->zero_point;
  return sizeof(uparams->scalar);
}

size_t xnn_init_qs8_add_minmax_scalar_params(
    struct xnn_qs8_add_minmax_params uparams[XNN_MIN_ELEMENTS(1)],
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization) {
  assert(a_quantization);
  assert(b_quantization);
  assert(output_quantization);
  const float a_output_scale = a_quantization->scale / output_quantization->scale;
  const float b_output_scale = b_quantization->scale / output_quantization->scale;
  const float abs_a_output_scale = fabsf(a_output_scale);
  const float abs_b_output_scale = fabsf(b_output_scale);
  assert(abs_a_output_scale >= 0x1.0p-10f);
  assert(abs_b_output_scale >= 0x1.0p-10f);
  assert(abs_a_output_scale < 0x1.0p+8f);
  assert(abs_b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_abs_output_scale = math_max_f32(abs_a_output_scale, abs_b_output_scale);
  assert(max_abs_output_scale >= 0x1.0p-10f);
  assert(max_abs_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = float_as_uint32(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(uint32_as_float(float_as_uint32(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(uint32_as_float(float_as_uint32(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  uparams->scalar.bias = rounding - a_multiplier * (int32_t)a_quantization->zero_point -
                         b_multiplier * (int32_t)b_quantization->zero_point;
  uparams->scalar.a_zero_point = a_quantization->zero_point;
  uparams->scalar.b_zero_point = b_quantization->zero_point;
  uparams->scalar.a_multiplier = a_multiplier;
  uparams->scalar.b_multiplier = b_multiplier;
  uparams->scalar.shift = shift;
  uparams->scalar.output_zero_point = (int32_t)output_quantization->zero_point;
  uparams->scalar.output_min = INT8_MIN;
  uparams->scalar.output_max = INT8_MAX;
  return sizeof(uparams->scalar);
}

size_t xnn_init_qu8_mul_minmax_scalar_params(
    union xnn_qu8_mul_minmax_params uparams[XNN_MIN_ELEMENTS(1)],
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization) {
  assert(a_quantization);
  assert(b_quantization);
  assert(output_quantization);
  const float product_scale = a_quantization->scale * b_quantization->scale;
  const float product_output_scale = product_scale / output_quantization->scale;
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  uparams->scalar.a_zero_point = a_quantization->zero_point;
  uparams->scalar.b_zero_point = b_quantization->zero_point;
  uparams->scalar.scale = product_output_scale;
  uparams->scalar.output_zero_point = output_quantization->zero_point;
  uparams->scalar.output_min = 0;
  uparams->scalar.output_max = UINT8_MAX;
  return sizeof(uparams->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qu8_mul_minmax_rndnu_neon_params(
    union xnn_qu8_mul_minmax_params uparams[XNN_MIN_ELEMENTS(1)],
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization) {
  assert(a_quantization);
  assert(b_quantization);
  assert(output_quantization);

  const float product_scale = a_quantization->scale * b_quantization->scale;
  const float product_output_scale = product_scale / output_quantization->scale;
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const uint32_t scale_bits = float_as_uint32(product_output_scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [-8, 15] range.
  const int32_t shift = 127 + 31 - 32 - (scale_bits >> 23);
  assert(shift >= -8);
  assert(shift < 16);

  // Split shift into pre_shift + post_shift, post_shift in [1, 15] range.
  const int32_t post_shift = math_max_s32(shift, 1);
  const int32_t pre_shift = shift - post_shift;

  uparams->rndnu_neon.a_zero_point = a_quantization->zero_point;
  uparams->rndnu_neon.b_zero_point = b_quantization->zero_point;
  uparams->rndnu_neon.left_pre_shift = -pre_shift;
  uparams->rndnu_neon.multiplier = multiplier;
  uparams->rndnu_neon.left_post_shift = -post_shift;
  uparams->rndnu_neon.output_zero_point = (int16_t)output_quantization->zero_point;
  uparams->rndnu_neon.output_min = 0;
  uparams->rndnu_neon.output_max = UINT8_MAX;
  return sizeof(uparams->rndnu_neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

size_t xnn_init_qs8_mul_minmax_scalar_params(
    union xnn_qs8_mul_minmax_params uparams[XNN_MIN_ELEMENTS(1)],
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization) {
  assert(a_quantization);
  assert(b_quantization);
  assert(output_quantization);
  const float product_scale = a_quantization->scale * b_quantization->scale;
  const float product_output_scale = product_scale / output_quantization->scale;
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  uparams->scalar.a_zero_point = a_quantization->zero_point;
  uparams->scalar.b_zero_point = b_quantization->zero_point;
  uparams->scalar.scale = product_output_scale;
  uparams->scalar.output_zero_point = output_quantization->zero_point;
  uparams->scalar.output_min = INT8_MIN;
  uparams->scalar.output_max = INT8_MAX;
  return sizeof(uparams->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_mul_minmax_rndnu_neon_params(
    union xnn_qs8_mul_minmax_params uparams[XNN_MIN_ELEMENTS(1)],
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization) {
  assert(a_quantization);
  assert(b_quantization);
  assert(output_quantization);
  const float product_scale = a_quantization->scale * b_quantization->scale;
  const float product_output_scale = product_scale / output_quantization->scale;
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const uint32_t scale_bits = float_as_uint32(product_output_scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [-8, 15] range.
  const int32_t shift = 127 + 31 - 32 - (scale_bits >> 23);
  assert(shift >= -8);
  assert(shift < 16);

  // Split shift into pre_shift + post_shift, post_shift in [1, 15] range.
  const int32_t post_shift = math_max_s32(shift, 1);
  const int32_t pre_shift = shift - post_shift;

  uparams->rndnu_neon.a_zero_point = a_quantization->zero_point;
  uparams->rndnu_neon.b_zero_point = b_quantization->zero_point;
  uparams->rndnu_neon.left_pre_shift = -pre_shift;
  uparams->rndnu_neon.multiplier = multiplier;
  uparams->rndnu_neon.left_post_shift = -post_shift;
  uparams->rndnu_neon.output_zero_point = (int16_t)output_quantization->zero_point;
  uparams->rndnu_neon.output_min = INT8_MIN;
  uparams->rndnu_neon.output_max = INT8_MAX;
  return sizeof(uparams->rndnu_neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

size_t xnn_init_f16_qs8_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->f16_qs8_cvt.scalar.scale = xnn_float16_from_float(1.0f / output_quantization->scale);
  params->f16_qs8_cvt.scalar.output_zero_point = output_quantization->zero_point;
  return sizeof(params->f16_qs8_cvt);
}

size_t xnn_init_f32_qs8_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->f32_qs8_cvt.scalar.scale = 1.0f / output_quantization->scale;
  params->f32_qs8_cvt.scalar.output_zero_point = (int16_t) output_quantization->zero_point;
  return sizeof(params->f32_qs8_cvt);
}

size_t xnn_init_qs8_reduce_minmax_scalar_params(
  struct xnn_qs8_reduce_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int32_t num_elements,
  int8_t input_zero_point,
  int8_t output_zero_point)
{
  params->scalar.scale = scale;
  params->scalar.num_elements = num_elements;
  params->scalar.input_zero_point = input_zero_point;
  params->scalar.output_zero_point = output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_qu8_reduce_minmax_scalar_params(
  struct xnn_qu8_reduce_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int32_t num_elements,
  uint8_t input_zero_point,
  uint8_t output_zero_point)
{
  params->scalar.scale = scale;
  params->scalar.num_elements = num_elements;
  params->scalar.input_zero_point = input_zero_point;
  params->scalar.output_zero_point = output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_f32_qu8_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->f32_qu8_cvt.scalar.scale = 1.0f / output_quantization->scale;
  params->f32_qu8_cvt.scalar.output_zero_point = (int16_t) output_quantization->zero_point;
  return sizeof(params->f32_qu8_cvt);
}

size_t xnn_init_s32_f32_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->s32_f32_cvt.scalar.zero_point = input_quantization->zero_point;
  return sizeof(params->s32_f32_cvt);
}

size_t xnn_init_u32_f32_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->u32_f32_cvt.scalar.zero_point = input_quantization->zero_point;
  return sizeof(params->u32_f32_cvt);
}

size_t xnn_init_qs8_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  const float input_output_scale = input_quantization->scale / output_quantization->scale;
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(256.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 32768L);
  params->qs8_cvt.scalar.input_zero_point = (int16_t) input_quantization->zero_point;
  params->qs8_cvt.scalar.multiplier = (int32_t) multiplier;
  params->qs8_cvt.scalar.output_zero_point = (int16_t) output_quantization->zero_point;
  return sizeof(params->qs8_cvt);
}

size_t xnn_init_qs16_qs8_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  const float input_output_scale = input_quantization->scale / output_quantization->scale;
  assert(input_output_scale >= 0x1.0p-16);
  assert(input_output_scale <= 0x1.0p+8);

  const long multiplier = lrintf(65536.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 0x01000000L);
  params->qs16_qs8_cvt.scalar.multiplier = (int32_t) multiplier;
  params->qs16_qs8_cvt.scalar.output_zero_point = (int32_t) output_quantization->zero_point;
  return sizeof(params->qs16_qs8_cvt);
}

size_t xnn_init_qs8_f32_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->qs8_f32_cvt.scalar.zero_point = (int32_t) input_quantization->zero_point;
  params->qs8_f32_cvt.scalar.scale = input_quantization->scale;
  return sizeof(params->qs8_f32_cvt);
}

size_t xnn_init_qs8_f16_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->qs8_f16_cvt.scalar.zero_point = (int16_t) input_quantization->zero_point;
  params->qs8_f16_cvt.scalar.scale = xnn_float16_from_float(input_quantization->scale);
  return sizeof(params->qs8_f16_cvt);
}

size_t xnn_init_qu8_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  const float input_output_scale = input_quantization->scale / output_quantization->scale;
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(256.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 32768L);
  params->qu8_cvt.scalar.input_zero_point = (uint16_t) input_quantization->zero_point;
  params->qu8_cvt.scalar.multiplier = (int32_t) multiplier;
  params->qu8_cvt.scalar.output_zero_point = (int16_t) output_quantization->zero_point;
  return sizeof(params->qu8_cvt);
}

size_t xnn_init_qu8_f32_cvt_scalar_params(
  union xnn_unary_uparams* params,
  const union xnn_unary_params* op_params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization)
{
  params->qu8_f32_cvt.scalar.zero_point = (int32_t) input_quantization->zero_point;
  params->qu8_f32_cvt.scalar.scale = input_quantization->scale;
  return sizeof(params->qu8_f32_cvt);
}
