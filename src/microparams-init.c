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


size_t xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_fmagic.init_bias = init_bias;
  params->fp32_scalar_fmagic.scale = scale;
  params->fp32_scalar_fmagic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_fmagic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_fmagic.magic_bias = 12582912.0f;
  params->fp32_scalar_fmagic.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_fmagic);
}

void xnn_update_qs8_avgpool_minmax_fp32_scalar_fmagic_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_fmagic.init_bias = init_bias;
  params->fp32_scalar_fmagic.scale = scale;
}

size_t xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_imagic.init_bias = init_bias;
  params->fp32_scalar_imagic.scale = scale;
  params->fp32_scalar_imagic.magic_bias = 12582912.0f;
  params->fp32_scalar_imagic.magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  params->fp32_scalar_imagic.magic_max = (int32_t) float_as_uint32(12582912.0f + output_max_less_zero_point);
  params->fp32_scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_imagic);
}

void xnn_update_qs8_avgpool_minmax_fp32_scalar_imagic_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_imagic.init_bias = init_bias;
  params->fp32_scalar_imagic.scale = scale;
}

size_t xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_lrintf.init_bias = init_bias;
  params->fp32_scalar_lrintf.scale = scale;
  params->fp32_scalar_lrintf.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_lrintf.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_lrintf.output_zero_point = (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_lrintf);
}

void xnn_update_qs8_avgpool_minmax_fp32_scalar_lrintf_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_lrintf.init_bias = init_bias;
  params->fp32_scalar_lrintf.scale = scale;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs8_avgpool_minmax_fp32_sse2_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.init_bias[i] = init_bias;
    params->fp32_sse2.scale[i] = scale;
    params->fp32_sse2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.output_zero_point[i] = (int16_t) output_zero_point;
    params->fp32_sse2.output_min[i] = (int16_t) output_min;
  }
  return sizeof(params->fp32_sse2);
}

void xnn_update_qs8_avgpool_minmax_fp32_sse2_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.init_bias[i] = init_bias;
    params->fp32_sse2.scale[i] = scale;
  }
}

// Same as NEON.  Used for rsum ssse3
size_t xnn_init_qs8_avgpool_minmax_fp32_ssse3_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_ssse3.init_bias = init_bias;
  params->fp32_ssse3.scale = scale;
  params->fp32_ssse3.magic_bias = 12582912.0f;
  params->fp32_ssse3.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_ssse3.output_min = output_min;
  params->fp32_ssse3.output_max = output_max;
  return sizeof(params->fp32_ssse3);
}

size_t xnn_init_qs8_avgpool_minmax_fp32_sse4_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse4.init_bias[i] = init_bias;
    params->fp32_sse4.scale[i] = scale;
    params->fp32_sse4.magic_bias[i] = 12582912.0f;
    params->fp32_sse4.magic_bias_less_output_zero_point[i] = INT32_C(0x4B400000) - (int32_t) output_zero_point;
    params->fp32_sse4.output_max_less_zero_point[i] = output_max_less_zero_point;
    params->fp32_sse4.magic_bias_less_output_zero_point[i] = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse4.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_sse4.output_min[i] = output_min;
    params->fp32_sse4.output_max[i] = output_max;
  }
  return sizeof(params->fp32_sse4);
}

size_t xnn_init_qs8_avgpool_minmax_fp32_avx2_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_avx2.init_bias[i] = init_bias;
    params->fp32_avx2.scale[i] = scale;
    params->fp32_avx2.magic_bias[i] = 12582912.0f;
    params->fp32_avx2.magic_bias_less_output_zero_point[i] = INT32_C(0x4B400000) - (int32_t) output_zero_point;
    params->fp32_avx2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx2.output_min[i] = output_min;
    params->fp32_avx2.output_max[i] = output_max;
  }
  return sizeof(params->fp32_avx2);
}

size_t xnn_init_qs8_avgpool_minmax_fp32_avx512_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx512.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx512.init_bias[i] = init_bias;
    params->fp32_avx512.scale[i] = scale;
    params->fp32_avx512.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 64; i++) {
    params->fp32_avx512.output_min[i] = output_min;
  }
  return sizeof(params->fp32_avx512);
}

void xnn_update_qs8_avgpool_minmax_fp32_sse4_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse4.init_bias[i] = init_bias;
    params->fp32_sse4.scale[i] = scale;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_avgpool_minmax_fp32_neon_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neon.init_bias = init_bias;
  params->fp32_neon.scale = scale;
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_neon.output_min = output_min;
  params->fp32_neon.output_max = output_max;
  return sizeof(params->fp32_neon);
}

void xnn_update_qs8_avgpool_minmax_fp32_neon_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neon.init_bias = init_bias;
  params->fp32_neon.scale = scale;
}

size_t xnn_init_qs8_avgpool_minmax_fp32_neonv8_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neonv8.init_bias = init_bias;
  params->fp32_neonv8.scale = scale;
  params->fp32_neonv8.output_zero_point = (int16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
  return sizeof(params->fp32_neonv8);
}

void xnn_update_qs8_avgpool_minmax_fp32_neonv8_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neonv8.init_bias = init_bias;
  params->fp32_neonv8.scale = scale;
}

size_t xnn_init_qs8_avgpool_minmax_rndnu_neon_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
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

  params->rndnu_neon.init_bias = init_bias;
  params->rndnu_neon.left_pre_shift = -pre_shift;
  params->rndnu_neon.multiplier = multiplier;
  params->rndnu_neon.left_post_shift = -post_shift;
  params->rndnu_neon.output_zero_point = (int16_t) output_zero_point;
  params->rndnu_neon.output_min = output_min;
  params->rndnu_neon.output_max = output_max;
  return sizeof(params->rndnu_neon);
}

void xnn_update_qs8_avgpool_minmax_rndnu_neon_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
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

  params->rndnu_neon.init_bias = init_bias;
  params->rndnu_neon.left_pre_shift = -pre_shift;
  params->rndnu_neon.multiplier = multiplier;
  params->rndnu_neon.left_post_shift = -post_shift;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  const int32_t magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 2; i++) {
    params->fp32_wasmsimd.init_bias[i] = init_bias;
    params->fp32_wasmsimd.scale[i] = scale;
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_min[i] = magic_min;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = magic_bias_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_wasmsimd.output_max[i] = output_max;
  }
  return sizeof(params->fp32_wasmsimd);
}

void xnn_update_qs8_avgpool_minmax_fp32_wasmsimd_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  for (uint32_t i = 0; i < 2; i++) {
    params->fp32_wasmsimd.init_bias[i] = init_bias;
    params->fp32_wasmsimd.scale[i] = scale;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qu8_avgpool_minmax_fp32_scalar_fmagic_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_fmagic.init_bias = init_bias;
  params->fp32_scalar_fmagic.scale = scale;
  params->fp32_scalar_fmagic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_fmagic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_fmagic.magic_bias = 12582912.0f;
  params->fp32_scalar_fmagic.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_fmagic);
}

void xnn_update_qu8_avgpool_minmax_fp32_scalar_fmagic_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_fmagic.init_bias = init_bias;
  params->fp32_scalar_fmagic.scale = scale;
}

size_t xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_imagic.init_bias = init_bias;
  params->fp32_scalar_imagic.scale = scale;
  params->fp32_scalar_imagic.magic_bias = 12582912.0f;
  params->fp32_scalar_imagic.magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  params->fp32_scalar_imagic.magic_max = (int32_t) float_as_uint32(12582912.0f + output_max_less_zero_point);
  params->fp32_scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_imagic);
}

void xnn_update_qu8_avgpool_minmax_fp32_scalar_imagic_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_imagic.init_bias = init_bias;
  params->fp32_scalar_imagic.scale = scale;
}

size_t xnn_init_qu8_avgpool_minmax_fp32_scalar_lrintf_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_lrintf.init_bias = init_bias;
  params->fp32_scalar_lrintf.scale = scale;
  params->fp32_scalar_lrintf.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_lrintf.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_lrintf.output_zero_point = (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_lrintf);
}

void xnn_update_qu8_avgpool_minmax_fp32_scalar_lrintf_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_lrintf.init_bias = init_bias;
  params->fp32_scalar_lrintf.scale = scale;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qu8_avgpool_minmax_fp32_sse2_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.init_bias[i] = init_bias;
    params->fp32_sse2.scale[i] = scale;
    params->fp32_sse2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_sse2.output_min[i] = output_min;
  }
  return sizeof(params->fp32_sse2);
}

void xnn_update_qu8_avgpool_minmax_fp32_sse2_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.init_bias[i] = init_bias;
    params->fp32_sse2.scale[i] = scale;
  }
}

size_t xnn_init_qu8_avgpool_minmax_fp32_sse4_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse4.init_bias[i] = init_bias;
    params->fp32_sse4.scale[i] = scale;
    params->fp32_sse4.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse4.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_sse4.output_min[i] = output_min;
  }
  return sizeof(params->fp32_sse4);
}

void xnn_update_qu8_avgpool_minmax_fp32_sse4_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse4.init_bias[i] = init_bias;
    params->fp32_sse4.scale[i] = scale;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qu8_avgpool_minmax_fp32_neon_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neon.init_bias = init_bias;
  params->fp32_neon.scale = scale;
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_neon.output_min = output_min;
  params->fp32_neon.output_max = output_max;
  return sizeof(params->fp32_neon);
}

void xnn_update_qu8_avgpool_minmax_fp32_neon_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neon.init_bias = init_bias;
  params->fp32_neon.scale = scale;
}

size_t xnn_init_qu8_avgpool_minmax_fp32_neonv8_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neonv8.init_bias = init_bias;
  params->fp32_neonv8.scale = scale;
  params->fp32_neonv8.output_zero_point = (int16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
  return sizeof(params->fp32_neonv8);
}

void xnn_update_qu8_avgpool_minmax_fp32_neonv8_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_neonv8.init_bias = init_bias;
  params->fp32_neonv8.scale = scale;
}

size_t xnn_init_qu8_avgpool_minmax_rndnu_neon_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
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

  params->rndnu_neon.init_bias = init_bias;
  params->rndnu_neon.left_pre_shift = -pre_shift;
  params->rndnu_neon.multiplier = multiplier;
  params->rndnu_neon.left_post_shift = -post_shift;
  params->rndnu_neon.output_zero_point = (int16_t) output_zero_point;
  params->rndnu_neon.output_min = output_min;
  params->rndnu_neon.output_max = output_max;
  return sizeof(params->rndnu_neon);
}

void xnn_update_qu8_avgpool_minmax_rndnu_neon_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
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

  params->rndnu_neon.init_bias = init_bias;
  params->rndnu_neon.left_pre_shift = -pre_shift;
  params->rndnu_neon.multiplier = multiplier;
  params->rndnu_neon.left_post_shift = -post_shift;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qu8_avgpool_minmax_fp32_wasmsimd_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  const int32_t magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 2; i++) {
    params->fp32_wasmsimd.init_bias[i] = init_bias;
    params->fp32_wasmsimd.scale[i] = scale;
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_min[i] = magic_min;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = magic_bias_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_wasmsimd.output_max[i] = output_max;
  }
  return sizeof(params->fp32_wasmsimd);
}

void xnn_update_qu8_avgpool_minmax_fp32_wasmsimd_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  for (uint32_t i = 0; i < 2; i++) {
    params->fp32_wasmsimd.init_bias[i] = init_bias;
    params->fp32_wasmsimd.scale[i] = scale;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

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

size_t xnn_init_f32_gavgpool_scalar_params(
  union xnn_f32_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  float multiplier,
  float output_min,
  float output_max,
  uint32_t width)
{
  params->scalar.multiplier = multiplier;
  params->scalar.output_min = output_min;
  params->scalar.output_max = output_max;

  const uint32_t w = (width - 1) & 3;
  params->scalar.mask[0] = UINT32_C(0xFFFFFFFF);
  params->scalar.mask[1] = -(int32_t) (w >= 1);
  params->scalar.mask[2] = -(int32_t) (w >= 2);
  params->scalar.mask[3] = -(int32_t) (w >= 3);
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f32_gavgpool_neon_params(
  union xnn_f32_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  float multiplier,
  float output_min,
  float output_max,
  uint32_t width)
{
  params->neon.multiplier = multiplier;
  params->neon.output_min = output_min;
  params->neon.output_max = output_max;

  const uint32_t w = (width - 1) & 3;
  params->neon.mask[0] = UINT32_C(0xFFFFFFFF);
  params->neon.mask[1] = -(uint32_t) (w >= 1);
  params->neon.mask[2] = -(uint32_t) (w >= 2);
  params->neon.mask[3] = -(uint32_t) (w >= 3);
  return sizeof(params->neon);
}
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_gavgpool_sse_params(
  union xnn_f32_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  float multiplier,
  float output_min,
  float output_max,
  uint32_t width)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.multiplier[i] = multiplier;
    params->sse.output_min[i] = output_min;
    params->sse.output_max[i] = output_max;
  }

  const uint32_t w = (width - 1) & 3;
  params->sse.mask[0] = UINT32_C(0xFFFFFFFF);
  params->sse.mask[1] = -(uint32_t) (w >= 1);
  params->sse.mask[2] = -(uint32_t) (w >= 2);
  params->sse.mask[3] = -(uint32_t) (w >= 3);
  return sizeof(params->sse);
}
#endif

size_t xnn_init_f16_gavgpool_scalar_params(
  union xnn_f16_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t multiplier,
  uint16_t output_min,
  uint16_t output_max,
  uint32_t width)
{
  params->scalar.multiplier = multiplier;
  params->scalar.output_min = output_min;
  params->scalar.output_max = output_max;

  const uint32_t w = (width - 1) & 7;
  params->scalar.mask[0] = UINT16_C(0xFFFF);
  params->scalar.mask[1] = -(uint16_t) (w >= 1);
  params->scalar.mask[2] = -(uint16_t) (w >= 2);
  params->scalar.mask[3] = -(uint16_t) (w >= 3);
  params->scalar.mask[4] = -(uint16_t) (w >= 4);
  params->scalar.mask[5] = -(uint16_t) (w >= 5);
  params->scalar.mask[6] = -(uint16_t) (w >= 6);
  params->scalar.mask[7] = -(uint16_t) (w >= 7);
  return sizeof(params->scalar);
}

void xnn_update_f32_gavgpool_params(
  union xnn_f32_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  float multiplier,
  uint32_t width)
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse.multiplier[i] = multiplier;
    }

    const uint32_t w = (width - 1) & 3;
    params->sse.mask[0] = UINT32_C(0xFFFFFFFF);
    params->sse.mask[1] = -(uint32_t) (w >= 1);
    params->sse.mask[2] = -(uint32_t) (w >= 2);
    params->sse.mask[3] = -(uint32_t) (w >= 3);
  #elif XNN_ARCH_ARM || XNN_ARCH_ARM64
    params->neon.multiplier = multiplier;

    const uint32_t w = (width - 1) & 3;
    params->neon.mask[0] = UINT32_C(0xFFFFFFFF);
    params->neon.mask[1] = -(uint32_t) (w >= 1);
    params->neon.mask[2] = -(uint32_t) (w >= 2);
    params->neon.mask[3] = -(uint32_t) (w >= 3);
  #else
    params->scalar.multiplier = multiplier;

    const uint32_t w = (width - 1) & 3;
    params->scalar.mask[0] = UINT32_C(0xFFFFFFFF);
    params->scalar.mask[1] = -(int32_t) (w >= 1);
    params->scalar.mask[2] = -(int32_t) (w >= 2);
    params->scalar.mask[3] = -(int32_t) (w >= 3);
  #endif
}

void xnn_update_f16_gavgpool_scalar_params(
  union xnn_f16_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t multiplier,
  uint32_t width)
{
  params->scalar.multiplier = multiplier;

  const uint32_t w = (width - 1) & 7;
  params->scalar.mask[0] = UINT16_C(0xFFFF);
  params->scalar.mask[1] = -(uint16_t) (w >= 1);
  params->scalar.mask[2] = -(uint16_t) (w >= 2);
  params->scalar.mask[3] = -(uint16_t) (w >= 3);
  params->scalar.mask[4] = -(uint16_t) (w >= 4);
  params->scalar.mask[5] = -(uint16_t) (w >= 5);
  params->scalar.mask[6] = -(uint16_t) (w >= 6);
  params->scalar.mask[7] = -(uint16_t) (w >= 7);
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

size_t xnn_init_qs8_hswish_scalar_params(
  union xnn_qs8_hswish_params params[XNN_MIN_ELEMENTS(1)],
  int16_t input_zero_point,
  int16_t output_zero_point,
  float input_scale,
  float output_scale)
{
  params->scalar.input_zero_point = (uint32_t) input_zero_point;
  params->scalar.output_zero_point= (int32_t) output_zero_point;
  const float divisor1 = 0x1.555556p-10f;
  const uint32_t input_scale_div = float_as_uint32(input_scale * divisor1);
  params->scalar.input_scale_div_exp = (int32_t) (input_scale_div >> 23) - 126;
  params->scalar.input_scale_div_mantissa = (int32_t) ((input_scale_div << 9) >> 18 | UINT16_C(0x4000));
  const float scale_ratio = input_scale / output_scale;
  assert(scale_ratio >= 0x1.0p-8f);
  assert(scale_ratio < 0x1.0p+7f);
  params->scalar.scale_ratio = (int32_t) lrintf(scale_ratio * 256.0f);
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs8_hswish_sse2_params(
  union xnn_qs8_hswish_params params[XNN_MIN_ELEMENTS(1)],
  int16_t input_zero_point,
  int16_t output_zero_point,
  float input_scale,
  float output_scale)
{
  const int16_t input_scale_div = (int16_t) -lrintf(256.0f * input_scale / 6.0f);
  const float scale_ratio = input_scale / output_scale;
  assert(scale_ratio >= 0x1.0p-8f);
  assert(scale_ratio < 0x1.0p+7f);
  const int16_t scale_ratio_param = (int16_t) -lrintf(scale_ratio * 256.0f);
  params->sse2.input_zero_point = input_zero_point;
  params->sse2.output_zero_point = output_zero_point;
  params->sse2.input_scale_div = input_scale_div;
  params->sse2.scale_ratio = scale_ratio_param;
  return sizeof(params->sse2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

size_t xnn_init_qu8_hswish_scalar_params(
  union xnn_qu8_hswish_params params[XNN_MIN_ELEMENTS(1)],
  int16_t input_zero_point,
  int16_t output_zero_point,
  float input_scale,
  float output_scale)
{
  params->scalar.input_zero_point = (uint32_t) input_zero_point;
  params->scalar.output_zero_point= (int32_t) output_zero_point;
  const float divisor1 = 0x1.555556p-10f;
  const uint32_t input_scale_div = float_as_uint32(input_scale * divisor1);
  params->scalar.input_scale_div_exp = (int32_t) (input_scale_div >> 23) - 126;
  params->scalar.input_scale_div_mantissa = (int32_t) ((input_scale_div << 9) >> 18 | UINT16_C(0x4000));
  const float scale_ratio = input_scale / output_scale;
  assert(scale_ratio >= 0x1.0p-8f);
  assert(scale_ratio < 0x1.0p+7f);
  params->scalar.scale_ratio = (int32_t) lrintf(scale_ratio * 256.0f);
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qu8_hswish_sse2_params(
  union xnn_qu8_hswish_params params[XNN_MIN_ELEMENTS(1)],
  int16_t input_zero_point,
  int16_t output_zero_point,
  float input_scale,
  float output_scale)
{
  const int16_t input_scale_div = (int16_t) -lrintf(256.0f * input_scale / 6.0f);
  const float scale_ratio = input_scale / output_scale;
  assert(scale_ratio >= 0x1.0p-8f);
  assert(scale_ratio < 0x1.0p+7f);
  const int16_t scale_ratio_param = (int16_t) -lrintf(scale_ratio * 256.0f);
  params->sse2.input_zero_point = input_zero_point;
  params->sse2.output_zero_point = output_zero_point;
  params->sse2.input_scale_div = input_scale_div;
  params->sse2.scale_ratio = scale_ratio_param;
  return sizeof(params->sse2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

size_t xnn_init_f16_elu_scalar_params(
  struct xnn_f16_elu_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 prescale,
  xnn_float16 alpha,
  xnn_float16 beta)
{
  params->scalar.prescale = prescale;
  params->scalar.alpha = alpha;
  params->scalar.beta = beta;
  return sizeof(params->scalar);
}


size_t xnn_init_f32_elu_scalar_params(
  struct xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->scalar.prescale = prescale;
  params->scalar.alpha = alpha;
  params->scalar.beta = beta;
  return sizeof(params->scalar);
}


size_t xnn_init_f16_lrelu_scalar_params(
  struct xnn_f16_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 slope)
{
  params->scalar.slope = slope;
  return sizeof(params->scalar);
}

size_t xnn_init_f32_lrelu_scalar_params(
  struct xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope)
{
  params->scalar.slope = slope;
  return sizeof(params->scalar);
}

size_t xnn_init_qs8_lrelu_scalar_params(
  struct xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float positive_scale,
  float negative_scale,
  int8_t input_zero_point,
  int8_t output_zero_point)
{
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
  params->scalar.input_zero_point = input_zero_point;
  params->scalar.positive_multiplier = positive_multiplier;
  params->scalar.negative_multiplier = negative_multiplier;
  params->scalar.output_zero_point = output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_qu8_lrelu_scalar_params(
  struct xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float positive_scale,
  float negative_scale,
  uint8_t input_zero_point,
  uint8_t output_zero_point)
{
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
  params->scalar.input_zero_point = input_zero_point;
  params->scalar.positive_multiplier = positive_multiplier;
  params->scalar.negative_multiplier = negative_multiplier;
  params->scalar.output_zero_point = output_zero_point;
  return sizeof(params->scalar);
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
  struct xnn_f16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->scalar.scale = scale;
  params->scalar.output_min = output_min;
  params->scalar.output_max = output_max;
  params->scalar.output_zero_point = output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_f32_qs8_cvt_scalar_params(
  struct xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->scalar.scale = scale;
  params->scalar.output_zero_point = (int16_t) output_zero_point;
  params->scalar.output_min = output_min;
  params->scalar.output_max = output_max;
  return sizeof(params->scalar);
}

size_t xnn_init_qs8_mean_minmax_scalar_params(
  struct xnn_qs8_mean_minmax_params params[XNN_MIN_ELEMENTS(1)],
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

size_t xnn_init_qu8_mean_minmax_scalar_params(
  struct xnn_qu8_mean_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  struct xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  params->scalar.scale = scale;
  params->scalar.output_zero_point = (int16_t) output_zero_point;
  params->scalar.output_min = output_min;
  params->scalar.output_max = output_max;
  return sizeof(params->scalar);
}

size_t xnn_init_s32_f32_cvt_scalar_params(
  struct xnn_s32_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  int32_t zero_point)
{
  params->scalar.zero_point = zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_u32_f32_cvt_scalar_params(
  struct xnn_u32_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  int32_t zero_point)
{
  params->scalar.zero_point = zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_qs8_cvt_scalar_params(
  struct xnn_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t input_zero_point,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(256.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 32768L);
  params->scalar.input_zero_point = (int16_t) input_zero_point;
  params->scalar.multiplier = (int32_t) multiplier;
  params->scalar.output_zero_point = (int16_t) output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_qs16_qs8_cvt_scalar_params(
  struct xnn_qs16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-16);
  assert(input_output_scale <= 0x1.0p+8);

  const long multiplier = lrintf(65536.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 0x01000000L);
  params->scalar.multiplier = (int32_t) multiplier;
  params->scalar.output_zero_point = (int32_t) output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_qs8_f32_cvt_scalar_params(
  struct xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  params->scalar.zero_point = (int32_t) zero_point;
  params->scalar.scale = scale;
  return sizeof(params->scalar);
}

size_t xnn_init_qs8_f16_cvt_scalar_params(
  struct xnn_qs8_f16_cvt_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale,
  int8_t zero_point)
{
  params->scalar.zero_point = (int16_t) zero_point;
  params->scalar.scale = scale;
  return sizeof(params->scalar);
}

size_t xnn_init_qu8_cvt_scalar_params(
  struct xnn_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  uint8_t input_zero_point,
  uint8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(256.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 32768L);
  params->scalar.input_zero_point = (uint16_t) input_zero_point;
  params->scalar.multiplier = (int32_t) multiplier;
  params->scalar.output_zero_point = (int16_t) output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_qu8_f32_cvt_scalar_params(
  struct xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  params->scalar.zero_point = (int32_t) zero_point;
  params->scalar.scale = scale;
  return sizeof(params->scalar);
}
