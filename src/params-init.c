// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <math.h>

#include <fp16.h>

#include <xnnpack/math.h>
#include <xnnpack/params-init.h>


void xnn_init_qu8_conv_minmax_gemmlowp_scalar_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  // Compute requantization parameters
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;

  params->gemmlowp_scalar.kernel_zero_point = (int32_t) (uint32_t) kernel_zero_point;
  params->gemmlowp_scalar.multiplier = multiplier;
  params->gemmlowp_scalar.remainder_mask = (int32_t) remainder_mask;
  params->gemmlowp_scalar.remainder_threshold = (int32_t) remainder_threshold;
  params->gemmlowp_scalar.shift = (uint32_t) shift;
  params->gemmlowp_scalar.output_min_less_zero_point =
    (int32_t) (uint32_t) output_min - (int32_t) (uint32_t) output_zero_point;
  params->gemmlowp_scalar.output_max_less_zero_point =
    (int32_t) (uint32_t) output_max - (int32_t) (uint32_t) output_zero_point;
  params->gemmlowp_scalar.output_zero_point = (int32_t) (uint32_t) output_zero_point;
}

void xnn_init_qu8_conv_minmax_fp32_scalar_lrint_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  params->fp32_scalar_lrint.kernel_zero_point = (int32_t) (uint32_t) kernel_zero_point;
  params->fp32_scalar_lrint.scale = scale;
  params->fp32_scalar_lrint.output_min_less_zero_point = (long) (int32_t) ((uint32_t) output_min - (uint32_t) output_zero_point);
  params->fp32_scalar_lrint.output_max_less_zero_point = (long) (int32_t) ((uint32_t) output_max - (uint32_t) output_zero_point);
  params->fp32_scalar_lrint.output_zero_point = (int32_t) (uint32_t) output_zero_point;
}

void xnn_init_qu8_conv_minmax_fp32_scalar_magic_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  params->fp32_scalar_magic.kernel_zero_point = (int32_t) (uint32_t) kernel_zero_point;
  params->fp32_scalar_magic.scale = scale;
  params->fp32_scalar_magic.output_min_less_zero_point = (float) (int32_t) ((uint32_t) output_min - (uint32_t) output_zero_point);
  params->fp32_scalar_magic.output_max_less_zero_point = (float) (int32_t) ((uint32_t) output_max - (uint32_t) output_zero_point);
  params->fp32_scalar_magic.magic_bias = 12582912.0f;
  params->fp32_scalar_magic.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) (uint32_t) output_zero_point;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qu8_conv_minmax_gemmlowp_sse2_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  for (uint32_t i = 0; i < 8; i++) {
    params->gemmlowp_sse2.kernel_zero_point[i] = (int16_t) (uint16_t) kernel_zero_point;
  }
  params->gemmlowp_sse2.multiplier[0] = multiplier;
  params->gemmlowp_sse2.multiplier[1] = multiplier;
  params->gemmlowp_sse2.multiplier[2] = multiplier;
  params->gemmlowp_sse2.multiplier[3] = multiplier;
  params->gemmlowp_sse2.rounding[0] = UINT64_C(0x40000000);
  params->gemmlowp_sse2.rounding[1] = UINT64_C(0x40000000);
  params->gemmlowp_sse2.remainder_mask[0] = (int32_t) remainder_mask;
  params->gemmlowp_sse2.remainder_mask[1] = (int32_t) remainder_mask;
  params->gemmlowp_sse2.remainder_mask[2] = (int32_t) remainder_mask;
  params->gemmlowp_sse2.remainder_mask[3] = (int32_t) remainder_mask;
  params->gemmlowp_sse2.remainder_threshold[0] = (int32_t) remainder_threshold;
  params->gemmlowp_sse2.remainder_threshold[1] = (int32_t) remainder_threshold;
  params->gemmlowp_sse2.remainder_threshold[2] = (int32_t) remainder_threshold;
  params->gemmlowp_sse2.remainder_threshold[3] = (int32_t) remainder_threshold;
  params->gemmlowp_sse2.shift[0] = (uint64_t) (uint32_t) shift;
  params->gemmlowp_sse2.shift[1] = (uint64_t) (uint32_t) shift;
  for (uint32_t i = 0; i < 8; i++) {
    params->gemmlowp_sse2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->gemmlowp_sse2.output_min[i] = output_min;
    params->gemmlowp_sse2.output_max[i] = output_max;
  }
}

void xnn_init_qu8_conv_minmax_fp32_sse2_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.scale[i] = scale;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.kernel_zero_point[i] = (int16_t) (uint16_t) kernel_zero_point;
    params->fp32_sse2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_sse2.output_min[i] = output_min;
    params->fp32_sse2.output_max[i] = output_max;
  }
}

void xnn_init_qu8_conv_minmax_fp32_avx2_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_avx2.scale[i] = scale;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx2.kernel_zero_point[i] = (int16_t) (uint16_t) kernel_zero_point;
    params->fp32_avx2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx2.output_min[i] = output_min;
    params->fp32_avx2.output_max[i] = output_max;
  }
}

void xnn_init_qu8_conv_minmax_fp32_avx512_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx512.scale[i] = scale;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx512.kernel_zero_point[i] = (int16_t) (uint16_t) kernel_zero_point;
    params->fp32_avx512.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 64; i++) {
    params->fp32_avx512.output_min[i] = output_min;
    params->fp32_avx512.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_qu8_conv_minmax_gemmlowp_neon_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  params->gemmlowp_neon.kernel_zero_point[0] = kernel_zero_point;
  params->gemmlowp_neon.kernel_zero_point[1] = kernel_zero_point;
  params->gemmlowp_neon.kernel_zero_point[2] = kernel_zero_point;
  params->gemmlowp_neon.kernel_zero_point[3] = kernel_zero_point;
  params->gemmlowp_neon.multiplier = multiplier;
  params->gemmlowp_neon.right_shift = -shift;
  params->gemmlowp_neon.output_zero_point = (int16_t) (uint16_t) output_zero_point;
  params->gemmlowp_neon.output_min = output_min;
  params->gemmlowp_neon.output_max = output_max;
}

void xnn_init_qu8_conv_minmax_fp32_neon_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  params->fp32_neon.kernel_zero_point[0] = kernel_zero_point;
  params->fp32_neon.kernel_zero_point[1] = kernel_zero_point;
  params->fp32_neon.kernel_zero_point[2] = kernel_zero_point;
  params->fp32_neon.kernel_zero_point[3] = kernel_zero_point;
  params->fp32_neon.scale = scale;
  params->fp32_neon.output_min_less_zero_point = (float) (int32_t) ((uint32_t) output_min - (uint32_t) output_zero_point);
  params->fp32_neon.output_max_less_zero_point = (float) (int32_t) ((uint32_t) output_max - (uint32_t) output_zero_point);
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) (uint32_t) output_zero_point;
}

void xnn_init_qu8_conv_minmax_fp32_neonv8_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  params->fp32_neonv8.kernel_zero_point[0] = kernel_zero_point;
  params->fp32_neonv8.kernel_zero_point[1] = kernel_zero_point;
  params->fp32_neonv8.kernel_zero_point[2] = kernel_zero_point;
  params->fp32_neonv8.kernel_zero_point[3] = kernel_zero_point;
  params->fp32_neonv8.scale = scale;
  params->fp32_neonv8.output_zero_point = (int16_t) (uint16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
}

void xnn_init_qu8_conv_minmax_rndnu_neon_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
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
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD
void xnn_init_qu8_conv_minmax_fp32_wasmsimd_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_wasmsimd.kernel_zero_point[i] = (int16_t) (uint16_t) kernel_zero_point;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_wasmsimd.scale[i] = scale;
    params->fp32_wasmsimd.output_min_less_zero_point[i] = (float) (int32_t) ((uint32_t) output_min - (uint32_t) output_zero_point);
    params->fp32_wasmsimd.output_max_less_zero_point[i] = (float) (int32_t) ((uint32_t) output_max - (uint32_t) output_zero_point);
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = INT32_C(0x4B400000) - (int32_t) (uint32_t) output_zero_point;
  }
}
#endif  // XNN_ARCH_WASMSIMD

void xnn_init_qs8_conv_minmax_gemmlowp_scalar_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t)(((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;

  params->gemmlowp_scalar.multiplier = multiplier;
  params->gemmlowp_scalar.remainder_mask = (int32_t) remainder_mask;
  params->gemmlowp_scalar.remainder_threshold = (int32_t) remainder_threshold;
  params->gemmlowp_scalar.shift = (uint32_t) shift;
  params->gemmlowp_scalar.output_min_less_zero_point = (int32_t) output_min - (int32_t) output_zero_point;
  params->gemmlowp_scalar.output_max_less_zero_point = (int32_t) output_max - (int32_t) output_zero_point;
  params->gemmlowp_scalar.output_zero_point = (int32_t) output_zero_point;
}

void xnn_init_qs8_conv_minmax_rndnu_scalar_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x00800000, 0x00FFFFFF] range.
  const int32_t multiplier = ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  assert(multiplier >= INT32_C(0x00800000));
  assert(multiplier <= INT32_C(0x00FFFFFF));

  // Shift is in [24, 56] range.
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 24);
  assert(shift < 56);
  const int64_t rounding = INT64_C(1) << (shift - 1);

  params->rndnu_scalar.multiplier = multiplier;
  params->rndnu_scalar.shift = shift;
  params->rndnu_scalar.rounding = rounding;
  params->rndnu_scalar.output_min_less_zero_point = (int32_t) output_min - (int32_t) output_zero_point;
  params->rndnu_scalar.output_max_less_zero_point = (int32_t) output_max - (int32_t) output_zero_point;
  params->rndnu_scalar.output_zero_point = (int32_t) output_zero_point;
}


void xnn_init_qs8_conv_minmax_fp32_scalar_lrint_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_scalar_lrint.scale = scale;
  params->fp32_scalar_lrint.output_min_less_zero_point = (long) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_lrint.output_max_less_zero_point = (long) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_lrint.output_zero_point = (int32_t) output_zero_point;
}

void xnn_init_qs8_conv_minmax_fp32_scalar_magic_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_scalar_magic.scale = scale;
  params->fp32_scalar_magic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_magic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_magic.magic_bias = 12582912.0f;
  params->fp32_scalar_magic.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qs8_conv_minmax_gemmlowp_sse2_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t)(((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  params->gemmlowp_sse2.multiplier[0] = multiplier;
  params->gemmlowp_sse2.multiplier[1] = multiplier;
  params->gemmlowp_sse2.multiplier[2] = multiplier;
  params->gemmlowp_sse2.multiplier[3] = multiplier;
  params->gemmlowp_sse2.rounding[0] = UINT64_C(0x40000000);
  params->gemmlowp_sse2.rounding[1] = UINT64_C(0x40000000);
  params->gemmlowp_sse2.remainder_mask[0] = (int32_t) remainder_mask;
  params->gemmlowp_sse2.remainder_mask[1] = (int32_t) remainder_mask;
  params->gemmlowp_sse2.remainder_mask[2] = (int32_t) remainder_mask;
  params->gemmlowp_sse2.remainder_mask[3] = (int32_t) remainder_mask;
  params->gemmlowp_sse2.remainder_threshold[0] = (int32_t) remainder_threshold;
  params->gemmlowp_sse2.remainder_threshold[1] = (int32_t) remainder_threshold;
  params->gemmlowp_sse2.remainder_threshold[2] = (int32_t) remainder_threshold;
  params->gemmlowp_sse2.remainder_threshold[3] = (int32_t) remainder_threshold;
  params->gemmlowp_sse2.shift[0] = (uint64_t) (uint32_t) shift;
  params->gemmlowp_sse2.shift[1] = (uint64_t) (uint32_t) shift;
  for (uint32_t i = 0; i < 8; i++) {
    params->gemmlowp_sse2.output_zero_point[i] = (int16_t) output_zero_point;
    params->gemmlowp_sse2.output_min[i] = (int16_t) output_min;
    params->gemmlowp_sse2.output_max[i] = (int16_t) output_max;
  }
}

void xnn_init_qs8_conv_minmax_gemmlowp_sse4_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t)(((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  params->gemmlowp_sse4.multiplier[0] = multiplier;
  params->gemmlowp_sse4.multiplier[1] = multiplier;
  params->gemmlowp_sse4.multiplier[2] = multiplier;
  params->gemmlowp_sse4.multiplier[3] = multiplier;
  params->gemmlowp_sse4.rounding[0] = UINT64_C(0x40000000);
  params->gemmlowp_sse4.rounding[1] = UINT64_C(0x40000000);
  params->gemmlowp_sse4.remainder_mask[0] = (int32_t) remainder_mask;
  params->gemmlowp_sse4.remainder_mask[1] = (int32_t) remainder_mask;
  params->gemmlowp_sse4.remainder_mask[2] = (int32_t) remainder_mask;
  params->gemmlowp_sse4.remainder_mask[3] = (int32_t) remainder_mask;
  params->gemmlowp_sse4.remainder_threshold[0] = (int32_t) remainder_threshold;
  params->gemmlowp_sse4.remainder_threshold[1] = (int32_t) remainder_threshold;
  params->gemmlowp_sse4.remainder_threshold[2] = (int32_t) remainder_threshold;
  params->gemmlowp_sse4.remainder_threshold[3] = (int32_t) remainder_threshold;
  params->gemmlowp_sse4.shift[0] = (uint64_t) (uint32_t) shift;
  params->gemmlowp_sse4.shift[1] = (uint64_t) (uint32_t) shift;
  for (uint32_t i = 0; i < 8; i++) {
    params->gemmlowp_sse4.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->gemmlowp_sse4.output_min[i] = output_min;
    params->gemmlowp_sse4.output_max[i] = output_max;
  }
}

void xnn_init_qs8_conv_minmax_gemmlowp_avx2_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t)(((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  for (uint32_t i = 0; i < 8; i++) {
    params->gemmlowp_avx2.multiplier[i] = multiplier;
  }
  params->gemmlowp_avx2.rounding[0] = UINT64_C(0x40000000);
  params->gemmlowp_avx2.rounding[1] = UINT64_C(0x40000000);
  params->gemmlowp_avx2.rounding[2] = UINT64_C(0x40000000);
  params->gemmlowp_avx2.rounding[3] = UINT64_C(0x40000000);
  for (uint32_t i = 0; i < 8; i++) {
    params->gemmlowp_avx2.remainder_mask[i] = (int32_t) remainder_mask;
    params->gemmlowp_avx2.remainder_threshold[i] = (int32_t) remainder_threshold;
  }
  params->gemmlowp_avx2.shift[0] = (uint64_t) (uint32_t) shift;
  params->gemmlowp_avx2.shift[1] = (uint64_t) (uint32_t) shift;
  params->gemmlowp_avx2.shift[2] = (uint64_t) (uint32_t) shift;
  params->gemmlowp_avx2.shift[3] = (uint64_t) (uint32_t) shift;
  for (uint32_t i = 0; i < 16; i++) {
    params->gemmlowp_avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->gemmlowp_avx2.output_min[i] = output_min;
    params->gemmlowp_avx2.output_max[i] = output_max;
  }
}

void xnn_init_qs8_conv_minmax_gemmlowp_avx512_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t)(((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  params->gemmlowp_avx512.multiplier = (int64_t) multiplier;
  params->gemmlowp_avx512.rounding = UINT64_C(0x40000000);
  params->gemmlowp_avx512.remainder_mask = (int32_t) remainder_mask;
  params->gemmlowp_avx512.remainder_threshold = (int32_t) remainder_threshold;
  params->gemmlowp_avx512.shift = (uint64_t) (uint32_t) shift;
  for (uint32_t i = 0; i < 32; i++) {
    params->gemmlowp_avx512.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 64; i++) {
    params->gemmlowp_avx512.output_min[i] = output_min;
    params->gemmlowp_avx512.output_max[i] = output_max;
  }
}

void xnn_init_qs8_conv_minmax_fp32_sse2_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.scale[i] = scale;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.output_zero_point[i] = (int16_t) output_zero_point;
    params->fp32_sse2.output_min[i] = (int16_t) output_min;
    params->fp32_sse2.output_max[i] = (int16_t) output_max;
  }
}

void xnn_init_qs8_conv_minmax_fp32_sse4_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse4.scale[i] = scale;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse4.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_sse4.output_min[i] = output_min;
    params->fp32_sse4.output_max[i] = output_max;
  }
}

void xnn_init_qs8_conv_minmax_fp32_avx2_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_avx2.scale[i] = scale;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx2.output_min[i] = output_min;
    params->fp32_avx2.output_max[i] = output_max;
  }
}

void xnn_init_qs8_conv_minmax_fp32_avx512_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx512.scale[i] = scale;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx512.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 64; i++) {
    params->fp32_avx512.output_min[i] = output_min;
    params->fp32_avx512.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_qs8_conv_minmax_gemmlowp_neon_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  params->gemmlowp_neon.multiplier = multiplier;
  params->gemmlowp_neon.right_shift = -shift;
  params->gemmlowp_neon.output_zero_point = (int16_t) output_zero_point;
  params->gemmlowp_neon.output_min = output_min;
  params->gemmlowp_neon.output_max = output_max;
}

void xnn_init_qs8_conv_minmax_fp32_neon_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_neon.scale = scale;
  params->fp32_neon.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_neon.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

void xnn_init_qs8_conv_minmax_fp32_neonv8_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_neonv8.scale = scale;
  params->fp32_neonv8.output_zero_point = (int16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
}

void xnn_init_qs8_conv_minmax_rndnu_neon_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t) (((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
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
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD
void xnn_init_qs8_conv_minmax_gemmlowp_wasmsimd_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x40000000, 0x7FFFFF80] range.
  const int32_t multiplier = (int32_t)(((scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000)) << 7);
  assert(multiplier >= INT32_C(0x40000000));
  assert(multiplier <= INT32_C(0x7FFFFF80));

  // Shift is in [0, 31] range.
  const int32_t shift = 127 + 31 - 32 - (fp32_to_bits(scale) >> 23);
  assert(shift >= 0);
  assert(shift < 32);

  const int64_t twice_multiplier = INT64_C(2) * (int64_t) multiplier;
  const uint32_t remainder_mask = (UINT32_C(1) << shift) - UINT32_C(1);
  const uint32_t remainder_threshold = remainder_mask >> 1;
  params->gemmlowp_wasmsimd.multiplier[0] = twice_multiplier;
  params->gemmlowp_wasmsimd.multiplier[1] = twice_multiplier;
  params->gemmlowp_wasmsimd.rounding[0] = INT64_C(0x80000000);
  params->gemmlowp_wasmsimd.rounding[1] = INT64_C(0x80000000);
  params->gemmlowp_wasmsimd.remainder_mask[0] = (int32_t) remainder_mask;
  params->gemmlowp_wasmsimd.remainder_mask[1] = (int32_t) remainder_mask;
  params->gemmlowp_wasmsimd.remainder_mask[2] = (int32_t) remainder_mask;
  params->gemmlowp_wasmsimd.remainder_mask[3] = (int32_t) remainder_mask;
  params->gemmlowp_wasmsimd.remainder_threshold[0] = (int32_t) remainder_threshold;
  params->gemmlowp_wasmsimd.remainder_threshold[1] = (int32_t) remainder_threshold;
  params->gemmlowp_wasmsimd.remainder_threshold[2] = (int32_t) remainder_threshold;
  params->gemmlowp_wasmsimd.remainder_threshold[3] = (int32_t) remainder_threshold;
  params->gemmlowp_wasmsimd.shift = shift;
  for (uint32_t i = 0; i < 8; i++) {
    params->gemmlowp_wasmsimd.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->gemmlowp_wasmsimd.output_min[i] = output_min;
    params->gemmlowp_wasmsimd.output_max[i] = output_max;
  }
}

void xnn_init_qs8_conv_minmax_fp32_wasmsimd_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_wasmsimd.scale[i] = scale;
    params->fp32_wasmsimd.output_min_less_zero_point[i] = (float) ((int32_t) output_min - (int32_t) output_zero_point);
    params->fp32_wasmsimd.output_max_less_zero_point[i] = (float) ((int32_t) output_max - (int32_t) output_zero_point);
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  }
}
#endif  // XNN_ARCH_WASMSIMD

void xnn_init_qc8_scale_fp32_params(
  size_t channels,
  size_t channels_tile,
  size_t stride,
  const float scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w)
{
  for (size_t tile_start = 0; tile_start < channels; tile_start += channels_tile) {
    const size_t tile_size = min(channels - tile_start, channels_tile);
    for (size_t tile_offset = 0; tile_offset < tile_size; tile_offset++) {
      ((float*) packed_w)[tile_offset] = scale[tile_start + tile_offset];
    }
    packed_w = (void*) ((uintptr_t) packed_w + stride);
  }
}

void xnn_init_qs8_minmax_scalar_lrint_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->scalar_lrint.output_min_less_zero_point = (long) ((int32_t) output_min - (int32_t) output_zero_point);
  params->scalar_lrint.output_max_less_zero_point = (long) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_lrint.output_zero_point = (int32_t) output_zero_point;
}

void xnn_init_qs8_minmax_scalar_magic_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->scalar_magic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->scalar_magic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_magic.magic_bias = 12582912.0f;
  params->scalar_magic.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qs8_minmax_sse2_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.output_zero_point[i] = (int16_t) output_zero_point;
    params->sse2.output_min[i] = (int16_t) output_min;
    params->sse2.output_max[i] = (int16_t) output_max;
  }
}

void xnn_init_qs8_minmax_sse4_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->sse4.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse4.output_min[i] = output_min;
    params->sse4.output_max[i] = output_max;
  }
}

void xnn_init_qs8_minmax_avx2_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->avx2.output_min[i] = output_min;
    params->avx2.output_max[i] = output_max;
  }
}

void xnn_init_qs8_minmax_avx512_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 32; i++) {
    params->avx512.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 64; i++) {
    params->avx512.output_min[i] = output_min;
    params->avx512.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_qs8_minmax_neon_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->neon.output_zero_point = (int16_t) output_zero_point;
  params->neon.output_min = output_min;
  params->neon.output_max = output_max;
}

void xnn_init_qs8_minmax_neon_fp32_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->neon_fp32.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->neon_fp32.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->neon_fp32.magic_bias = 12582912.0f;
  params->neon_fp32.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD
void xnn_init_qs8_minmax_wasmsimd_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.output_min_less_zero_point[i] = (float) ((int32_t) output_min - (int32_t) output_zero_point);
    params->wasmsimd.output_max_less_zero_point[i] = (float) ((int32_t) output_max - (int32_t) output_zero_point);
    params->wasmsimd.magic_bias[i] = 12582912.0f;
    params->wasmsimd.magic_bias_less_output_zero_point[i] = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  }
}
#endif  // XNN_ARCH_WASMSIMD

void xnn_init_qu8_avgpool_params(
  union xnn_qu8_avgpool_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  // Compute requantization parameters.
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x00800000, 0x00FFFFFF] range.
  const int32_t multiplier = ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  assert(multiplier >= INT32_C(0x00800000));
  assert(multiplier <= INT32_C(0x00FFFFFF));

  // Shift is in [16, 55] range.
  const int32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 64);

  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    const uint32_t right_shift = (uint32_t) shift;
    const uint64_t rounding = UINT64_C(1) << (right_shift - 1);
    params->sse2.bias[0] = bias;
    params->sse2.bias[1] = bias;
    params->sse2.bias[2] = bias;
    params->sse2.bias[3] = bias;
    params->sse2.multiplier[0] = (uint32_t) multiplier;
    params->sse2.multiplier[1] = (uint32_t) multiplier;
    params->sse2.multiplier[2] = (uint32_t) multiplier;
    params->sse2.multiplier[3] = (uint32_t) multiplier;
    params->sse2.rounding[0] = rounding;
    params->sse2.rounding[1] = rounding;
    params->sse2.right_shift[0] = (uint64_t) right_shift;
    params->sse2.right_shift[1] = (uint64_t) right_shift;
    for (uint32_t i = 0; i < 8; i++) {
      params->sse2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
    }
    for (uint32_t i = 0; i < 16; i++) {
      params->sse2.output_min[i] = output_min;
      params->sse2.output_max[i] = output_max;
    }
  #elif XNN_ARCH_ARM || XNN_ARCH_ARM64
    params->neon.bias = bias;
    params->neon.multiplier = multiplier;
    params->neon.left_shift = (int64_t) -shift;
    params->neon.output_zero_point = (int16_t) (uint16_t) output_zero_point;
    params->neon.output_min = output_min;
    params->neon.output_max = output_max;
  #else
    const uint32_t right_shift = (uint32_t) shift;
    const int64_t rounding = INT64_C(1) << (right_shift - 1);
    params->scalar.bias = bias;
    params->scalar.multiplier = multiplier;
    params->scalar.rounding = rounding;
    params->scalar.right_shift = right_shift;
    params->scalar.output_min_less_zero_point =
      (int32_t) (uint32_t) output_min - (int32_t) (uint32_t) output_zero_point;
    params->scalar.output_max_less_zero_point =
      (int32_t) (uint32_t) output_max - (int32_t) (uint32_t) output_zero_point;
    params->scalar.output_zero_point = (int32_t) (uint32_t) output_zero_point;
  #endif
}

void xnn_init_scalar_qu8_avgpool_params(
  union xnn_qu8_avgpool_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  // Compute requantization parameters.
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x00800000, 0x00FFFFFF] range.
  const int32_t multiplier = ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  assert(multiplier >= INT32_C(0x00800000));
  assert(multiplier <= INT32_C(0x00FFFFFF));

  // Shift is in [16, 55] range.
  const int32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 64);

  const uint32_t right_shift = (uint32_t) shift;
  const int64_t rounding = INT64_C(1) << (right_shift - 1);
  params->scalar.bias = bias;
  params->scalar.rounding = rounding;
  params->scalar.multiplier = multiplier;
  params->scalar.right_shift = right_shift;
  params->scalar.output_min_less_zero_point =
    (int32_t) (uint32_t) output_min - (int32_t) (uint32_t) output_zero_point;
  params->scalar.output_max_less_zero_point =
    (int32_t) (uint32_t) output_max - (int32_t) (uint32_t) output_zero_point;
  params->scalar.output_zero_point = (int32_t) (uint32_t) output_zero_point;
}

void xnn_update_qu8_avgpool_params(
  union xnn_qu8_avgpool_params* params,
  int32_t bias,
  float scale)
{
  // Compute requantization parameters.
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x00800000, 0x00FFFFFF] range.
  const int32_t multiplier = ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  assert(multiplier >= INT32_C(0x00800000));
  assert(multiplier <= INT32_C(0x00FFFFFF));

  // Shift is in [16, 55] range.
  const int32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 64);

  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    const uint64_t rounding = UINT64_C(1) << ((uint32_t) shift - 1);
    params->sse2.bias[0] = bias;
    params->sse2.bias[1] = bias;
    params->sse2.bias[2] = bias;
    params->sse2.bias[3] = bias;
    params->sse2.multiplier[0] = (uint32_t) multiplier;
    params->sse2.multiplier[1] = (uint32_t) multiplier;
    params->sse2.multiplier[2] = (uint32_t) multiplier;
    params->sse2.multiplier[3] = (uint32_t) multiplier;
    params->sse2.rounding[0] = rounding;
    params->sse2.rounding[1] = rounding;
    params->sse2.right_shift[0] = (uint64_t) (uint32_t) shift;
    params->sse2.right_shift[1] = (uint64_t) (uint32_t) shift;
  #elif XNN_ARCH_ARM || XNN_ARCH_ARM64
    params->neon.bias = bias;
    params->neon.multiplier = multiplier;
    params->neon.left_shift = (int64_t) -shift;
  #else
    const int64_t rounding = INT64_C(1) << ((uint32_t) shift - 1);
    params->scalar.bias = bias;
    params->scalar.multiplier = multiplier;
    params->scalar.rounding = rounding;
    params->scalar.right_shift = (uint32_t) shift;
  #endif
}

void xnn_init_qs8_avgpool_params(
  union xnn_qs8_avgpool_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters.
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x00800000, 0x00FFFFFF] range.
  const int32_t multiplier = ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  assert(multiplier >= INT32_C(0x00800000));
  assert(multiplier <= INT32_C(0x00FFFFFF));

  // Shift is in [16, 55] range.
  const int32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 64);

  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    const uint64_t rounding = UINT64_C(1) << ((uint32_t) shift - 1);
    params->sse2.bias[0] = bias;
    params->sse2.bias[1] = bias;
    params->sse2.bias[2] = bias;
    params->sse2.bias[3] = bias;
    params->sse2.multiplier[0] = (uint32_t) multiplier;
    params->sse2.multiplier[1] = (uint32_t) multiplier;
    params->sse2.multiplier[2] = (uint32_t) multiplier;
    params->sse2.multiplier[3] = (uint32_t) multiplier;
    params->sse2.rounding[0] = rounding;
    params->sse2.rounding[1] = rounding;
    params->sse2.shift[0] = (uint64_t) (uint32_t) shift;
    params->sse2.shift[1] = (uint64_t) (uint32_t) shift;
    for (uint32_t i = 0; i < 8; i++) {
      params->sse2.output_zero_point[i] = (int16_t) output_zero_point;
      params->sse2.output_min[i] = (int16_t) output_min;
      params->sse2.output_max[i] = (int16_t) output_max;
    }
  #elif XNN_ARCH_ARM || XNN_ARCH_ARM64
    params->neon.bias = bias;
    params->neon.multiplier = multiplier;
    params->neon.left_shift = (int64_t) -shift;
    params->neon.output_zero_point = (int16_t) output_zero_point;
    params->neon.output_min = output_min;
    params->neon.output_max = output_max;
  #elif XNN_ARCH_WASMSIMD
    const int64_t rounding = INT64_C(1) << ((uint32_t) shift - 1);
    params->wasmsimd.bias[0] = bias;
    params->wasmsimd.bias[1] = bias;
    params->wasmsimd.bias[2] = bias;
    params->wasmsimd.bias[3] = bias;
    params->wasmsimd.multiplier[0] = (int64_t) multiplier;
    params->wasmsimd.multiplier[1] = (int64_t) multiplier;
    params->wasmsimd.rounding[0] = rounding;
    params->wasmsimd.rounding[1] = rounding;
    params->wasmsimd.shift = shift;
    for (uint32_t i = 0; i < 8; i++) {
      params->wasmsimd.output_zero_point[i] = (int16_t) output_zero_point;
    }
    for (uint32_t i = 0; i < 16; i++) {
      params->wasmsimd.output_min[i] = output_min;
      params->wasmsimd.output_max[i] = output_max;
    }
  #else
    const int64_t rounding = INT64_C(1) << ((uint32_t) shift - 1);
    params->scalar.bias = bias;
    params->scalar.multiplier = multiplier;
    params->scalar.rounding = rounding;
    params->scalar.shift = (uint32_t) shift;
    params->scalar.output_min_less_zero_point = (int32_t) output_min - (int32_t) output_zero_point;
    params->scalar.output_max_less_zero_point = (int32_t) output_max - (int32_t) output_zero_point;
    params->scalar.output_zero_point = (int32_t) output_zero_point;
  #endif
}

void xnn_init_scalar_qs8_avgpool_params(
  union xnn_qs8_avgpool_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  // Compute requantization parameters.
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x00800000, 0x00FFFFFF] range.
  const int32_t multiplier = ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  assert(multiplier >= INT32_C(0x00800000));
  assert(multiplier <= INT32_C(0x00FFFFFF));

  // Shift is in [16, 55] range.
  const int32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 64);

  const int64_t rounding = INT64_C(1) << ((uint32_t) shift - 1);
  params->scalar.bias = bias;
  params->scalar.rounding = rounding;
  params->scalar.multiplier = multiplier;
  params->scalar.shift = shift;
  params->scalar.output_min_less_zero_point = (int32_t) output_min - (int32_t) output_zero_point;
  params->scalar.output_max_less_zero_point = (int32_t) output_max - (int32_t) output_zero_point;
  params->scalar.output_zero_point = (int32_t) output_zero_point;
}

void xnn_update_qs8_avgpool_params(
  union xnn_qs8_avgpool_params* params,
  int32_t bias,
  float scale)
{
  // Compute requantization parameters.
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);
  const uint32_t scale_bits = fp32_to_bits(scale);

  // Multiplier is in [0x00800000, 0x00FFFFFF] range.
  const int32_t multiplier = ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  assert(multiplier >= INT32_C(0x00800000));
  assert(multiplier <= INT32_C(0x00FFFFFF));

  // Shift is in [16, 55] range.
  const int32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 64);

  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    const uint64_t rounding = UINT64_C(1) << ((uint32_t) shift - 1);
    params->sse2.bias[0] = bias;
    params->sse2.bias[1] = bias;
    params->sse2.bias[2] = bias;
    params->sse2.bias[3] = bias;
    params->sse2.multiplier[0] = (uint32_t) multiplier;
    params->sse2.multiplier[1] = (uint32_t) multiplier;
    params->sse2.multiplier[2] = (uint32_t) multiplier;
    params->sse2.multiplier[3] = (uint32_t) multiplier;
    params->sse2.rounding[0] = rounding;
    params->sse2.rounding[1] = rounding;
    params->sse2.shift[0] = (uint64_t) (uint32_t) shift;
    params->sse2.shift[1] = (uint64_t) (uint32_t) shift;
  #elif XNN_ARCH_ARM || XNN_ARCH_ARM64
    params->neon.bias = bias;
    params->neon.multiplier = multiplier;
    params->neon.left_shift = (int64_t) -shift;
  #elif XNN_ARCH_WASMSIMD
    const int64_t rounding = INT64_C(1) << ((uint32_t) shift - 1);
    params->wasmsimd.bias[0] = bias;
    params->wasmsimd.bias[1] = bias;
    params->wasmsimd.bias[2] = bias;
    params->wasmsimd.bias[3] = bias;
    params->wasmsimd.multiplier[0] = (int64_t) multiplier;
    params->wasmsimd.multiplier[1] = (int64_t) multiplier;
    params->wasmsimd.rounding[0] = rounding;
    params->wasmsimd.rounding[1] = rounding;
    params->wasmsimd.shift = shift;
  #else
    const int64_t rounding = INT64_C(1) << ((uint32_t) shift - 1);
    params->scalar.bias = bias;
    params->scalar.multiplier = multiplier;
    params->scalar.rounding = rounding;
    params->scalar.shift = (uint32_t) shift;
  #endif
}

void xnn_update_f16_scaleminmax_params(
  struct xnn_f16_scaleminmax_params* params,
  uint16_t scale)
{
  params->scale = scale;
}

void xnn_update_f32_scaleminmax_params(
  union xnn_f32_scaleminmax_params* params,
  float scale)
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse2.scale[i] = scale;
    }
  #else
    params->scalar.scale = scale;
  #endif
}

void xnn_init_f16_scaleminmax_params(
  struct xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  uint16_t min,
  uint16_t max)
{
  params->scale = scale;
  params->min = min;
  params->max = max;
  params->pad = 0;  // unused.
}

void xnn_init_f32_scaleminmax_params(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max)
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse2.scale[i] = scale;
      params->sse2.min[i] = min;
      params->sse2.max[i] = max;
    }
  #else
    params->scalar.scale = scale;
    params->scalar.min = min;
    params->scalar.max = max;
  #endif
}

void xnn_init_f32_gavgpool_params(
  union xnn_f32_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  float multiplier,
  float output_min,
  float output_max,
  uint32_t width)
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
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
  #elif XNN_ARCH_ARM || XNN_ARCH_ARM64
    params->neon.multiplier = multiplier;
    params->neon.output_min = output_min;
    params->neon.output_max = output_max;

    const uint32_t w = (width - 1) & 3;
    params->neon.mask[0] = UINT32_C(0xFFFFFFFF);
    params->neon.mask[1] = -(uint32_t) (w >= 1);
    params->neon.mask[2] = -(uint32_t) (w >= 2);
    params->neon.mask[3] = -(uint32_t) (w >= 3);
  #else
    params->scalar.multiplier = multiplier;
    params->scalar.output_min = output_min;
    params->scalar.output_max = output_max;

    const uint32_t w = (width - 1) & 3;
    params->scalar.mask[0] = UINT32_C(0xFFFFFFFF);
    params->scalar.mask[1] = -(int32_t) (w >= 1);
    params->scalar.mask[2] = -(int32_t) (w >= 2);
    params->scalar.mask[3] = -(int32_t) (w >= 3);
  #endif
}

void xnn_update_f32_gavgpool_params(
  union xnn_f32_gavgpool_params* params,
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

void xnn_init_scalar_f32_scaleminmax_params(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max)
{
  params->scalar.scale = scale;
  params->scalar.min = min;
  params->scalar.max = max;
}

void xnn_init_scalar_f32_gavgpool_params(
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
}

void xnn_init_f16_minmax_params(
  struct xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t min,
  uint16_t max)
{
  params->min = min;
  params->max = max;
}

void xnn_init_f32_minmax_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse.min[i] = output_min;
      params->sse.max[i] = output_max;
    }
  #else
    params->scalar.min = output_min;
    params->scalar.max = output_max;
  #endif
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_minmax_sse_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.min[i] = output_min;
    params->sse.max[i] = output_max;
  }
}

void xnn_init_f32_minmax_avx_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.min[i] = output_min;
    params->avx.max[i] = output_max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

void xnn_init_f32_minmax_scalar_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  params->scalar.min = output_min;
  params->scalar.max = output_max;
}

void xnn_init_f16_hswish_params(
  struct xnn_f16_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  params->sixth = UINT16_C(0x3155);
  params->three = UINT16_C(0x4200);
  params->six = UINT16_C(0x4600);
}

void xnn_init_f32_hswish_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse.sixth[i] = 0x1.555556p-3f;
      params->sse.half[i] = 0.5f;
      params->sse.one[i] = 1.0f;
    }
  #else
    params->scalar.sixth = 0x1.555556p-3f;
    params->scalar.three = 3.0f;
    params->scalar.six = 6.0f;
  #endif
}

void xnn_init_scalar_f32_hswish_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar.sixth = 0x1.555556p-3f;
  params->scalar.three = 3.0f;
  params->scalar.six = 6.0f;
}

void xnn_init_f32_abs_params(
  union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)])
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse.nonsign_mask[i] = math_nonsign_mask_f32();
    }
  #elif XNN_ARCH_WASMSIMD
    params->wasmsimd.nonsign_mask = math_nonsign_mask_f32();
  #endif
}

void xnn_init_scalar_f32_abs_params(
  union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)])
{
}

void xnn_init_f32_neg_params(
  union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)])
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse.sign_mask[i] = -0.0f;
    }
  #elif XNN_ARCH_WASMSIMD
    params->wasmsimd.sign_mask = -0.0f;
  #endif
}

void xnn_init_scalar_f32_neg_params(
  union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)])
{
}

void xnn_init_f32_rnd_params(
  union xnn_f32_rnd_params params[XNN_MIN_ELEMENTS(1)])
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse2.sign_mask[i] = -0.0f;
    }
    for (uint32_t i = 0; i < 4; i++) {
      params->sse2.one[i] = 1.0f;
    }
  #endif
}

void xnn_init_scalar_f32_rnd_params(
  union xnn_f32_rnd_params params[XNN_MIN_ELEMENTS(1)])
{
}

void xnn_init_f32_elu_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse.prescale[i] = prescale;
      params->sse.alpha[i] = alpha;
      params->sse.beta[i] = beta;
    }
  #else
    params->scalar.prescale = prescale;
    params->scalar.alpha = alpha;
    params->scalar.beta = beta;
  #endif
}

void xnn_init_scalar_f32_elu_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->scalar.prescale = prescale;
  params->scalar.alpha = alpha;
  params->scalar.beta = beta;
}

void xnn_init_f32_lrelu_params(
  union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope)
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse.slope[i] = slope;
    }
  #else
    params->scalar.slope = slope;
  #endif
}

void xnn_init_scalar_f32_lrelu_params(
  union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope)
{
  params->scalar.slope = slope;
}

void xnn_init_f32_sqrt_params(
  union xnn_f32_sqrt_params params[XNN_MIN_ELEMENTS(1)])
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    params->fma.half = 0.5f;
  #endif
}

void xnn_init_scalar_f32_sqrt_params(
  union xnn_f32_sqrt_params params[XNN_MIN_ELEMENTS(1)])
{
}

void xnn_init_f32_chw_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max)
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 4; i++) {
      params->sse.min[i] = output_min;
      params->sse.max[i] = output_max;
    }

    const uint32_t w4 = (width - 1) & 3;
    params->sse.mask[0] = UINT32_C(0xFFFFFFFF);
    params->sse.mask[1] = -(uint32_t) (w4 >= 1);
    params->sse.mask[2] = -(uint32_t) (w4 >= 2);
    params->sse.mask[3] = -(uint32_t) (w4 >= 3);

    const uint32_t w8 = (width - 1) & 7;
    params->sse.mask_even[0] = UINT32_C(0xFFFFFFFF);
    params->sse.mask_even[1] = -(uint32_t) (w8 >= 2);
    params->sse.mask_even[2] = -(uint32_t) (w8 >= 4);
    params->sse.mask_even[3] = -(uint32_t) (w8 >= 6);
    params->sse.mask_odd[0] = -(uint32_t) (w8 >= 1);
    params->sse.mask_odd[1] = -(uint32_t) (w8 >= 3);
    params->sse.mask_odd[2] = -(uint32_t) (w8 >= 5);
    params->sse.mask_odd[3] = -(uint32_t) (w8 >= 7);
  #elif XNN_ARCH_ARM || XNN_ARCH_ARM64
    params->neon.min = output_min;
    params->neon.max = output_max;

    const uint32_t w4 = (width - 1) & 3;
    params->neon.mask[0] = UINT32_C(0xFFFFFFFF);
    params->neon.mask[1] = -(uint32_t) (w4 >= 1);
    params->neon.mask[2] = -(uint32_t) (w4 >= 2);
    params->neon.mask[3] = -(uint32_t) (w4 >= 3);

    const uint32_t w8 = (width - 1) & 7;
    params->neon.mask_even[0] = UINT32_C(0xFFFFFFFF);
    params->neon.mask_even[1] = -(uint32_t) (w8 >= 2);
    params->neon.mask_even[2] = -(uint32_t) (w8 >= 4);
    params->neon.mask_even[3] = -(uint32_t) (w8 >= 6);
    params->neon.mask_odd[0] = -(uint32_t) (w8 >= 1);
    params->neon.mask_odd[1] = -(uint32_t) (w8 >= 3);
    params->neon.mask_odd[2] = -(uint32_t) (w8 >= 5);
    params->neon.mask_odd[3] = -(uint32_t) (w8 >= 7);
  #else
    params->scalar.min = output_min;
    params->scalar.max = output_max;

    const uint32_t w4 = (width - 1) & 3;
    params->scalar.mask[0] = UINT32_C(0xFFFFFFFF);
    params->scalar.mask[1] = -(uint32_t) (w4 >= 1);
    params->scalar.mask[2] = -(uint32_t) (w4 >= 2);
    params->scalar.mask[3] = -(uint32_t) (w4 >= 3);

    const uint32_t w8 = (width - 1) & 7;
    params->scalar.mask_even[0] = UINT32_C(0xFFFFFFFF);
    params->scalar.mask_even[1] = -(uint32_t) (w8 >= 2);
    params->scalar.mask_even[2] = -(uint32_t) (w8 >= 4);
    params->scalar.mask_even[3] = -(uint32_t) (w8 >= 6);
    params->scalar.mask_odd[0] = -(uint32_t) (w8 >= 1);
    params->scalar.mask_odd[1] = -(uint32_t) (w8 >= 3);
    params->scalar.mask_odd[2] = -(uint32_t) (w8 >= 5);
    params->scalar.mask_odd[3] = -(uint32_t) (w8 >= 7);
  #endif
}

void xnn_update_f32_chw_params(
  union xnn_f32_chw_params* params,
  uint32_t width)
{
  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    const uint32_t w4 = (width - 1) & 3;
    params->sse.mask[0] = UINT32_C(0xFFFFFFFF);
    params->sse.mask[1] = -(uint32_t) (w4 >= 1);
    params->sse.mask[2] = -(uint32_t) (w4 >= 2);
    params->sse.mask[3] = -(uint32_t) (w4 >= 3);

    const uint32_t w8 = (width - 1) & 7;
    params->sse.mask_even[0] = UINT32_C(0xFFFFFFFF);
    params->sse.mask_even[1] = -(uint32_t) (w8 >= 2);
    params->sse.mask_even[2] = -(uint32_t) (w8 >= 4);
    params->sse.mask_even[3] = -(uint32_t) (w8 >= 6);
    params->sse.mask_odd[0] = -(uint32_t) (w8 >= 1);
    params->sse.mask_odd[1] = -(uint32_t) (w8 >= 3);
    params->sse.mask_odd[2] = -(uint32_t) (w8 >= 5);
    params->sse.mask_odd[3] = -(uint32_t) (w8 >= 7);
  #elif XNN_ARCH_ARM || XNN_ARCH_ARM64
    const uint32_t w4 = (width - 1) & 3;
    params->neon.mask[0] = UINT32_C(0xFFFFFFFF);
    params->neon.mask[1] = -(uint32_t) (w4 >= 1);
    params->neon.mask[2] = -(uint32_t) (w4 >= 2);
    params->neon.mask[3] = -(uint32_t) (w4 >= 3);

    const uint32_t w8 = (width - 1) & 7;
    params->neon.mask_even[0] = UINT32_C(0xFFFFFFFF);
    params->neon.mask_even[1] = -(uint32_t) (w8 >= 2);
    params->neon.mask_even[2] = -(uint32_t) (w8 >= 4);
    params->neon.mask_even[3] = -(uint32_t) (w8 >= 6);
    params->neon.mask_odd[0] = -(uint32_t) (w8 >= 1);
    params->neon.mask_odd[1] = -(uint32_t) (w8 >= 3);
    params->neon.mask_odd[2] = -(uint32_t) (w8 >= 5);
    params->neon.mask_odd[3] = -(uint32_t) (w8 >= 7);
  #else
    const uint32_t w4 = (width - 1) & 3;
    params->scalar.mask[0] = UINT32_C(0xFFFFFFFF);
    params->scalar.mask[1] = -(uint32_t) (w4 >= 1);
    params->scalar.mask[2] = -(uint32_t) (w4 >= 2);
    params->scalar.mask[3] = -(uint32_t) (w4 >= 3);

    const uint32_t w8 = (width - 1) & 7;
    params->scalar.mask_even[0] = UINT32_C(0xFFFFFFFF);
    params->scalar.mask_even[1] = -(uint32_t) (w8 >= 2);
    params->scalar.mask_even[2] = -(uint32_t) (w8 >= 4);
    params->scalar.mask_even[3] = -(uint32_t) (w8 >= 6);
    params->scalar.mask_odd[0] = -(uint32_t) (w8 >= 1);
    params->scalar.mask_odd[1] = -(uint32_t) (w8 >= 3);
    params->scalar.mask_odd[2] = -(uint32_t) (w8 >= 5);
    params->scalar.mask_odd[3] = -(uint32_t) (w8 >= 7);
  #endif
}

void xnn_init_scalar_f32_chw_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max)
{
  params->scalar.min = output_min;
  params->scalar.max = output_max;

  const uint32_t w4 = (width - 1) & 3;
  params->scalar.mask[0] = UINT32_C(0xFFFFFFFF);
  params->scalar.mask[1] = -(uint32_t) (w4 >= 1);
  params->scalar.mask[2] = -(uint32_t) (w4 >= 2);
  params->scalar.mask[3] = -(uint32_t) (w4 >= 3);

  const uint32_t w8 = (width - 1) & 7;
  params->scalar.mask_even[0] = UINT32_C(0xFFFFFFFF);
  params->scalar.mask_even[1] = -(uint32_t) (w8 >= 2);
  params->scalar.mask_even[2] = -(uint32_t) (w8 >= 4);
  params->scalar.mask_even[3] = -(uint32_t) (w8 >= 6);
  params->scalar.mask_odd[0] = -(uint32_t) (w8 >= 1);
  params->scalar.mask_odd[1] = -(uint32_t) (w8 >= 3);
  params->scalar.mask_odd[2] = -(uint32_t) (w8 >= 5);
  params->scalar.mask_odd[3] = -(uint32_t) (w8 >= 7);
}

void xnn_init_u8_minmax_params(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  #if XNN_ARCH_X86 || XNN_ARCH_X86_64
    for (uint32_t i = 0; i < 16; i++) {
      params->sse2.min[i] = output_min;
      params->sse2.max[i] = output_max;
    }
  #else
    params->scalar.min = output_min;
    params->scalar.max = output_max;
  #endif
}

void xnn_init_scalar_u8_minmax_params(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  params->scalar.min = (int32_t) (uint32_t) output_min;
  params->scalar.max = (int32_t) (uint32_t) output_max;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qu8_add_minmax_sse2_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) a_zero_point + b_multiplier * (int32_t) b_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.bias[i] = bias;
  }
  const uint16_t a_multiplier_lo = (uint16_t) a_multiplier;
  const uint16_t a_multiplier_hi = (uint16_t) ((uint32_t) a_multiplier >> 16);
  const uint16_t b_multiplier_lo = (uint16_t) b_multiplier;
  const uint16_t b_multiplier_hi = (uint16_t) ((uint32_t) b_multiplier >> 16);
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.a_multiplier_lo[i] = a_multiplier_lo;
    params->sse2.a_multiplier_hi[i] = a_multiplier_hi;
    params->sse2.b_multiplier_lo[i] = b_multiplier_lo;
    params->sse2.b_multiplier_hi[i] = b_multiplier_hi;
  }
  params->sse2.shift = shift;
  params->sse2.b_multiplier = (uint32_t) b_multiplier;
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.rounding[i] = rounding;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse2.output_min[i] = output_min;
    params->sse2.output_max[i] = output_max;
  }
}

void xnn_init_qu8_add_minmax_sse4_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) (uint32_t) a_zero_point + b_multiplier * (int32_t) (uint32_t) b_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4.bias[i] = bias;
    params->sse4.a_multiplier[i] = a_multiplier;
    params->sse4.b_multiplier[i] = b_multiplier;
    params->sse4.rounding[i] = rounding;
    params->sse4.shift[i] = shift;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse4.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse4.output_min[i] = output_min;
    params->sse4.output_max[i] = output_max;
  }
}

void xnn_init_qu8_add_minmax_avx2_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) (uint32_t) a_zero_point + b_multiplier * (int32_t) (uint32_t) b_zero_point);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2.bias[i] = bias;
    params->avx2.a_multiplier[i] = a_multiplier;
    params->avx2.b_multiplier[i] = b_multiplier;
    params->avx2.rounding[i] = rounding;
    params->avx2.shift[i] = shift;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
    params->avx2.output_min[i] = output_min;
    params->avx2.output_max[i] = output_max;
  }
}

void xnn_init_qu8_add_minmax_avx512_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) (uint32_t) a_zero_point + b_multiplier * (int32_t) (uint32_t) b_zero_point);
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.bias[i] = bias;
    params->avx512.a_multiplier[i] = a_multiplier;
    params->avx512.b_multiplier[i] = b_multiplier;
    params->avx512.rounding[i] = rounding;
    params->avx512.shift[i] = shift;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->avx512.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
    params->avx512.output_min[i] = output_min;
    params->avx512.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_qu8_add_minmax_neon_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  params->neon.a_zero_point = a_zero_point;
  params->neon.b_zero_point = b_zero_point;
  params->neon.a_multiplier = (int32_t) a_multiplier;
  params->neon.b_multiplier = (int32_t) b_multiplier;
  params->neon.right_shift = (int32_t) -shift;
  params->neon.output_zero_point = (int16_t) (uint16_t) output_zero_point;
  params->neon.output_min = output_min;
  params->neon.output_max = output_max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD
void xnn_init_qu8_add_minmax_wasmsimd_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) (uint32_t) a_zero_point + b_multiplier * (int32_t) (uint32_t) b_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.bias[i] = bias;
    params->wasmsimd.a_multiplier[i] = a_multiplier;
    params->wasmsimd.b_multiplier[i] = b_multiplier;
    params->wasmsimd.rounding[i] = rounding;
  }
  params->wasmsimd.shift = shift;
  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->wasmsimd.output_min[i] = output_min;
    params->wasmsimd.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD

void xnn_init_qu8_add_minmax_scalar_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  params->scalar.bias = (int32_t) -(a_multiplier * (int32_t) (uint32_t) a_zero_point + b_multiplier * (int32_t) (uint32_t) b_zero_point);
  params->scalar.a_multiplier = a_multiplier;
  params->scalar.b_multiplier = b_multiplier;
  params->scalar.rounding = rounding;
  params->scalar.shift = shift;
  params->scalar.output_min_less_zero_point = (int32_t) (uint32_t) output_min - (int32_t) (uint32_t) output_zero_point;
  params->scalar.output_max_less_zero_point = (int32_t) (uint32_t) output_max - (int32_t) (uint32_t) output_zero_point;
  params->scalar.output_zero_point = (int32_t) (uint32_t) output_zero_point;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qs8_add_minmax_sse2_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) a_zero_point + b_multiplier * (int32_t) b_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.bias[i] = bias;
  }
  const uint16_t a_multiplier_lo = (uint16_t) a_multiplier;
  const uint16_t a_multiplier_hi = (uint16_t) ((uint32_t) a_multiplier >> 16);
  const uint16_t b_multiplier_lo = (uint16_t) b_multiplier;
  const uint16_t b_multiplier_hi = (uint16_t) ((uint32_t) b_multiplier >> 16);
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.a_multiplier_lo[i] = a_multiplier_lo;
    params->sse2.a_multiplier_hi[i] = a_multiplier_hi;
    params->sse2.b_multiplier_lo[i] = b_multiplier_lo;
    params->sse2.b_multiplier_hi[i] = b_multiplier_hi;
  }
  params->sse2.shift = shift;
  params->sse2.b_multiplier = (uint32_t) b_multiplier;
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.rounding[i] = rounding;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.output_zero_point[i] = (int16_t) output_zero_point;
    params->sse2.output_min[i] = (int16_t) output_min;
    params->sse2.output_max[i] = (int16_t) output_max;
  }
}

void xnn_init_qs8_add_minmax_sse4_mul16_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) a_zero_point + b_multiplier * (int32_t) b_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4_mul16.bias[i] = bias;
  }
  const uint16_t a_multiplier_lo = (uint16_t) a_multiplier;
  const uint16_t a_multiplier_hi = (uint16_t) ((uint32_t) a_multiplier >> 16);
  const uint16_t b_multiplier_lo = (uint16_t) b_multiplier;
  const uint16_t b_multiplier_hi = (uint16_t) ((uint32_t) b_multiplier >> 16);
  for (uint32_t i = 0; i < 8; i++) {
    params->sse4_mul16.a_multiplier_lo[i] = a_multiplier_lo;
    params->sse4_mul16.a_multiplier_hi[i] = a_multiplier_hi;
    params->sse4_mul16.b_multiplier_lo[i] = b_multiplier_lo;
    params->sse4_mul16.b_multiplier_hi[i] = b_multiplier_hi;
  }
  params->sse4_mul16.shift = shift;
  params->sse4_mul16.b_multiplier = (uint32_t) b_multiplier;
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4_mul16.rounding[i] = rounding;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse4_mul16.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse4_mul16.output_min[i] = output_min;
    params->sse4_mul16.output_max[i] = output_max;
  }
}

void xnn_init_qs8_add_minmax_sse4_mul32_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) a_zero_point + b_multiplier * (int32_t) b_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4_mul32.bias[i] = bias;
    params->sse4_mul32.a_multiplier[i] = a_multiplier;
    params->sse4_mul32.b_multiplier[i] = b_multiplier;
    params->sse4_mul32.rounding[i] = rounding;
    params->sse4_mul32.shift[i] = shift;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse4_mul32.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse4_mul32.output_min[i] = output_min;
    params->sse4_mul32.output_max[i] = output_max;
  }
}

void xnn_init_qs8_add_minmax_avx2_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) a_zero_point + b_multiplier * (int32_t) b_zero_point);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2.bias[i] = bias;
    params->avx2.a_multiplier[i] = a_multiplier;
    params->avx2.b_multiplier[i] = b_multiplier;
    params->avx2.rounding[i] = rounding;
    params->avx2.shift[i] = shift;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.output_zero_point[i] = (int16_t) output_zero_point;
    params->avx2.output_min[i] = output_min;
    params->avx2.output_max[i] = output_max;
  }
}

void xnn_init_qs8_add_minmax_avx512_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) a_zero_point + b_multiplier * (int32_t) b_zero_point);
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.bias[i] = bias;
    params->avx512.a_multiplier[i] = a_multiplier;
    params->avx512.b_multiplier[i] = b_multiplier;
    params->avx512.rounding[i] = rounding;
    params->avx512.shift[i] = shift;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->avx512.output_zero_point[i] = (int16_t) output_zero_point;
    params->avx512.output_min[i] = output_min;
    params->avx512.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_qs8_add_minmax_neon_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  params->neon.a_zero_point = a_zero_point;
  params->neon.b_zero_point = b_zero_point;
  params->neon.a_multiplier = (int32_t) a_multiplier;
  params->neon.b_multiplier = (int32_t) b_multiplier;
  params->neon.right_shift = (int32_t) -shift;
  params->neon.output_zero_point = (int16_t) output_zero_point;
  params->neon.output_min = output_min;
  params->neon.output_max = output_max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD
void xnn_init_qs8_add_minmax_wasmsimd_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = (int32_t) -(a_multiplier * (int32_t) a_zero_point + b_multiplier * (int32_t) b_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.bias[i] = bias;
    params->wasmsimd.a_multiplier[i] = a_multiplier;
    params->wasmsimd.b_multiplier[i] = b_multiplier;
    params->wasmsimd.rounding[i] = rounding;
  }
  params->wasmsimd.shift = shift;
  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->wasmsimd.output_min[i] = output_min;
    params->wasmsimd.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD

void xnn_init_qs8_add_minmax_scalar_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  assert(a_output_scale >= 0x1.0p-10f);
  assert(b_output_scale >= 0x1.0p-10f);
  assert(a_output_scale < 0x1.0p+8f);
  assert(b_output_scale < 0x1.0p+8f);

  // Compute requantization parameters.
  const float max_output_scale = math_max_f32(a_output_scale, b_output_scale);
  assert(max_output_scale >= 0x1.0p-10f);
  assert(max_output_scale < 0x1.0p+8f);
  const uint32_t max_scale_bits = fp32_to_bits(max_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(a_output_scale) + (shift << 23)));
  const int32_t b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(b_output_scale) + (shift << 23)));
  assert(math_max_s32(a_multiplier, b_multiplier) >= INT32_C(0x00100000));
  assert(a_multiplier < INT32_C(0x00200000));
  assert(b_multiplier < INT32_C(0x00200000));

  const int32_t rounding = INT32_C(1) << (shift - 1);
  params->scalar.bias = (int32_t) -(a_multiplier * (int32_t) a_zero_point + b_multiplier * (int32_t) b_zero_point);
  params->scalar.a_multiplier = a_multiplier;
  params->scalar.b_multiplier = b_multiplier;
  params->scalar.rounding = rounding;
  params->scalar.shift = shift;
  params->scalar.output_min_less_zero_point = (int32_t) output_min - (int32_t) output_zero_point;
  params->scalar.output_max_less_zero_point = (int32_t) output_max - (int32_t) output_zero_point;
  params->scalar.output_zero_point = (int32_t) output_zero_point;
}

void xnn_init_qu8_mul_minmax_fp32_scalar_params(
  union xnn_qu8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float product_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  params->fp32_scalar.a_zero_point = (int16_t) (uint16_t) a_zero_point;
  params->fp32_scalar.b_zero_point = (int16_t) (uint16_t) b_zero_point;
  params->fp32_scalar.scale = product_output_scale;
  params->fp32_scalar.output_min_less_zero_point = (float) (int32_t) ((uint32_t) output_min - (uint32_t) output_zero_point);
  params->fp32_scalar.output_max_less_zero_point = (float) (int32_t) ((uint32_t) output_max - (uint32_t) output_zero_point);
  params->fp32_scalar.magic_bias = 12582912.0f;
  params->fp32_scalar.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) (uint32_t) output_zero_point;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_qu8_mul_minmax_fp32_neon_params(
  union xnn_qu8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float product_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  params->fp32_neon.a_zero_point[0] = a_zero_point;
  params->fp32_neon.a_zero_point[1] = a_zero_point;
  params->fp32_neon.b_zero_point[0] = b_zero_point;
  params->fp32_neon.b_zero_point[1] = b_zero_point;
  params->fp32_neon.scale = product_output_scale;
  params->fp32_neon.output_min_less_zero_point = (float) (int32_t) ((uint32_t) output_min - (uint32_t) output_zero_point);
  params->fp32_neon.output_max_less_zero_point = (float) (int32_t) ((uint32_t) output_max - (uint32_t) output_zero_point);
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) (uint32_t) output_zero_point;
}

void xnn_init_qu8_mul_minmax_fp32_neonv8_params(
  union xnn_qu8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float product_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  params->fp32_neonv8.a_zero_point[0] = a_zero_point;
  params->fp32_neonv8.a_zero_point[1] = a_zero_point;
  params->fp32_neonv8.b_zero_point[0] = b_zero_point;
  params->fp32_neonv8.b_zero_point[1] = b_zero_point;
  params->fp32_neonv8.scale = product_output_scale;
  params->fp32_neonv8.output_zero_point = (int16_t) (uint16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qu8_mul_minmax_fp32_sse2_params(
  union xnn_qu8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float product_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.a_zero_point[i] = (int16_t) (uint16_t) a_zero_point;
    params->fp32_sse2.b_zero_point[i] = (int16_t) (uint16_t) b_zero_point;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.scale[i] = product_output_scale;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_sse2.output_min[i] = output_min;
    params->fp32_sse2.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
void xnn_init_qu8_mul_minmax_fp32_wasmsimd_params(
  union xnn_qu8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float product_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_wasmsimd.a_zero_point[i] = (int16_t) (uint16_t) a_zero_point;
    params->fp32_wasmsimd.b_zero_point[i] = (int16_t) (uint16_t) b_zero_point;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_wasmsimd.scale[i] = product_output_scale;
    params->fp32_wasmsimd.output_min_less_zero_point[i] = (float) (int32_t) ((uint32_t) output_min - (uint32_t) output_zero_point);
    params->fp32_wasmsimd.output_max_less_zero_point[i] = (float) (int32_t) ((uint32_t) output_max - (uint32_t) output_zero_point);
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = INT32_C(0x4B400000) - (int32_t) (uint32_t) output_zero_point;
  }
}
#endif  // XNN_ARCH_WASMSIMD

void xnn_init_qs8_mul_minmax_fp32_scalar_params(
  union xnn_qs8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float product_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_scalar.a_zero_point = (int16_t) a_zero_point;
  params->fp32_scalar.b_zero_point = (int16_t) b_zero_point;
  params->fp32_scalar.scale = product_output_scale;
  params->fp32_scalar.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar.magic_bias = 12582912.0f;
  params->fp32_scalar.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_qs8_mul_minmax_fp32_neon_params(
  union xnn_qs8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float product_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_neon.a_zero_point[0] = a_zero_point;
  params->fp32_neon.a_zero_point[1] = a_zero_point;
  params->fp32_neon.b_zero_point[0] = b_zero_point;
  params->fp32_neon.b_zero_point[1] = b_zero_point;
  params->fp32_neon.scale = product_output_scale;
  params->fp32_neon.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_neon.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

void xnn_init_qs8_mul_minmax_fp32_neonv8_params(
  union xnn_qs8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float product_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_neonv8.a_zero_point[0] = a_zero_point;
  params->fp32_neonv8.a_zero_point[1] = a_zero_point;
  params->fp32_neonv8.b_zero_point[0] = b_zero_point;
  params->fp32_neonv8.b_zero_point[1] = b_zero_point;
  params->fp32_neonv8.scale = product_output_scale;
  params->fp32_neonv8.output_zero_point = (int16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qs8_mul_minmax_fp32_sse2_params(
  union xnn_qs8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float product_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.a_zero_point[i] = (int16_t) a_zero_point;
    params->fp32_sse2.b_zero_point[i] = (int16_t) b_zero_point;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.scale[i] = product_output_scale;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.output_min[i] = (int16_t) output_min;
    params->fp32_sse2.output_max[i] = (int16_t) output_max;
  }
}

void xnn_init_qs8_mul_minmax_fp32_sse4_params(
  union xnn_qs8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float product_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse4.a_zero_point[i] = (int16_t) a_zero_point;
    params->fp32_sse4.b_zero_point[i] = (int16_t) b_zero_point;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse4.scale[i] = product_output_scale;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse4.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_sse4.output_min[i] = output_min;
    params->fp32_sse4.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
void xnn_init_qs8_mul_minmax_fp32_wasmsimd_params(
  union xnn_qs8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float product_output_scale,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_wasmsimd.a_zero_point[i] = (int16_t) a_zero_point;
    params->fp32_wasmsimd.b_zero_point[i] = (int16_t) b_zero_point;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_wasmsimd.scale[i] = product_output_scale;
    params->fp32_wasmsimd.output_min_less_zero_point[i] = (float) ((int32_t) output_min - (int32_t) output_zero_point);
    params->fp32_wasmsimd.output_max_less_zero_point[i] = (float) ((int32_t) output_max - (int32_t) output_zero_point);
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  }
}
#endif  // XNN_ARCH_WASMSIMD
