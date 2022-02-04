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


void xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_fmagic.kernel_zero_point = (int32_t) kernel_zero_point;
  params->fp32_scalar_fmagic.scale = scale;
  params->fp32_scalar_fmagic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_fmagic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_fmagic.magic_bias = 12582912.0f;
  params->fp32_scalar_fmagic.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

void xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_imagic.kernel_zero_point = (int32_t) kernel_zero_point;
  params->fp32_scalar_imagic.scale = scale;
  params->fp32_scalar_imagic.magic_bias = 12582912.0f;
  params->fp32_scalar_imagic.magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  params->fp32_scalar_imagic.magic_max = (int32_t) fp32_to_bits(12582912.0f + output_max_less_zero_point);
  params->fp32_scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

void xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_lrintf.kernel_zero_point = (int32_t) kernel_zero_point;
  params->fp32_scalar_lrintf.scale = scale;
  params->fp32_scalar_lrintf.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_lrintf.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_lrintf.output_zero_point = (int32_t) output_zero_point;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qu8_conv_minmax_fp32_sse2_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.scale[i] = scale;
    params->fp32_sse2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.kernel_zero_point[i] = (int16_t) kernel_zero_point;
    params->fp32_sse2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_sse2.output_min[i] = output_min;
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
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_avx2.scale[i] = scale;
    params->fp32_avx2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx2.kernel_zero_point[i] = (int16_t) kernel_zero_point;
    params->fp32_avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx2.output_min[i] = output_min;
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
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx512.scale[i] = scale;
    params->fp32_avx512.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx512.kernel_zero_point[i] = (int16_t) (uint16_t) kernel_zero_point;
    params->fp32_avx512.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 64; i++) {
    params->fp32_avx512.output_min[i] = output_min;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_qu8_conv_minmax_fp32_neon_params(
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
}

void xnn_init_qu8_conv_minmax_fp32_neonv8_params(
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
}

void xnn_init_qu8_conv_minmax_rndnu_neon_params(
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
  const uint32_t scale_bits = fp32_to_bits(scale);

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
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_qu8_conv_minmax_fp32_wasmsimd_params(
  union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t kernel_zero_point,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  const int32_t magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_wasmsimd.kernel_zero_point[i] = (int16_t) (uint16_t) kernel_zero_point;
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->fp32_wasmsimd.scale[i] = scale;
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_min[i] = magic_min;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = magic_bias_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_wasmsimd.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

void xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_fmagic.scale = scale;
  params->fp32_scalar_fmagic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_fmagic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_fmagic.magic_bias = 12582912.0f;
  params->fp32_scalar_fmagic.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

void xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_imagic.scale = scale;
  params->fp32_scalar_imagic.magic_bias = 12582912.0f;
  params->fp32_scalar_imagic.magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  params->fp32_scalar_imagic.magic_max = (int32_t) fp32_to_bits(12582912.0f + output_max_less_zero_point);
  params->fp32_scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

void xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_scalar_lrintf.scale = scale;
  params->fp32_scalar_lrintf.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_lrintf.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_lrintf.output_zero_point = (int32_t) output_zero_point;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qs8_conv_minmax_fp32_sse2_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.scale[i] = scale;
    params->fp32_sse2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.output_zero_point[i] = (int16_t) output_zero_point;
    params->fp32_sse2.output_min[i] = (int16_t) output_min;
  }
}

void xnn_init_qs8_conv_minmax_fp32_sse4_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse4.scale[i] = scale;
    params->fp32_sse4.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse4.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_sse4.output_min[i] = output_min;
  }
}

void xnn_init_qs8_conv_minmax_fp32_avx2_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_avx2.scale[i] = scale;
    params->fp32_avx2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx2.output_min[i] = output_min;
  }
}

void xnn_init_qs8_conv_minmax_fp32_avx512_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx512.scale[i] = scale;
    params->fp32_avx512.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx512.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 64; i++) {
    params->fp32_avx512.output_min[i] = output_min;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_qs8_conv_minmax_fp32_neon_params(
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
}

void xnn_init_qs8_conv_minmax_fp32_neonv8_params(
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
}

void xnn_init_qs8_conv_minmax_rndnu_neon_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

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
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_qs8_conv_minmax_fp32_wasmsimd_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  const int32_t magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 2; i++) {
    params->fp32_wasmsimd.scale[i] = scale;
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_min[i] = magic_min;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = magic_bias_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_wasmsimd.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

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

void xnn_init_qs8_minmax_scalar_fmagic_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->scalar_fmagic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->scalar_fmagic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_fmagic.magic_bias = 12582912.0f;
  params->scalar_fmagic.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

void xnn_init_qs8_minmax_scalar_imagic_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_imagic.magic_bias = 12582912.0f;
  params->scalar_imagic.magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  params->scalar_imagic.magic_max = (int32_t) fp32_to_bits(12582912.0f + output_max_less_zero_point);
  params->scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

void xnn_init_qs8_minmax_scalar_lrintf_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->scalar_lrintf.output_min_less_zero_point = (long) ((int32_t) output_min - (int32_t) output_zero_point);
  params->scalar_lrintf.output_max_less_zero_point = (long) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_lrintf.output_zero_point = (int32_t) output_zero_point;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qs8_minmax_sse2_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.output_zero_point[i] = (int16_t) output_zero_point;
    params->sse2.output_min[i] = (int16_t) output_min;
  }
}

void xnn_init_qs8_minmax_sse4_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse4.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse4.output_min[i] = output_min;
  }
}

void xnn_init_qs8_minmax_avx2_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->avx2.output_min[i] = output_min;
  }
}

void xnn_init_qs8_minmax_avx512_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->avx512.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 64; i++) {
    params->avx512.output_min[i] = output_min;
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
  params->neon.magic_bias = 12582912.0f;
  params->neon.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->neon.output_min = output_min;
  params->neon.output_max = output_max;
}

void xnn_init_qs8_minmax_neonv8_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->neonv8.output_zero_point = (int16_t) output_zero_point;
  params->neonv8.output_min = output_min;
  params->neonv8.output_max = output_max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_qs8_minmax_wasmsimd_params(
  union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  const int32_t magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd.magic_bias[i] = 12582912.0f;
    params->wasmsimd.magic_min[i] = magic_min;
    params->wasmsimd.magic_bias_less_output_zero_point[i] = magic_bias_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

void xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params(
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

void xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params(
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
  params->fp32_scalar_imagic.magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  params->fp32_scalar_imagic.magic_max = (int32_t) fp32_to_bits(12582912.0f + output_max_less_zero_point);
  params->fp32_scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
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

void xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params(
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
void xnn_init_qs8_avgpool_minmax_fp32_sse2_params(
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

void xnn_init_qs8_avgpool_minmax_fp32_sse4_params(
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
    params->fp32_sse4.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse4.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_sse4.output_min[i] = output_min;
  }
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
void xnn_init_qs8_avgpool_minmax_fp32_neon_params(
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

void xnn_init_qs8_avgpool_minmax_fp32_neonv8_params(
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

void xnn_init_qs8_avgpool_minmax_rndnu_neon_params(
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
  const uint32_t scale_bits = fp32_to_bits(scale);

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
}

void xnn_update_qs8_avgpool_minmax_rndnu_neon_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

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
void xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params(
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
  const int32_t magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
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

void xnn_init_qu8_avgpool_minmax_fp32_scalar_fmagic_params(
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

void xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params(
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
  params->fp32_scalar_imagic.magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  params->fp32_scalar_imagic.magic_max = (int32_t) fp32_to_bits(12582912.0f + output_max_less_zero_point);
  params->fp32_scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
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

void xnn_init_qu8_avgpool_minmax_fp32_scalar_lrintf_params(
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
void xnn_init_qu8_avgpool_minmax_fp32_sse2_params(
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

void xnn_init_qu8_avgpool_minmax_fp32_sse4_params(
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
void xnn_init_qu8_avgpool_minmax_fp32_neon_params(
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

void xnn_init_qu8_avgpool_minmax_fp32_neonv8_params(
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

void xnn_init_qu8_avgpool_minmax_rndnu_neon_params(
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
  const uint32_t scale_bits = fp32_to_bits(scale);

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
}

void xnn_update_qu8_avgpool_minmax_rndnu_neon_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t init_bias,
  float scale)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(scale);

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
void xnn_init_qu8_avgpool_minmax_fp32_wasmsimd_params(
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
  const int32_t magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
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

void xnn_init_qu8_avgpool_minmax_scalar_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
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

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_qu8_avgpool_minmax_neon_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
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

  params->neon.bias = bias;
  params->neon.multiplier = multiplier;
  params->neon.left_shift = (int64_t) -shift;
  params->neon.output_zero_point = (int16_t) (uint16_t) output_zero_point;
  params->neon.output_min = output_min;
  params->neon.output_max = output_max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qu8_avgpool_minmax_sse2_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

void xnn_update_qu8_avgpool_minmax_scalar_params(
  union xnn_qu8_avgpool_minmax_params* params,
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

  const int64_t rounding = INT64_C(1) << ((uint32_t) shift - 1);
  params->scalar.bias = bias;
  params->scalar.multiplier = multiplier;
  params->scalar.rounding = rounding;
  params->scalar.right_shift = (uint32_t) shift;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_update_qu8_avgpool_minmax_neon_params(
  union xnn_qu8_avgpool_minmax_params* params,
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

  params->neon.bias = bias;
  params->neon.multiplier = multiplier;
  params->neon.left_shift = (int64_t) -shift;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_update_qu8_avgpool_minmax_sse2_params(
  union xnn_qu8_avgpool_minmax_params* params,
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
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

void xnn_update_f32_scaleminmax_scalar_params(
  union xnn_f32_scaleminmax_params* params,
  float scale)
{
  params->scalar.scale = scale;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_update_f32_scaleminmax_sse_params(
  union xnn_f32_scaleminmax_params* params,
  float scale)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.scale[i] = scale;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_f16_scaleminmax_neon_params(
  union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  uint16_t min,
  uint16_t max)
{
  params->neon.scale = scale;
  params->neon.min = min;
  params->neon.max = max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f16_scaleminmax_avx_params(
  union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  uint16_t min,
  uint16_t max)
{
  const float scale_f32 = fp16_ieee_to_fp32_value(scale);
  const float min_f32 = fp16_ieee_to_fp32_value(min);
  const float max_f32 = fp16_ieee_to_fp32_value(max);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.scale[i] = scale_f32;
    params->avx.min[i] = min_f32;
    params->avx.max[i] = max_f32;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_update_f16_scaleminmax_neon_params(
  union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale)
{
  params->neon.scale = scale;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_update_f16_scaleminmax_avx_params(
  union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale)
{
  const float scale_f32 = fp16_ieee_to_fp32_value(scale);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.scale[i] = scale_f32;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

void xnn_init_f32_scaleminmax_scalar_params(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max)
{
  params->scalar.scale = scale;
  params->scalar.min = min;
  params->scalar.max = max;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_scaleminmax_sse_params(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.scale[i] = scale;
    params->sse.min[i] = min;
    params->sse.max[i] = max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

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

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_f16_minmax_neon_params(
  union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t min,
  uint16_t max)
{
  params->neon.min = min;
  params->neon.max = max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f16_minmax_avx_params(
  union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t min,
  uint16_t max)
{
  const float min_f32 = fp16_ieee_to_fp32_value(min);
  const float max_f32 = fp16_ieee_to_fp32_value(max);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.min[i] = min_f32;
    params->avx.max[i] = max_f32;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_default_avx_params(
  union xnn_f32_default_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

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
  #elif XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
    params->wasmsimd.min[0] = output_min;
    params->wasmsimd.min[1] = output_min;
    params->wasmsimd.max[0] = output_max;
    params->wasmsimd.max[1] = output_max;
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
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_f32_minmax_wasmsimd_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  params->wasmsimd.min[0] = output_min;
  params->wasmsimd.min[1] = output_min;
  params->wasmsimd.max[0] = output_max;
  params->wasmsimd.max[1] = output_max;
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

void xnn_init_f32_minmax_scalar_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  params->scalar.min = output_min;
  params->scalar.max = output_max;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_f16_hswish_neon_params(
  union xnn_f16_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon.sixth = UINT16_C(0x3155);
  params->neon.three = UINT16_C(0x4200);
  params->neon.six = UINT16_C(0x4600);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f16_hswish_avx_params(
  union xnn_f16_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.sixth[i] = 0x1.554000p-3f;
    params->avx.three[i] = 3.0f;
    params->avx.six[i] = UINT16_C(0x4600);
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

void xnn_init_f32_hswish_scalar_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar.sixth = 0x1.555556p-3f;
  params->scalar.three = 3.0f;
  params->scalar.six = 6.0f;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_hswish_sse_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.sixth[i] = 0x1.555556p-3f;
    params->sse.half[i] = 0.5f;
    params->sse.one[i] = 1.0f;
  }
}

void xnn_init_f32_hswish_avx_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.sixth[i] = 0x1.555556p-3f;
    params->avx.half[i] = 0.5f;
    params->avx.one[i] = 1.0f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
}

void xnn_init_f32_hswish_avx512_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512.sixth = 0x1.555556p-3f;
  params->avx512.half = 0.5f;
  params->avx512.one = 1.0f;
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_f32_hswish_wasmsimd_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd.sixth[i] = 0x1.555556p-3f;
    params->wasmsimd.three[i] = 3.0f;
    params->wasmsimd.six[i] = 6.0f;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

void xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_rr2_lut64_p2.magic_bias = 0x1.800000p17f;
  params->scalar_rr2_lut64_p2.minus_log2e = -0x1.715476p0f;
  params->scalar_rr2_lut64_p2.ln2_hi = 0x1.630000p-1f;
  params->scalar_rr2_lut64_p2.ln2_lo = -0x1.BD0106p-13f;
  params->scalar_rr2_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->scalar_rr2_lut64_p2.one = 1.0f;
  params->scalar_rr2_lut64_p2.denorm_cutoff = 0x1.5D589Ep+6f;
}

void xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_rr2_lut2048_p1.magic_bias = 0x1.800000p12f;
  params->scalar_rr2_lut2048_p1.minus_log2e = -0x1.715476p0f;
  params->scalar_rr2_lut2048_p1.ln2_hi = 0x1.600000p-1f;
  params->scalar_rr2_lut2048_p1.ln2_lo = 0x1.7217F8p-8f;
  params->scalar_rr2_lut2048_p1.c1 = -0x1.FFFFFEp-1f;
  params->scalar_rr2_lut2048_p1.one = 1.0f;
  params->scalar_rr2_lut2048_p1.denorm_cutoff = 0x1.5D589Ep+6f;
}

void xnn_init_f32_sigmoid_scalar_rr2_p5_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_rr2_p5.magic_bias = 0x1.8000FEp23f;
  params->scalar_rr2_p5.minus_log2e = -0x1.715476p0f;
  params->scalar_rr2_p5.ln2_hi = 0x1.62E400p-1f;
  params->scalar_rr2_p5.ln2_lo = 0x1.7F7D1Cp-20f;
  params->scalar_rr2_p5.c5 = -0x1.0F9F9Cp-7f;
  params->scalar_rr2_p5.c4 = 0x1.573A1Ap-5f;
  params->scalar_rr2_p5.c3 = -0x1.555A80p-3f;
  params->scalar_rr2_p5.c2 = 0x1.FFFDC6p-2f;
  params->scalar_rr2_p5.c1 = -0x1.FFFFF6p-1f;
  params->scalar_rr2_p5.one = 1.0f;
  params->scalar_rr2_p5.denorm_cutoff = 0x1.5D589Ep+6f;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon_rr2_lut64_p2.magic_bias = 0x1.800000p17f;
  params->neon_rr2_lut64_p2.minus_log2e = -0x1.715476p0f;
  params->neon_rr2_lut64_p2.ln2_hi = 0x1.630000p-1f;
  params->neon_rr2_lut64_p2.ln2_lo = -0x1.BD0106p-13f;
  params->neon_rr2_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->neon_rr2_lut64_p2.denorm_cutoff = 0x1.5D589Ep+6f;
}

void xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon_rr2_lut2048_p1.magic_bias = 0x1.800000p12f;
  params->neon_rr2_lut2048_p1.minus_log2e = -0x1.715476p0f;
  params->neon_rr2_lut2048_p1.ln2_hi = 0x1.600000p-1f;
  params->neon_rr2_lut2048_p1.ln2_lo = 0x1.7217F8p-8f;
  params->neon_rr2_lut2048_p1.c1 = -0x1.FFFFFEp-1f;
  params->neon_rr2_lut2048_p1.denorm_cutoff = 0x1.5D589Ep+6f;
}

void xnn_init_f32_sigmoid_neon_rr2_p5_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon_rr2_p5.magic_bias = 0x1.8000FEp23f;
  params->neon_rr2_p5.minus_log2e = -0x1.715476p0f;
  params->neon_rr2_p5.ln2_hi = 0x1.62E400p-1f;
  params->neon_rr2_p5.ln2_lo = 0x1.7F7D1Cp-20f;
  params->neon_rr2_p5.c5 = -0x1.0F9F9Cp-7f;
  params->neon_rr2_p5.c4 = 0x1.573A1Ap-5f;
  params->neon_rr2_p5.c3 = -0x1.555A80p-3f;
  params->neon_rr2_p5.c2 = 0x1.FFFDC6p-2f;
  params->neon_rr2_p5.c1 = -0x1.FFFFF6p-1f;
  params->neon_rr2_p5.denorm_cutoff = 0x1.5D589Ep+6f;
}

void xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neonfma_rr1_lut2048_p1.magic_bias = 0x1.800000p12f;
  params->neonfma_rr1_lut2048_p1.minus_log2e = -0x1.715476p0f;
  params->neonfma_rr1_lut2048_p1.ln2 = 0x1.62E430p-1f;
  params->neonfma_rr1_lut2048_p1.c1 = -0x1.FFFFFEp-1f;
  params->neonfma_rr1_lut2048_p1.denorm_cutoff = 0x1.5D589Ep+6f;
}

void xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neonfma_rr1_lut64_p2.magic_bias = 0x1.800000p17f;
  params->neonfma_rr1_lut64_p2.minus_log2e = -0x1.715476p0f;
  params->neonfma_rr1_lut64_p2.ln2 = 0x1.62E430p-1f;
  params->neonfma_rr1_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->neonfma_rr1_lut64_p2.denorm_cutoff = 0x1.5D589Ep+6f;
}

void xnn_init_f32_sigmoid_neonfma_rr1_p5_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neonfma_rr1_p5.magic_bias = 0x1.8000FEp23f;
  params->neonfma_rr1_p5.minus_log2e = -0x1.715476p0f;
  params->neonfma_rr1_p5.ln2 = 0x1.62E430p-1f;
  params->neonfma_rr1_p5.c5 = -0x1.0F9F9Cp-7f;
  params->neonfma_rr1_p5.c4 = 0x1.573A1Ap-5f;
  params->neonfma_rr1_p5.c3 = -0x1.555A80p-3f;
  params->neonfma_rr1_p5.c2 = 0x1.FFFDC6p-2f;
  params->neonfma_rr1_p5.c1 = -0x1.FFFFF6p-1f;
  params->neonfma_rr1_p5.denorm_cutoff = 0x1.5D589Ep+6f;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2_rr2_lut64_p2.sign_mask[i] = -0.0f;
    params->sse2_rr2_lut64_p2.magic_bias[i] = 0x1.800000p17f;
    params->sse2_rr2_lut64_p2.log2e[i] = 0x1.715476p0f;
    params->sse2_rr2_lut64_p2.index_mask[i] = UINT32_C(0x3F);
    params->sse2_rr2_lut64_p2.minus_ln2_hi[i] = -0x1.630000p-1f;
    params->sse2_rr2_lut64_p2.minus_ln2_lo[i] = 0x1.BD0106p-13f;
    params->sse2_rr2_lut64_p2.c2[i] = 0x1.FFFF0Ap-2f;
    params->sse2_rr2_lut64_p2.one[i] = 1.0f;
    params->sse2_rr2_lut64_p2.denorm_cutoff[i] = -0x1.5D589Ep+6f;
  }
}

void xnn_init_f32_sigmoid_sse2_rr2_p5_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2_rr2_p5.sign_mask[i] = -0.0f;
    params->sse2_rr2_p5.magic_bias[i] = 0x1.8000FEp23f;
    params->sse2_rr2_p5.log2e[i] = 0x1.715476p0f;
    params->sse2_rr2_p5.minus_ln2_hi[i] = -0x1.62E400p-1f;
    params->sse2_rr2_p5.minus_ln2_lo[i] = -0x1.7F7D1Cp-20f;
    params->sse2_rr2_p5.c5[i] = 0x1.0F9F9Cp-7f;
    params->sse2_rr2_p5.c4[i] = 0x1.573A1Ap-5f;
    params->sse2_rr2_p5.c3[i] = 0x1.555A80p-3f;
    params->sse2_rr2_p5.c2[i] = 0x1.FFFDC6p-2f;
    params->sse2_rr2_p5.c1[i] = 0x1.FFFFF6p-1f;
    params->sse2_rr2_p5.one[i] = 1.0f;
    params->sse2_rr2_p5.denorm_cutoff[i] = -0x1.5D589Ep+6f;
  }
}

void xnn_init_f32_sigmoid_avx_rr2_p5_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_rr2_p5.sign_mask[i] = -0.0f;
    params->avx_rr2_p5.magic_bias[i] = 0x1.8000FEp23f;
    params->avx_rr2_p5.log2e[i] = 0x1.715476p0f;
    params->avx_rr2_p5.minus_ln2_hi[i] = -0x1.62E400p-1f;
    params->avx_rr2_p5.minus_ln2_lo[i] = -0x1.7F7D1Cp-20f;
    params->avx_rr2_p5.c5[i] = 0x1.0F9F9Cp-7f;
    params->avx_rr2_p5.c4[i] = 0x1.573A1Ap-5f;
    params->avx_rr2_p5.c3[i] = 0x1.555A80p-3f;
    params->avx_rr2_p5.c2[i] = 0x1.FFFDC6p-2f;
    params->avx_rr2_p5.c1[i] = 0x1.FFFFF6p-1f;
    params->avx_rr2_p5.one[i] = 1.0f;
    params->avx_rr2_p5.two[i] = 2.0f;
    params->avx_rr2_p5.denorm_cutoff[i] = -0x1.5D589Ep+6f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx_rr2_p5.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx_rr2_p5.mask_table[i] = 0;
  }
}

void xnn_init_f32_sigmoid_avx2_rr1_p5_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_p5.sign_mask[i] = -0.0f;
    params->avx2_rr1_p5.magic_bias[i] = 0x1.8000FEp23f;
    params->avx2_rr1_p5.log2e[i] = 0x1.715476p0f;
    params->avx2_rr1_p5.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx2_rr1_p5.c5[i] = 0x1.0F9F9Cp-7f;
    params->avx2_rr1_p5.c4[i] = 0x1.573A1Ap-5f;
    params->avx2_rr1_p5.c3[i] = 0x1.555A80p-3f;
    params->avx2_rr1_p5.c2[i] = 0x1.FFFDC6p-2f;
    params->avx2_rr1_p5.c1[i] = 0x1.FFFFF6p-1f;
    params->avx2_rr1_p5.one[i] = 1.0f;
    params->avx2_rr1_p5.denorm_cutoff[i] = -0x1.5D589Ep+6f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx2_rr1_p5.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx2_rr1_p5.mask_table[i] = 0;
  }
}

void xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512_rr1_lut16_p3.sign_mask = UINT32_C(0x80000000);
  params->avx512_rr1_lut16_p3.magic_bias = 0x1.800000p19f;
  params->avx512_rr1_lut16_p3.log2e = 0x1.715476p0f;
  params->avx512_rr1_lut16_p3.minus_ln2 = -0x1.62E430p-1f;
  params->avx512_rr1_lut16_p3.c3 = 0x1.55559Ap-3f;
  params->avx512_rr1_lut16_p3.c2 = 0x1.00021Ep-1f;
  params->avx512_rr1_lut16_p3.one = 1.0f;
  params->avx512_rr1_lut16_p3.table[ 0] = 0x1.000000p+0f;
  params->avx512_rr1_lut16_p3.table[ 1] = 0x1.0B5586p+0f;
  params->avx512_rr1_lut16_p3.table[ 2] = 0x1.172B84p+0f;
  params->avx512_rr1_lut16_p3.table[ 3] = 0x1.2387A6p+0f;
  params->avx512_rr1_lut16_p3.table[ 4] = 0x1.306FE0p+0f;
  params->avx512_rr1_lut16_p3.table[ 5] = 0x1.3DEA64p+0f;
  params->avx512_rr1_lut16_p3.table[ 6] = 0x1.4BFDAEp+0f;
  params->avx512_rr1_lut16_p3.table[ 7] = 0x1.5AB07Ep+0f;
  params->avx512_rr1_lut16_p3.table[ 8] = 0x1.6A09E6p+0f;
  params->avx512_rr1_lut16_p3.table[ 9] = 0x1.7A1148p+0f;
  params->avx512_rr1_lut16_p3.table[10] = 0x1.8ACE54p+0f;
  params->avx512_rr1_lut16_p3.table[11] = 0x1.9C4918p+0f;
  params->avx512_rr1_lut16_p3.table[12] = 0x1.AE89FAp+0f;
  params->avx512_rr1_lut16_p3.table[13] = 0x1.C199BEp+0f;
  params->avx512_rr1_lut16_p3.table[14] = 0x1.D5818Ep+0f;
  params->avx512_rr1_lut16_p3.table[15] = 0x1.EA4AFAp+0f;
}

void xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512_rr2_lut32_p2.sign_mask = UINT32_C(0x80000000);
  params->avx512_rr2_lut32_p2.magic_bias = 0x1.800000p18f;
  params->avx512_rr2_lut32_p2.log2e = 0x1.715476p0f;
  params->avx512_rr2_lut32_p2.minus_ln2_hi = -0x1.62E430p-1f;
  params->avx512_rr2_lut32_p2.minus_ln2_lo = 0x1.05C61p-29f;
  params->avx512_rr2_lut32_p2.c2 = 0x1.000000p-1f;
  params->avx512_rr2_lut32_p2.c1 = 0x1.0000F6p-0f;
  params->avx512_rr2_lut32_p2.one = 1.0f;

  params->avx512_rr2_lut32_p2.table_lo[ 0] = 0x1.000000p+0f;
  params->avx512_rr2_lut32_p2.table_lo[ 1] = 0x1.059B0Ep+0f;
  params->avx512_rr2_lut32_p2.table_lo[ 2] = 0x1.0B5586p+0f;
  params->avx512_rr2_lut32_p2.table_lo[ 3] = 0x1.11301Ep+0f;
  params->avx512_rr2_lut32_p2.table_lo[ 4] = 0x1.172B84p+0f;
  params->avx512_rr2_lut32_p2.table_lo[ 5] = 0x1.1D4874p+0f;
  params->avx512_rr2_lut32_p2.table_lo[ 6] = 0x1.2387A6p+0f;
  params->avx512_rr2_lut32_p2.table_lo[ 7] = 0x1.29E9E0p+0f;
  params->avx512_rr2_lut32_p2.table_lo[ 8] = 0x1.306FE0p+0f;
  params->avx512_rr2_lut32_p2.table_lo[ 9] = 0x1.371A74p+0f;
  params->avx512_rr2_lut32_p2.table_lo[10] = 0x1.3DEA64p+0f;
  params->avx512_rr2_lut32_p2.table_lo[11] = 0x1.44E086p+0f;
  params->avx512_rr2_lut32_p2.table_lo[12] = 0x1.4BFDAEp+0f;
  params->avx512_rr2_lut32_p2.table_lo[13] = 0x1.5342B6p+0f;
  params->avx512_rr2_lut32_p2.table_lo[14] = 0x1.5AB07Ep+0f;
  params->avx512_rr2_lut32_p2.table_lo[15] = 0x1.6247ECp+0f;

  params->avx512_rr2_lut32_p2.table_hi[ 0] = 0x1.6A09E6p+0f;
  params->avx512_rr2_lut32_p2.table_hi[ 1] = 0x1.71F75Ep+0f;
  params->avx512_rr2_lut32_p2.table_hi[ 2] = 0x1.7A1148p+0f;
  params->avx512_rr2_lut32_p2.table_hi[ 3] = 0x1.82589Ap+0f;
  params->avx512_rr2_lut32_p2.table_hi[ 4] = 0x1.8ACE54p+0f;
  params->avx512_rr2_lut32_p2.table_hi[ 5] = 0x1.93737Cp+0f;
  params->avx512_rr2_lut32_p2.table_hi[ 6] = 0x1.9C4918p+0f;
  params->avx512_rr2_lut32_p2.table_hi[ 7] = 0x1.A5503Cp+0f;
  params->avx512_rr2_lut32_p2.table_hi[ 8] = 0x1.AE89FAp+0f;
  params->avx512_rr2_lut32_p2.table_hi[ 9] = 0x1.B7F770p+0f;
  params->avx512_rr2_lut32_p2.table_hi[10] = 0x1.C199BEp+0f;
  params->avx512_rr2_lut32_p2.table_hi[11] = 0x1.CB720Ep+0f;
  params->avx512_rr2_lut32_p2.table_hi[12] = 0x1.D5818Ep+0f;
  params->avx512_rr2_lut32_p2.table_hi[13] = 0x1.DFC974p+0f;
  params->avx512_rr2_lut32_p2.table_hi[14] = 0x1.EA4AFAp+0f;
  params->avx512_rr2_lut32_p2.table_hi[15] = 0x1.F50766p+0f;
}

void xnn_init_f32_sigmoid_avx512_rr1_p5_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512_rr1_p5.sign_mask = UINT32_C(0x80000000);
  params->avx512_rr1_p5.log2e = 0x1.715476p0f;
  params->avx512_rr1_p5.minus_ln2 = -0x1.62E430p-1f;
  params->avx512_rr1_p5.c5 = 0x1.0F9F9Cp-7f;
  params->avx512_rr1_p5.c4 = 0x1.573A1Ap-5f;
  params->avx512_rr1_p5.c3 = 0x1.555A80p-3f;
  params->avx512_rr1_p5.c2 = 0x1.FFFDC6p-2f;
  params->avx512_rr1_p5.c1 = 0x1.FFFFF6p-1f;
  params->avx512_rr1_p5.one = 1.0f;
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_rr2_lut64_p2.magic_bias[i] = 0x1.800000p17f;
    params->wasmsimd_rr2_lut64_p2.minus_log2e[i] = -0x1.715476p0f;
    params->wasmsimd_rr2_lut64_p2.index_mask[i] = UINT32_C(0x3F);
    params->wasmsimd_rr2_lut64_p2.ln2_hi[i] = 0x1.630000p-1f;
    params->wasmsimd_rr2_lut64_p2.ln2_lo[i] = -0x1.BD0106p-13f;
    params->wasmsimd_rr2_lut64_p2.c2[i] = 0x1.FFFF0Ap-2f;
    params->wasmsimd_rr2_lut64_p2.one[i] = 1.0f;
    params->wasmsimd_rr2_lut64_p2.denorm_cutoff[i] = 0x1.5D589Ep+6f;
  }
}

void xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_rr2_p5.magic_bias[i] = 0x1.8000FEp23f;
    params->wasmsimd_rr2_p5.minus_log2e[i] = -0x1.715476p+0f;
    params->wasmsimd_rr2_p5.ln2_hi[i] = 0x1.62E400p-1f;
    params->wasmsimd_rr2_p5.ln2_lo[i] = 0x1.7F7D1Cp-20f;
    params->wasmsimd_rr2_p5.c5[i] = -0x1.0F9F9Cp-7f;
    params->wasmsimd_rr2_p5.c4[i] = 0x1.573A1Ap-5f;
    params->wasmsimd_rr2_p5.c3[i] = -0x1.555A80p-3f;
    params->wasmsimd_rr2_p5.c2[i] = 0x1.FFFDC6p-2f;
    params->wasmsimd_rr2_p5.c1[i] = -0x1.FFFFF6p-1f;
    params->wasmsimd_rr2_p5.one[i] = 1.0f;
    params->wasmsimd_rr2_p5.denorm_cutoff[i] = 0x1.5D589Ep+6f;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_abs_sse_params(
  union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.nonsign_mask[i] = math_nonsign_mask_f32();
  }
}

void xnn_init_f32_abs_avx_params(
  union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.nonsign_mask[i] = math_nonsign_mask_f32();
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
}

void xnn_init_f32_abs_avx512_params(
  union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512.nonsign_mask = UINT32_C(0x7FFFFFFF);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_f32_abs_wasmsimd_params(
  union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)])
{
  params->wasmsimd.nonsign_mask[0] = math_nonsign_mask_f32();
  params->wasmsimd.nonsign_mask[1] = math_nonsign_mask_f32();
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_neg_sse_params(
  union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.sign_mask[i] = -0.0f;
  }
}

void xnn_init_f32_neg_avx_params(
  union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.sign_mask[i] = -0.0f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
}

void xnn_init_f32_neg_avx512_params(
  union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512.sign_mask = UINT32_C(0x80000000);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_f32_neg_wasmsimd_params(
  union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)])
{
  params->wasmsimd.sign_mask[0] = -0.0f;
  params->wasmsimd.sign_mask[1] = -0.0f;
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_rnd_sse2_params(
  union xnn_f32_rnd_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.sign_mask[i] = -0.0f;
    params->sse2.one[i] = 1.0f;
  }
}

void xnn_init_f32_rnd_avx_params(
  union xnn_f32_rnd_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_f32_rnd_wasmsimd_params(
  union xnn_f32_rnd_params params[XNN_MIN_ELEMENTS(1)])
{
  params->wasmsimd.sign_mask[0] = -0.0f;
  params->wasmsimd.sign_mask[1] = -0.0f;
  params->wasmsimd.magic_bias[0] = 0x1.000000p+23f;
  params->wasmsimd.magic_bias[1] = 0x1.000000p+23f;
  params->wasmsimd.one[0] = 1.0f;
  params->wasmsimd.one[1] = 1.0f;
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

void xnn_init_f32_elu_scalar_rr2_lut16_p3_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->scalar_rr2_lut16_p3.prescale = prescale;
  params->scalar_rr2_lut16_p3.alpha = alpha;
  params->scalar_rr2_lut16_p3.beta = beta;
  params->scalar_rr2_lut16_p3.sat_cutoff = -0x1.154246p+4f;
  params->scalar_rr2_lut16_p3.magic_bias = 0x1.800000p19f;
  params->scalar_rr2_lut16_p3.log2e = 0x1.715476p+0f;
  params->scalar_rr2_lut16_p3.minus_ln2_hi = -0x1.62E400p-1f;
  params->scalar_rr2_lut16_p3.minus_ln2_lo = -0x1.7F7D1Cp-20f;
  params->scalar_rr2_lut16_p3.c3 = 0x1.55561Cp-3f;
  params->scalar_rr2_lut16_p3.c2 = 0x1.0001ECp-1f;
  params->scalar_rr2_lut16_p3.one = 1.0f;
}

void xnn_init_f32_elu_scalar_rr2_p6_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->scalar_rr2_p6.prescale = prescale;
  params->scalar_rr2_p6.alpha = alpha;
  params->scalar_rr2_p6.beta = beta;
  params->scalar_rr2_p6.sat_cutoff = -0x1.154246p+4f;
  params->scalar_rr2_p6.magic_bias = 0x1.8000FEp23f;
  params->scalar_rr2_p6.log2e = 0x1.715476p+0f;
  params->scalar_rr2_p6.minus_ln2_hi = -0x1.62E440p-1f;
  params->scalar_rr2_p6.minus_ln2_lo = 0x1.0105C6p-21f;
  params->scalar_rr2_p6.c6 = 0x1.6b7338p-10f;
  params->scalar_rr2_p6.c5 = 0x1.12278Ep-7f;
  params->scalar_rr2_p6.c4 = 0x1.555716p-5f;
  params->scalar_rr2_p6.c3 = 0x1.5554B0p-3f;
  params->scalar_rr2_p6.c2 = 0x1.FFFFFEp-2f;
  params->scalar_rr2_p6.one = 1.0f;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_f32_elu_neon_rr2_lut16_p3_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->neon_rr2_lut16_p3.prescale = prescale;
  params->neon_rr2_lut16_p3.alpha = alpha;
  params->neon_rr2_lut16_p3.beta = beta;
  params->neon_rr2_lut16_p3.sat_cutoff = -0x1.154246p+4f;
  params->neon_rr2_lut16_p3.magic_bias = 0x1.800000p19f;
  params->neon_rr2_lut16_p3.log2e = 0x1.715476p+0f;
  params->neon_rr2_lut16_p3.minus_ln2_hi = -0x1.62E400p-1f;
  params->neon_rr2_lut16_p3.minus_ln2_lo = -0x1.7F7D1Cp-20f;
  params->neon_rr2_lut16_p3.c3 = 0x1.55561Cp-3f;
  params->neon_rr2_lut16_p3.c2 = 0x1.0001ECp-1f;
}

void xnn_init_f32_elu_neon_rr2_p6_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->neon_rr2_p6.prescale = prescale;
  params->neon_rr2_p6.alpha = alpha;
  params->neon_rr2_p6.beta = beta;
  params->neon_rr2_p6.sat_cutoff = -0x1.154246p+4f;
  params->neon_rr2_p6.magic_bias = 0x1.8000FEp23f;
  params->neon_rr2_p6.log2e = 0x1.715476p+0f;
  params->neon_rr2_p6.minus_ln2_hi = -0x1.62E440p-1f;
  params->neon_rr2_p6.minus_ln2_lo = 0x1.0105C6p-21f;
  params->neon_rr2_p6.c6 = 0x1.6b7338p-10f;
  params->neon_rr2_p6.c5 = 0x1.12278Ep-7f;
  params->neon_rr2_p6.c4 = 0x1.555716p-5f;
  params->neon_rr2_p6.c3 = 0x1.5554B0p-3f;
  params->neon_rr2_p6.c2 = 0x1.FFFFFEp-2f;
}

void xnn_init_f32_elu_neonfma_rr1_lut16_p3_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->neonfma_rr1_lut16_p3.prescale = prescale;
  params->neonfma_rr1_lut16_p3.alpha = alpha;
  params->neonfma_rr1_lut16_p3.beta = beta;
  params->neonfma_rr1_lut16_p3.sat_cutoff = -0x1.154246p+4f;
  params->neonfma_rr1_lut16_p3.magic_bias = 0x1.800000p19f;
  params->neonfma_rr1_lut16_p3.log2e = 0x1.715476p+0f;
  params->neonfma_rr1_lut16_p3.minus_ln2 = -0x1.62E430p-1f;
  params->neonfma_rr1_lut16_p3.c3 = 0x1.55561Cp-3f;
  params->neonfma_rr1_lut16_p3.c2 = 0x1.0001ECp-1f;
}

void xnn_init_f32_elu_neonfma_rr1_p6_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->neonfma_rr1_p6.prescale = prescale;
  params->neonfma_rr1_p6.alpha = alpha;
  params->neonfma_rr1_p6.beta = beta;
  params->neonfma_rr1_p6.sat_cutoff = -0x1.154246p+4f;
  params->neonfma_rr1_p6.magic_bias = 0x1.8000FEp23f;
  params->neonfma_rr1_p6.log2e = 0x1.715476p+0f;
  params->neonfma_rr1_p6.minus_ln2 = -0x1.62E430p-1f;
  params->neonfma_rr1_p6.c6 = 0x1.6b7338p-10f;
  params->neonfma_rr1_p6.c5 = 0x1.12278Ep-7f;
  params->neonfma_rr1_p6.c4 = 0x1.555716p-5f;
  params->neonfma_rr1_p6.c3 = 0x1.5554B0p-3f;
  params->neonfma_rr1_p6.c2 = 0x1.FFFFFEp-2f;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_elu_sse2_rr2_lut16_p3_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2_rr2_lut16_p3.prescale[i] = prescale;
    params->sse2_rr2_lut16_p3.alpha[i] = alpha;
    params->sse2_rr2_lut16_p3.beta[i] = beta;
    params->sse2_rr2_lut16_p3.sat_cutoff[i] = -0x1.154246p+4f;
    params->sse2_rr2_lut16_p3.magic_bias[i] = 0x1.800000p19f;
    params->sse2_rr2_lut16_p3.log2e[i] = 0x1.715476p+0f;
    params->sse2_rr2_lut16_p3.index_mask[i] = UINT32_C(0xF);
    params->sse2_rr2_lut16_p3.minus_ln2_hi[i] = -0x1.62E400p-1f;
    params->sse2_rr2_lut16_p3.minus_ln2_lo[i] = -0x1.7F7D1Cp-20f;
    params->sse2_rr2_lut16_p3.c3[i] = 0x1.55561Cp-3f;
    params->sse2_rr2_lut16_p3.c2[i] = 0x1.0001ECp-1f;
    params->sse2_rr2_lut16_p3.one[i] = 1.0f;
  }
}

void xnn_init_f32_elu_sse2_rr2_p6_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2_rr2_p6.prescale[i] = prescale;
    params->sse2_rr2_p6.alpha[i] = alpha;
    params->sse2_rr2_p6.beta[i] = beta;
    params->sse2_rr2_p6.sat_cutoff[i] = -0x1.154246p+4f;
    params->sse2_rr2_p6.magic_bias[i] = 0x1.8000FEp23f;
    params->sse2_rr2_p6.log2e[i] = 0x1.715476p+0f;
    params->sse2_rr2_p6.minus_ln2_hi[i] = -0x1.62E440p-1f;
    params->sse2_rr2_p6.minus_ln2_lo[i] = 0x1.0105C6p-21f;
    params->sse2_rr2_p6.c6[i] = 0x1.6b7338p-10f;
    params->sse2_rr2_p6.c5[i] = 0x1.12278Ep-7f;
    params->sse2_rr2_p6.c4[i] = 0x1.555716p-5f;
    params->sse2_rr2_p6.c3[i] = 0x1.5554B0p-3f;
    params->sse2_rr2_p6.c2[i] = 0x1.FFFFFEp-2f;
    params->sse2_rr2_p6.one[i] = 1.0f;
  }
}

void xnn_init_f32_elu_avx_rr2_lut16_p3_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_rr2_lut16_p3.prescale[i] = prescale;
    params->avx_rr2_lut16_p3.alpha[i] = alpha;
    params->avx_rr2_lut16_p3.beta[i] = beta;
    params->avx_rr2_lut16_p3.sat_cutoff[i] = -0x1.154246p+4f;
    params->avx_rr2_lut16_p3.magic_bias[i] = 0x1.800000p19f;
    params->avx_rr2_lut16_p3.log2e[i] = 0x1.715476p+0f;
    params->avx_rr2_lut16_p3.index_mask[i] = UINT32_C(0xF);
    params->avx_rr2_lut16_p3.minus_ln2_hi[i] = -0x1.62E400p-1f;
    params->avx_rr2_lut16_p3.minus_ln2_lo[i] = -0x1.7F7D1Cp-20f;
    params->avx_rr2_lut16_p3.c3[i] = 0x1.55561Cp-3f;
    params->avx_rr2_lut16_p3.c2[i] = 0x1.0001ECp-1f;
    params->avx_rr2_lut16_p3.one[i] = 1.0f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx_rr2_lut16_p3.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx_rr2_lut16_p3.mask_table[i] = 0;
  }
}

void xnn_init_f32_elu_avx_rr2_lut4_p4_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_rr2_lut4_p4.prescale[i] = prescale;
    params->avx_rr2_lut4_p4.alpha[i] = alpha;
    params->avx_rr2_lut4_p4.beta[i] = beta;
    params->avx_rr2_lut4_p4.sat_cutoff[i] = -0x1.154246p+4f;
    params->avx_rr2_lut4_p4.magic_bias[i] = 0x1.8003F8p21f;
    params->avx_rr2_lut4_p4.log2e[i] = 0x1.715476p+0f;
    params->avx_rr2_lut4_p4.index_mask[i] = UINT32_C(0x3);
  }
  params->avx_rr2_lut4_p4.table[0] = 0x1.000000p+0f;
  params->avx_rr2_lut4_p4.table[1] = 0x1.306FE0p+0f;
  params->avx_rr2_lut4_p4.table[2] = 0x1.6A09E6p+0f;
  params->avx_rr2_lut4_p4.table[3] = 0x1.AE89FAp+0f;
  params->avx_rr2_lut4_p4.table[4] = 0x1.000000p+0f;
  params->avx_rr2_lut4_p4.table[5] = 0x1.306FE0p+0f;
  params->avx_rr2_lut4_p4.table[6] = 0x1.6A09E6p+0f;
  params->avx_rr2_lut4_p4.table[7] = 0x1.AE89FAp+0f;
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_rr2_lut4_p4.minus_ln2_hi[i] = -0x1.62E400p-1f;
    params->avx_rr2_lut4_p4.minus_ln2_lo[i] = -0x1.7F7D1Cp-20f;
    params->avx_rr2_lut4_p4.c4[i] = 0x1.554F9Ap-5f;
    params->avx_rr2_lut4_p4.c3[i] = 0x1.557082p-3f;
    params->avx_rr2_lut4_p4.c2[i] = 0x1.000002p-1f;
    params->avx_rr2_lut4_p4.one[i] = 1.0f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx_rr2_lut4_p4.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx_rr2_lut4_p4.mask_table[i] = 0;
  }
}

void xnn_init_f32_elu_avx_rr2_p6_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_rr2_p6.prescale[i] = prescale;
    params->avx_rr2_p6.alpha[i] = alpha;
    params->avx_rr2_p6.beta[i] = beta;
    params->avx_rr2_p6.sat_cutoff[i] = -0x1.154246p+4f;
    params->avx_rr2_p6.magic_bias[i] = 0x1.8000FEp23f;
    params->avx_rr2_p6.log2e[i] = 0x1.715476p+0f;
    params->avx_rr2_p6.minus_ln2_hi[i] = -0x1.62E440p-1f;
    params->avx_rr2_p6.minus_ln2_lo[i] = 0x1.0105C6p-21f;
    params->avx_rr2_p6.c6[i] = 0x1.6b7338p-10f;
    params->avx_rr2_p6.c5[i] = 0x1.12278Ep-7f;
    params->avx_rr2_p6.c4[i] = 0x1.555716p-5f;
    params->avx_rr2_p6.c3[i] = 0x1.5554B0p-3f;
    params->avx_rr2_p6.c2[i] = 0x1.FFFFFEp-2f;
    params->avx_rr2_p6.one[i] = 1.0f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx_rr2_p6.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx_rr2_p6.mask_table[i] = 0;
  }
}

void xnn_init_f32_elu_avx2_rr1_lut16_p3_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_lut16_p3.prescale[i] = prescale;
    params->avx2_rr1_lut16_p3.alpha[i] = alpha;
    params->avx2_rr1_lut16_p3.beta[i] = beta;
    params->avx2_rr1_lut16_p3.sat_cutoff[i] = -0x1.154246p+4f;
    params->avx2_rr1_lut16_p3.magic_bias[i] = 0x1.800000p19f;
    params->avx2_rr1_lut16_p3.log2e[i] = 0x1.715476p+0f;
    params->avx2_rr1_lut16_p3.index_mask[i] = UINT32_C(0xF);
    params->avx2_rr1_lut16_p3.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx2_rr1_lut16_p3.c3[i] = 0x1.55561Cp-3f;
    params->avx2_rr1_lut16_p3.c2[i] = 0x1.0001ECp-1f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx2_rr1_lut16_p3.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx2_rr1_lut16_p3.mask_table[i] = 0;
  }
}

void xnn_init_f32_elu_avx2_rr1_lut8_p4_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_lut8_p4.prescale[i] = prescale;
    params->avx2_rr1_lut8_p4.alpha[i] = alpha;
    params->avx2_rr1_lut8_p4.beta[i] = beta;
    params->avx2_rr1_lut8_p4.sat_cutoff[i] = -0x1.154246p+4f;
    params->avx2_rr1_lut8_p4.magic_bias[i] = 0x1.800000p20f;
    params->avx2_rr1_lut8_p4.log2e[i] = 0x1.715476p+0f;
  }
  params->avx2_rr1_lut8_p4.table[0] = UINT32_C(0x3F800000);
  params->avx2_rr1_lut8_p4.table[1] = UINT32_C(0x3F7B95C2);
  params->avx2_rr1_lut8_p4.table[2] = UINT32_C(0x3F7837F0);
  params->avx2_rr1_lut8_p4.table[3] = UINT32_C(0x3F75FED7);
  params->avx2_rr1_lut8_p4.table[4] = UINT32_C(0x3F7504F3);
  params->avx2_rr1_lut8_p4.table[5] = UINT32_C(0x3F75672A);
  params->avx2_rr1_lut8_p4.table[6] = UINT32_C(0x3F7744FD);
  params->avx2_rr1_lut8_p4.table[7] = UINT32_C(0x3F7AC0C7);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_lut8_p4.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx2_rr1_lut8_p4.c4[i] = 0x1.5558ECp-5f;
    params->avx2_rr1_lut8_p4.c3[i] = 0x1.555C20p-3f;
    params->avx2_rr1_lut8_p4.c2[i] = 0x1.000000p-1f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx2_rr1_lut8_p4.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx2_rr1_lut8_p4.mask_table[i] = 0;
  }
}

void xnn_init_f32_elu_avx2_rr1_lut4_p4_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_lut4_p4.prescale[i] = prescale;
    params->avx2_rr1_lut4_p4.alpha[i] = alpha;
    params->avx2_rr1_lut4_p4.beta[i] = beta;
    params->avx2_rr1_lut4_p4.sat_cutoff[i] = -0x1.154246p+4f;
    params->avx2_rr1_lut4_p4.magic_bias[i] = 0x1.800000p21f;
    params->avx2_rr1_lut4_p4.log2e[i] = 0x1.715476p+0f;
  }
  params->avx2_rr1_lut4_p4.table[0] = 0x1.000000p+0f;
  params->avx2_rr1_lut4_p4.table[1] = 0x1.F06FE0p-1f;
  params->avx2_rr1_lut4_p4.table[2] = 0x1.EA09E6p-1f;
  params->avx2_rr1_lut4_p4.table[3] = 0x1.EE89FAp-1f;
  params->avx2_rr1_lut4_p4.table[4] = 0x1.000000p+0f;
  params->avx2_rr1_lut4_p4.table[5] = 0x1.F06FE0p-1f;
  params->avx2_rr1_lut4_p4.table[6] = 0x1.EA09E6p-1f;
  params->avx2_rr1_lut4_p4.table[7] = 0x1.EE89FAp-1f;
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_lut4_p4.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx2_rr1_lut4_p4.c4[i] = 0x1.554F9Ap-5f;
    params->avx2_rr1_lut4_p4.c3[i] = 0x1.557082p-3f;
    params->avx2_rr1_lut4_p4.c2[i] = 0x1.000002p-1f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx2_rr1_lut4_p4.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx2_rr1_lut4_p4.mask_table[i] = 0;
  }
}

void xnn_init_f32_elu_avx2_rr1_p6_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_p6.prescale[i] = prescale;
    params->avx2_rr1_p6.alpha[i] = alpha;
    params->avx2_rr1_p6.beta[i] = beta;
    params->avx2_rr1_p6.sat_cutoff[i] = -0x1.154246p+4f;
    params->avx2_rr1_p6.magic_bias[i] = 0x1.8000FEp23f;
    params->avx2_rr1_p6.log2e[i] = 0x1.715476p+0f;
    params->avx2_rr1_p6.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx2_rr1_p6.c6[i] = 0x1.6B7338p-10f;
    params->avx2_rr1_p6.c5[i] = 0x1.12278Ep-7f;
    params->avx2_rr1_p6.c4[i] = 0x1.555716p-5f;
    params->avx2_rr1_p6.c3[i] = 0x1.5554B0p-3f;
    params->avx2_rr1_p6.c2[i] = 0x1.FFFFFEp-2f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx2_rr1_p6.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx2_rr1_p6.mask_table[i] = 0;
  }
}

void xnn_init_f32_elu_avx512_rr1_lut16_p3_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->avx512_rr1_lut16_p3.prescale = prescale;
  params->avx512_rr1_lut16_p3.alpha = alpha;
  params->avx512_rr1_lut16_p3.beta = beta;
  params->avx512_rr1_lut16_p3.sat_cutoff = -0x1.154246p+4f;
  params->avx512_rr1_lut16_p3.magic_bias = 0x1.800000p19f;
  params->avx512_rr1_lut16_p3.log2e = 0x1.715476p+0f;
  params->avx512_rr1_lut16_p3.minus_ln2 = -0x1.62E430p-1f;
  params->avx512_rr1_lut16_p3.c3 = 0x1.55561Cp-3f;
  params->avx512_rr1_lut16_p3.c2 = 0x1.0001ECp-1f;
  params->avx512_rr1_lut16_p3.table[ 0] = UINT32_C(0x3F800000);
  params->avx512_rr1_lut16_p3.table[ 1] = UINT32_C(0x3F7DAAC3);
  params->avx512_rr1_lut16_p3.table[ 2] = UINT32_C(0x3F7B95C2);
  params->avx512_rr1_lut16_p3.table[ 3] = UINT32_C(0x3F79C3D3);
  params->avx512_rr1_lut16_p3.table[ 4] = UINT32_C(0x3F7837F0);
  params->avx512_rr1_lut16_p3.table[ 5] = UINT32_C(0x3F76F532);
  params->avx512_rr1_lut16_p3.table[ 6] = UINT32_C(0x3F75FED7);
  params->avx512_rr1_lut16_p3.table[ 7] = UINT32_C(0x3F75583F);
  params->avx512_rr1_lut16_p3.table[ 8] = UINT32_C(0x3F7504F3);
  params->avx512_rr1_lut16_p3.table[ 9] = UINT32_C(0x3F7508A4);
  params->avx512_rr1_lut16_p3.table[10] = UINT32_C(0x3F75672A);
  params->avx512_rr1_lut16_p3.table[11] = UINT32_C(0x3F76248C);
  params->avx512_rr1_lut16_p3.table[12] = UINT32_C(0x3F7744FD);
  params->avx512_rr1_lut16_p3.table[13] = UINT32_C(0x3F78CCDF);
  params->avx512_rr1_lut16_p3.table[14] = UINT32_C(0x3F7AC0C7);
  params->avx512_rr1_lut16_p3.table[15] = UINT32_C(0x3F7D257D);
}

void xnn_init_f32_elu_avx512_rr1_p6_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->avx512_rr1_p6.prescale = prescale;
  params->avx512_rr1_p6.alpha = alpha;
  params->avx512_rr1_p6.beta = beta;
  params->avx512_rr1_p6.sat_cutoff = -0x1.154246p+4f;
  params->avx512_rr1_p6.magic_bias = 0x1.8000FEp23f;
  params->avx512_rr1_p6.log2e = 0x1.715476p+0f;
  params->avx512_rr1_p6.minus_ln2 = -0x1.62E430p-1f;
  params->avx512_rr1_p6.c6 = 0x1.6B7338p-10f;
  params->avx512_rr1_p6.c5 = 0x1.12278Ep-7f;
  params->avx512_rr1_p6.c4 = 0x1.555716p-5f;
  params->avx512_rr1_p6.c3 = 0x1.5554B0p-3f;
  params->avx512_rr1_p6.c2 = 0x1.FFFFFEp-2f;
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_rr2_lut16_p3.prescale[i] = prescale;
    params->wasmsimd_rr2_lut16_p3.alpha[i] = alpha;
    params->wasmsimd_rr2_lut16_p3.beta[i] = beta;
    params->wasmsimd_rr2_lut16_p3.sat_cutoff[i] = -0x1.154246p+4f;
    params->wasmsimd_rr2_lut16_p3.magic_bias[i] = 0x1.800000p19f;
    params->wasmsimd_rr2_lut16_p3.log2e[i] = 0x1.715476p+0f;
    params->wasmsimd_rr2_lut16_p3.index_mask[i] = UINT32_C(0xF);
    params->wasmsimd_rr2_lut16_p3.minus_ln2_hi[i] = -0x1.62E400p-1f;
    params->wasmsimd_rr2_lut16_p3.minus_ln2_lo[i] = -0x1.7F7D1Cp-20f;
    params->wasmsimd_rr2_lut16_p3.c3[i] = 0x1.55561Cp-3f;
    params->wasmsimd_rr2_lut16_p3.c2[i] = 0x1.0001ECp-1f;
    params->wasmsimd_rr2_lut16_p3.one[i] = 1.0f;
  }
}

void xnn_init_f32_elu_wasmsimd_rr2_p6_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_rr2_p6.prescale[i] = prescale;
    params->wasmsimd_rr2_p6.alpha[i] = alpha;
    params->wasmsimd_rr2_p6.beta[i] = beta;
    params->wasmsimd_rr2_p6.sat_cutoff[i] = -0x1.154246p+4f;
    params->wasmsimd_rr2_p6.magic_bias[i] = 0x1.8000FEp23f;
    params->wasmsimd_rr2_p6.log2e[i] = 0x1.715476p+0f;
    params->wasmsimd_rr2_p6.minus_ln2_hi[i] = -0x1.62E440p-1f;
    params->wasmsimd_rr2_p6.minus_ln2_lo[i] = 0x1.0105C6p-21f;
    params->wasmsimd_rr2_p6.c6[i] = 0x1.6b7338p-10f;
    params->wasmsimd_rr2_p6.c5[i] = 0x1.12278Ep-7f;
    params->wasmsimd_rr2_p6.c4[i] = 0x1.555716p-5f;
    params->wasmsimd_rr2_p6.c3[i] = 0x1.5554B0p-3f;
    params->wasmsimd_rr2_p6.c2[i] = 0x1.FFFFFEp-2f;
    params->wasmsimd_rr2_p6.one[i] = 1.0f;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

void xnn_init_f32_expminus_scalar_rr2_p5_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_rr2_p5.log2e = 0x1.715476p+0f;
  params->scalar_rr2_p5.magic_bias = 0x1.8000FEp23f;
  params->scalar_rr2_p5.minus_ln2_hi = -0x1.62E400p-1f;
  params->scalar_rr2_p5.minus_ln2_lo = -0x1.7F7D1Cp-20f;
  params->scalar_rr2_p5.c5 = 0x1.0F9F9Cp-7f;
  params->scalar_rr2_p5.c4 = 0x1.573A1Ap-5f;
  params->scalar_rr2_p5.c3 = 0x1.555A80p-3f;
  params->scalar_rr2_p5.c2 = 0x1.FFFDC6p-2f;
  params->scalar_rr2_p5.c1 = 0x1.FFFFF6p-1f;
  params->scalar_rr2_p5.denorm_cutoff = -0x1.5D589Ep6f;
}

void xnn_init_f32_expminus_scalar_rr2_lut64_p2_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_rr2_lut64_p2.log2e  = 0x1.715476p0f;
  params->scalar_rr2_lut64_p2.magic_bias = 0x1.800000p17f;
  params->scalar_rr2_lut64_p2.minus_ln2_hi = -0x1.630000p-1f;
  params->scalar_rr2_lut64_p2.minus_ln2_lo = 0x1.BD0106p-13f;
  params->scalar_rr2_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->scalar_rr2_lut64_p2.denorm_cutoff = -0x1.5D589Ep6f;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_f32_expminus_neon_rr2_p5_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon_rr2_p5.log2e = 0x1.715476p+0f;
  params->neon_rr2_p5.magic_bias = 0x1.8000FEp23f;
  params->neon_rr2_p5.minus_ln2_hi = -0x1.62E400p-1f;
  params->neon_rr2_p5.minus_ln2_lo = -0x1.7F7D1Cp-20f;
  params->neon_rr2_p5.c5 = 0x1.0F9F9Cp-7f;
  params->neon_rr2_p5.c4 = 0x1.573A1Ap-5f;
  params->neon_rr2_p5.c3 = 0x1.555A80p-3f;
  params->neon_rr2_p5.c2 = 0x1.FFFDC6p-2f;
  params->neon_rr2_p5.c1 = 0x1.FFFFF6p-1f;
  params->neon_rr2_p5.denorm_cutoff = -0x1.5D589Ep6f;
}

void xnn_init_f32_expminus_neon_rr2_lut64_p2_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon_rr2_lut64_p2.log2e = 0x1.715476p+0f;
  params->neon_rr2_lut64_p2.magic_bias = 0x1.800000p17f;
  params->neon_rr2_lut64_p2.minus_ln2_hi = -0x1.62E400p-1f;
  params->neon_rr2_lut64_p2.minus_ln2_lo = -0x1.7F7D1Cp-20f;
  params->neon_rr2_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->neon_rr2_lut64_p2.denorm_cutoff = -0x1.5D589Ep6f;
}

void xnn_init_f32_expminus_neonfma_rr1_p5_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neonfma_rr1_p5.log2e = 0x1.715476p+0f;
  params->neonfma_rr1_p5.magic_bias = 0x1.8000FEp23f;
  params->neonfma_rr1_p5.minus_ln2 = -0x1.62E430p-1f;
  params->neonfma_rr1_p5.c5 = 0x1.0F9F9Cp-7f;
  params->neonfma_rr1_p5.c4 = 0x1.573A1Ap-5f;
  params->neonfma_rr1_p5.c3 = 0x1.555A80p-3f;
  params->neonfma_rr1_p5.c2 = 0x1.FFFDC6p-2f;
  params->neonfma_rr1_p5.c1 = 0x1.FFFFF6p-1f;
  params->neonfma_rr1_p5.denorm_cutoff = -0x1.5D589Ep6f;
}

void xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neonfma_rr1_lut64_p2.log2e = 0x1.715476p+0f;
  params->neonfma_rr1_lut64_p2.magic_bias = 0x1.800000p17f;
  params->neonfma_rr1_lut64_p2.minus_ln2 = -0x1.62E430p-1f;
  params->neonfma_rr1_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->neonfma_rr1_lut64_p2.denorm_cutoff = -0x1.5D589Ep6f;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_expminus_sse2_rr2_p5_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2_rr2_p5.log2e[i] = 0x1.715476p+0f;
    params->sse2_rr2_p5.magic_bias[i] = 0x1.8000FEp23f;
    params->sse2_rr2_p5.minus_ln2_hi[i] = -0x1.62E400p-1f;
    params->sse2_rr2_p5.minus_ln2_lo[i] = -0x1.7F7D1Cp-20f;
    params->sse2_rr2_p5.c5[i] = 0x1.0F9F9Cp-7f;
    params->sse2_rr2_p5.c4[i] = 0x1.573A1Ap-5f;
    params->sse2_rr2_p5.c3[i] = 0x1.555A80p-3f;
    params->sse2_rr2_p5.c2[i] = 0x1.FFFDC6p-2f;
    params->sse2_rr2_p5.c1[i] = 0x1.FFFFF6p-1f;
    params->sse2_rr2_p5.denorm_cutoff[i] = -0x1.5D589Ep6f;
  }
}

void xnn_init_f32_expminus_avx2_rr1_p5_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_p5.log2e[i] = 0x1.715476p+0f;
    params->avx2_rr1_p5.magic_bias[i] = 0x1.8000FEp23f;
    params->avx2_rr1_p5.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx2_rr1_p5.c5[i] = 0x1.0F9F9Cp-7f;
    params->avx2_rr1_p5.c4[i] = 0x1.573A1Ap-5f;
    params->avx2_rr1_p5.c3[i] = 0x1.555A80p-3f;
    params->avx2_rr1_p5.c2[i] = 0x1.FFFDC6p-2f;
    params->avx2_rr1_p5.c1[i] = 0x1.FFFFF6p-1f;
    params->avx2_rr1_p5.denorm_cutoff[i] = -0x1.5D589Ep6f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx2_rr1_p5.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx2_rr1_p5.mask_table[i] = 0;
  }
}

void xnn_init_f32_expminus_avx512_rr1_p5_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512_rr1_p5.log2e = 0x1.715476p+0f;
  params->avx512_rr1_p5.minus_ln2 = -0x1.62E430p-1f;
  params->avx512_rr1_p5.c5 = 0x1.0F9F9Cp-7f;
  params->avx512_rr1_p5.c4 = 0x1.573A1Ap-5f;
  params->avx512_rr1_p5.c3 = 0x1.555A80p-3f;
  params->avx512_rr1_p5.c2 = 0x1.FFFDC6p-2f;
  params->avx512_rr1_p5.c1 = 0x1.FFFFF6p-1f;
  params->avx512_rr1_p5.c0 = 1.0f;
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_f32_expminus_wasmsimd_rr2_p5_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_rr2_p5.log2e[i] = 0x1.715476p+0f;
    params->wasmsimd_rr2_p5.magic_bias[i] = 0x1.8000FEp23f;
    params->wasmsimd_rr2_p5.minus_ln2_hi[i] = -0x1.62E400p-1f;
    params->wasmsimd_rr2_p5.minus_ln2_lo[i] = -0x1.7F7D1Cp-20f;
    params->wasmsimd_rr2_p5.c5[i] = 0x1.0F9F9Cp-7f;
    params->wasmsimd_rr2_p5.c4[i] = 0x1.573A1Ap-5f;
    params->wasmsimd_rr2_p5.c3[i] = 0x1.555A80p-3f;
    params->wasmsimd_rr2_p5.c2[i] = 0x1.FFFDC6p-2f;
    params->wasmsimd_rr2_p5.c1[i] = 0x1.FFFFF6p-1f;
    params->wasmsimd_rr2_p5.denorm_cutoff[i] = -0x1.5D589Ep6f;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_f16_lrelu_neon_params(
  union xnn_f16_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t slope)
{
  params->neon.slope = slope;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f16_lrelu_avx_params(
  union xnn_f16_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t slope)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.slope[i] = fp16_ieee_to_fp32_value(slope);
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

void xnn_init_f32_lrelu_scalar_params(
  union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope)
{
  params->scalar.slope = slope;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_lrelu_sse_params(
  union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.slope[i] = slope;
  }
}

void xnn_init_f32_lrelu_avx_params(
  union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.slope[i] = slope;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_f32_lrelu_wasmsimd_params(
  union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope)
{
  params->wasmsimd.slope[0] = slope;
  params->wasmsimd.slope[1] = slope;
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_f32_sqrt_avx_params(
  union xnn_f32_sqrt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
}

void xnn_init_f32_sqrt_fma_params(
  union xnn_f32_sqrt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->fma.half[i] = 0.5f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->fma.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->fma.mask_table[i] = 0;
  }
}

void xnn_init_f32_sqrt_avx512_params(
  union xnn_f32_sqrt_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512.half = 0.5f;
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

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

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_s8_minmax_sse2_params(
  union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max)
{
  assert(output_min < output_max);

  const uint8_t output_min_with_bias = UINT8_C(0x80) ^ (uint8_t) output_min;
  const uint8_t output_max_with_bias = UINT8_C(0x80) ^ (uint8_t) output_max;
  for (uint32_t i = 0; i < 16; i++) {
    params->sse2.bias[i] = UINT8_C(0x80);
    params->sse2.min_with_bias[i] = output_min_with_bias;
    params->sse2.max_with_bias[i] = output_max_with_bias;
  }
}

void xnn_init_s8_minmax_sse4_params(
  union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max)
{
  assert(output_min < output_max);

  for (uint32_t i = 0; i < 16; i++) {
    params->sse4.min[i] = output_min;
    params->sse4.max[i] = output_max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_s8_minmax_neon_params(
  union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max)
{
  assert(output_min < output_max);

  params->neon.min = output_min;
  params->neon.max = output_max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_s8_minmax_wasmsimd_params(
  union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max)
{
  assert(output_min < output_max);

  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd.min[i] = output_min;
    params->wasmsimd.max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

void xnn_init_s8_minmax_scalar_params(
  union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max)
{
  assert(output_min < output_max);

  params->scalar.min = (int32_t) output_min;
  params->scalar.max = (int32_t) output_max;
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
  #elif XNN_ARCH_ARM || XNN_ARCH_ARM64
    params->neon.min = output_min;
    params->neon.max = output_max;
  #else
    params->scalar.min = (uint32_t) output_min;
    params->scalar.max = (uint32_t) output_max;
  #endif
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_u8_minmax_sse2_params(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  for (uint32_t i = 0; i < 16; i++) {
    params->sse2.min[i] = output_min;
    params->sse2.max[i] = output_max;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_u8_minmax_wasmsimd_params(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd.min[i] = output_min;
    params->wasmsimd.max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_init_u8_minmax_neon_params(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  params->neon.min = output_min;
  params->neon.max = output_max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

void xnn_init_u8_minmax_scalar_params(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  params->scalar.min = (uint32_t) output_min;
  params->scalar.max = (uint32_t) output_max;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qu8_add_minmax_sse2_params(
  union xnn_qu8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
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
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse2.output_min[i] = output_min;
    params->sse2.output_max[i] = output_max;
  }
}

void xnn_init_qu8_add_minmax_sse4_params(
  union xnn_qu8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) (uint32_t) a_zero_point - b_multiplier * (int32_t) (uint32_t) b_zero_point;
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4.bias[i] = bias;
    params->sse4.a_multiplier[i] = a_multiplier;
    params->sse4.b_multiplier[i] = b_multiplier;
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
  union xnn_qu8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) (uint32_t) a_zero_point - b_multiplier * (int32_t) (uint32_t) b_zero_point;
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2.bias[i] = bias;
    params->avx2.a_multiplier[i] = a_multiplier;
    params->avx2.b_multiplier[i] = b_multiplier;
    params->avx2.shift[i] = shift;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
    params->avx2.output_min[i] = output_min;
    params->avx2.output_max[i] = output_max;
  }
}

void xnn_init_qu8_add_minmax_avx512_params(
  union xnn_qu8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) (uint32_t) a_zero_point - b_multiplier * (int32_t) (uint32_t) b_zero_point;
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.bias[i] = bias;
    params->avx512.a_multiplier[i] = a_multiplier;
    params->avx512.b_multiplier[i] = b_multiplier;
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
  union xnn_qu8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

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

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_qu8_add_minmax_wasmsimd_params(
  union xnn_qu8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) (uint32_t) a_zero_point - b_multiplier * (int32_t) (uint32_t) b_zero_point;
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd.bias[i] = bias;
    params->wasmsimd.a_multiplier[i] = a_multiplier;
    params->wasmsimd.b_multiplier[i] = b_multiplier;
  }
  params->wasmsimd.shift = shift;
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd.output_min[i] = output_min;
    params->wasmsimd.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

void xnn_init_qu8_add_minmax_scalar_params(
  union xnn_qu8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t a_zero_point,
  uint8_t b_zero_point,
  uint8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  uint8_t output_min,
  uint8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  params->scalar.bias = rounding - a_multiplier * (int32_t) (uint32_t) a_zero_point - b_multiplier * (int32_t) (uint32_t) b_zero_point;
  params->scalar.a_multiplier = a_multiplier;
  params->scalar.b_multiplier = b_multiplier;
  params->scalar.shift = shift;
  params->scalar.output_min_less_zero_point = (int32_t) (uint32_t) output_min - (int32_t) (uint32_t) output_zero_point;
  params->scalar.output_max_less_zero_point = (int32_t) (uint32_t) output_max - (int32_t) (uint32_t) output_zero_point;
  params->scalar.output_zero_point = (int32_t) (uint32_t) output_zero_point;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_init_qs8_add_minmax_sse2_params(
  union xnn_qs8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
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
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.output_zero_point[i] = (int16_t) output_zero_point;
    params->sse2.output_min[i] = (int16_t) output_min;
    params->sse2.output_max[i] = (int16_t) output_max;
  }
}

void xnn_init_qs8_add_minmax_sse4_mul16_params(
  union xnn_qs8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
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
  for (uint32_t i = 0; i < 8; i++) {
    params->sse4_mul16.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse4_mul16.output_min[i] = output_min;
    params->sse4_mul16.output_max[i] = output_max;
  }
}

void xnn_init_qs8_add_minmax_sse4_mul32_params(
  union xnn_qs8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4_mul32.bias[i] = bias;
    params->sse4_mul32.a_multiplier[i] = a_multiplier;
    params->sse4_mul32.b_multiplier[i] = b_multiplier;
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
  union xnn_qs8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2.bias[i] = bias;
    params->avx2.a_multiplier[i] = a_multiplier;
    params->avx2.b_multiplier[i] = b_multiplier;
    params->avx2.shift[i] = shift;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.output_zero_point[i] = (int16_t) output_zero_point;
    params->avx2.output_min[i] = output_min;
    params->avx2.output_max[i] = output_max;
  }
}

void xnn_init_qs8_add_minmax_avx512_params(
  union xnn_qs8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.bias[i] = bias;
    params->avx512.a_multiplier[i] = a_multiplier;
    params->avx512.b_multiplier[i] = b_multiplier;
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
  union xnn_qs8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

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

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_qs8_add_minmax_wasmsimd_params(
  union xnn_qs8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  const int32_t bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd.bias[i] = bias;
    params->wasmsimd.a_multiplier[i] = a_multiplier;
    params->wasmsimd.b_multiplier[i] = b_multiplier;
  }
  params->wasmsimd.shift = shift;
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd.output_min[i] = output_min;
    params->wasmsimd.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

void xnn_init_qs8_add_minmax_scalar_params(
  union xnn_qs8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t a_zero_point,
  int8_t b_zero_point,
  int8_t output_zero_point,
  float a_output_scale,
  float b_output_scale,
  int8_t output_min,
  int8_t output_max)
{
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
  const uint32_t max_scale_bits = fp32_to_bits(max_abs_output_scale);
  const int32_t max_scale_exponent = (int32_t) (max_scale_bits >> 23) - 127;

  // Shift is in [12, 30] range.
  const uint32_t shift = (uint32_t) (20 /* multiplier bits */ - max_scale_exponent);
  assert(shift <= 30);
  assert(shift >= 12);

  // Multipliers are in [0, 2**21) range, largest multiplier is in [2**20, 2**21) range.
  const int32_t abs_a_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_a_output_scale) + (shift << 23)));
  const int32_t abs_b_multiplier = (int32_t) lrintf(fp32_from_bits(fp32_to_bits(abs_b_output_scale) + (shift << 23)));
  assert(math_max_s32(abs_a_multiplier, abs_b_multiplier) >= INT32_C(0x00100000));
  assert(abs_a_multiplier <= INT32_C(0x00200000));
  assert(abs_b_multiplier <= INT32_C(0x00200000));

  const int32_t a_multiplier = signbit(a_output_scale) ? -abs_a_multiplier : abs_a_multiplier;
  const int32_t b_multiplier = signbit(b_output_scale) ? -abs_b_multiplier : abs_b_multiplier;

  const int32_t rounding = INT32_C(1) << (shift - 1);
  params->scalar.bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
  params->scalar.a_multiplier = a_multiplier;
  params->scalar.b_multiplier = b_multiplier;
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
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

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
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  params->fp32_neon.a_zero_point[0] = a_zero_point;
  params->fp32_neon.a_zero_point[1] = a_zero_point;
  params->fp32_neon.b_zero_point[0] = b_zero_point;
  params->fp32_neon.b_zero_point[1] = b_zero_point;
  params->fp32_neon.scale = product_output_scale;
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_neon.output_min = output_min;
  params->fp32_neon.output_max = output_max;
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
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  params->fp32_neonv8.a_zero_point[0] = a_zero_point;
  params->fp32_neonv8.a_zero_point[1] = a_zero_point;
  params->fp32_neonv8.b_zero_point[0] = b_zero_point;
  params->fp32_neonv8.b_zero_point[1] = b_zero_point;
  params->fp32_neonv8.scale = product_output_scale;
  params->fp32_neonv8.output_zero_point = (int16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
}

void xnn_init_qu8_mul_minmax_rndnu_neon_params(
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

  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(product_output_scale);

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

  params->rndnu_neon.a_zero_point[0] = a_zero_point;
  params->rndnu_neon.a_zero_point[1] = a_zero_point;
  params->rndnu_neon.b_zero_point[0] = b_zero_point;
  params->rndnu_neon.b_zero_point[1] = b_zero_point;
  params->rndnu_neon.left_pre_shift = -pre_shift;
  params->rndnu_neon.multiplier = multiplier;
  params->rndnu_neon.left_post_shift = -post_shift;
  params->rndnu_neon.output_zero_point = (int16_t) output_zero_point;
  params->rndnu_neon.output_min = output_min;
  params->rndnu_neon.output_max = output_max;
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

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_qu8_mul_minmax_fp32_wasmsimd_params(
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

  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  const int32_t magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_wasmsimd.a_zero_point[i] = (int16_t) a_zero_point;
    params->fp32_wasmsimd.b_zero_point[i] = (int16_t) b_zero_point;
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->fp32_wasmsimd.scale[i] = product_output_scale;
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_min[i] = magic_min;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = magic_bias_less_output_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_wasmsimd.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

void xnn_init_qs8_mul_minmax_fp32_scalar_params(
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
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  params->fp32_neon.a_zero_point[0] = a_zero_point;
  params->fp32_neon.a_zero_point[1] = a_zero_point;
  params->fp32_neon.b_zero_point[0] = b_zero_point;
  params->fp32_neon.b_zero_point[1] = b_zero_point;
  params->fp32_neon.scale = product_output_scale;
  params->fp32_neon.magic_bias = 12582912.0f;
  params->fp32_neon.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->fp32_neon.output_min = output_min;
  params->fp32_neon.output_max = output_max;
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
  assert(product_output_scale >= 0x1.0p-16f);
  assert(product_output_scale < 0x1.0p+8f);

  params->fp32_neonv8.a_zero_point[0] = a_zero_point;
  params->fp32_neonv8.a_zero_point[1] = a_zero_point;
  params->fp32_neonv8.b_zero_point[0] = b_zero_point;
  params->fp32_neonv8.b_zero_point[1] = b_zero_point;
  params->fp32_neonv8.scale = product_output_scale;
  params->fp32_neonv8.output_zero_point = (int16_t) output_zero_point;
  params->fp32_neonv8.output_min = output_min;
  params->fp32_neonv8.output_max = output_max;
}

void xnn_init_qs8_mul_minmax_rndnu_neon_params(
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

  // Compute requantization parameters.
  const uint32_t scale_bits = fp32_to_bits(product_output_scale);

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

  params->rndnu_neon.a_zero_point[0] = a_zero_point;
  params->rndnu_neon.a_zero_point[1] = a_zero_point;
  params->rndnu_neon.b_zero_point[0] = b_zero_point;
  params->rndnu_neon.b_zero_point[1] = b_zero_point;
  params->rndnu_neon.left_pre_shift = -pre_shift;
  params->rndnu_neon.multiplier = multiplier;
  params->rndnu_neon.left_post_shift = -post_shift;
  params->rndnu_neon.output_zero_point = (int16_t) output_zero_point;
  params->rndnu_neon.output_min = output_min;
  params->rndnu_neon.output_max = output_max;
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

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_init_qs8_mul_minmax_fp32_wasmsimd_params(
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

  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  const int32_t magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_wasmsimd.a_zero_point[i] = (int16_t) a_zero_point;
    params->fp32_wasmsimd.b_zero_point[i] = (int16_t) b_zero_point;
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->fp32_wasmsimd.scale[i] = product_output_scale;
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_min[i] = magic_min;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = magic_bias_less_output_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_wasmsimd.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_INTERNAL void xnn_init_f16_f32_cvt_scalar_params(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar.sign_mask = UINT32_C(0x80000000);
  params->scalar.exp_offset = UINT32_C(0x70000000);
  params->scalar.exp_scale = 0x1.0p-112f;
  params->scalar.magic_mask = UINT32_C(0x3F000000);
  params->scalar.magic_bias = 0.5f;
  params->scalar.denorm_cutoff = UINT32_C(0x08000000);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_INTERNAL void xnn_init_f16_f32_cvt_neon_params(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon.exp_scale = 0x1.0p-112f;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_INTERNAL void xnn_init_f16_f32_cvt_sse_int16_params(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->sse_int16.sign_mask[i] = UINT16_C(0x8000);
    params->sse_int16.exp_offset[i] = UINT16_C(0x7000);
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->sse_int16.exp_scale[i] = 0x1.0p-112f;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse_int16.magic_mask[i] = UINT16_C(0x3F00);
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->sse_int16.magic_bias[i] = 0.5f;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse_int16.denorm_cutoff[i] = INT16_C(0x0400);
  }
}

XNN_INTERNAL void xnn_init_f16_f32_cvt_sse_int32_params(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse_int32.sign_mask[i] = UINT32_C(0x80000000);
    params->sse_int32.exp_offset[i] = UINT32_C(0x70000000);
    params->sse_int32.exp_scale[i] = 0x1.0p-112f;
    params->sse_int32.magic_bias[i] = UINT32_C(0x3F000000);
    params->sse_int32.denorm_cutoff[i] = INT32_C(0x04000000);
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_INTERNAL void xnn_init_f16_f32_cvt_wasmsimd_int16_params(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd_int16.sign_mask[i] = UINT16_C(0x8000);
    params->wasmsimd_int16.exp_offset[i] = UINT16_C(0x7000);
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_int16.exp_scale[i] = 0x1.0p-112f;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd_int16.magic_mask[i] = UINT16_C(0x3F00);
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_int16.magic_bias[i] = 0.5f;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd_int16.denorm_cutoff[i] = INT16_C(0x0400);
  }
}

XNN_INTERNAL void xnn_init_f16_f32_cvt_wasmsimd_int32_params(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_int32.sign_mask[i] = UINT32_C(0x80000000);
    params->wasmsimd_int32.exp_offset[i] = UINT32_C(0x70000000);
    params->wasmsimd_int32.exp_scale[i] = 0x1.0p-112f;
    params->wasmsimd_int32.magic_bias[i] = UINT32_C(0x3F000000);
    params->wasmsimd_int32.denorm_cutoff[i] = INT32_C(0x04000000);
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_INTERNAL void xnn_init_f32_f16_cvt_scalar_bitcast_params(
  union xnn_f32_f16_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_bitcast.nonsign_mask = UINT32_C(0x7FFFFFFF);
  params->scalar_bitcast.exp_bias = UINT32_C(0x07800000);
  params->scalar_bitcast.scale_to_inf = 0x1.0p+112f;
  params->scalar_bitcast.expw_max = UINT32_C(0x7F800000);
  params->scalar_bitcast.scale_to_zero = 0x1.0p-110f;
  params->scalar_bitcast.bias_min = UINT32_C(0x40000000);
  params->scalar_bitcast.exph_mask = UINT16_C(0x7C00);
  params->scalar_bitcast.manth_mask = UINT16_C(0x0FFF);
  params->scalar_bitcast.nanh = UINT16_C(0x7E00);
}

XNN_INTERNAL void xnn_init_f32_f16_cvt_scalar_fabsf_params(
  union xnn_f32_f16_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_fabsf.scale_to_inf = 0x1.0p+112f;
  params->scalar_fabsf.exp_bias = UINT32_C(0x07800000);
  params->scalar_fabsf.scale_to_zero = 0x1.0p-110f;
  params->scalar_fabsf.expw_max = UINT32_C(0x7F800000);
  params->scalar_fabsf.bias_min = UINT32_C(0x40000000);
  params->scalar_fabsf.exph_mask = UINT16_C(0x7C00);
  params->scalar_fabsf.manth_mask = UINT16_C(0x0FFF);
  params->scalar_fabsf.nanh = UINT16_C(0x7E00);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_INTERNAL void xnn_init_f32_f16_cvt_neon_params(
  union xnn_f32_f16_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon.exp_bias = UINT32_C(0x07800000);
  params->neon.scale_to_inf = 0x1.0p+112f;
  params->neon.expw_max = UINT32_C(0x7F800000);
  params->neon.scale_to_zero = 0x1.0p-110f;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_INTERNAL void xnn_init_f32_f16_cvt_sse2_params(
  union xnn_f32_f16_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.nonsign_mask[i] = UINT32_C(0x7FFFFFFF);
    params->sse2.exp_bias[i] = UINT32_C(0x07800000);
    params->sse2.scale_to_inf[i] = 0x1.0p+112f;
    params->sse2.expw_max[i] = UINT32_C(0x7F800000);
    params->sse2.scale_to_zero[i] = 0x1.0p-110f;
  }
  params->sse2.bias_min[0] = INT16_C(0x8000);
  params->sse2.bias_min[1] = INT16_C(0x4000);
  params->sse2.bias_min[2] = INT16_C(0x8000);
  params->sse2.bias_min[3] = INT16_C(0x4000);
  params->sse2.bias_min[4] = INT16_C(0x8000);
  params->sse2.bias_min[5] = INT16_C(0x4000);
  params->sse2.bias_min[6] = INT16_C(0x8000);
  params->sse2.bias_min[7] = INT16_C(0x4000);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.manth_mask[i] = UINT32_C(0x00000FFF);
    params->sse2.exph_mask[i] = UINT32_C(0x00007C00);
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.nanh[i] = UINT16_C(0x7E00);
  }
}

XNN_INTERNAL void xnn_init_f32_f16_cvt_f16c_params(
  union xnn_f32_f16_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 7; i++) {
    params->f16c.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->f16c.mask_table[i] = 0;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_INTERNAL void xnn_init_f32_f16_cvt_wasmsimd_params(
  union xnn_f32_f16_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd.exp_bias[i] = UINT32_C(0x07800000);
    params->wasmsimd.scale_to_inf[i] = 0x1.0p+112f;
    params->wasmsimd.expw_max[i] = UINT32_C(0x7F800000);
    params->wasmsimd.scale_to_zero[i] = 0x1.0p-110f;
  }
  params->wasmsimd.bias_min[0] = INT16_C(0x8000);
  params->wasmsimd.bias_min[1] = INT16_C(0x4000);
  params->wasmsimd.bias_min[2] = INT16_C(0x8000);
  params->wasmsimd.bias_min[3] = INT16_C(0x4000);
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd.manth_mask[i] = UINT32_C(0x00000FFF);
    params->wasmsimd.exph_mask[i] = UINT32_C(0x00007C00);
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.nanh[i] = UINT16_C(0x7E00);
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_INTERNAL void xnn_init_f32_qs8_cvt_scalar_fmagic_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->scalar_fmagic.scale = scale;
  params->scalar_fmagic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->scalar_fmagic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_fmagic.magic_bias = 12582912.0f;
  params->scalar_fmagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

XNN_INTERNAL void xnn_init_f32_qs8_cvt_scalar_imagic_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_imagic.scale = scale;
  params->scalar_imagic.magic_bias = 12582912.0f;
  params->scalar_imagic.magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  params->scalar_imagic.magic_max = (int32_t) fp32_to_bits(12582912.0f + output_max_less_zero_point);
  params->scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

XNN_INTERNAL void xnn_init_f32_qs8_cvt_scalar_lrintf_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->scalar_lrintf.scale = scale;
  params->scalar_lrintf.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->scalar_lrintf.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_lrintf.output_zero_point = (int32_t) output_zero_point;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_INTERNAL void xnn_init_f32_qs8_cvt_neon_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->neon.scale = scale;
  params->neon.magic_bias = 12582912.0f;
  params->neon.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->neon.output_min = output_min;
  params->neon.output_max = output_max;
}

XNN_INTERNAL void xnn_init_f32_qs8_cvt_neonv8_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->neonv8.scale = scale;
  params->neonv8.output_zero_point = (int16_t) output_zero_point;
  params->neonv8.output_min = output_min;
  params->neonv8.output_max = output_max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_INTERNAL void xnn_init_f32_qs8_cvt_sse2_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.scale[i] = scale;
    params->sse2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.output_zero_point[i] = (int16_t) output_zero_point;
    params->sse2.output_min[i] = (int16_t) output_min;
  }
}

XNN_INTERNAL void xnn_init_f32_qs8_cvt_sse4_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4.scale[i] = scale;
    params->sse4.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse4.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse4.output_min[i] = output_min;
  }
}

XNN_INTERNAL void xnn_init_f32_qs8_cvt_avx_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.scale[i] = scale;
    params->avx.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx.output_min[i] = output_min;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
}

XNN_INTERNAL void xnn_init_f32_qs8_cvt_avx2_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2.scale[i] = scale;
    params->avx2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  params->avx2.shuffle_mask[0] = 0;
  params->avx2.shuffle_mask[1] = 4;
  params->avx2.shuffle_mask[2] = 1;
  params->avx2.shuffle_mask[3] = 5;
  params->avx2.shuffle_mask[4] = 2;
  params->avx2.shuffle_mask[5] = 6;
  params->avx2.shuffle_mask[6] = 3;
  params->avx2.shuffle_mask[7] = 7;
  for (uint32_t i = 0; i < 32; i++) {
    params->avx2.output_min[i] = output_min;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx2.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx2.mask_table[i] = 0;
  }
}

XNN_INTERNAL void xnn_init_f32_qs8_cvt_avx512_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.scale[i] = scale;
    params->avx512.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->avx512.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 64; i++) {
    params->avx512.output_min[i] = output_min;
  }
  params->avx512.shuffle512_mask[0] = 0;
  params->avx512.shuffle512_mask[1] = 4;
  params->avx512.shuffle512_mask[2] = 8;
  params->avx512.shuffle512_mask[3] = 12;
  params->avx512.shuffle512_mask[4] = 1;
  params->avx512.shuffle512_mask[5] = 5;
  params->avx512.shuffle512_mask[6] = 9;
  params->avx512.shuffle512_mask[7] = 13;
  params->avx512.shuffle512_mask[8] = 2;
  params->avx512.shuffle512_mask[9] = 6;
  params->avx512.shuffle512_mask[10] = 10;
  params->avx512.shuffle512_mask[11] = 14;
  params->avx512.shuffle512_mask[12] = 3;
  params->avx512.shuffle512_mask[13] = 7;
  params->avx512.shuffle512_mask[14] = 11;
  params->avx512.shuffle512_mask[15] = 15;
  params->avx512.shuffle256_mask[0] = 0;
  params->avx512.shuffle256_mask[1] = 4;
  params->avx512.shuffle256_mask[2] = 2;
  params->avx512.shuffle256_mask[3] = 6;
  params->avx512.shuffle256_mask[4] = 1;
  params->avx512.shuffle256_mask[5] = 5;
  params->avx512.shuffle256_mask[6] = 3;
  params->avx512.shuffle256_mask[7] = 7;
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_INTERNAL void xnn_init_f32_qs8_cvt_wasmsimd_cvt_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_cvt.scale[i] = scale;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd_cvt.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd_cvt.output_min[i] = output_min;
    params->wasmsimd_cvt.output_max[i] = output_max;
  }
}

XNN_INTERNAL void xnn_init_f32_qs8_cvt_wasmsimd_magic_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  const int32_t magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_magic.scale[i] = scale;
    params->wasmsimd_magic.magic_bias[i] = 12582912.0f;
    params->wasmsimd_magic.magic_min[i] = magic_min;
    params->wasmsimd_magic.magic_bias_less_zero_point[i] = magic_bias_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd_magic.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_INTERNAL void xnn_init_f32_qu8_cvt_scalar_fmagic_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  params->scalar_fmagic.scale = scale;
  params->scalar_fmagic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->scalar_fmagic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_fmagic.magic_bias = 12582912.0f;
  params->scalar_fmagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

XNN_INTERNAL void xnn_init_f32_qu8_cvt_scalar_imagic_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_imagic.scale = scale;
  params->scalar_imagic.magic_bias = 12582912.0f;
  params->scalar_imagic.magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  params->scalar_imagic.magic_max = (int32_t) fp32_to_bits(12582912.0f + output_max_less_zero_point);
  params->scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
}

XNN_INTERNAL void xnn_init_f32_qu8_cvt_scalar_lrintf_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  params->scalar_lrintf.scale = scale;
  params->scalar_lrintf.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->scalar_lrintf.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_lrintf.output_zero_point = (int32_t) output_zero_point;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_INTERNAL void xnn_init_f32_qu8_cvt_neon_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  params->neon.scale = scale;
  params->neon.magic_bias = 12582912.0f;
  params->neon.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  params->neon.output_min = output_min;
  params->neon.output_max = output_max;
}

XNN_INTERNAL void xnn_init_f32_qu8_cvt_neonv8_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  params->neonv8.scale = scale;
  params->neonv8.output_zero_point = (int16_t) output_zero_point;
  params->neonv8.output_min = output_min;
  params->neonv8.output_max = output_max;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_INTERNAL void xnn_init_f32_qu8_cvt_sse2_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.scale[i] = scale;
    params->sse2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse2.output_min[i] = output_min;
  }
}

XNN_INTERNAL void xnn_init_f32_qu8_cvt_avx_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.scale[i] = scale;
    params->avx.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx.output_min[i] = output_min;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
}

XNN_INTERNAL void xnn_init_f32_qu8_cvt_avx2_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2.scale[i] = scale;
    params->avx2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  params->avx2.shuffle_mask[0] = 0;
  params->avx2.shuffle_mask[1] = 4;
  params->avx2.shuffle_mask[2] = 1;
  params->avx2.shuffle_mask[3] = 5;
  params->avx2.shuffle_mask[4] = 2;
  params->avx2.shuffle_mask[5] = 6;
  params->avx2.shuffle_mask[6] = 3;
  params->avx2.shuffle_mask[7] = 7;
  for (uint32_t i = 0; i < 32; i++) {
    params->avx2.output_min[i] = output_min;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx2.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx2.mask_table[i] = 0;
  }
}

XNN_INTERNAL void xnn_init_f32_qu8_cvt_avx512_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.scale[i] = scale;
    params->avx512.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->avx512.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 64; i++) {
    params->avx512.output_min[i] = output_min;
  }
  params->avx512.shuffle512_mask[0] = 0;
  params->avx512.shuffle512_mask[1] = 4;
  params->avx512.shuffle512_mask[2] = 8;
  params->avx512.shuffle512_mask[3] = 12;
  params->avx512.shuffle512_mask[4] = 1;
  params->avx512.shuffle512_mask[5] = 5;
  params->avx512.shuffle512_mask[6] = 9;
  params->avx512.shuffle512_mask[7] = 13;
  params->avx512.shuffle512_mask[8] = 2;
  params->avx512.shuffle512_mask[9] = 6;
  params->avx512.shuffle512_mask[10] = 10;
  params->avx512.shuffle512_mask[11] = 14;
  params->avx512.shuffle512_mask[12] = 3;
  params->avx512.shuffle512_mask[13] = 7;
  params->avx512.shuffle512_mask[14] = 11;
  params->avx512.shuffle512_mask[15] = 15;
  params->avx512.shuffle256_mask[0] = 0;
  params->avx512.shuffle256_mask[1] = 4;
  params->avx512.shuffle256_mask[2] = 2;
  params->avx512.shuffle256_mask[3] = 6;
  params->avx512.shuffle256_mask[4] = 1;
  params->avx512.shuffle256_mask[5] = 5;
  params->avx512.shuffle256_mask[6] = 3;
  params->avx512.shuffle256_mask[7] = 7;
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_INTERNAL void xnn_init_f32_qu8_cvt_wasmsimd_cvt_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_cvt.scale[i] = scale;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd_cvt.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd_cvt.output_min[i] = output_min;
    params->wasmsimd_cvt.output_max[i] = output_max;
  }
}

XNN_INTERNAL void xnn_init_f32_qu8_cvt_wasmsimd_magic_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) fp32_to_bits(12582912.0f + output_min_less_zero_point);
  const int32_t magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_magic.scale[i] = scale;
    params->wasmsimd_magic.magic_bias[i] = 12582912.0f;
    params->wasmsimd_magic.magic_min[i] = magic_min;
    params->wasmsimd_magic.magic_bias_less_zero_point[i] = magic_bias_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd_magic.output_max[i] = output_max;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_INTERNAL void xnn_init_qs8_f32_cvt_scalar_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  params->scalar.zero_point = (int32_t) zero_point;
  params->scalar.scale = scale;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_INTERNAL void xnn_init_qs8_f32_cvt_neon_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  params->neon.minus_zero_point[0] = -(int16_t) zero_point;
  params->neon.minus_zero_point[1] = -(int16_t) zero_point;
  params->neon.scale = scale;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_INTERNAL void xnn_init_qs8_f32_cvt_sse2_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  for (uint32_t i = 0; i < 16; i++) {
    params->sse2.sign_mask[i] = UINT8_C(0x80);
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.magic_exp[i] = UINT16_C(0x4B00);
  }
  const float magic_bias = (float) (INT32_C(0x00800080) + (int32_t) zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.magic_bias[i] = magic_bias;
    params->sse2.scale[i] = scale;
  }
}

XNN_INTERNAL void xnn_init_qs8_f32_cvt_sse4_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4.minus_zero_point[i] = -(int32_t) zero_point;
    params->sse4.scale[i] = scale;
  }
}

XNN_INTERNAL void xnn_init_qs8_f32_cvt_avx_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.minus_zero_point[i] = -(int32_t) zero_point;
    params->avx.scale[i] = scale;
  }
}

XNN_INTERNAL void xnn_init_qs8_f32_cvt_avx512_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.minus_zero_point[i] = -(int32_t) zero_point;
    params->avx512.scale[i] = scale;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_INTERNAL void xnn_init_qs8_f32_cvt_wasmsimd_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.minus_zero_point[i] = -(int16_t) zero_point;
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd.scale[i] = scale;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_INTERNAL void xnn_init_qu8_f32_cvt_scalar_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  params->scalar.zero_point = (int32_t) zero_point;
  params->scalar.scale = scale;
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_INTERNAL void xnn_init_qu8_f32_cvt_neon_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  params->neon.minus_zero_point[0] = -(int16_t) zero_point;
  params->neon.minus_zero_point[1] = -(int16_t) zero_point;
  params->neon.scale = scale;
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_INTERNAL void xnn_init_qu8_f32_cvt_sse2_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.magic_exp[i] = UINT16_C(0x4B00);
  }
  const float magic_bias = (float) (INT32_C(0x00800000) + (int32_t) zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.magic_bias[i] = magic_bias;
    params->sse2.scale[i] = scale;
  }
}

XNN_INTERNAL void xnn_init_qu8_f32_cvt_sse4_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4.minus_zero_point[i] = -(int32_t) zero_point;
    params->sse4.scale[i] = scale;
  }
}

XNN_INTERNAL void xnn_init_qu8_f32_cvt_avx_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.minus_zero_point[i] = -(int32_t) zero_point;
    params->avx.scale[i] = scale;
  }
}

XNN_INTERNAL void xnn_init_qu8_f32_cvt_avx512_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.minus_zero_point[i] = -(int32_t) zero_point;
    params->avx512.scale[i] = scale;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_INTERNAL void xnn_init_qu8_f32_cvt_wasmsimd_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.minus_zero_point[i] = -(int16_t) zero_point;
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd.scale[i] = scale;
  }
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
