// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/microparams.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/unaligned.h>

#include <fp16/fp16.h>

size_t xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_scalar_fmagic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_fmagic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_fmagic.magic_bias = 12582912.0f;
  params->fp32_scalar_fmagic.magic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_fmagic);
}

size_t xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_imagic.magic_bias = 12582912.0f;
  params->fp32_scalar_imagic.magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  params->fp32_scalar_imagic.magic_max = (int32_t) float_as_uint32(12582912.0f + output_max_less_zero_point);
  params->fp32_scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_imagic);
}

size_t xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_scalar_lrintf.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->fp32_scalar_lrintf.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_scalar_lrintf.output_zero_point = (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_lrintf);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
    params->fp32_sse2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_sse2.output_zero_point[i] = (int16_t) output_zero_point;
    params->fp32_sse2.output_min[i] = (int16_t) output_min;
  }
  return sizeof(params->fp32_sse2);
}

size_t xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 4; i++) {
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

size_t xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_avx2.output_max_less_zero_point[i] = output_max_less_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx2.output_min[i] = output_min;
  }
  return sizeof(params->fp32_avx2);
}

size_t xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_avx512.output_max_less_zero_point = output_max_less_zero_point;
  params->fp32_avx512.output_zero_point = (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx512.output_min[i] = output_min;
  }
  return sizeof(params->fp32_avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

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

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  const int32_t magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 2; i++) {
    params->fp32_wasmsimd.magic_bias[i] = 12582912.0f;
    params->fp32_wasmsimd.magic_min[i] = magic_min;
    params->fp32_wasmsimd.magic_bias_less_output_zero_point[i] = magic_bias_less_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fp32_wasmsimd.output_max[i] = output_max;
  }
  return sizeof(params->fp32_wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params(
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
  return sizeof(params->fp32_scalar_fmagic);
}

size_t xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params(
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
  params->fp32_scalar_imagic.magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  params->fp32_scalar_imagic.magic_max = (int32_t) float_as_uint32(12582912.0f + output_max_less_zero_point);
  params->fp32_scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_imagic);
}

size_t xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params(
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
  return sizeof(params->fp32_scalar_lrintf);
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
  const int32_t output_min_less_zero_point = (int32_t) output_min - (int32_t) output_zero_point;
  const int32_t output_max_less_zero_point = (int32_t) output_max - (int32_t) output_zero_point;

  params->rndnu_scalar.multiplier = multiplier;
  params->rndnu_scalar.shift = shift;
  params->rndnu_scalar.rounding = rounding;
  params->rndnu_scalar.output_min_less_zero_point = output_min_less_zero_point;
  params->rndnu_scalar.output_max_less_zero_point = output_max_less_zero_point;
  params->rndnu_scalar.output_zero_point = (int32_t) output_zero_point;
  return sizeof(params->rndnu_scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs8_conv_minmax_fp32_sse2_params(
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
  return sizeof(params->fp32_sse2);
}

size_t xnn_init_qs8_conv_minmax_fp32_sse4_params(
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
  return sizeof(params->fp32_sse4);
}

size_t xnn_init_qs8_conv_minmax_fp32_avx2_params(
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
  return sizeof(params->fp32_avx2);
}

size_t xnn_init_qs8_conv_minmax_fp32_avx512_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_avx512.output_max_less_zero_point = output_max_less_zero_point;
  params->fp32_avx512.output_zero_point = (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx512.scale[i] = scale;
    params->fp32_avx512.output_min[i] = output_min;
  }
  return sizeof(params->fp32_avx512);
}

size_t xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_avx512vnni.sign_mask = 0x80;
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_avx512vnni.output_max_less_zero_point = output_max_less_zero_point;
  params->fp32_avx512vnni.output_zero_point = (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx512vnni.output_min[i] = output_min;
  }
  return sizeof(params->fp32_avx512vnni);
}

size_t xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params(
  union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->fp32_avxvnni.sign_mask = 0x80;
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_avxvnni.output_max_less_zero_point = output_max_less_zero_point;
  params->fp32_avxvnni.output_zero_point = (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avxvnni.output_min[i] = output_min;
  }
  return sizeof(params->fp32_avxvnni);
}

size_t xnn_init_qs8_conv_minmax_fp32_avx512vnni_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  assert(scale >= 0x1.0p-32f);
  assert(scale < 256.0f);

  params->fp32_avx512vnni.sign_mask = 0x80;
  params->fp32_avx512vnni.mask = 0xF0;
  params->fp32_avx512vnni.gfni_shl4 = INT64_C(0x01020408);
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->fp32_avx512vnni.output_max_less_zero_point = output_max_less_zero_point;
  params->fp32_avx512vnni.output_zero_point = (int32_t) output_zero_point;
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx512vnni.scale[i] = scale;
    params->fp32_avx512vnni.output_min[i] = output_min;
  }
  return sizeof(params->fp32_avx512vnni);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

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

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qs8_conv_minmax_fp32_wasmsimd_params(
  union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params(
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
  return sizeof(params->fp32_scalar_fmagic);
}

size_t xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params(
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
  params->fp32_scalar_imagic.magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  params->fp32_scalar_imagic.magic_max = (int32_t) float_as_uint32(12582912.0f + output_max_less_zero_point);
  params->fp32_scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->fp32_scalar_imagic);
}

size_t xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params(
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
  return sizeof(params->fp32_scalar_lrintf);
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
  const int32_t output_min_less_zero_point = (int32_t) output_min - (int32_t) output_zero_point;
  const int32_t output_max_less_zero_point = (int32_t) output_max - (int32_t) output_zero_point;

  params->rndnu_scalar.kernel_zero_point = (int32_t) kernel_zero_point;
  params->rndnu_scalar.multiplier = multiplier;
  params->rndnu_scalar.rounding = rounding;
  params->rndnu_scalar.shift = shift;
  params->rndnu_scalar.output_min_less_zero_point = output_min_less_zero_point;
  params->rndnu_scalar.output_max_less_zero_point = output_max_less_zero_point;
  params->rndnu_scalar.output_zero_point = (int32_t) output_zero_point;
  return sizeof(params->rndnu_scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qu8_conv_minmax_fp32_sse2_params(
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
  return sizeof(params->fp32_sse2);
}

size_t xnn_init_qu8_conv_minmax_fp32_avx2_params(
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
  return sizeof(params->fp32_avx2);
}

size_t xnn_init_qu8_conv_minmax_fp32_avx512_params(
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
  params->fp32_avx512.output_max_less_zero_point = output_max_less_zero_point;
  params->fp32_avx512.output_zero_point = (int32_t) (uint32_t) output_zero_point;
  for (uint32_t i = 0; i < 16; i++) {
    params->fp32_avx512.scale[i] = scale;
    params->fp32_avx512.output_min[i] = output_min;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->fp32_avx512.kernel_zero_point[i] = (int16_t) (uint16_t) kernel_zero_point;
  }
  return sizeof(params->fp32_avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

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

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qu8_conv_minmax_fp32_wasmsimd_params(
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
  const int32_t magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
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
  return sizeof(params->fp32_wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

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

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_scale_fp16arith_params(
  union xnn_f16_scale_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale)
{
  params->fp16arith.scale = scale;
  return sizeof(params->fp16arith);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

size_t xnn_init_f16_f32acc_scale_scalar_params(
  union xnn_f16_f32acc_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  params->scalar.scale = scale;
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_f32acc_scale_avx_params(
  union xnn_f16_f32acc_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
  params->avx.scale = scale;
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

size_t xnn_init_f32_scale_scalar_params(
  union xnn_f32_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  params->scalar.scale = scale;
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_scale_avx_params(
  union xnn_f32_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
  params->avx.scale = scale;
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

void xnn_update_f32_scaleminmax_scalar_params(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  params->scalar.scale = scale;
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_update_f32_scaleminmax_sse_params(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.scale[i] = scale;
  }
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_scaleminmax_fp16arith_params(
  union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  uint16_t min,
  uint16_t max)
{
  params->fp16arith.scale = scale;
  params->fp16arith.min = min;
  params->fp16arith.max = max;
  return sizeof(params->fp16arith);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_scaleminmax_avx_params(
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
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_update_f16_scaleminmax_fp16arith_params(
  union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale)
{
  params->fp16arith.scale = scale;
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

size_t xnn_init_f32_scaleminmax_scalar_params(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max)
{
  params->scalar.scale = scale;
  params->scalar.min = min;
  params->scalar.max = max;
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_scaleminmax_sse_params(
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
  return sizeof(params->sse);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

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

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_gavgpool_neonfp16arith_params(
  union xnn_f16_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t multiplier,
  uint16_t output_min,
  uint16_t output_max,
  uint32_t width)
{
  params->neonfp16arith.multiplier = multiplier;
  params->neonfp16arith.output_min = output_min;
  params->neonfp16arith.output_max = output_max;

  const uint32_t w = (width - 1) & 7;
  params->neonfp16arith.mask[0] = UINT16_C(0xFFFF);
  params->neonfp16arith.mask[1] = -(uint16_t) (w >= 1);
  params->neonfp16arith.mask[2] = -(uint16_t) (w >= 2);
  params->neonfp16arith.mask[3] = -(uint16_t) (w >= 3);
  params->neonfp16arith.mask[4] = -(uint16_t) (w >= 4);
  params->neonfp16arith.mask[5] = -(uint16_t) (w >= 5);
  params->neonfp16arith.mask[6] = -(uint16_t) (w >= 6);
  params->neonfp16arith.mask[7] = -(uint16_t) (w >= 7);
  return sizeof(params->neonfp16arith);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

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

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_update_f16_gavgpool_neonfp16arith_params(
  union xnn_f16_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t multiplier,
  uint32_t width)
{
  params->neonfp16arith.multiplier = multiplier;

  const uint32_t w = (width - 1) & 7;
  params->neonfp16arith.mask[0] = UINT16_C(0xFFFF);
  params->neonfp16arith.mask[1] = -(uint16_t) (w >= 1);
  params->neonfp16arith.mask[2] = -(uint16_t) (w >= 2);
  params->neonfp16arith.mask[3] = -(uint16_t) (w >= 3);
  params->neonfp16arith.mask[4] = -(uint16_t) (w >= 4);
  params->neonfp16arith.mask[5] = -(uint16_t) (w >= 5);
  params->neonfp16arith.mask[6] = -(uint16_t) (w >= 6);
  params->neonfp16arith.mask[7] = -(uint16_t) (w >= 7);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

size_t xnn_init_bf16_minmax_scalar_params(
  union xnn_bf16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t output_min,
  uint16_t output_max)
{
  params->scalar.min = uint32_as_float((uint32_t) output_min << 16);
  params->scalar.max = uint32_as_float((uint32_t) output_max << 16);
  return sizeof(params->scalar);
}

size_t xnn_init_f16_minmax_fp16arith_params(
  union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t min,
  uint16_t max)
{
  params->fp16arith.min = min;
  params->fp16arith.max = max;
  return sizeof(params->fp16arith);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_minmax_avx_params(
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
  return sizeof(params->avx);
}

size_t xnn_init_f16_minmax_avxvnni_params(
  union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t min,
  uint16_t max)
{
  const float min_f32 = fp16_ieee_to_fp32_value(min);
  const float max_f32 = fp16_ieee_to_fp32_value(max);
  params->avxvnni.min = min_f32;
  params->avxvnni.max = max_f32;
  params->avxvnni.sign_mask = 0x80;
  return sizeof(params->avxvnni);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_qc4w_minmax_scalar_params(
  union xnn_f16_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t output_min,
  uint16_t output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->fp16arith.min = output_min;
  params->fp16arith.max = output_max;
  params->fp16arith.minus_kernel_zero_point = -(int32_t) kernel_zero_point;
  return sizeof(params->fp16arith);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_qc4w_minmax_avx_params(
  union xnn_f16_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t output_min,
  uint16_t output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  const float min_f32 = fp16_ieee_to_fp32_value(output_min);
  const float max_f32 = fp16_ieee_to_fp32_value(output_max);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.min[i] = min_f32;
    params->avx.max[i] = max_f32;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx.mask[i] = 0xF0;
  }
  return sizeof(params->avx);
}

size_t xnn_init_f16_qc4w_minmax_avxvnni_params(
  union xnn_f16_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t output_min,
  uint16_t output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  const float min_f32 = fp16_ieee_to_fp32_value(output_min);
  const float max_f32 = fp16_ieee_to_fp32_value(output_max);
  params->avxvnni.min = min_f32;
  params->avxvnni.max = max_f32;
  params->avxvnni.sign_mask = 0x80;
  params->avxvnni.mask = 0xF0;
  params->avxvnni.gfni_shl4 = INT64_C(0x01020408);
  return sizeof(params->avxvnni);
}

size_t xnn_init_f32_default_avx_params(
  union xnn_f32_default_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_minmax_sse_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.min[i] = output_min;
    params->sse.max[i] = output_max;
  }
  return sizeof(params->sse);
}

size_t xnn_init_f32_minmax_avx_params(
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
  return sizeof(params->avx);
}

size_t xnn_init_f32_minmax_avx512vnni_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max) {
  params->avx512vnni.min = output_min;
  params->avx512vnni.max = output_max;
  params->avx512vnni.sign_mask = 0x80;
  return sizeof(params->avx512vnni);
}

size_t xnn_init_f32_minmax_avxvnni_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max) {
  params->avxvnni.min = output_min;
  params->avxvnni.max = output_max;
  params->avxvnni.sign_mask = 0x80;
  return sizeof(params->avxvnni);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_minmax_wasmsimd_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  params->wasmsimd.min[0] = output_min;
  params->wasmsimd.min[1] = output_min;
  params->wasmsimd.max[0] = output_max;
  params->wasmsimd.max[1] = output_max;
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_f32_minmax_scalar_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  params->scalar.min = output_min;
  params->scalar.max = output_max;
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_qc4w_minmax_sse_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.min[i] = output_min;
    params->sse.max[i] = output_max;
    params->sse.magic_bias_c0[i] = 0x4B0000F0;
    params->sse.magic_bias_c1[i] = 0x4900000F;
    params->sse.magic_bias_plus_kernel_zero_point_c0[i] = 0x1.0001E0p+23f + (float) kernel_zero_point;
    params->sse.magic_bias_plus_kernel_zero_point_c1[i] = 0x1.00001Ep+19f + (float) kernel_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse.mask[i] = 0xF0;
  }
  return sizeof(params->sse);
}

size_t xnn_init_f32_qc4w_minmax_xop_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  for (uint32_t i = 0; i < 4; i++) {
    params->xop.min[i] = output_min;
    params->xop.max[i] = output_max;
    params->xop.magic_bias_c0[i] = 0x4B0000F0;
    params->xop.magic_bias_c1[i] = 0x4900000F;
    params->xop.magic_bias_plus_kernel_zero_point_c0[i] = 0x1.0001E0p+23f + (float) kernel_zero_point;
    params->xop.magic_bias_plus_kernel_zero_point_c1[i] = 0x1.00001Ep+19f + (float) kernel_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->xop.mask[i] = 0xF0;
    params->xop.shift[i] = 4;
  }
  return sizeof(params->xop);
}

size_t xnn_init_f32_qc4w_minmax_avx_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.min[i] = output_min;
    params->avx.max[i] = output_max;
    params->avx.magic_bias_c0[i] = 0x4B0000F0;
    params->avx.magic_bias_c1[i] = 0x4900000F;
    params->avx.magic_bias_plus_kernel_zero_point_c0[i] = 0x1.0001E0p+23f + (float) kernel_zero_point;
    params->avx.magic_bias_plus_kernel_zero_point_c1[i] = 0x1.00001Ep+19f + (float) kernel_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx.mask[i] = 0xF0;
  }
  return sizeof(params->avx);
}

size_t xnn_init_f32_qc4w_minmax_avx512_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->avx512.min = output_min;
  params->avx512.max = output_max;
  params->avx512.magic_bias_c0 = 0x4B0000F0;
  params->avx512.magic_bias_c1 = 0x4900000F;
  params->avx512.magic_bias_plus_kernel_zero_point_c0 = 0x1.0001E0p+23f + (float) kernel_zero_point;
  params->avx512.magic_bias_plus_kernel_zero_point_c1 = 0x1.00001Ep+19f + (float) kernel_zero_point;
  return sizeof(params->avx512);
}

size_t xnn_init_f32_qc4w_minmax_avx512vnni_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->avx512vnni.min = output_min;
  params->avx512vnni.max = output_max;
  params->avx512vnni.sign_mask = 0x80;
  params->avx512vnni.mask = 0xF0;
  params->avx512vnni.gfni_shl4 = INT64_C(0x01020408);
  return sizeof(params->avx512vnni);
}


size_t xnn_init_f32_qc4w_minmax_avxvnni_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->avxvnni.min = output_min;
  params->avxvnni.max = output_max;
  params->avxvnni.sign_mask = 0x80;
  params->avxvnni.mask = 0xF0;
  params->avxvnni.gfni_shl4 = INT64_C(0x01020408);
  return sizeof(params->avxvnni);
}

#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_qc4w_minmax_wasmsimd_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->wasmsimd.min[0] = output_min;
  params->wasmsimd.min[1] = output_min;
  params->wasmsimd.max[0] = output_max;
  params->wasmsimd.max[1] = output_max;
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.minus_kernel_zero_point[i] = -(int32_t) kernel_zero_point;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd.mask[i] = 0xF0;
  }
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_f32_qc4w_minmax_scalar_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->scalar.min = output_min;
  params->scalar.max = output_max;
  params->scalar.minus_kernel_zero_point = -(int32_t) kernel_zero_point;
  params->scalar.mask = 0xF0;
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_hswish_fp16arith_params(
  union xnn_f16_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  params->fp16arith.sixth = UINT16_C(0x3155);
  params->fp16arith.three = UINT16_C(0x4200);
  params->fp16arith.six = UINT16_C(0x4600);
  return sizeof(params->fp16arith);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_hswish_avx_params(
  union xnn_f16_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.sixth[i] = 0x1.554000p-3f;
    params->avx.three[i] = 3.0f;
    params->avx.six[i] = UINT16_C(0x4600);
  }
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

size_t xnn_init_f32_hswish_scalar_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar.sixth = 0x1.555556p-3f;
  params->scalar.three = 3.0f;
  params->scalar.six = 6.0f;
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_hswish_sse_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.sixth[i] = 0x1.555556p-3f;
    params->sse.half[i] = 0.5f;
    params->sse.one[i] = 1.0f;
  }
  return sizeof(params->sse);
}

size_t xnn_init_f32_hswish_avx_params(
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
  return sizeof(params->avx);
}

size_t xnn_init_f32_hswish_avx512_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512.sixth = 0x1.555556p-3f;
  params->avx512.half = 0.5f;
  params->avx512.one = 1.0f;
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_hswish_wasmsimd_params(
  union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd.sixth[i] = 0x1.555556p-3f;
    params->wasmsimd.three[i] = 3.0f;
    params->wasmsimd.six[i] = 6.0f;
  }
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

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

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_hswish_neon_params(
  union xnn_qs8_hswish_params params[XNN_MIN_ELEMENTS(1)],
  int16_t input_zero_point,
  int16_t output_zero_point,
  float input_scale,
  float output_scale)
{
  params->neon.input_zero_point = input_zero_point;
  params->neon.output_zero_point= output_zero_point;
  const float divisor1 = 0x1.555556p-10f;
  const uint32_t input_scale_div = float_as_uint32(input_scale * divisor1);
  params->neon.input_scale_div_exp = (int16_t) (input_scale_div >> 23) - 111;
  params->neon.input_scale_div_mantissa = (int16_t) ((input_scale_div << 9) >> 18 | UINT16_C(0x4000));
  const float scale_ratio = input_scale / output_scale;
  assert(scale_ratio >= 0x1.0p-8f);
  assert(scale_ratio < 0x1.0p+7f);
  params->neon.scale_ratio = (int16_t) lrintf(scale_ratio * 256.0f);
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

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
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.input_zero_point[i] = input_zero_point;
    params->sse2.output_zero_point[i] = output_zero_point;
    params->sse2.input_scale_div[i] = input_scale_div;
    params->sse2.scale_ratio[i] = scale_ratio_param;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.half[i] = 0x4000;
  }
  return sizeof(params->sse2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qs8_hswish_wasmsimd_params(
  union xnn_qs8_hswish_params params[XNN_MIN_ELEMENTS(1)],
  int16_t input_zero_point,
  int16_t output_zero_point,
  float input_scale,
  float output_scale)
{
  const float divisor1 = 0x1.555556p-10f;
  const uint32_t input_scale_div = float_as_uint32(input_scale * divisor1);
  const int16_t input_scale_div_exp = (int16_t) (input_scale_div >> 23) - 111;
  assert(input_scale_div_exp >= 0);
  assert(input_scale_div_exp <=15);
  params->wasmsimd.input_scale_div_exp = (uint32_t) input_scale_div_exp;
  const int16_t input_scale_div_mantissa = (int16_t) ((input_scale_div << 9) >> 18 | UINT16_C(0x4000));
  const float scale_ratio = input_scale / output_scale;
  assert(scale_ratio >= 0x1.0p-8f);
  assert(scale_ratio < 0x1.0p+7f);
  const int16_t scale_ratio_int = (int16_t) lrintf(scale_ratio * 256.0f);
  int16_t shift_max = (int16_t) 1 << (15 - input_scale_div_exp);
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.input_zero_point[i] = input_zero_point;
    params->wasmsimd.output_zero_point[i] = output_zero_point;
    params->wasmsimd.input_scale_div_mantissa[i] = input_scale_div_mantissa;
    params->wasmsimd.scale_ratio[i] = scale_ratio_int;
    params->wasmsimd.shift_max[i] = shift_max;
    params->wasmsimd.shift_min[i] = -shift_max;
    params->wasmsimd.max_val[i] = 0x7FFF;
    params->wasmsimd.min_val[i] = 0x8000;
    params->wasmsimd.half[i] = 0x4000;
    params->wasmsimd.zero[i] = 0x0000;
  }
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

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
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qu8_hswish_neon_params(
  union xnn_qu8_hswish_params params[XNN_MIN_ELEMENTS(1)],
  int16_t input_zero_point,
  int16_t output_zero_point,
  float input_scale,
  float output_scale)
{
  params->neon.input_zero_point = input_zero_point;
  params->neon.output_zero_point= output_zero_point;
  const float divisor1 = 0x1.555556p-10f;
  const uint32_t input_scale_div = float_as_uint32(input_scale * divisor1);
  params->neon.input_scale_div_exp = (int16_t) (input_scale_div >> 23) - 111;
  params->neon.input_scale_div_mantissa = (int16_t) ((input_scale_div << 9) >> 18 | UINT16_C(0x4000));
  const float scale_ratio = input_scale / output_scale;
  assert(scale_ratio >= 0x1.0p-8f);
  assert(scale_ratio < 0x1.0p+7f);
  params->neon.scale_ratio = (int16_t) lrintf(scale_ratio * 256.0f);
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qu8_hswish_wasmsimd_params(
  union xnn_qu8_hswish_params params[XNN_MIN_ELEMENTS(1)],
  int16_t input_zero_point,
  int16_t output_zero_point,
  float input_scale,
  float output_scale)
{
  const float divisor1 = 0x1.555556p-10f;
  const uint32_t input_scale_div = float_as_uint32(input_scale * divisor1);
  const int16_t input_scale_div_exp = (int16_t) (input_scale_div >> 23) - 111;
  assert(input_scale_div_exp >= 0);
  assert(input_scale_div_exp <=15);
  params->wasmsimd.input_scale_div_exp = (uint32_t) input_scale_div_exp;
  const int16_t input_scale_div_mantissa = (int16_t) ((input_scale_div << 9) >> 18 | UINT16_C(0x4000));
  const float scale_ratio = input_scale / output_scale;
  assert(scale_ratio >= 0x1.0p-8f);
  assert(scale_ratio < 0x1.0p+7f);
  const int16_t scale_ratio_int = (int16_t) lrintf(scale_ratio * 256.0f);
  const int16_t shift_max = (int16_t) 1 << (15 - input_scale_div_exp);
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.input_zero_point[i] = input_zero_point;
    params->wasmsimd.output_zero_point[i] = output_zero_point;
    params->wasmsimd.input_scale_div_mantissa[i] = input_scale_div_mantissa;
    params->wasmsimd.scale_ratio[i] = scale_ratio_int;
    params->wasmsimd.shift_max[i] = shift_max;
    params->wasmsimd.shift_min[i] = -shift_max;
    params->wasmsimd.max_val[i] = 0x7FFF;
    params->wasmsimd.min_val[i] = 0x8000;
    params->wasmsimd.half[i] = 0x4000;
    params->wasmsimd.zero[i] = 0x0000;
  }
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

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
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.input_zero_point[i] = input_zero_point;
    params->sse2.output_zero_point[i] = output_zero_point;
    params->sse2.input_scale_div[i] = input_scale_div;
    params->sse2.scale_ratio[i] = scale_ratio_param;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.half[i] = 0x4000;
  }
  return sizeof(params->sse2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_sigmoid_fp16arith_rr2_p2_params(
  union xnn_f16_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->fp16arith_rr2_p2.magic_bias = UINT16_C(0x660F);  // 0x1.83Cp+10h
  params->fp16arith_rr2_p2.minus_log2e = UINT16_C(0xBDC5);  // -0x1.714p+0h
  params->fp16arith_rr2_p2.ln2_hi = UINT16_C(0x398C);  // 0x1.630p-1h
  params->fp16arith_rr2_p2.ln2_lo = UINT16_C(0x8AF4);  // -0x1.BD0p-13h
  params->fp16arith_rr2_p2.c2 = UINT16_C(0x37F9);  // 0x1.FE4p-2h
  params->fp16arith_rr2_p2.c1 = UINT16_C(0xBC0E);  // -0x1.038p+0h
  params->fp16arith_rr2_p2.denorm_cutoff = UINT16_C(0xC8DA);  // -0x1.368p+3h
  return sizeof(params->fp16arith_rr2_p2);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_sigmoid_avx2_rr1_p2_params(
  union xnn_f16_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_p2.sign_mask[i] = -0.0f;
    params->avx2_rr1_p2.magic_bias[i] = 0x1.8000FEp23f;
    params->avx2_rr1_p2.log2e[i] = 0x1.715476p0f;
    params->avx2_rr1_p2.minus_ln2[i] = -0x1.62E43p-1f;
    params->avx2_rr1_p2.c2[i] = 0x1.FF3A32p-2f;
    params->avx2_rr1_p2.c1[i] = 0x1.039E10p+0f;
    params->avx2_rr1_p2.one[i] = 1.0f;
    params->avx2_rr1_p2.denorm_cutoff[i] = -0x1.368000p+3f;
  }
  return sizeof(params->avx2_rr1_p2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

size_t xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_rr2_lut64_p2.magic_bias = 0x1.800000p17f;
  params->scalar_rr2_lut64_p2.minus_log2e = -0x1.715476p0f;
  params->scalar_rr2_lut64_p2.ln2_hi = 0x1.630000p-1f;
  params->scalar_rr2_lut64_p2.ln2_lo = -0x1.BD0106p-13f;
  params->scalar_rr2_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->scalar_rr2_lut64_p2.one = 1.0f;
  params->scalar_rr2_lut64_p2.denorm_cutoff = 0x1.5D589Ep+6f;
  return sizeof(params->scalar_rr2_lut64_p2);
}

size_t xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_rr2_lut2048_p1.magic_bias = 0x1.800000p12f;
  params->scalar_rr2_lut2048_p1.minus_log2e = -0x1.715476p0f;
  params->scalar_rr2_lut2048_p1.ln2_hi = 0x1.600000p-1f;
  params->scalar_rr2_lut2048_p1.ln2_lo = 0x1.7217F8p-8f;
  params->scalar_rr2_lut2048_p1.c1 = -0x1.FFFFFEp-1f;
  params->scalar_rr2_lut2048_p1.one = 1.0f;
  params->scalar_rr2_lut2048_p1.denorm_cutoff = 0x1.5D589Ep+6f;
  return sizeof(params->scalar_rr2_lut2048_p1);
}

size_t xnn_init_f32_sigmoid_scalar_rr2_p5_params(
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
  return sizeof(params->scalar_rr2_p5);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon_rr2_lut64_p2.magic_bias = 0x1.800000p17f;
  params->neon_rr2_lut64_p2.minus_log2e = -0x1.715476p0f;
  params->neon_rr2_lut64_p2.ln2_hi = 0x1.630000p-1f;
  params->neon_rr2_lut64_p2.ln2_lo = -0x1.BD0106p-13f;
  params->neon_rr2_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->neon_rr2_lut64_p2.denorm_cutoff = 0x1.5D589Ep+6f;
  return sizeof(params->neon_rr2_lut64_p2);
}

size_t xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon_rr2_lut2048_p1.magic_bias = 0x1.800000p12f;
  params->neon_rr2_lut2048_p1.minus_log2e = -0x1.715476p0f;
  params->neon_rr2_lut2048_p1.ln2_hi = 0x1.600000p-1f;
  params->neon_rr2_lut2048_p1.ln2_lo = 0x1.7217F8p-8f;
  params->neon_rr2_lut2048_p1.c1 = -0x1.FFFFFEp-1f;
  params->neon_rr2_lut2048_p1.denorm_cutoff = 0x1.5D589Ep+6f;
  return sizeof(params->neon_rr2_lut2048_p1);
}

size_t xnn_init_f32_sigmoid_neon_rr2_p5_params(
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
  return sizeof(params->neon_rr2_p5);
}

size_t xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neonfma_rr1_lut2048_p1.magic_bias = 0x1.800000p12f;
  params->neonfma_rr1_lut2048_p1.minus_log2e = -0x1.715476p0f;
  params->neonfma_rr1_lut2048_p1.ln2 = 0x1.62E430p-1f;
  params->neonfma_rr1_lut2048_p1.c1 = -0x1.FFFFFEp-1f;
  params->neonfma_rr1_lut2048_p1.denorm_cutoff = 0x1.5D589Ep+6f;
  return sizeof(params->neonfma_rr1_lut2048_p1);
}

size_t xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params(
  union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neonfma_rr1_lut64_p2.magic_bias = 0x1.800000p17f;
  params->neonfma_rr1_lut64_p2.minus_log2e = -0x1.715476p0f;
  params->neonfma_rr1_lut64_p2.ln2 = 0x1.62E430p-1f;
  params->neonfma_rr1_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->neonfma_rr1_lut64_p2.denorm_cutoff = 0x1.5D589Ep+6f;
  return sizeof(params->neonfma_rr1_lut64_p2);
}

size_t xnn_init_f32_sigmoid_neonfma_rr1_p5_params(
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
  return sizeof(params->neonfma_rr1_p5);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params(
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
  return sizeof(params->sse2_rr2_lut64_p2);
}

size_t xnn_init_f32_sigmoid_sse2_rr2_p5_params(
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
  return sizeof(params->sse2_rr2_p5);
}

size_t xnn_init_f32_sigmoid_avx_rr2_p5_params(
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
  return sizeof(params->avx_rr2_p5);
}

size_t xnn_init_f32_sigmoid_avx2_rr1_p5_params(
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
  return sizeof(params->avx2_rr1_p5);
}

size_t xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params(
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
  return sizeof(params->avx512_rr1_lut16_p3);
}

size_t xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params(
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
  return sizeof(params->avx512_rr2_lut32_p2);
}

size_t xnn_init_f32_sigmoid_avx512_rr1_p5_params(
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
  return sizeof(params->avx512_rr1_p5);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params(
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
  return sizeof(params->wasmsimd_rr2_lut64_p2);
}

size_t xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params(
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
  return sizeof(params->wasmsimd_rr2_p5);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params(
  union xnn_f16_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 16; i++) {
    params->avx_expm1minus_rr1_p3h2.sign_mask[i] = UINT16_C(0x8000);
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_expm1minus_rr1_p3h2.sat_cutoff[i] = -0x1.208000p+2f;
    params->avx_expm1minus_rr1_p3h2.log2e[i] = 0x1.715476p+0f;
    params->avx_expm1minus_rr1_p3h2.magic_bias[i] = 0x1.8000FEp+22f;
    params->avx_expm1minus_rr1_p3h2.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx_expm1minus_rr1_p3h2.c3[i] = 0x1.560722p+0f;
    params->avx_expm1minus_rr1_p3h2.c2[i] = 0x1.01E2A2p+1f;
    params->avx_expm1minus_rr1_p3h2.two[i] = 2.0f;
    params->avx_expm1minus_rr1_p3h2.minus_one[i] = -1.0f;
  }
  return sizeof(params->avx_expm1minus_rr1_p3h2);
}

size_t xnn_init_f16_tanh_avx_polynomial_p19h9t2_params(
  union xnn_f16_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_polynomial_p19h9t2.neg_sat_cutoff[i] = -0x1.1F0000p+2f;
    params->avx_polynomial_p19h9t2.pos_sat_cutoff[i] = 0x1.1F0000p+2f;
    params->avx_polynomial_p19h9t2.c19[i] = -0x1.1D841Cp-32f;
    params->avx_polynomial_p19h9t2.c17[i] = 0x1.C4FC88p-26f;
    params->avx_polynomial_p19h9t2.c15[i] = -0x1.332066p-20f;
    params->avx_polynomial_p19h9t2.c13[i] = 0x1.D1AEA2p-16f;
    params->avx_polynomial_p19h9t2.c11[i] = -0x1.B2782Ep-12f;
    params->avx_polynomial_p19h9t2.c9[i] = 0x1.03CAEAp-8f;
    params->avx_polynomial_p19h9t2.c7[i] = -0x1.967628p-6f;
    params->avx_polynomial_p19h9t2.c5[i] = 0x1.ABC35Cp-4f;
    params->avx_polynomial_p19h9t2.c3[i] = -0x1.499D08p-2f;
  }
  return sizeof(params->avx_polynomial_p19h9t2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

size_t xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_expm1minus_rr1_lut8_p4h3.sat_cutoff = 0x1.205968p+3f;
  params->scalar_expm1minus_rr1_lut8_p4h3.minus_log2e = -0x1.715476p+0f;
  params->scalar_expm1minus_rr1_lut8_p4h3.magic_bias = 0x1.800000p+19f;
  params->scalar_expm1minus_rr1_lut8_p4h3.ln2 = 0x1.62E430p-1f;
  params->scalar_expm1minus_rr1_lut8_p4h3.c4 = 0x1.5558ECp-1f;
  params->scalar_expm1minus_rr1_lut8_p4h3.c3 = -0x1.555C20p+0f;
  params->scalar_expm1minus_rr1_lut8_p4h3.c2 = 0x1.000000p+1f;
  params->scalar_expm1minus_rr1_lut8_p4h3.minus_two = -2.0f;
  params->scalar_expm1minus_rr1_lut8_p4h3.one = 1.0f;
  return sizeof(params->scalar_expm1minus_rr1_lut8_p4h3);
}

size_t xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_expm1minus_rr1_p6h5.sat_cutoff = 0x1.205968p+3f;
  params->scalar_expm1minus_rr1_p6h5.minus_log2e = -0x1.715476p+0f;
  params->scalar_expm1minus_rr1_p6h5.magic_bias = 0x1.8000FEp+22f;
  params->scalar_expm1minus_rr1_p6h5.ln2 = 0x1.62E430p-1f;
  params->scalar_expm1minus_rr1_p6h5.c6 = 0x1.6B7338p-4f;
  params->scalar_expm1minus_rr1_p6h5.c5 = -0x1.12278Ep-2f;
  params->scalar_expm1minus_rr1_p6h5.c4 = 0x1.555716p-1f;
  params->scalar_expm1minus_rr1_p6h5.c3 = -0x1.5554B0p+0f;
  params->scalar_expm1minus_rr1_p6h5.c2 = 0x1.FFFFFEp+0f;
  params->scalar_expm1minus_rr1_p6h5.minus_two = -2.0f;
  params->scalar_expm1minus_rr1_p6h5.one = 1.0f;
  return sizeof(params->scalar_expm1minus_rr1_p6h5);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_tanh_sse_expm1minus_rr1_lut8_p4h3_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse_expm1minus_rr1_lut8_p4h3.sign_mask[i] = -0.0f;
    params->sse_expm1minus_rr1_lut8_p4h3.sat_cutoff[i] = -0x1.205968p+3f;
    params->sse_expm1minus_rr1_lut8_p4h3.log2e[i] = 0x1.715476p+0f;
    params->sse_expm1minus_rr1_lut8_p4h3.magic_bias[i] = 0x1.800000p+19f;
    params->sse_expm1minus_rr1_lut8_p4h3.index_mask[i] = UINT32_C(0x7);
    params->sse_expm1minus_rr1_lut8_p4h3.minus_ln2[i] = -0x1.62E430p-1f;
    params->sse_expm1minus_rr1_lut8_p4h3.c4[i] = 0x1.5558ECp-1f;
    params->sse_expm1minus_rr1_lut8_p4h3.c3[i] = 0x1.555C20p+0f;
    params->sse_expm1minus_rr1_lut8_p4h3.c2[i] = 0x1.000000p+1f;
    params->sse_expm1minus_rr1_lut8_p4h3.minus_two[i] = -2.0f;
    params->sse_expm1minus_rr1_lut8_p4h3.minus_one[i] = -1.0f;
  }
  return sizeof(params->sse_expm1minus_rr1_lut8_p4h3);
}

size_t xnn_init_f32_tanh_sse_expm1minus_rr1_p6h5_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse_expm1minus_rr1_p6h5.sign_mask[i] = -0.0f;
    params->sse_expm1minus_rr1_p6h5.sat_cutoff[i] = -0x1.205968p+3f;
    params->sse_expm1minus_rr1_p6h5.log2e[i] = 0x1.715476p+0f;
    params->sse_expm1minus_rr1_p6h5.magic_bias[i] = 0x1.8000FEp+22f;
    params->sse_expm1minus_rr1_p6h5.minus_ln2[i] = -0x1.62E430p-1f;
    params->sse_expm1minus_rr1_p6h5.c6[i] = 0x1.6B7338p-4f;
    params->sse_expm1minus_rr1_p6h5.c5[i] = 0x1.12278Ep-2f;
    params->sse_expm1minus_rr1_p6h5.c4[i] = 0x1.555716p-1f;
    params->sse_expm1minus_rr1_p6h5.c3[i] = 0x1.5554B0p+0f;
    params->sse_expm1minus_rr1_p6h5.c2[i] = 0x1.FFFFFEp+0f;
    params->sse_expm1minus_rr1_p6h5.minus_two[i] = -2.0f;
    params->sse_expm1minus_rr1_p6h5.minus_one[i] = -1.0f;
  }
  return sizeof(params->sse_expm1minus_rr1_p6h5);
}

size_t xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h2_perm_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_expm1minus_rr1_lut4_p4h2_perm.sign_mask[i] = -0.0f;
    params->avx_expm1minus_rr1_lut4_p4h2_perm.sat_cutoff[i] = -0x1.205968p+3f;
    params->avx_expm1minus_rr1_lut4_p4h2_perm.log2e[i] = 0x1.715476p+0f;
    params->avx_expm1minus_rr1_lut4_p4h2_perm.magic_bias[i] = 0x1.800000p+20f;
    params->avx_expm1minus_rr1_lut4_p4h2_perm.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx_expm1minus_rr1_lut4_p4h2_perm.c4[i] = 0x1.554F9Ap-2f;
    params->avx_expm1minus_rr1_lut4_p4h2_perm.c3[i] = 0x1.557082p-1f;
    params->avx_expm1minus_rr1_lut4_p4h2_perm.c2[i] = 0x1.000002p+0f;
    params->avx_expm1minus_rr1_lut4_p4h2_perm.two[i] = 2.0f;
    params->avx_expm1minus_rr1_lut4_p4h2_perm.minus_one[i] = -1.0f;
  }
  params->avx_expm1minus_rr1_lut4_p4h2_perm.table[0] = 0x1.000000p+0f;
  params->avx_expm1minus_rr1_lut4_p4h2_perm.table[1] = 0x1.F06FE0p-1f;
  params->avx_expm1minus_rr1_lut4_p4h2_perm.table[2] = 0x1.EA09E6p-1f;
  params->avx_expm1minus_rr1_lut4_p4h2_perm.table[3] = 0x1.EE89FAp-1f;
  params->avx_expm1minus_rr1_lut4_p4h2_perm.table[4] = 0x1.000000p+0f;
  params->avx_expm1minus_rr1_lut4_p4h2_perm.table[5] = 0x1.F06FE0p-1f;
  params->avx_expm1minus_rr1_lut4_p4h2_perm.table[6] = 0x1.EA09E6p-1f;
  params->avx_expm1minus_rr1_lut4_p4h2_perm.table[7] = 0x1.EE89FAp-1f;
  for (uint32_t i = 0; i < 7; i++) {
    params->avx_expm1minus_rr1_lut4_p4h2_perm.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx_expm1minus_rr1_lut4_p4h2_perm.mask_table[i] = 0;
  }
  return sizeof(params->avx_expm1minus_rr1_lut4_p4h2_perm);
}

size_t xnn_init_f32_tanh_avx_expm1minus_rr1_lut4_p4h3_perm_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_expm1minus_rr1_lut4_p4h3_perm.sign_mask[i] = -0.0f;
    params->avx_expm1minus_rr1_lut4_p4h3_perm.sat_cutoff[i] = -0x1.205968p+3f;
    params->avx_expm1minus_rr1_lut4_p4h3_perm.log2e[i] = 0x1.715476p+0f;
    params->avx_expm1minus_rr1_lut4_p4h3_perm.magic_bias[i] = 0x1.800000p+20f;
    params->avx_expm1minus_rr1_lut4_p4h3_perm.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx_expm1minus_rr1_lut4_p4h3_perm.c4[i] = 0x1.554F9Ap-1f;
    params->avx_expm1minus_rr1_lut4_p4h3_perm.c3[i] = 0x1.557082p+0f;
    params->avx_expm1minus_rr1_lut4_p4h3_perm.c2[i] = 0x1.000002p+1f;
    params->avx_expm1minus_rr1_lut4_p4h3_perm.two[i] = 2.0f;
    params->avx_expm1minus_rr1_lut4_p4h3_perm.minus_one[i] = -1.0f;
  }
  params->avx_expm1minus_rr1_lut4_p4h3_perm.table[0] = 0x1.000000p+0f;
  params->avx_expm1minus_rr1_lut4_p4h3_perm.table[1] = 0x1.F06FE0p-1f;
  params->avx_expm1minus_rr1_lut4_p4h3_perm.table[2] = 0x1.EA09E6p-1f;
  params->avx_expm1minus_rr1_lut4_p4h3_perm.table[3] = 0x1.EE89FAp-1f;
  params->avx_expm1minus_rr1_lut4_p4h3_perm.table[4] = 0x1.000000p+0f;
  params->avx_expm1minus_rr1_lut4_p4h3_perm.table[5] = 0x1.F06FE0p-1f;
  params->avx_expm1minus_rr1_lut4_p4h3_perm.table[6] = 0x1.EA09E6p-1f;
  params->avx_expm1minus_rr1_lut4_p4h3_perm.table[7] = 0x1.EE89FAp-1f;
  for (uint32_t i = 0; i < 7; i++) {
    params->avx_expm1minus_rr1_lut4_p4h3_perm.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx_expm1minus_rr1_lut4_p4h3_perm.mask_table[i] = 0;
  }
  return sizeof(params->avx_expm1minus_rr1_lut4_p4h3_perm);
}

size_t xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_perm_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_expm1minus_rr1_lut8_p4h3_perm.sign_mask[i] = -0.0f;
    params->avx_expm1minus_rr1_lut8_p4h3_perm.sat_cutoff[i] = -0x1.205968p+3f;
    params->avx_expm1minus_rr1_lut8_p4h3_perm.log2e[i] = 0x1.715476p+0f;
    params->avx_expm1minus_rr1_lut8_p4h3_perm.magic_bias[i] = 0x1.800000p+19f;
    params->avx_expm1minus_rr1_lut8_p4h3_perm.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx_expm1minus_rr1_lut8_p4h3_perm.c4[i] = 0x1.5558ECp-1f;
    params->avx_expm1minus_rr1_lut8_p4h3_perm.c3[i] = 0x1.555C20p+0f;
    params->avx_expm1minus_rr1_lut8_p4h3_perm.c2[i] = 0x1.000000p+1f;
    params->avx_expm1minus_rr1_lut8_p4h3_perm.two[i] = 2.0f;
    params->avx_expm1minus_rr1_lut8_p4h3_perm.minus_one[i] = -1.0f;
  }
  params->avx_expm1minus_rr1_lut8_p4h3_perm.table[0] = UINT32_C(0x3F800000);
  params->avx_expm1minus_rr1_lut8_p4h3_perm.table[1] = UINT32_C(0x3F7B95C2);
  params->avx_expm1minus_rr1_lut8_p4h3_perm.table[2] = UINT32_C(0x3F7837F0);
  params->avx_expm1minus_rr1_lut8_p4h3_perm.table[3] = UINT32_C(0x3F75FED7);
  params->avx_expm1minus_rr1_lut8_p4h3_perm.table[4] = UINT32_C(0x3F7504F3);
  params->avx_expm1minus_rr1_lut8_p4h3_perm.table[5] = UINT32_C(0x3F75672A);
  params->avx_expm1minus_rr1_lut8_p4h3_perm.table[6] = UINT32_C(0x3F7744FD);
  params->avx_expm1minus_rr1_lut8_p4h3_perm.table[7] = UINT32_C(0x3F7AC0C7);
  for (uint32_t i = 0; i < 7; i++) {
    params->avx_expm1minus_rr1_lut8_p4h3_perm.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx_expm1minus_rr1_lut8_p4h3_perm.mask_table[i] = 0;
  }
  return sizeof(params->avx_expm1minus_rr1_lut8_p4h3_perm);
}

size_t xnn_init_f32_tanh_avx_expm1minus_rr1_lut8_p4h3_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_expm1minus_rr1_lut8_p4h3.sign_mask[i] = -0.0f;
    params->avx_expm1minus_rr1_lut8_p4h3.sat_cutoff[i] = -0x1.205968p+3f;
    params->avx_expm1minus_rr1_lut8_p4h3.log2e[i] = 0x1.715476p+0f;
    params->avx_expm1minus_rr1_lut8_p4h3.magic_bias[i] = 0x1.800000p+19f;
    params->avx_expm1minus_rr1_lut8_p4h3.index_mask[i] = UINT32_C(0x7);
    params->avx_expm1minus_rr1_lut8_p4h3.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx_expm1minus_rr1_lut8_p4h3.c4[i] = 0x1.5558ECp-1f;
    params->avx_expm1minus_rr1_lut8_p4h3.c3[i] = 0x1.555C20p+0f;
    params->avx_expm1minus_rr1_lut8_p4h3.c2[i] = 0x1.000000p+1f;
    params->avx_expm1minus_rr1_lut8_p4h3.two[i] = 2.0f;
    params->avx_expm1minus_rr1_lut8_p4h3.minus_one[i] = -1.0f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx_expm1minus_rr1_lut8_p4h3.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx_expm1minus_rr1_lut8_p4h3.mask_table[i] = 0;
  }
  return sizeof(params->avx_expm1minus_rr1_lut8_p4h3);
}

size_t xnn_init_f32_tanh_avx_expm1minus_rr1_p6h5_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx_expm1minus_rr1_p6h5.sign_mask[i] = -0.0f;
    params->avx_expm1minus_rr1_p6h5.sat_cutoff[i] = -0x1.205968p+3f;
    params->avx_expm1minus_rr1_p6h5.log2e[i] = 0x1.715476p+0f;
    params->avx_expm1minus_rr1_p6h5.magic_bias[i] = 0x1.8000FEp+22f;
    params->avx_expm1minus_rr1_p6h5.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx_expm1minus_rr1_p6h5.c6[i] = 0x1.6B7338p-4f;
    params->avx_expm1minus_rr1_p6h5.c5[i] = 0x1.12278Ep-2f;
    params->avx_expm1minus_rr1_p6h5.c4[i] = 0x1.555716p-1f;
    params->avx_expm1minus_rr1_p6h5.c3[i] = 0x1.5554B0p+0f;
    params->avx_expm1minus_rr1_p6h5.c2[i] = 0x1.FFFFFEp+0f;
    params->avx_expm1minus_rr1_p6h5.two[i] = 2.0f;
    params->avx_expm1minus_rr1_p6h5.minus_one[i] = -1.0f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx_expm1minus_rr1_p6h5.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx_expm1minus_rr1_p6h5.mask_table[i] = 0;
  }
  return sizeof(params->avx_expm1minus_rr1_p6h5);
}

size_t xnn_init_f32_tanh_avx512_expm1minus_rr1_lut4_p4h3_perm_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.sat_cutoff = 0x1.205968p+3f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.minus_log2e = -0x1.715476p+0f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.magic_bias = 0x1.800000p+20f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.ln2 = 0x1.62E430p-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.c4 = 0x1.554F9Ap-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.c3 = -0x1.557082p+0f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.c2 = 0x1.000002p+1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.minus_two = -2.0f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.one = 1.0f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.sign_mask = UINT32_C(0x80000000);
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[ 0] = 0x1.000000p+0f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[ 1] = 0x1.F06FE0p-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[ 2] = 0x1.EA09E6p-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[ 3] = 0x1.EE89FAp-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[ 4] = 0x1.000000p+0f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[ 5] = 0x1.F06FE0p-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[ 6] = 0x1.EA09E6p-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[ 7] = 0x1.EE89FAp-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[ 8] = 0x1.000000p+0f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[ 9] = 0x1.F06FE0p-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[10] = 0x1.EA09E6p-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[11] = 0x1.EE89FAp-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[12] = 0x1.000000p+0f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[13] = 0x1.F06FE0p-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[14] = 0x1.EA09E6p-1f;
  params->avx512_expm1minus_rr1_lut4_p4h3_perm.table[15] = 0x1.EE89FAp-1f;
  return sizeof(params->avx512_expm1minus_rr1_lut4_p4h3_perm);
}

size_t xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_perm_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.sat_cutoff = 0x1.205968p+3f;
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.minus_log2e = -0x1.715476p+0f;
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.magic_bias = 0x1.800000p+19f;
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.ln2 = 0x1.62E430p-1f;
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.c4 = 0x1.5558ECp-1f;
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.c3 = -0x1.555C20p+0f;
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.c2 = 0x1.000000p+1f;
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.minus_two = -2.0f;
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.one = 1.0f;
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.sign_mask = UINT32_C(0x80000000);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[ 0] = UINT32_C(0x3F800000);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[ 1] = UINT32_C(0x3F7B95C2);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[ 2] = UINT32_C(0x3F7837F0);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[ 3] = UINT32_C(0x3F75FED7);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[ 4] = UINT32_C(0x3F7504F3);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[ 5] = UINT32_C(0x3F75672A);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[ 6] = UINT32_C(0x3F7744FD);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[ 7] = UINT32_C(0x3F7AC0C7);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[ 8] = UINT32_C(0x3F800000);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[ 9] = UINT32_C(0x3F7B95C2);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[10] = UINT32_C(0x3F7837F0);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[11] = UINT32_C(0x3F75FED7);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[12] = UINT32_C(0x3F7504F3);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[13] = UINT32_C(0x3F75672A);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[14] = UINT32_C(0x3F7744FD);
  params->avx512_expm1minus_rr1_lut8_p4h3_perm.table[15] = UINT32_C(0x3F7AC0C7);
  return sizeof(params->avx512_expm1minus_rr1_lut8_p4h3_perm);
}

size_t xnn_init_f32_tanh_avx512_expm1minus_rr1_lut8_p4h3_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512_expm1minus_rr1_lut8_p4h3.sat_cutoff = 0x1.205968p+3f;
  params->avx512_expm1minus_rr1_lut8_p4h3.minus_log2e = -0x1.715476p+0f;
  params->avx512_expm1minus_rr1_lut8_p4h3.magic_bias = 0x1.800000p+19f;
  params->avx512_expm1minus_rr1_lut8_p4h3.index_mask = UINT32_C(0x7);
  params->avx512_expm1minus_rr1_lut8_p4h3.ln2 = 0x1.62E430p-1f;
  params->avx512_expm1minus_rr1_lut8_p4h3.c4 = 0x1.5558ECp-1f;
  params->avx512_expm1minus_rr1_lut8_p4h3.c3 = -0x1.555C20p+0f;
  params->avx512_expm1minus_rr1_lut8_p4h3.c2 = 0x1.000000p+1f;
  params->avx512_expm1minus_rr1_lut8_p4h3.minus_two = -2.0f;
  params->avx512_expm1minus_rr1_lut8_p4h3.one = 1.0f;
  params->avx512_expm1minus_rr1_lut8_p4h3.sign_mask = UINT32_C(0x80000000);
  return sizeof(params->avx512_expm1minus_rr1_lut8_p4h3);
}

size_t xnn_init_f32_tanh_avx512_expm1minus_rr1_p6h5_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512_expm1minus_rr1_p6h5.sign_mask = UINT32_C(0x80000000);
  params->avx512_expm1minus_rr1_p6h5.sat_cutoff = 0x1.205968p+3f;
  params->avx512_expm1minus_rr1_p6h5.minus_log2e = -0x1.715476p+0f;
  params->avx512_expm1minus_rr1_p6h5.magic_bias = 0x1.8000FEp+22f;
  params->avx512_expm1minus_rr1_p6h5.ln2 = 0x1.62E430p-1f;
  params->avx512_expm1minus_rr1_p6h5.c6 = 0x1.6B7338p-4f;
  params->avx512_expm1minus_rr1_p6h5.c5 = -0x1.12278Ep-2f;
  params->avx512_expm1minus_rr1_p6h5.c4 = 0x1.555716p-1f;
  params->avx512_expm1minus_rr1_p6h5.c3 = -0x1.5554B0p+0f;
  params->avx512_expm1minus_rr1_p6h5.c2 = 0x1.FFFFFEp+0f;
  params->avx512_expm1minus_rr1_p6h5.minus_two = -2.0f;
  params->avx512_expm1minus_rr1_p6h5.one = 1.0f;
  return sizeof(params->avx512_expm1minus_rr1_p6h5);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.sat_cutoff[i] = 0x1.205968p+3f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.minus_log2e[i] = -0x1.715476p+0f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.magic_bias[i] = 0x1.800000p+19f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.index_mask[i] = UINT32_C(0x7);
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.ln2[i] = 0x1.62E430p-1f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.c4[i] = 0x1.5558ECp-1f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.c3[i] = -0x1.555C20p+0f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.c2[i] = 0x1.000000p+1f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.minus_two[i] = -2.0f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.one[i] = 1.0f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs.sign_mask[i] = -0.0f;
  }
  return sizeof(params->wasmsimd_expm1minus_rr1_lut8_p4h3_abs);
}

size_t xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_expm1minus_rr1_p6h5_abs.sat_cutoff[i] = 0x1.205968p+3f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.minus_log2e[i] = -0x1.715476p+0f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.magic_bias[i] = 0x1.8000FEp+22f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.ln2[i] = 0x1.62E430p-1f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.c6[i] = 0x1.6B7338p-4f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.c5[i] = -0x1.12278Ep-2f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.c4[i] = 0x1.555716p-1f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.c3[i] = -0x1.5554B0p+0f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.c2[i] = 0x1.FFFFFEp+0f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.minus_two[i] = -2.0f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.one[i] = 1.0f;
    params->wasmsimd_expm1minus_rr1_p6h5_abs.sign_mask[i] = -0.0f;
  }
  return sizeof(params->wasmsimd_expm1minus_rr1_p6h5_abs);
}

size_t xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.sign_mask[i] = -0.0f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.sat_cutoff[i] = -0x1.205968p+3f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.log2e[i] = 0x1.715476p+0f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.magic_bias[i] = 0x1.800000p+19f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.index_mask[i] = UINT32_C(0x7);
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.minus_ln2[i] = -0x1.62E430p-1f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.c4[i] = 0x1.5558ECp-1f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.c3[i] = 0x1.555C20p+0f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.c2[i] = 0x1.000000p+1f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.two[i] = 2.0f;
    params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs.one[i] = 1.0f;
  }
  return sizeof(params->wasmsimd_expm1minus_rr1_lut8_p4h3_nabs);
}

size_t xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.sign_mask[i] = -0.0f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.sat_cutoff[i] = -0x1.205968p+3f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.log2e[i] = 0x1.715476p+0f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.magic_bias[i] = 0x1.8000FEp+22f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.minus_ln2[i] = -0x1.62E430p-1f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.c6[i] = 0x1.6B7338p-4f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.c5[i] = 0x1.12278Ep-2f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.c4[i] = 0x1.555716p-1f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.c3[i] = 0x1.5554B0p+0f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.c2[i] = 0x1.FFFFFEp+0f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.two[i] = 2.0f;
    params->wasmsimd_expm1minus_rr1_p6h5_nabs.one[i] = 1.0f;
  }
  return sizeof(params->wasmsimd_expm1minus_rr1_p6h5_nabs);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon_expm1minus_rr1_lut8_p4h3.sat_cutoff = 0x1.205968p+3f;
  params->neon_expm1minus_rr1_lut8_p4h3.minus_log2e = -0x1.715476p+0f;
  params->neon_expm1minus_rr1_lut8_p4h3.magic_bias = 0x1.800000p+19f;
  params->neon_expm1minus_rr1_lut8_p4h3.ln2 = 0x1.62E430p-1f;
  params->neon_expm1minus_rr1_lut8_p4h3.c4 = 0x1.5558ECp-1f;
  params->neon_expm1minus_rr1_lut8_p4h3.c3 = -0x1.555C20p+0f;
  params->neon_expm1minus_rr1_lut8_p4h3.c2 = 0x1.000000p+1f;
  return sizeof(params->neon_expm1minus_rr1_lut8_p4h3);
}

size_t xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params(
  union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon_expm1minus_rr1_p6h5.sat_cutoff = 0x1.205968p+3f;
  params->neon_expm1minus_rr1_p6h5.minus_log2e = -0x1.715476p+0f;
  params->neon_expm1minus_rr1_p6h5.magic_bias = 0x1.8000FEp+22f;
  params->neon_expm1minus_rr1_p6h5.ln2 = 0x1.62E430p-1f;
  params->neon_expm1minus_rr1_p6h5.c6 = 0x1.6B7338p-4f;
  params->neon_expm1minus_rr1_p6h5.c5 = -0x1.12278Ep-2f;
  params->neon_expm1minus_rr1_p6h5.c4 = 0x1.555716p-1f;
  params->neon_expm1minus_rr1_p6h5.c3 = -0x1.5554B0p+0f;
  params->neon_expm1minus_rr1_p6h5.c2 = 0x1.FFFFFEp+0f;
  return sizeof(params->neon_expm1minus_rr1_p6h5);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_abs_sse_params(
  union xnn_f16_abs_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->sse.nonsign_mask[i] = UINT16_C(0x7FFF);
  }
  return sizeof(params->sse);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_abs_sse_params(
  union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.nonsign_mask[i] = math_nonsign_mask_f32();
  }
  return sizeof(params->sse);
}

size_t xnn_init_f32_abs_avx_params(
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
  return sizeof(params->avx);
}

size_t xnn_init_f32_abs_avx512_params(
  union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512.nonsign_mask = UINT32_C(0x7FFFFFFF);
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_abs_wasmsimd_params(
  union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)])
{
  params->wasmsimd.nonsign_mask[0] = math_nonsign_mask_f32();
  params->wasmsimd.nonsign_mask[1] = math_nonsign_mask_f32();
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_neg_sse_params(
  union xnn_f16_neg_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->sse.sign_mask[i] = UINT16_C(0x8000);
  }
  return sizeof(params->sse);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_neg_sse_params(
  union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.sign_mask[i] = -0.0f;
  }
  return sizeof(params->sse);
}

size_t xnn_init_f32_neg_avx_params(
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
  return sizeof(params->avx);
}

size_t xnn_init_f32_neg_avx512_params(
  union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512.sign_mask = UINT32_C(0x80000000);
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_neg_wasmsimd_params(
  union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)])
{
  params->wasmsimd.sign_mask[0] = -0.0f;
  params->wasmsimd.sign_mask[1] = -0.0f;
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_rnd_sse2_params(
  union xnn_f32_rnd_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.sign_mask[i] = -0.0f;
    params->sse2.one[i] = 1.0f;
  }
  return sizeof(params->sse2);
}

size_t xnn_init_f32_rnd_avx_params(
  union xnn_f32_rnd_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_elu_fp16arith_rr1_p3_params(
  union xnn_f16_elu_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t prescale,
  uint16_t alpha,
  uint16_t beta)
{
  params->fp16arith_rr1_p3.prescale = prescale;
  params->fp16arith_rr1_p3.sat_cutoff = UINT16_C(0xC829);  // -0x1.0A4p+3h;
  params->fp16arith_rr1_p3.magic_bias = UINT16_C(0x660F);  // 0x1.83Cp+10h
  params->fp16arith_rr1_p3.log2e = UINT16_C(0x3DC5);  // 0x1.714p+0h
  params->fp16arith_rr1_p3.minus_ln2 = UINT16_C(0xB98C);  // -0x1.62E430p-1h
  params->fp16arith_rr1_p3.c3 = UINT16_C(0x315B);  // 0x1.56Cp-3h
  params->fp16arith_rr1_p3.c2 = UINT16_C(0x3808);  // 0x1.020p-1h
  params->fp16arith_rr1_p3.minus_alpha = alpha ^ UINT16_C(0x8000);
  params->fp16arith_rr1_p3.beta = beta;
  return sizeof(params->fp16arith_rr1_p3);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_elu_avx2_rr1_p3_params(
  union xnn_f16_elu_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t prescale,
  uint16_t alpha,
  uint16_t beta)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_p3.prescale[i] = fp16_ieee_to_fp32_value(prescale);
    params->avx2_rr1_p3.sat_cutoff[i] = -0x1.0A4000p+3f;
    params->avx2_rr1_p3.magic_bias[i] = 0x1.8000FEp23f;
    params->avx2_rr1_p3.log2e[i] = 0x1.715476p+0f;
    params->avx2_rr1_p3.minus_ln2[i] = -0x1.62E430p-1f;
    params->avx2_rr1_p3.c3[i] = 0x1.5554DCp-3f;
    params->avx2_rr1_p3.c2[i] = 0x1.01EBB2p-1f;
    params->avx2_rr1_p3.c1[i] = 0x1.0002F2p+0f;
    params->avx2_rr1_p3.alpha[i] = fp16_ieee_to_fp32_value(alpha);
    params->avx2_rr1_p3.beta[i] = fp16_ieee_to_fp32_value(beta);
  }
  return sizeof(params->avx2_rr1_p3);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

size_t xnn_init_f32_elu_scalar_rr2_lut16_p3_params(
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
  return sizeof(params->scalar_rr2_lut16_p3);
}

size_t xnn_init_f32_elu_scalar_rr2_p6_params(
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
  return sizeof(params->scalar_rr2_p6);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f32_elu_neon_rr2_lut16_p3_params(
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
  return sizeof(params->neon_rr2_lut16_p3);
}

size_t xnn_init_f32_elu_neon_rr2_p6_params(
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
  return sizeof(params->neon_rr2_p6);
}

size_t xnn_init_f32_elu_neonfma_rr1_lut16_p3_params(
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
  return sizeof(params->neonfma_rr1_lut16_p3);
}

size_t xnn_init_f32_elu_neonfma_rr1_p6_params(
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
  return sizeof(params->neonfma_rr1_p6);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_elu_sse2_rr2_lut16_p3_params(
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
  return sizeof(params->sse2_rr2_lut16_p3);
}

size_t xnn_init_f32_elu_sse2_rr2_p6_params(
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
  return sizeof(params->sse2_rr2_p6);
}

size_t xnn_init_f32_elu_avx_rr2_lut16_p3_params(
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
  return sizeof(params->avx_rr2_lut16_p3);
}

size_t xnn_init_f32_elu_avx_rr2_lut4_p4_params(
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
  return sizeof(params->avx_rr2_lut4_p4);
}

size_t xnn_init_f32_elu_avx_rr2_p6_params(
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
  return sizeof(params->avx_rr2_p6);
}

size_t xnn_init_f32_elu_avx2_rr1_lut16_p3_params(
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
  return sizeof(params->avx2_rr1_lut16_p3);
}

size_t xnn_init_f32_elu_avx2_rr1_lut8_p4_params(
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
  return sizeof(params->avx2_rr1_lut8_p4);
}

size_t xnn_init_f32_elu_avx2_rr1_lut4_p4_params(
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
  return sizeof(params->avx2_rr1_lut4_p4);
}

size_t xnn_init_f32_elu_avx2_rr1_p6_params(
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
  return sizeof(params->avx2_rr1_p6);
}

size_t xnn_init_f32_elu_avx512_rr1_lut16_p3_params(
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
  return sizeof(params->avx512_rr1_lut16_p3);
}

size_t xnn_init_f32_elu_avx512_rr1_p6_params(
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
  return sizeof(params->avx512_rr1_p6);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params(
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
  return sizeof(params->wasmsimd_rr2_lut16_p3);
}

size_t xnn_init_f32_elu_wasmsimd_rr2_p6_params(
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
  return sizeof(params->wasmsimd_rr2_p6);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_expminus_fp16arith_rr2_p2_params(
  union xnn_f16_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->fp16arith_rr2_p2.magic_bias = UINT16_C(0x660F);  // 0x1.83Cp+10h
  params->fp16arith_rr2_p2.log2e = UINT16_C(0x3DC5);  // 0x1.714p+0h
  params->fp16arith_rr2_p2.minus_ln2_hi = UINT16_C(0xB98C);  // -0x1.630p-1h
  params->fp16arith_rr2_p2.minus_ln2_lo = UINT16_C(0x0AF4);  // 0x1.BD0p-13h
  params->fp16arith_rr2_p2.c2 = UINT16_C(0x37F9);  // 0x1.FE4p-2h
  params->fp16arith_rr2_p2.c1 = UINT16_C(0x3C0E);  // 0x1.038p+0h
  params->fp16arith_rr2_p2.denorm_cutoff = UINT16_C(0xC8DA);  // -0x1.368p+3h
  return sizeof(params->fp16arith_rr2_p2);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_RISCV
size_t xnn_init_f32_expminus_rvv_rr2_p6_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->rvv_rr2_p6.x_min = -0x1.5ebb82p6;
  params->rvv_rr2_p6.log2e = 0x1.715476p+0f;
  params->rvv_rr2_p6.ln2_hi = 0x1.62E400p-1f;
  params->rvv_rr2_p6.ln2_lo = 0x1.7F7D1Cp-20f;
  params->rvv_rr2_p6.c6 = 0x1.6850e4p-10f;
  params->rvv_rr2_p6.c5 = 0x1.123bccp-7;
  params->rvv_rr2_p6.c4 = 0x1.555b98p-5f;
  params->rvv_rr2_p6.c3 = 0x1.55548ep-3f;
  params->rvv_rr2_p6.c2 = 0x1.fffff8p-2f;
  return sizeof(params->rvv_rr2_p6);
}
#endif  // XNN_ARCH_RISCV

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_expminus_avx2_rr1_p2_params(
  union xnn_f16_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2_rr1_p2.magic_bias[i] = 0x1.8000FEp23f;
    params->avx2_rr1_p2.log2e[i] = 0x1.715476p0f;
    params->avx2_rr1_p2.minus_ln2[i] = -0x1.62E43p-1f;
    params->avx2_rr1_p2.c2[i] = 0x1.FF3A32p-2f;
    params->avx2_rr1_p2.c1[i] = 0x1.039E10p+0f;
    params->avx2_rr1_p2.denorm_cutoff[i] = -0x1.368000p+3f;
  }
  return sizeof(params->avx2_rr1_p2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

size_t xnn_init_f32_expminus_scalar_rr2_p5_params(
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
  return sizeof(params->scalar_rr2_p5);
}

size_t xnn_init_f32_expminus_scalar_rr2_lut64_p2_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar_rr2_lut64_p2.log2e  = 0x1.715476p0f;
  params->scalar_rr2_lut64_p2.magic_bias = 0x1.800000p17f;
  params->scalar_rr2_lut64_p2.minus_ln2_hi = -0x1.630000p-1f;
  params->scalar_rr2_lut64_p2.minus_ln2_lo = 0x1.BD0106p-13f;
  params->scalar_rr2_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->scalar_rr2_lut64_p2.denorm_cutoff = -0x1.5D589Ep6f;
  return sizeof(params->scalar_rr2_lut64_p2);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f32_expminus_neon_rr2_p5_params(
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
  return sizeof(params->neon_rr2_p5);
}

size_t xnn_init_f32_expminus_neon_rr2_lut64_p2_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon_rr2_lut64_p2.log2e = 0x1.715476p+0f;
  params->neon_rr2_lut64_p2.magic_bias = 0x1.800000p17f;
  params->neon_rr2_lut64_p2.minus_ln2_hi = -0x1.62E400p-1f;
  params->neon_rr2_lut64_p2.minus_ln2_lo = -0x1.7F7D1Cp-20f;
  params->neon_rr2_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->neon_rr2_lut64_p2.denorm_cutoff = -0x1.5D589Ep6f;
  return sizeof(params->neon_rr2_lut64_p2);
}

size_t xnn_init_f32_expminus_neonfma_rr1_p5_params(
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
  return sizeof(params->neonfma_rr1_p5);
}

size_t xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params(
  union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neonfma_rr1_lut64_p2.log2e = 0x1.715476p+0f;
  params->neonfma_rr1_lut64_p2.magic_bias = 0x1.800000p17f;
  params->neonfma_rr1_lut64_p2.minus_ln2 = -0x1.62E430p-1f;
  params->neonfma_rr1_lut64_p2.c2 = 0x1.FFFF0Ap-2f;
  params->neonfma_rr1_lut64_p2.denorm_cutoff = -0x1.5D589Ep6f;
  return sizeof(params->neonfma_rr1_lut64_p2);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_expminus_sse2_rr2_p5_params(
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
  return sizeof(params->sse2_rr2_p5);
}

size_t xnn_init_f32_expminus_avx2_rr1_p5_params(
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
  return sizeof(params->avx2_rr1_p5);
}

size_t xnn_init_f32_expminus_avx512_rr1_p5_params(
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
  return sizeof(params->avx512_rr1_p5);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_expminus_wasmsimd_rr2_p5_params(
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
  return sizeof(params->wasmsimd_rr2_p5);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_lrelu_fp16arith_params(
  union xnn_f16_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t slope)
{
  params->fp16arith.slope = slope;
  return sizeof(params->fp16arith);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_lrelu_avx_params(
  union xnn_f16_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t slope)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.slope[i] = fp16_ieee_to_fp32_value(slope);
  }
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

size_t xnn_init_f32_lrelu_scalar_params(
  union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope)
{
  params->scalar.slope = slope;
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_lrelu_sse_params(
  union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.slope[i] = slope;
  }
  return sizeof(params->sse);
}

size_t xnn_init_f32_lrelu_avx_params(
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
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_lrelu_wasmsimd_params(
  union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)],
  float slope)
{
  params->wasmsimd.slope[0] = slope;
  params->wasmsimd.slope[1] = slope;
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qs8_lrelu_scalar_select_params(
  union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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
  params->scalar_select.input_zero_point = (int32_t) input_zero_point;
  params->scalar_select.positive_multiplier = (int32_t) positive_multiplier;
  params->scalar_select.negative_multiplier = (int32_t) negative_multiplier;
  params->scalar_select.bias = ((int32_t) output_zero_point << 8) + INT32_C(0x80);
  return sizeof(params->scalar_select);
}

size_t xnn_init_qs8_lrelu_scalar_andxor_params(
  union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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
  params->scalar_andxor.input_zero_point = (int32_t) input_zero_point;
  params->scalar_andxor.multiplier_base = (int32_t) positive_multiplier;
  params->scalar_andxor.multiplier_diff = (int32_t) negative_multiplier ^ (int32_t) positive_multiplier;
  params->scalar_andxor.bias = ((int32_t) output_zero_point << 8) + INT32_C(0x80);
  return sizeof(params->scalar_andxor);
}

#if XNN_ARCH_ARM
size_t xnn_init_qs8_lrelu_armsimd32_params(
  union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  params->armsimd32.input_zero_point = (uint32_t) (uint16_t) (int16_t) input_zero_point * UINT32_C(0x00010001);
  params->armsimd32.positive_multiplier = (uint32_t) (uint16_t) (int16_t) positive_multiplier * UINT32_C(0x00010001);
  params->armsimd32.negative_multiplier = (uint32_t) (uint16_t) (int16_t) negative_multiplier * UINT32_C(0x00010001);
  params->armsimd32.bias = ((int32_t) output_zero_point << 8) + INT32_C(0x80);
  return sizeof(params->armsimd32);
}
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_lrelu_neon_params(
  union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  params->neon.input_zero_point = (int16_t) input_zero_point;
  params->neon.positive_multiplier = (int16_t) positive_multiplier;
  params->neon.negative_multiplier = (int16_t) negative_multiplier;
  params->neon.output_zero_point = (int16_t) output_zero_point;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs8_lrelu_sse2_params(
  union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  const int16_t multiplier_base = (int16_t) negative_multiplier;
  const int16_t multiplier_diff = (int16_t) positive_multiplier ^ (int16_t) negative_multiplier;
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.input_zero_point[i] = (int16_t) input_zero_point;
    params->sse2.multiplier_diff[i] = multiplier_diff;
    params->sse2.multiplier_base[i] = multiplier_base;
    params->sse2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->sse2);
}

size_t xnn_init_qs8_lrelu_avx_params(
  union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.input_zero_point[i] = (int16_t) input_zero_point;
    params->avx.positive_multiplier[i] = (int16_t) positive_multiplier;
    params->avx.negative_multiplier[i] = (int16_t) negative_multiplier;
    params->avx.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->avx);
}

size_t xnn_init_qs8_lrelu_avx2_params(
  union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.input_zero_point[i] = (int16_t) input_zero_point;
    params->avx2.positive_multiplier[i] = (int16_t) positive_multiplier;
    params->avx2.negative_multiplier[i] = (int16_t) negative_multiplier;
    params->avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->avx2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qs8_lrelu_wasmsimd_arm_params(
  union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd_arm.input_zero_point[i] = (int16_t) input_zero_point;
    params->wasmsimd_arm.positive_multiplier[i] = (int16_t) positive_multiplier;
    params->wasmsimd_arm.negative_multiplier[i] = (int16_t) negative_multiplier;
    params->wasmsimd_arm.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->wasmsimd_arm);
}

size_t xnn_init_qs8_lrelu_wasmsimd_x86_params(
  union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  const int16_t multiplier_base = (int16_t) negative_multiplier;
  const int16_t multiplier_diff = (int16_t) positive_multiplier ^ (int16_t) negative_multiplier;
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd_x86.input_zero_point[i] = (int16_t) input_zero_point;
    params->wasmsimd_x86.multiplier_diff[i] = multiplier_diff;
    params->wasmsimd_x86.multiplier_base[i] = multiplier_base;
    params->wasmsimd_x86.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->wasmsimd_x86);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qu8_lrelu_scalar_select_params(
  union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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
  params->scalar_select.input_zero_point = (int32_t) input_zero_point;
  params->scalar_select.positive_multiplier = (int32_t) positive_multiplier;
  params->scalar_select.negative_multiplier = (int32_t) negative_multiplier;
  params->scalar_select.bias = ((int32_t) output_zero_point << 8) + INT32_C(0x80);
  return sizeof(params->scalar_select);
}

size_t xnn_init_qu8_lrelu_scalar_andxor_params(
  union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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
  params->scalar_andxor.input_zero_point = (int32_t) input_zero_point;
  params->scalar_andxor.multiplier_base = (int32_t) positive_multiplier;
  params->scalar_andxor.multiplier_diff = (int32_t) negative_multiplier ^ (int32_t) positive_multiplier;
  params->scalar_andxor.bias = ((int32_t) output_zero_point << 8) + INT32_C(0x80);
  return sizeof(params->scalar_andxor);
}

#if XNN_ARCH_ARM
size_t xnn_init_qu8_lrelu_armsimd32_params(
  union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  params->armsimd32.input_zero_point = (uint32_t) input_zero_point * UINT32_C(0x00010001);
  params->armsimd32.positive_multiplier = (uint32_t) (uint16_t) (int16_t) positive_multiplier * UINT32_C(0x00010001);
  params->armsimd32.negative_multiplier = (uint32_t) (uint16_t) (int16_t) negative_multiplier * UINT32_C(0x00010001);
  params->armsimd32.bias = ((int32_t) output_zero_point << 8) + INT32_C(0x80);
  return sizeof(params->armsimd32);
}
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qu8_lrelu_neon_params(
  union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  params->neon.input_zero_point = (uint16_t) input_zero_point;
  params->neon.positive_multiplier = (int16_t) positive_multiplier;
  params->neon.negative_multiplier = (int16_t) negative_multiplier;
  params->neon.output_zero_point = (int16_t) output_zero_point;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qu8_lrelu_sse2_params(
  union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  const int16_t multiplier_base = (int16_t) negative_multiplier;
  const int16_t multiplier_diff = (int16_t) positive_multiplier ^ (int16_t) negative_multiplier;
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.input_zero_point[i] = (int16_t) (uint16_t) input_zero_point;
    params->sse2.multiplier_diff[i] = multiplier_diff;
    params->sse2.multiplier_base[i] = multiplier_base;
    params->sse2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  return sizeof(params->sse2);
}

size_t xnn_init_qu8_lrelu_avx_params(
  union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.input_zero_point[i] = (int16_t) (uint16_t) input_zero_point;
    params->avx.positive_multiplier[i] = (int16_t) positive_multiplier;
    params->avx.negative_multiplier[i] = (int16_t) negative_multiplier;
    params->avx.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  return sizeof(params->avx);
}

size_t xnn_init_qu8_lrelu_avx2_params(
  union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.input_zero_point[i] = (int16_t) (uint16_t) input_zero_point;
    params->avx2.positive_multiplier[i] = (int16_t) positive_multiplier;
    params->avx2.negative_multiplier[i] = (int16_t) negative_multiplier;
    params->avx2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  return sizeof(params->avx2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qu8_lrelu_wasmsimd_arm_params(
  union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd_arm.input_zero_point[i] = (int16_t) (uint16_t) input_zero_point;
    params->wasmsimd_arm.positive_multiplier[i] = (int16_t) positive_multiplier;
    params->wasmsimd_arm.negative_multiplier[i] = (int16_t) negative_multiplier;
    params->wasmsimd_arm.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  return sizeof(params->wasmsimd_arm);
}

size_t xnn_init_qu8_lrelu_wasmsimd_x86_params(
  union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)],
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

  const long positive_multiplier = lrintf(-256.0f * positive_scale);
  assert(positive_multiplier <= -1L);
  assert(positive_multiplier >= -32768L);
  const long negative_multiplier = lrintf(-256.0f * negative_scale);
  assert(negative_multiplier >= -32768L);
  assert(negative_multiplier <= 32767L);
  assert(negative_multiplier != 0L);
  const int16_t multiplier_base = (int16_t) negative_multiplier;
  const int16_t multiplier_diff = (int16_t) positive_multiplier ^ (int16_t) negative_multiplier;
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd_x86.input_zero_point[i] = (int16_t) (uint16_t) input_zero_point;
    params->wasmsimd_x86.multiplier_diff[i] = multiplier_diff;
    params->wasmsimd_x86.multiplier_base[i] = multiplier_base;
    params->wasmsimd_x86.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  return sizeof(params->wasmsimd_x86);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_sqrt_avx_params(
  union xnn_f32_sqrt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
  return sizeof(params->avx);
}

size_t xnn_init_f32_sqrt_fma_params(
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
  return sizeof(params->fma);
}

size_t xnn_init_f32_sqrt_avx512_params(
  union xnn_f32_sqrt_params params[XNN_MIN_ELEMENTS(1)])
{
  params->avx512.half = 0.5f;
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_rsqrt_sse_params(
    union xnn_f32_rsqrt_params params[XNN_MIN_ELEMENTS(1)]) {
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.three[i] = 3.0f;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.half[i] = 0.5f;
  }
  return sizeof(params->sse);
}
size_t xnn_init_f32_rsqrt_avx_params(
    union xnn_f32_rsqrt_params params[XNN_MIN_ELEMENTS(1)]) {
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.three[i] = 3.0f;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.half[i] = 0.5f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->avx.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->avx.mask_table[i] = 0;
  }
  return sizeof(params->avx);
}
size_t xnn_init_f32_rsqrt_fma3_params(
    union xnn_f32_rsqrt_params params[XNN_MIN_ELEMENTS(1)]) {
  for (uint32_t i = 0; i < 8; i++) {
    params->fma3.three[i] = 3.0f;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->fma3.neg_half[i] = -0.5f;
  }
  for (uint32_t i = 0; i < 7; i++) {
    params->fma3.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->fma3.mask_table[i] = 0;
  }
  return sizeof(params->avx);
}
size_t xnn_init_f32_rsqrt_avx512_params(
    union xnn_f32_rsqrt_params params[XNN_MIN_ELEMENTS(1)]) {
  params->avx512.three = 3.0f;
  params->avx512.neg_half = -0.5f;
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_chw_neonfp16arith_stride1_params(
  union xnn_f16_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  uint16_t output_min,
  uint16_t output_max)
{
  params->neonfp16arith_stride1.min = output_min;
  params->neonfp16arith_stride1.max = output_max;

  const uint32_t w8 = (width - 1) & 7;
  params->neonfp16arith_stride1.mask[0] = UINT16_C(0xFFFF);
  params->neonfp16arith_stride1.mask[1] = -(uint16_t) (w8 >= 1);
  params->neonfp16arith_stride1.mask[2] = -(uint16_t) (w8 >= 2);
  params->neonfp16arith_stride1.mask[3] = -(uint16_t) (w8 >= 3);
  params->neonfp16arith_stride1.mask[4] = -(uint16_t) (w8 >= 4);
  params->neonfp16arith_stride1.mask[5] = -(uint16_t) (w8 >= 5);
  params->neonfp16arith_stride1.mask[6] = -(uint16_t) (w8 >= 6);
  params->neonfp16arith_stride1.mask[7] = -(uint16_t) (w8 >= 7);

  return sizeof(params->neonfp16arith_stride1);
}

size_t xnn_init_f16_chw_neonfp16arith_stride2_params(
  union xnn_f16_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  uint16_t output_min,
  uint16_t output_max)
{
  params->neonfp16arith_stride1.min = output_min;
  params->neonfp16arith_stride1.max = output_max;

  const uint32_t w16 = (width - 1) & 15;
  params->neonfp16arith_stride2.mask_even[0] = UINT16_C(0xFFFF);
  params->neonfp16arith_stride2.mask_even[1] = -(uint16_t) (w16 >= 2);
  params->neonfp16arith_stride2.mask_even[2] = -(uint16_t) (w16 >= 4);
  params->neonfp16arith_stride2.mask_even[3] = -(uint16_t) (w16 >= 6);
  params->neonfp16arith_stride2.mask_even[4] = -(uint16_t) (w16 >= 8);
  params->neonfp16arith_stride2.mask_even[5] = -(uint16_t) (w16 >= 10);
  params->neonfp16arith_stride2.mask_even[6] = -(uint16_t) (w16 >= 12);
  params->neonfp16arith_stride2.mask_even[7] = -(uint16_t) (w16 >= 14);
  params->neonfp16arith_stride2.mask_odd[0] = -(uint16_t) (w16 >= 1);
  params->neonfp16arith_stride2.mask_odd[1] = -(uint16_t) (w16 >= 3);
  params->neonfp16arith_stride2.mask_odd[2] = -(uint16_t) (w16 >= 5);
  params->neonfp16arith_stride2.mask_odd[3] = -(uint16_t) (w16 >= 7);
  params->neonfp16arith_stride2.mask_odd[4] = -(uint16_t) (w16 >= 9);
  params->neonfp16arith_stride2.mask_odd[5] = -(uint16_t) (w16 >= 11);
  params->neonfp16arith_stride2.mask_odd[6] = -(uint16_t) (w16 >= 13);
  params->neonfp16arith_stride2.mask_odd[7] = -(uint16_t) (w16 >= 15);

  return sizeof(params->neonfp16arith_stride2);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

size_t xnn_init_f32_chw_scalar_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max)
{
  params->scalar.min = output_min;
  params->scalar.max = output_max;
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f32_chw_neon_stride1_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max)
{
  params->neon_stride1.min = output_min;
  params->neon_stride1.max = output_max;

  const uint32_t w4 = (width - 1) & 3;
  params->neon_stride1.mask[0] = UINT32_C(0xFFFFFFFF);
  params->neon_stride1.mask[1] = -(uint32_t) (w4 >= 1);
  params->neon_stride1.mask[2] = -(uint32_t) (w4 >= 2);
  params->neon_stride1.mask[3] = -(uint32_t) (w4 >= 3);

  return sizeof(params->neon_stride1);
}

size_t xnn_init_f32_chw_neon_stride2_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max)
{
  params->neon_stride2.min = output_min;
  params->neon_stride2.max = output_max;

  const uint32_t w8 = (width - 1) & 7;
  params->neon_stride2.mask_even[0] = UINT32_C(0xFFFFFFFF);
  params->neon_stride2.mask_even[1] = -(uint32_t) (w8 >= 2);
  params->neon_stride2.mask_even[2] = -(uint32_t) (w8 >= 4);
  params->neon_stride2.mask_even[3] = -(uint32_t) (w8 >= 6);
  params->neon_stride2.mask_odd[0] = -(uint32_t) (w8 >= 1);
  params->neon_stride2.mask_odd[1] = -(uint32_t) (w8 >= 3);
  params->neon_stride2.mask_odd[2] = -(uint32_t) (w8 >= 5);
  params->neon_stride2.mask_odd[3] = -(uint32_t) (w8 >= 7);

  return sizeof(params->neon_stride2);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_chw_sse_stride1_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse_stride1.min[i] = output_min;
    params->sse_stride1.max[i] = output_max;
  }

  const uint32_t w4 = (width - 1) & 3;
  params->sse_stride1.mask[0] = UINT32_C(0xFFFFFFFF);
  params->sse_stride1.mask[1] = -(uint32_t) (w4 >= 1);
  params->sse_stride1.mask[2] = -(uint32_t) (w4 >= 2);
  params->sse_stride1.mask[3] = -(uint32_t) (w4 >= 3);

  return sizeof(params->sse_stride1);
}

size_t xnn_init_f32_chw_sse_stride2_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse_stride2.min[i] = output_min;
    params->sse_stride2.max[i] = output_max;
  }

  const uint32_t w8 = (width - 1) & 7;
  params->sse_stride2.mask_even[0] = UINT32_C(0xFFFFFFFF);
  params->sse_stride2.mask_even[1] = -(uint32_t) (w8 >= 2);
  params->sse_stride2.mask_even[2] = -(uint32_t) (w8 >= 4);
  params->sse_stride2.mask_even[3] = -(uint32_t) (w8 >= 6);
  params->sse_stride2.mask_odd[0] = -(uint32_t) (w8 >= 1);
  params->sse_stride2.mask_odd[1] = -(uint32_t) (w8 >= 3);
  params->sse_stride2.mask_odd[2] = -(uint32_t) (w8 >= 5);
  params->sse_stride2.mask_odd[3] = -(uint32_t) (w8 >= 7);

  return sizeof(params->sse_stride2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_chw_wasmsimd_stride1_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max)
{
  params->wasmsimd_stride1.min[0] = output_min;
  params->wasmsimd_stride1.min[1] = output_min;
  params->wasmsimd_stride1.max[0] = output_max;
  params->wasmsimd_stride1.max[1] = output_max;

  const uint32_t w4 = (width - 1) & 3;
  params->wasmsimd_stride1.mask[0] = UINT32_C(0xFFFFFFFF);
  params->wasmsimd_stride1.mask[1] = -(uint32_t) (w4 >= 1);
  params->wasmsimd_stride1.mask[2] = -(uint32_t) (w4 >= 2);
  params->wasmsimd_stride1.mask[3] = -(uint32_t) (w4 >= 3);

  return sizeof(params->wasmsimd_stride1);
}

size_t xnn_init_f32_chw_wasmsimd_stride2_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max)
{
  params->wasmsimd_stride2.min[0] = output_min;
  params->wasmsimd_stride2.min[1] = output_min;
  params->wasmsimd_stride2.max[0] = output_max;
  params->wasmsimd_stride2.max[1] = output_max;

  const uint32_t w8 = (width - 1) & 7;
  params->wasmsimd_stride2.mask_even[0] = UINT32_C(0xFFFFFFFF);
  params->wasmsimd_stride2.mask_even[1] = -(uint32_t) (w8 >= 2);
  params->wasmsimd_stride2.mask_even[2] = -(uint32_t) (w8 >= 4);
  params->wasmsimd_stride2.mask_even[3] = -(uint32_t) (w8 >= 6);
  params->wasmsimd_stride2.mask_odd[0] = -(uint32_t) (w8 >= 1);
  params->wasmsimd_stride2.mask_odd[1] = -(uint32_t) (w8 >= 3);
  params->wasmsimd_stride2.mask_odd[2] = -(uint32_t) (w8 >= 5);
  params->wasmsimd_stride2.mask_odd[3] = -(uint32_t) (w8 >= 7);

  return sizeof(params->wasmsimd_stride2);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_update_f16_chw_neonfp16arith_stride1_params(
  union xnn_f16_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width)
{
  const uint32_t w8 = (width - 1) & 7;
  params->neonfp16arith_stride1.mask[0] = UINT16_C(0xFFFF);
  params->neonfp16arith_stride1.mask[1] = -(uint16_t) (w8 >= 1);
  params->neonfp16arith_stride1.mask[2] = -(uint16_t) (w8 >= 2);
  params->neonfp16arith_stride1.mask[3] = -(uint16_t) (w8 >= 3);
  params->neonfp16arith_stride1.mask[4] = -(uint16_t) (w8 >= 4);
  params->neonfp16arith_stride1.mask[5] = -(uint16_t) (w8 >= 5);
  params->neonfp16arith_stride1.mask[6] = -(uint16_t) (w8 >= 6);
  params->neonfp16arith_stride1.mask[7] = -(uint16_t) (w8 >= 7);
}

void xnn_update_f16_chw_neonfp16arith_stride2_params(
  union xnn_f16_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width)
{
  const uint32_t w16 = (width - 1) & 15;
  params->neonfp16arith_stride2.mask_even[0] = UINT16_C(0xFFFF);
  params->neonfp16arith_stride2.mask_even[1] = -(uint16_t) (w16 >= 2);
  params->neonfp16arith_stride2.mask_even[2] = -(uint16_t) (w16 >= 4);
  params->neonfp16arith_stride2.mask_even[3] = -(uint16_t) (w16 >= 6);
  params->neonfp16arith_stride2.mask_even[4] = -(uint16_t) (w16 >= 8);
  params->neonfp16arith_stride2.mask_even[5] = -(uint16_t) (w16 >= 10);
  params->neonfp16arith_stride2.mask_even[6] = -(uint16_t) (w16 >= 12);
  params->neonfp16arith_stride2.mask_even[7] = -(uint16_t) (w16 >= 14);
  params->neonfp16arith_stride2.mask_odd[0] = -(uint16_t) (w16 >= 1);
  params->neonfp16arith_stride2.mask_odd[1] = -(uint16_t) (w16 >= 3);
  params->neonfp16arith_stride2.mask_odd[2] = -(uint16_t) (w16 >= 5);
  params->neonfp16arith_stride2.mask_odd[3] = -(uint16_t) (w16 >= 7);
  params->neonfp16arith_stride2.mask_odd[4] = -(uint16_t) (w16 >= 9);
  params->neonfp16arith_stride2.mask_odd[5] = -(uint16_t) (w16 >= 11);
  params->neonfp16arith_stride2.mask_odd[6] = -(uint16_t) (w16 >= 13);
  params->neonfp16arith_stride2.mask_odd[7] = -(uint16_t) (w16 >= 15);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
void xnn_update_f32_chw_neon_stride1_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width)
{
  const uint32_t w4 = (width - 1) & 3;
  params->neon_stride1.mask[0] = UINT32_C(0xFFFFFFFF);
  params->neon_stride1.mask[1] = -(uint32_t) (w4 >= 1);
  params->neon_stride1.mask[2] = -(uint32_t) (w4 >= 2);
  params->neon_stride1.mask[3] = -(uint32_t) (w4 >= 3);
}

void xnn_update_f32_chw_neon_stride2_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width)
{
  const uint32_t w8 = (width - 1) & 7;
  params->neon_stride2.mask_even[0] = UINT32_C(0xFFFFFFFF);
  params->neon_stride2.mask_even[1] = -(uint32_t) (w8 >= 2);
  params->neon_stride2.mask_even[2] = -(uint32_t) (w8 >= 4);
  params->neon_stride2.mask_even[3] = -(uint32_t) (w8 >= 6);
  params->neon_stride2.mask_odd[0] = -(uint32_t) (w8 >= 1);
  params->neon_stride2.mask_odd[1] = -(uint32_t) (w8 >= 3);
  params->neon_stride2.mask_odd[2] = -(uint32_t) (w8 >= 5);
  params->neon_stride2.mask_odd[3] = -(uint32_t) (w8 >= 7);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
void xnn_update_f32_chw_sse_stride1_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width)
{
  const uint32_t w4 = (width - 1) & 3;
  params->sse_stride1.mask[0] = UINT32_C(0xFFFFFFFF);
  params->sse_stride1.mask[1] = -(uint32_t) (w4 >= 1);
  params->sse_stride1.mask[2] = -(uint32_t) (w4 >= 2);
  params->sse_stride1.mask[3] = -(uint32_t) (w4 >= 3);
}

void xnn_update_f32_chw_sse_stride2_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width)
{
  const uint32_t w8 = (width - 1) & 7;
  params->sse_stride2.mask_even[0] = UINT32_C(0xFFFFFFFF);
  params->sse_stride2.mask_even[1] = -(uint32_t) (w8 >= 2);
  params->sse_stride2.mask_even[2] = -(uint32_t) (w8 >= 4);
  params->sse_stride2.mask_even[3] = -(uint32_t) (w8 >= 6);
  params->sse_stride2.mask_odd[0] = -(uint32_t) (w8 >= 1);
  params->sse_stride2.mask_odd[1] = -(uint32_t) (w8 >= 3);
  params->sse_stride2.mask_odd[2] = -(uint32_t) (w8 >= 5);
  params->sse_stride2.mask_odd[3] = -(uint32_t) (w8 >= 7);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
void xnn_update_f32_chw_wasmsimd_stride1_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width)
{
  const uint32_t w4 = (width - 1) & 3;
  params->wasmsimd_stride1.mask[0] = UINT32_C(0xFFFFFFFF);
  params->wasmsimd_stride1.mask[1] = -(uint32_t) (w4 >= 1);
  params->wasmsimd_stride1.mask[2] = -(uint32_t) (w4 >= 2);
  params->wasmsimd_stride1.mask[3] = -(uint32_t) (w4 >= 3);
}

void xnn_update_f32_chw_wasmsimd_stride2_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width)
{
  const uint32_t w8 = (width - 1) & 7;
  params->wasmsimd_stride2.mask_even[0] = UINT32_C(0xFFFFFFFF);
  params->wasmsimd_stride2.mask_even[1] = -(uint32_t) (w8 >= 2);
  params->wasmsimd_stride2.mask_even[2] = -(uint32_t) (w8 >= 4);
  params->wasmsimd_stride2.mask_even[3] = -(uint32_t) (w8 >= 6);
  params->wasmsimd_stride2.mask_odd[0] = -(uint32_t) (w8 >= 1);
  params->wasmsimd_stride2.mask_odd[1] = -(uint32_t) (w8 >= 3);
  params->wasmsimd_stride2.mask_odd[2] = -(uint32_t) (w8 >= 5);
  params->wasmsimd_stride2.mask_odd[3] = -(uint32_t) (w8 >= 7);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_s8_minmax_sse2_params(
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
  return sizeof(params->sse2);
}

size_t xnn_init_s8_minmax_sse4_params(
  union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max)
{
  assert(output_min < output_max);

  for (uint32_t i = 0; i < 16; i++) {
    params->sse4.min[i] = output_min;
    params->sse4.max[i] = output_max;
  }
  return sizeof(params->sse4);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_s8_minmax_neon_params(
  union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max)
{
  assert(output_min < output_max);

  params->neon.min = output_min;
  params->neon.max = output_max;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_s8_minmax_wasmsimd_params(
  union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max)
{
  assert(output_min < output_max);

  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd.min[i] = output_min;
    params->wasmsimd.max[i] = output_max;
  }
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_s8_minmax_scalar_params(
  union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max)
{
  assert(output_min < output_max);

  params->scalar.min = (int32_t) output_min;
  params->scalar.max = (int32_t) output_max;
  return sizeof(params->scalar);
}

size_t xnn_init_u8_minmax_params(
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
    return sizeof(params->sse2);
  #elif XNN_ARCH_ARM || XNN_ARCH_ARM64
    params->neon.min = output_min;
    params->neon.max = output_max;
    return sizeof(params->neon);
  #else
    params->scalar.min = (uint32_t) output_min;
    params->scalar.max = (uint32_t) output_max;
    return sizeof(params->scalar);
  #endif
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_u8_minmax_sse2_params(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  for (uint32_t i = 0; i < 16; i++) {
    params->sse2.min[i] = output_min;
    params->sse2.max[i] = output_max;
  }
  return sizeof(params->sse2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_u8_minmax_wasmsimd_params(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  for (uint32_t i = 0; i < 8; i++) {
    params->wasmsimd.min[i] = output_min;
    params->wasmsimd.max[i] = output_max;
  }
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_u8_minmax_neon_params(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  params->neon.min = output_min;
  params->neon.max = output_max;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

size_t xnn_init_u8_minmax_scalar_params(
  union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max)
{
  assert(output_min < output_max);

  params->scalar.min = (uint32_t) output_min;
  params->scalar.max = (uint32_t) output_max;
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qu8_add_minmax_sse2_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  return sizeof(params->sse2);
}

size_t xnn_init_qu8_add_minmax_sse4_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  const int32_t bias = rounding - a_multiplier * (int32_t) (uint32_t) a_zero_point - b_multiplier * (int32_t) (uint32_t) b_zero_point;
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4.bias[i] = bias;
    params->sse4.a_multiplier[i] = a_multiplier;
    params->sse4.b_multiplier[i] = b_multiplier;
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->sse4.shift[i] = (uint64_t) shift;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse4.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse4.output_min[i] = output_min;
    params->sse4.output_max[i] = output_max;
  }
  return sizeof(params->sse4);
}

size_t xnn_init_qu8_add_minmax_avx2_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  const int32_t bias = rounding - a_multiplier * (int32_t) (uint32_t) a_zero_point - b_multiplier * (int32_t) (uint32_t) b_zero_point;
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2.bias[i] = bias;
    params->avx2.a_multiplier[i] = a_multiplier;
    params->avx2.b_multiplier[i] = b_multiplier;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->avx2.shift[i] = (uint64_t) shift;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
    params->avx2.output_min[i] = output_min;
    params->avx2.output_max[i] = output_max;
  }
  return sizeof(params->avx2);
}

size_t xnn_init_qu8_add_minmax_avx512_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  const int32_t bias = rounding - a_multiplier * (int32_t) (uint32_t) a_zero_point - b_multiplier * (int32_t) (uint32_t) b_zero_point;
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.bias[i] = bias;
    params->avx512.a_multiplier[i] = a_multiplier;
    params->avx512.b_multiplier[i] = b_multiplier;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->avx512.shift[i] = (uint64_t) shift;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->avx512.output_zero_point[i] = (int16_t) (uint16_t) output_zero_point;
    params->avx512.output_min[i] = output_min;
    params->avx512.output_max[i] = output_max;
  }
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qu8_add_minmax_neon_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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

  params->neon.a_zero_point = a_zero_point;
  params->neon.b_zero_point = b_zero_point;
  params->neon.a_multiplier = (int32_t) a_multiplier;
  params->neon.b_multiplier = (int32_t) b_multiplier;
  params->neon.right_shift = (int32_t) -shift;
  params->neon.output_zero_point = (int16_t) (uint16_t) output_zero_point;
  params->neon.output_min = output_min;
  params->neon.output_max = output_max;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qu8_add_minmax_wasmsimd_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qu8_add_minmax_scalar_params(
  union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  params->scalar.bias = rounding - a_multiplier * (int32_t) (uint32_t) a_zero_point - b_multiplier * (int32_t) (uint32_t) b_zero_point;
  params->scalar.a_multiplier = a_multiplier;
  params->scalar.b_multiplier = b_multiplier;
  params->scalar.shift = shift;
  params->scalar.output_min_less_zero_point = (int32_t) (uint32_t) output_min - (int32_t) (uint32_t) output_zero_point;
  params->scalar.output_max_less_zero_point = (int32_t) (uint32_t) output_max - (int32_t) (uint32_t) output_zero_point;
  params->scalar.output_zero_point = (int32_t) (uint32_t) output_zero_point;
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs8_add_minmax_sse2_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  return sizeof(params->sse2);
}

size_t xnn_init_qs8_add_minmax_sse4_mul16_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  return sizeof(params->sse4_mul16);
}

size_t xnn_init_qs8_add_minmax_sse4_mul32_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  const int32_t bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4_mul32.bias[i] = bias;
    params->sse4_mul32.a_multiplier[i] = a_multiplier;
    params->sse4_mul32.b_multiplier[i] = b_multiplier;
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->sse4_mul32.shift[i] = (uint64_t) shift;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->sse4_mul32.output_zero_point[i] = (int16_t) output_zero_point;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->sse4_mul32.output_min[i] = output_min;
    params->sse4_mul32.output_max[i] = output_max;
  }
  return sizeof(params->sse4_mul32);
}

size_t xnn_init_qs8_add_minmax_avx2_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  const int32_t bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
  for (uint32_t i = 0; i < 8; i++) {
    params->avx2.bias[i] = bias;
    params->avx2.a_multiplier[i] = a_multiplier;
    params->avx2.b_multiplier[i] = b_multiplier;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->avx2.shift[i] = (uint64_t) shift;
  }
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.output_zero_point[i] = (int16_t) output_zero_point;
    params->avx2.output_min[i] = output_min;
    params->avx2.output_max[i] = output_max;
  }
  return sizeof(params->avx2);
}

size_t xnn_init_qs8_add_minmax_avx512_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  const int32_t bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.bias[i] = bias;
    params->avx512.a_multiplier[i] = a_multiplier;
    params->avx512.b_multiplier[i] = b_multiplier;
  }
  for (uint32_t i = 0; i < 8; i++) {
    params->avx512.shift[i] = (uint64_t) shift;
  }
  for (uint32_t i = 0; i < 32; i++) {
    params->avx512.output_zero_point[i] = (int16_t) output_zero_point;
    params->avx512.output_min[i] = output_min;
    params->avx512.output_max[i] = output_max;
  }
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_add_minmax_neon_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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

  params->neon.a_zero_point = a_zero_point;
  params->neon.b_zero_point = b_zero_point;
  params->neon.a_multiplier = (int32_t) a_multiplier;
  params->neon.b_multiplier = (int32_t) b_multiplier;
  params->neon.right_shift = (int32_t) -shift;
  params->neon.output_zero_point = (int16_t) output_zero_point;
  params->neon.output_min = output_min;
  params->neon.output_max = output_max;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qs8_add_minmax_wasmsimd_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qs8_add_minmax_scalar_params(
  union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)],
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
  params->scalar.bias = rounding - a_multiplier * (int32_t) a_zero_point - b_multiplier * (int32_t) b_zero_point;
  params->scalar.a_multiplier = a_multiplier;
  params->scalar.b_multiplier = b_multiplier;
  params->scalar.shift = shift;
  params->scalar.output_min_less_zero_point = (int32_t) output_min - (int32_t) output_zero_point;
  params->scalar.output_max_less_zero_point = (int32_t) output_max - (int32_t) output_zero_point;
  params->scalar.output_zero_point = (int32_t) output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_qu8_mul_minmax_fp32_scalar_params(
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
  return sizeof(params->fp32_scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qu8_mul_minmax_fp32_neon_params(
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
  return sizeof(params->fp32_neon);
}

size_t xnn_init_qu8_mul_minmax_fp32_neonv8_params(
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
  return sizeof(params->fp32_neonv8);
}

size_t xnn_init_qu8_mul_minmax_rndnu_neon_params(
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
  return sizeof(params->rndnu_neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qu8_mul_minmax_fp32_sse2_params(
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
  return sizeof(params->fp32_sse2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qu8_mul_minmax_fp32_wasmsimd_params(
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
  const int32_t magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
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
  return sizeof(params->fp32_wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qs8_mul_minmax_fp32_scalar_params(
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
  return sizeof(params->fp32_scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_mul_minmax_fp32_neon_params(
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
  return sizeof(params->fp32_neon);
}

size_t xnn_init_qs8_mul_minmax_fp32_neonv8_params(
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
  return sizeof(params->fp32_neonv8);
}

size_t xnn_init_qs8_mul_minmax_rndnu_neon_params(
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
  return sizeof(params->rndnu_neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs8_mul_minmax_fp32_sse2_params(
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
  return sizeof(params->fp32_sse2);
}

size_t xnn_init_qs8_mul_minmax_fp32_sse4_params(
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
  return sizeof(params->fp32_sse4);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qs8_mul_minmax_fp32_wasmsimd_params(
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
  const int32_t magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
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
  return sizeof(params->fp32_wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_f16_f32_cvt_scalar_params(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  params->scalar.sign_mask = UINT32_C(0x80000000);
  params->scalar.exp_offset = UINT32_C(0x70000000);
  params->scalar.exp_scale = 0x1.0p-112f;
  params->scalar.magic_mask = UINT32_C(0x3F000000);
  params->scalar.magic_bias = 0.5f;
  params->scalar.denorm_cutoff = UINT32_C(0x08000000);
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_f32_cvt_neon_params(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon.exp_scale = 0x1.0p-112f;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_f32_cvt_sse_int16_params(
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
  return sizeof(params->sse_int16);
}

size_t xnn_init_f16_f32_cvt_sse_int32_params(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse_int32.sign_mask[i] = UINT32_C(0x80000000);
    params->sse_int32.exp_offset[i] = UINT32_C(0x70000000);
    params->sse_int32.exp_scale[i] = 0x1.0p-112f;
    params->sse_int32.magic_bias[i] = UINT32_C(0x3F000000);
    params->sse_int32.denorm_cutoff[i] = INT32_C(0x04000000);
  }
  return sizeof(params->sse_int32);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f16_f32_cvt_wasmsimd_int16_params(
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
  return sizeof(params->wasmsimd_int16);
}

size_t xnn_init_f16_f32_cvt_wasmsimd_int32_params(
  union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 2; i++) {
    params->wasmsimd_int32.sign_mask[i] = UINT32_C(0x80000000);
    params->wasmsimd_int32.exp_offset[i] = UINT32_C(0x70000000);
    params->wasmsimd_int32.exp_scale[i] = 0x1.0p-112f;
    params->wasmsimd_int32.magic_bias[i] = UINT32_C(0x3F000000);
    params->wasmsimd_int32.denorm_cutoff[i] = INT32_C(0x04000000);
  }
  return sizeof(params->wasmsimd_int32);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_f32_f16_cvt_scalar_bitcast_params(
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
  return sizeof(params->scalar_bitcast);
}

size_t xnn_init_f32_f16_cvt_scalar_fabsf_params(
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
  return sizeof(params->scalar_fabsf);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f32_f16_cvt_neon_params(
  union xnn_f32_f16_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  params->neon.exp_bias = UINT32_C(0x07800000);
  params->neon.scale_to_inf = 0x1.0p+112f;
  params->neon.expw_max = UINT32_C(0x7F800000);
  params->neon.scale_to_zero = 0x1.0p-110f;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_f16_cvt_sse2_params(
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
  return sizeof(params->sse2);
}

size_t xnn_init_f32_f16_cvt_f16c_params(
  union xnn_f32_f16_cvt_params params[XNN_MIN_ELEMENTS(1)])
{
  for (uint32_t i = 0; i < 7; i++) {
    params->f16c.mask_table[i] = -1;
  }
  for (uint32_t i = 7; i < 14; i++) {
    params->f16c.mask_table[i] = 0;
  }
  return sizeof(params->f16c);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_f16_cvt_wasmsimd_params(
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
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_f16_qs8_cvt_scalar_fmagic_params(
  union xnn_f16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->scalar_fmagic.scale = fp16_ieee_to_fp32_value(scale);
  params->scalar_fmagic.output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  params->scalar_fmagic.output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_fmagic.magic_bias = 12582912.0f;
  params->scalar_fmagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->scalar_fmagic);
}

size_t xnn_init_f16_qs8_cvt_scalar_imagic_params(
  union xnn_f16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const float output_max_less_zero_point = (float) ((int32_t) output_max - (int32_t) output_zero_point);
  params->scalar_imagic.scale = fp16_ieee_to_fp32_value(scale);
  params->scalar_imagic.magic_bias = 12582912.0f;
  params->scalar_imagic.magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  params->scalar_imagic.magic_max = (int32_t) float_as_uint32(12582912.0f + output_max_less_zero_point);
  params->scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->scalar_imagic);
}

size_t xnn_init_f32_qs8_cvt_scalar_fmagic_params(
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
  return sizeof(params->scalar_fmagic);
}

size_t xnn_init_f32_qs8_cvt_scalar_imagic_params(
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
  params->scalar_imagic.magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  params->scalar_imagic.magic_max = (int32_t) float_as_uint32(12582912.0f + output_max_less_zero_point);
  params->scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->scalar_imagic);
}

size_t xnn_init_f32_qs8_cvt_scalar_lrintf_params(
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
  return sizeof(params->scalar_lrintf);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f16_qs8_cvt_neonfp16arith_params(
  union xnn_f16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->neonfp16arith.scale = scale;
  params->neonfp16arith.output_zero_point = (int16_t) output_zero_point;
  params->neonfp16arith.output_min = output_min;
  params->neonfp16arith.output_max = output_max;
  return sizeof(params->neonfp16arith);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f32_qs8_cvt_neon_params(
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
  return sizeof(params->neon);
}

size_t xnn_init_f32_qs8_cvt_neonv8_params(
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
  return sizeof(params->neonv8);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_qs8_cvt_sse2_params(
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
  return sizeof(params->sse2);
}

size_t xnn_init_f32_qs8_cvt_sse4_params(
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
  return sizeof(params->sse4);
}

size_t xnn_init_f32_qs8_cvt_avx_params(
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
  return sizeof(params->avx);
}

size_t xnn_init_f32_qs8_cvt_avx2_params(
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
  return sizeof(params->avx2);
}

size_t xnn_init_f32_qs8_cvt_avx512_params(
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
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_qs8_cvt_wasmsimd_cvt_params(
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
  return sizeof(params->wasmsimd_cvt);
}

size_t xnn_init_f32_qs8_cvt_wasmsimd_magic_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
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
  return sizeof(params->wasmsimd_magic);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_f32_qu8_cvt_scalar_fmagic_params(
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
  return sizeof(params->scalar_fmagic);
}

size_t xnn_init_f32_qu8_cvt_scalar_imagic_params(
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
  params->scalar_imagic.magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  params->scalar_imagic.magic_max = (int32_t) float_as_uint32(12582912.0f + output_max_less_zero_point);
  params->scalar_imagic.magic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) output_zero_point;
  return sizeof(params->scalar_imagic);
}

size_t xnn_init_f32_qu8_cvt_scalar_lrintf_params(
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
  return sizeof(params->scalar_lrintf);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_f32_qu8_cvt_neon_params(
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
  return sizeof(params->neon);
}

size_t xnn_init_f32_qu8_cvt_neonv8_params(
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
  return sizeof(params->neonv8);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_qu8_cvt_sse2_params(
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
  return sizeof(params->sse2);
}

size_t xnn_init_f32_qu8_cvt_avx_params(
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
  return sizeof(params->avx);
}

size_t xnn_init_f32_qu8_cvt_avx2_params(
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
  return sizeof(params->avx2);
}

size_t xnn_init_f32_qu8_cvt_avx512_params(
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
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_qu8_cvt_wasmsimd_cvt_params(
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
  return sizeof(params->wasmsimd_cvt);
}

size_t xnn_init_f32_qu8_cvt_wasmsimd_magic_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max)
{
  const float output_min_less_zero_point = (float) ((int32_t) output_min - (int32_t) output_zero_point);
  const int32_t magic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
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
  return sizeof(params->wasmsimd_magic);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qs8_cvt_scalar_params(
  union xnn_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t input_zero_point,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(256.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 32768L);
  params->scalar.bias = ((int32_t) output_zero_point << 8) - (int32_t) multiplier * (int32_t) input_zero_point + INT32_C(0x80);
  params->scalar.multiplier = (int32_t) multiplier;
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM
size_t xnn_init_qs8_cvt_armsimd32_params(
  union xnn_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t input_zero_point,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(131072.0f * input_output_scale);
  assert(multiplier >= 512L);
  assert(multiplier <= 16777216L);
  const uint16_t minus_input_zero_point = -(int16_t) input_zero_point;
  params->armsimd32.minus_input_zero_point = (uint32_t) minus_input_zero_point * UINT32_C(0x00010001);
  params->armsimd32.multiplier = (int32_t) multiplier;
  params->armsimd32.bias = ((int32_t) output_zero_point << 1) + INT32_C(1);
  return sizeof(params->armsimd32);
}
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_cvt_neon_params(
  union xnn_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t input_zero_point,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(-256.0f * input_output_scale);
  assert(multiplier <= -1L);
  assert(multiplier >= -32768L);
  params->neon.input_zero_point = (int16_t) input_zero_point;
  params->neon.multiplier = (int16_t) multiplier;
  params->neon.output_zero_point = (int16_t) output_zero_point;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs8_cvt_sse2_params(
  union xnn_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t input_zero_point,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(-256.0f * input_output_scale);
  assert(multiplier <= -1L);
  assert(multiplier >= -32768L);
  const int32_t bias = ((int32_t) output_zero_point << 8) + (int32_t) multiplier * (int32_t) input_zero_point + INT32_C(0x80);
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.multiplier[i] = (int16_t) multiplier;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.bias[i] = bias;
  }
  return sizeof(params->sse2);
}

size_t xnn_init_qs8_cvt_ssse3_params(
  union xnn_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t input_zero_point,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(-256.0f * input_output_scale);
  assert(multiplier <= -1L);
  assert(multiplier >= -32768L);
  for (uint32_t i = 0; i < 8; i++) {
    params->ssse3.input_zero_point[i] = (int16_t) input_zero_point;
    params->ssse3.multiplier[i] = (int16_t) multiplier;
    params->ssse3.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->ssse3);
}

size_t xnn_init_qs8_cvt_avx2_params(
  union xnn_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t input_zero_point,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(-256.0f * input_output_scale);
  assert(multiplier <= -1L);
  assert(multiplier >= -32768L);
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.input_zero_point[i] = (int16_t) input_zero_point;
    params->avx2.multiplier[i] = (int16_t) multiplier;
    params->avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->avx2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qs8_cvt_wasmsimd_params(
  union xnn_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t input_zero_point,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(-256.0f * input_output_scale);
  assert(multiplier <= -1L);
  assert(multiplier >= -32768L);
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.input_zero_point[i] = (int16_t) input_zero_point;
    params->wasmsimd.multiplier[i] = (int16_t) multiplier;
    params->wasmsimd.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qs16_qs8_cvt_scalar_params(
  union xnn_qs16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-16);
  assert(input_output_scale <= 0x1.0p+8);

  const long multiplier = lrintf(65536.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 0x01000000L);
  params->scalar.multiplier = (int32_t) multiplier;
  params->scalar.bias = ((int32_t) output_zero_point << 16) + INT32_C(0x8000);
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs16_qs8_cvt_neon_params(
  union xnn_qs16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-16);
  assert(input_output_scale <= 0x1.0p+8);

  const long multiplier = lrintf(65536.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 0x01000000L);
  params->neon.multiplier = (int32_t) multiplier;
  params->neon.output_zero_point = (int16_t) output_zero_point;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs16_qs8_cvt_sse2_params(
  union xnn_qs16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-16);
  assert(input_output_scale <= 0x1.0p+8);

  const long multiplier = lrintf(65536.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 0x01000000L);
  const int64_t bias = (int64_t) ((uint64_t) output_zero_point << 32) + INT64_C(0x80000000) -
      (INT64_C(0x80000000) * (int64_t) multiplier);

  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.input_bias[i] = UINT16_C(0x8000);
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.multiplier[i] = (int32_t) multiplier;
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->sse2.bias[i] = (int64_t) bias;
  }
  return sizeof(params->sse2);
}

size_t xnn_init_qs16_qs8_cvt_ssse3_params(
  union xnn_qs16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-16);
  assert(input_output_scale <= 0x1.0p+8);

  const long multiplier = lrintf(65536.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 0x01000000L);
  const int64_t bias = (int64_t) ((uint64_t) output_zero_point << 32) + INT64_C(0x80000000) -
      (INT64_C(0x80000000) * (int64_t) multiplier);

  for (uint32_t i = 0; i < 8; i++) {
    params->ssse3.input_bias[i] = UINT16_C(0x8000);
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->ssse3.multiplier[i] = (int32_t) multiplier;
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->ssse3.bias[i] = (int64_t) bias;
  }
  params->ssse3.shuffle01[0]  = 0x80;
  params->ssse3.shuffle01[1]  = 0x80;
  params->ssse3.shuffle01[2]  = 0;
  params->ssse3.shuffle01[3]  = 1;
  params->ssse3.shuffle01[4]  = 0x80;
  params->ssse3.shuffle01[5]  = 0x80;
  params->ssse3.shuffle01[6]  = 0x80;
  params->ssse3.shuffle01[7]  = 0x80;
  params->ssse3.shuffle01[8]  = 0x80;
  params->ssse3.shuffle01[9]  = 0x80;
  params->ssse3.shuffle01[10] = 2;
  params->ssse3.shuffle01[11] = 3;
  params->ssse3.shuffle01[12] = 0x80;
  params->ssse3.shuffle01[13] = 0x80;
  params->ssse3.shuffle01[14] = 0x80;
  params->ssse3.shuffle01[15] = 0x80;

  params->ssse3.shuffle23[0]  = 0x80;
  params->ssse3.shuffle23[1]  = 0x80;
  params->ssse3.shuffle23[2]  = 4;
  params->ssse3.shuffle23[3]  = 5;
  params->ssse3.shuffle23[4]  = 0x80;
  params->ssse3.shuffle23[5]  = 0x80;
  params->ssse3.shuffle23[6]  = 0x80;
  params->ssse3.shuffle23[7]  = 0x80;
  params->ssse3.shuffle23[8]  = 0x80;
  params->ssse3.shuffle23[9]  = 0x80;
  params->ssse3.shuffle23[10] = 6;
  params->ssse3.shuffle23[11] = 7;
  params->ssse3.shuffle23[12] = 0x80;
  params->ssse3.shuffle23[13] = 0x80;
  params->ssse3.shuffle23[14] = 0x80;
  params->ssse3.shuffle23[15] = 0x80;

  params->ssse3.shuffle45[0]  = 0x80;
  params->ssse3.shuffle45[1]  = 0x80;
  params->ssse3.shuffle45[2]  = 8;
  params->ssse3.shuffle45[3]  = 9;
  params->ssse3.shuffle45[4]  = 0x80;
  params->ssse3.shuffle45[5]  = 0x80;
  params->ssse3.shuffle45[6]  = 0x80;
  params->ssse3.shuffle45[7]  = 0x80;
  params->ssse3.shuffle45[8]  = 0x80;
  params->ssse3.shuffle45[9]  = 0x80;
  params->ssse3.shuffle45[10] = 10;
  params->ssse3.shuffle45[11] = 11;
  params->ssse3.shuffle45[12] = 0x80;
  params->ssse3.shuffle45[13] = 0x80;
  params->ssse3.shuffle45[14] = 0x80;
  params->ssse3.shuffle45[15] = 0x80;

  params->ssse3.shuffle67[0]  = 0x80;
  params->ssse3.shuffle67[1]  = 0x80;
  params->ssse3.shuffle67[2]  = 12;
  params->ssse3.shuffle67[3]  = 13;
  params->ssse3.shuffle67[4]  = 0x80;
  params->ssse3.shuffle67[5]  = 0x80;
  params->ssse3.shuffle67[6]  = 0x80;
  params->ssse3.shuffle67[7]  = 0x80;
  params->ssse3.shuffle67[8]  = 0x80;
  params->ssse3.shuffle67[9]  = 0x80;
  params->ssse3.shuffle67[10] = 14;
  params->ssse3.shuffle67[11] = 15;
  params->ssse3.shuffle67[12] = 0x80;
  params->ssse3.shuffle67[13] = 0x80;
  params->ssse3.shuffle67[14] = 0x80;
  params->ssse3.shuffle67[15] = 0x80;
  return sizeof(params->ssse3);
}

size_t xnn_init_qs16_qs8_cvt_sse4_params(
  union xnn_qs16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-16);
  assert(input_output_scale <= 0x1.0p+8);

  const long multiplier = lrintf(65536.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 0x01000000L);
  const int64_t bias = (int64_t) ((uint64_t) output_zero_point << 32) + INT64_C(0x80000000);

  for (uint32_t i = 0; i < 4; i++) {
    params->sse4.multiplier[i] = (int32_t) multiplier;
  }
  for (uint32_t i = 0; i < 2; i++) {
    params->sse4.bias[i] = (int64_t) bias;
  }
  params->sse4.shuffle01[0]  = 0x80;
  params->sse4.shuffle01[1]  = 0x80;
  params->sse4.shuffle01[2]  = 0;
  params->sse4.shuffle01[3]  = 1;
  params->sse4.shuffle01[4]  = 0x80;
  params->sse4.shuffle01[5]  = 0x80;
  params->sse4.shuffle01[6]  = 0x80;
  params->sse4.shuffle01[7]  = 0x80;
  params->sse4.shuffle01[8]  = 0x80;
  params->sse4.shuffle01[9]  = 0x80;
  params->sse4.shuffle01[10] = 2;
  params->sse4.shuffle01[11] = 3;
  params->sse4.shuffle01[12] = 0x80;
  params->sse4.shuffle01[13] = 0x80;
  params->sse4.shuffle01[14] = 0x80;
  params->sse4.shuffle01[15] = 0x80;

  params->sse4.shuffle23[0]  = 0x80;
  params->sse4.shuffle23[1]  = 0x80;
  params->sse4.shuffle23[2]  = 4;
  params->sse4.shuffle23[3]  = 5;
  params->sse4.shuffle23[4]  = 0x80;
  params->sse4.shuffle23[5]  = 0x80;
  params->sse4.shuffle23[6]  = 0x80;
  params->sse4.shuffle23[7]  = 0x80;
  params->sse4.shuffle23[8]  = 0x80;
  params->sse4.shuffle23[9]  = 0x80;
  params->sse4.shuffle23[10] = 6;
  params->sse4.shuffle23[11] = 7;
  params->sse4.shuffle23[12] = 0x80;
  params->sse4.shuffle23[13] = 0x80;
  params->sse4.shuffle23[14] = 0x80;
  params->sse4.shuffle23[15] = 0x80;

  params->sse4.shuffle45[0]  = 0x80;
  params->sse4.shuffle45[1]  = 0x80;
  params->sse4.shuffle45[2]  = 8;
  params->sse4.shuffle45[3]  = 9;
  params->sse4.shuffle45[4]  = 0x80;
  params->sse4.shuffle45[5]  = 0x80;
  params->sse4.shuffle45[6]  = 0x80;
  params->sse4.shuffle45[7]  = 0x80;
  params->sse4.shuffle45[8]  = 0x80;
  params->sse4.shuffle45[9]  = 0x80;
  params->sse4.shuffle45[10] = 10;
  params->sse4.shuffle45[11] = 11;
  params->sse4.shuffle45[12] = 0x80;
  params->sse4.shuffle45[13] = 0x80;
  params->sse4.shuffle45[14] = 0x80;
  params->sse4.shuffle45[15] = 0x80;

  params->sse4.shuffle67[0]  = 0x80;
  params->sse4.shuffle67[1]  = 0x80;
  params->sse4.shuffle67[2]  = 12;
  params->sse4.shuffle67[3]  = 13;
  params->sse4.shuffle67[4]  = 0x80;
  params->sse4.shuffle67[5]  = 0x80;
  params->sse4.shuffle67[6]  = 0x80;
  params->sse4.shuffle67[7]  = 0x80;
  params->sse4.shuffle67[8]  = 0x80;
  params->sse4.shuffle67[9]  = 0x80;
  params->sse4.shuffle67[10] = 14;
  params->sse4.shuffle67[11] = 15;
  params->sse4.shuffle67[12] = 0x80;
  params->sse4.shuffle67[13] = 0x80;
  params->sse4.shuffle67[14] = 0x80;
  params->sse4.shuffle67[15] = 0x80;
  return sizeof(params->sse4);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qs16_qs8_cvt_wasmsimd_params(
  union xnn_qs16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  int8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-16);
  assert(input_output_scale <= 0x1.0p+8);

  const long multiplier = lrintf(65536.0f * input_output_scale);
  const int64_t bias = ((int32_t) output_zero_point << 16) + INT32_C(0x8000);
  assert(multiplier >= 1L);
  assert(multiplier <= 0x01000000L);

  params->wasmsimd.multiplier[0] = (int32_t) multiplier;
  params->wasmsimd.multiplier[1] = (int32_t) multiplier;
  params->wasmsimd.bias = bias;
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qs8_f32_cvt_scalar_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  params->scalar.zero_point = (int32_t) zero_point;
  params->scalar.scale = scale;
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_f32_cvt_neon_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  params->neon.minus_zero_point[0] = -(int16_t) zero_point;
  params->neon.minus_zero_point[1] = -(int16_t) zero_point;
  params->neon.scale = scale;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qs8_f16_cvt_neonfp16arith_params(
  union xnn_qs8_f16_cvt_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  int8_t zero_point)
{
  params->neon.minus_zero_point = -(int16_t) zero_point;
  params->neon.scale = scale;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs8_f16_cvt_avx_params(
  union xnn_qs8_f16_cvt_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  int8_t zero_point)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.minus_zero_point[i] = -(int32_t) zero_point;
    params->avx.scale[i] = fp16_ieee_to_fp32_value(scale);
  }
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qs8_f32_cvt_sse2_params(
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
  return sizeof(params->sse2);
}

size_t xnn_init_qs8_f32_cvt_sse4_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4.minus_zero_point[i] = -(int32_t) zero_point;
    params->sse4.scale[i] = scale;
  }
  return sizeof(params->sse4);
}

size_t xnn_init_qs8_f32_cvt_avx_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.minus_zero_point[i] = -(int32_t) zero_point;
    params->avx.scale[i] = scale;
  }
  return sizeof(params->avx);
}

size_t xnn_init_qs8_f32_cvt_avx512_params(
  union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  int8_t zero_point)
{
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.minus_zero_point[i] = -(int32_t) zero_point;
    params->avx512.scale[i] = scale;
  }
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qs8_f32_cvt_wasmsimd_params(
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
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qu8_cvt_scalar_params(
  union xnn_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  uint8_t input_zero_point,
  uint8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(256.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 32768L);
  params->scalar.bias = ((int32_t) output_zero_point << 8) - (int32_t) multiplier * (int32_t) input_zero_point + INT32_C(0x80);
  params->scalar.multiplier = (int32_t) multiplier;
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM
size_t xnn_init_qu8_cvt_armsimd32_params(
  union xnn_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  uint8_t input_zero_point,
  uint8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(131072.0f * input_output_scale);
  assert(multiplier >= 512L);
  assert(multiplier <= 16777216L);
  const uint16_t minus_input_zero_point = -(int16_t) input_zero_point;
  params->armsimd32.minus_input_zero_point = (uint32_t) minus_input_zero_point * UINT32_C(0x00010001);
  params->armsimd32.multiplier = (int32_t) multiplier;
  params->armsimd32.bias = ((int32_t) output_zero_point << 1) + INT32_C(1);
  return sizeof(params->armsimd32);
}
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qu8_cvt_neon_params(
  union xnn_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  uint8_t input_zero_point,
  uint8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(-256.0f * input_output_scale);
  assert(multiplier <= -1L);
  assert(multiplier >= -32768L);
  params->neon.input_zero_point = (uint16_t) input_zero_point;
  params->neon.multiplier = (int16_t) multiplier;
  params->neon.output_zero_point = (int16_t) output_zero_point;
  return sizeof(params->neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qu8_cvt_sse2_params(
  union xnn_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  uint8_t input_zero_point,
  uint8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(256.0f * input_output_scale);
  assert(multiplier >= 1L);
  assert(multiplier <= 32768L);
  const int32_t bias = ((int32_t) output_zero_point << 8) - (int32_t) multiplier * (int32_t) input_zero_point + INT32_C(0x80);
  for (uint32_t i = 0; i < 8; i++) {
    params->sse2.multiplier[i] = (uint16_t) multiplier;
  }
  for (uint32_t i = 0; i < 4; i++) {
    params->sse2.bias[i] = bias;
  }
  return sizeof(params->sse2);
}

size_t xnn_init_qu8_cvt_ssse3_params(
  union xnn_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  uint8_t input_zero_point,
  uint8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(-256.0f * input_output_scale);
  assert(multiplier <= -1L);
  assert(multiplier >= -32768L);
  for (uint32_t i = 0; i < 8; i++) {
    params->ssse3.input_zero_point[i] = (uint16_t) input_zero_point;
    params->ssse3.multiplier[i] = (int16_t) multiplier;
    params->ssse3.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->ssse3);
}

size_t xnn_init_qu8_cvt_avx2_params(
  union xnn_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  uint8_t input_zero_point,
  uint8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(-256.0f * input_output_scale);
  assert(multiplier <= -1L);
  assert(multiplier >= -32768L);
  for (uint32_t i = 0; i < 16; i++) {
    params->avx2.input_zero_point[i] = (uint16_t) input_zero_point;
    params->avx2.multiplier[i] = (int16_t) multiplier;
    params->avx2.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->avx2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qu8_cvt_wasmsimd_params(
  union xnn_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float input_output_scale,
  uint8_t input_zero_point,
  uint8_t output_zero_point)
{
  assert(input_output_scale >= 0x1.0p-8);
  assert(input_output_scale <= 0x1.0p+7);

  const long multiplier = lrintf(-256.0f * input_output_scale);
  assert(multiplier <= -1L);
  assert(multiplier >= -32768L);
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.input_zero_point[i] = (uint16_t) input_zero_point;
    params->wasmsimd.multiplier[i] = (int16_t) multiplier;
    params->wasmsimd.output_zero_point[i] = (int16_t) output_zero_point;
  }
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_qu8_f32_cvt_scalar_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  params->scalar.zero_point = (int32_t) zero_point;
  params->scalar.scale = scale;
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_qu8_f32_cvt_neon_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  params->neon.minus_zero_point[0] = -(int16_t) zero_point;
  params->neon.minus_zero_point[1] = -(int16_t) zero_point;
  params->neon.scale = scale;
  return sizeof(params->neon);
}

#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_x24_transpose_ssse3_params(union xnn_x24_transpose_params params[XNN_MIN_ELEMENTS(1)]) {
  memset(&params->ssse3.pos0, -1, sizeof(params->ssse3.pos0));
  memset(&params->ssse3.pos1, -1, sizeof(params->ssse3.pos1));
  memset(&params->ssse3.pos2, -1, sizeof(params->ssse3.pos2));
  memset(&params->ssse3.pos3, -1, sizeof(params->ssse3.pos3));
  memset(&params->ssse3.pos4, -1, sizeof(params->ssse3.pos4));
  memset(&params->ssse3.pos5, -1, sizeof(params->ssse3.pos5));
  params->ssse3.pos0[0] = 0;
  params->ssse3.pos0[1] = 4;
  params->ssse3.pos0[2] = 8;
  params->ssse3.pos0[3] = 2;
  params->ssse3.pos0[4] = 6;
  params->ssse3.pos0[5] = 10;
  params->ssse3.pos0[6] = 1;
  params->ssse3.pos0[7] = 5;
  params->ssse3.pos0[8] = 9;
  params->ssse3.pos0[9] = 3;
  params->ssse3.pos0[10] = 7;
  params->ssse3.pos0[11] = 11;

  params->ssse3.pos1[0] = 4;
  params->ssse3.pos1[1] = 8;
  params->ssse3.pos1[2] = 12;
  params->ssse3.pos1[3] = 6;
  params->ssse3.pos1[4] = 10;
  params->ssse3.pos1[5] = 14;
  params->ssse3.pos1[6] = 5;
  params->ssse3.pos1[7] = 9;
  params->ssse3.pos1[8] = 13;
  params->ssse3.pos1[9] = 7;
  params->ssse3.pos1[10] = 11;
  params->ssse3.pos1[11] = 15;

  params->ssse3.pos2[0] = 12;
  params->ssse3.pos2[3] = 14;
  params->ssse3.pos2[6] = 13;
  params->ssse3.pos2[9] = 15;

  params->ssse3.pos3[1] = 0;
  params->ssse3.pos3[2] = 4;
  params->ssse3.pos3[4] = 2;
  params->ssse3.pos3[5] = 6;
  params->ssse3.pos3[7] = 1;
  params->ssse3.pos3[8] = 5;
  params->ssse3.pos3[10] = 3;
  params->ssse3.pos3[11] = 7;

  params->ssse3.pos4[0] = 8;
  params->ssse3.pos4[1] = 12;
  params->ssse3.pos4[3] = 10;
  params->ssse3.pos4[4] = 14;
  params->ssse3.pos4[6] = 9;
  params->ssse3.pos4[7] = 13;
  params->ssse3.pos4[9] = 11;
  params->ssse3.pos4[10] = 15;

  params->ssse3.pos5[2] = 0;
  params->ssse3.pos5[5] = 2;
  params->ssse3.pos5[8] = 1;
  params->ssse3.pos5[11] = 3;
  return sizeof(params->ssse3);
}

size_t xnn_init_x8_transpose_avx2_params(union xnn_x8_transpose_params params[XNN_MIN_ELEMENTS(1)]) {
  memset(&params->avx2.mask_table[0], -1, sizeof(uint32_t) * 8);
  memset(&params->avx2.mask_table[8], 0, sizeof(uint32_t) * 7);
  return sizeof(params->avx2);
}

size_t xnn_init_x16_transpose_avx2_params(union xnn_x16_transpose_params params[XNN_MIN_ELEMENTS(1)]) {
  memset(&params->avx2.mask_table[0], -1, sizeof(uint32_t) * 8);
  memset(&params->avx2.mask_table[8], 0, sizeof(uint32_t) * 7);
  return sizeof(params->avx2);
}

size_t xnn_init_x32_transpose_avx_params(union xnn_x32_transpose_params params[XNN_MIN_ELEMENTS(1)]) {
  memset(&params->avx.mask_table[0], -1, sizeof(uint32_t) * 8);
  memset(&params->avx.mask_table[8], 0, sizeof(uint32_t) * 7);
  return sizeof(params->avx);
}

size_t xnn_init_x64_transpose_avx_params(union xnn_x64_transpose_params params[XNN_MIN_ELEMENTS(1)]) {
  memset(&params->avx.mask_table[0], -1, sizeof(uint64_t) * 4);
  memset(&params->avx.mask_table[4], 0, sizeof(uint64_t) * 3);
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
size_t xnn_init_x24_transpose_neon_tbl64_params(union xnn_x24_transpose_params params[XNN_MIN_ELEMENTS(1)]) {
  memset(&params->neon_tbl64.pos0, 0, sizeof(params->neon_tbl64.pos0));
  memset(&params->neon_tbl64.pos1, 0, sizeof(params->neon_tbl64.pos1));
  params->neon_tbl64.pos0[0] = 0;
  params->neon_tbl64.pos0[1] = 1;
  params->neon_tbl64.pos0[2] = 2;
  params->neon_tbl64.pos0[3] = 8;
  params->neon_tbl64.pos0[4] = 9;
  params->neon_tbl64.pos0[5] = 10;
  params->neon_tbl64.pos1[0] = 3;
  params->neon_tbl64.pos1[1] = 4;
  params->neon_tbl64.pos1[2] = 5;
  params->neon_tbl64.pos1[3] = 11;
  params->neon_tbl64.pos1[4] = 12;
  params->neon_tbl64.pos1[5] = 13;
  return sizeof(params->neon_tbl64);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
size_t xnn_init_x24_transpose_neon_tbl128_params(union xnn_x24_transpose_params params[XNN_MIN_ELEMENTS(1)]) {
  memset(&params->neon_tbl128.pos0, 0, sizeof(params->neon_tbl128.pos0));
  memset(&params->neon_tbl128.pos1, 0, sizeof(params->neon_tbl128.pos1));
  memset(&params->neon_tbl128.pos2, 0, sizeof(params->neon_tbl128.pos2));
  memset(&params->neon_tbl128.pos3, 0, sizeof(params->neon_tbl128.pos3));
  params->neon_tbl128.pos0[0] = 0;
  params->neon_tbl128.pos0[1] = 1;
  params->neon_tbl128.pos0[2] = 2;
  params->neon_tbl128.pos0[3] = 16;
  params->neon_tbl128.pos0[4] = 17;
  params->neon_tbl128.pos0[5] = 18;
  params->neon_tbl128.pos0[6] = 32;
  params->neon_tbl128.pos0[7] = 33;
  params->neon_tbl128.pos0[8] = 34;
  params->neon_tbl128.pos0[9] = 48;
  params->neon_tbl128.pos0[10] = 49;
  params->neon_tbl128.pos0[11] = 50;
  params->neon_tbl128.pos1[0] = 3;
  params->neon_tbl128.pos1[1] = 4;
  params->neon_tbl128.pos1[2] = 5;
  params->neon_tbl128.pos1[3] = 19;
  params->neon_tbl128.pos1[4] = 20;
  params->neon_tbl128.pos1[5] = 21;
  params->neon_tbl128.pos1[6] = 35;
  params->neon_tbl128.pos1[7] = 36;
  params->neon_tbl128.pos1[8] = 37;
  params->neon_tbl128.pos1[9] = 51;
  params->neon_tbl128.pos1[10] = 52;
  params->neon_tbl128.pos1[11] = 53;
  params->neon_tbl128.pos2[0] = 6;
  params->neon_tbl128.pos2[1] = 7;
  params->neon_tbl128.pos2[2] = 8;
  params->neon_tbl128.pos2[3] = 22;
  params->neon_tbl128.pos2[4] = 23;
  params->neon_tbl128.pos2[5] = 24;
  params->neon_tbl128.pos2[6] = 38;
  params->neon_tbl128.pos2[7] = 39;
  params->neon_tbl128.pos2[8] = 40;
  params->neon_tbl128.pos2[9] = 54;
  params->neon_tbl128.pos2[10] = 55;
  params->neon_tbl128.pos2[11] = 56;
  params->neon_tbl128.pos3[0] = 9;
  params->neon_tbl128.pos3[1] = 10;
  params->neon_tbl128.pos3[2] = 11;
  params->neon_tbl128.pos3[3] = 25;
  params->neon_tbl128.pos3[4] = 26;
  params->neon_tbl128.pos3[5] = 27;
  params->neon_tbl128.pos3[6] = 41;
  params->neon_tbl128.pos3[7] = 42;
  params->neon_tbl128.pos3[8] = 43;
  params->neon_tbl128.pos3[9] = 57;
  params->neon_tbl128.pos3[10] = 58;
  params->neon_tbl128.pos3[11] = 59;
  return sizeof(params->neon_tbl128);
}

size_t xnn_init_x32_transpose_neon_tbl128_params(union xnn_x32_transpose_params params[XNN_MIN_ELEMENTS(1)]) {
  params->neon_tbl128.pos0[0] = 0;
  params->neon_tbl128.pos0[1] = 1;
  params->neon_tbl128.pos0[2] = 2;
  params->neon_tbl128.pos0[3] = 3;
  params->neon_tbl128.pos0[4] = 16;
  params->neon_tbl128.pos0[5] = 17;
  params->neon_tbl128.pos0[6] = 18;
  params->neon_tbl128.pos0[7] = 19;
  params->neon_tbl128.pos0[8] = 32;
  params->neon_tbl128.pos0[9] = 33;
  params->neon_tbl128.pos0[10] = 34;
  params->neon_tbl128.pos0[11] = 35;
  params->neon_tbl128.pos0[12] = 48;
  params->neon_tbl128.pos0[13] = 49;
  params->neon_tbl128.pos0[14] = 50;
  params->neon_tbl128.pos0[15] = 51;
  params->neon_tbl128.pos1[0] = 4;
  params->neon_tbl128.pos1[1] = 5;
  params->neon_tbl128.pos1[2] = 6;
  params->neon_tbl128.pos1[3] = 7;
  params->neon_tbl128.pos1[4] = 20;
  params->neon_tbl128.pos1[5] = 21;
  params->neon_tbl128.pos1[6] = 22;
  params->neon_tbl128.pos1[7] = 23;
  params->neon_tbl128.pos1[8] = 36;
  params->neon_tbl128.pos1[9] = 37;
  params->neon_tbl128.pos1[10] = 38;
  params->neon_tbl128.pos1[11] = 39;
  params->neon_tbl128.pos1[12] = 52;
  params->neon_tbl128.pos1[13] = 53;
  params->neon_tbl128.pos1[14] = 54;
  params->neon_tbl128.pos1[15] = 55;
  params->neon_tbl128.pos2[0] = 8;
  params->neon_tbl128.pos2[1] = 9;
  params->neon_tbl128.pos2[2] = 10;
  params->neon_tbl128.pos2[3] = 11;
  params->neon_tbl128.pos2[4] = 24;
  params->neon_tbl128.pos2[5] = 25;
  params->neon_tbl128.pos2[6] = 26;
  params->neon_tbl128.pos2[7] = 27;
  params->neon_tbl128.pos2[8] = 40;
  params->neon_tbl128.pos2[9] = 41;
  params->neon_tbl128.pos2[10] = 42;
  params->neon_tbl128.pos2[11] = 43;
  params->neon_tbl128.pos2[12] = 56;
  params->neon_tbl128.pos2[13] = 57;
  params->neon_tbl128.pos2[14] = 58;
  params->neon_tbl128.pos2[15] = 59;
  params->neon_tbl128.pos3[0] = 12;
  params->neon_tbl128.pos3[1] = 13;
  params->neon_tbl128.pos3[2] = 14;
  params->neon_tbl128.pos3[3] = 15;
  params->neon_tbl128.pos3[4] = 28;
  params->neon_tbl128.pos3[5] = 29;
  params->neon_tbl128.pos3[6] = 30;
  params->neon_tbl128.pos3[7] = 31;
  params->neon_tbl128.pos3[8] = 44;
  params->neon_tbl128.pos3[9] = 45;
  params->neon_tbl128.pos3[10] = 46;
  params->neon_tbl128.pos3[11] = 47;
  params->neon_tbl128.pos3[12] = 60;
  params->neon_tbl128.pos3[13] = 61;
  params->neon_tbl128.pos3[14] = 62;
  params->neon_tbl128.pos3[15] = 63;
  return sizeof(params->neon_tbl128);
}
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_qu8_f32_cvt_sse2_params(
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
  return sizeof(params->sse2);
}

size_t xnn_init_qu8_f32_cvt_sse4_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  for (uint32_t i = 0; i < 4; i++) {
    params->sse4.minus_zero_point[i] = -(int32_t) zero_point;
    params->sse4.scale[i] = scale;
  }
  return sizeof(params->sse4);
}

size_t xnn_init_qu8_f32_cvt_avx_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.minus_zero_point[i] = -(int32_t) zero_point;
    params->avx.scale[i] = scale;
  }
  return sizeof(params->avx);
}

size_t xnn_init_qu8_f32_cvt_avx512_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  for (uint32_t i = 0; i < 16; i++) {
    params->avx512.minus_zero_point[i] = -(int32_t) zero_point;
    params->avx512.scale[i] = scale;
  }
  return sizeof(params->avx512);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_qu8_f32_cvt_wasmsimd_params(
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
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
