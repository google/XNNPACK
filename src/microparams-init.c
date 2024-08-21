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

#include <fp16/fp16.h>
#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"
#include "xnnpack/unaligned.h"

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
  const uint16_t scale[XNN_MIN_ELEMENTS(1)],
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
        float scale_16 = math_cvt_bf16_fp32(math_cvt_fp32_bf16(scale[scale_index]) / 16.0f);
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
        float scale_16 = math_cvt_bf16_fp32(math_cvt_fp32_bf16(scale[scale_index]) / 16.0f);
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

size_t xnn_init_f16_scale_fp16arith_params(
  union xnn_f16_scale_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale)
{
  params->scale = scale;
  return sizeof(params[0]);
}

size_t xnn_init_f16_f32acc_scale_scalar_params(
  union xnn_f16_f32acc_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  params->scale = scale;
  return sizeof(params[0]);
}

size_t xnn_init_f32_scale_scalar_params(
  union xnn_f32_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale)
{
  params->scale = scale;
  return sizeof(params[0]);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_scaleminmax_avx_params(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max)
{
  params->avx.scale = scale;
  params->avx.min = min;
  params->avx.max = max;
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
  params->sse.scale = scale;
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
  params->avx.scale = scale_f32;
  params->avx.min = min_f32;
  params->avx.max = max_f32;
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
  params->avx.scale = scale_f32;
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
  params->sse.scale = scale;
  params->sse.min = min;
  params->sse.max = max;
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
  params->avx.min = min_f32;
  params->avx.max = max_f32;
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
  return sizeof(params->avxvnni);
}

size_t xnn_init_f16_minmax_scalar_params(
  union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t min,
  uint16_t max)
{
  const float min_f32 = fp16_ieee_to_fp32_value(min);
  const float max_f32 = fp16_ieee_to_fp32_value(max);
  params->scalar.min = min_f32;
  params->scalar.max = max_f32;
  return sizeof(params->scalar);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

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
  params->avx.min = min_f32;
  params->avx.max = max_f32;
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
  return sizeof(params->avxvnni);
}

size_t xnn_init_f16_qc4w_minmax_avxvnni_madd_params(
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
  return sizeof(params->avxvnni);
}
#endif

size_t xnn_init_f16_qb4w_minmax_scalar_params(
  union xnn_f16_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t output_min,
  uint16_t output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  params->fp16arith.min = output_min;
  params->fp16arith.max = output_max;
  params->fp16arith.minus_kernel_zero_point = -(int32_t) kernel_zero_point;
  params->fp16arith.blocksize = blocksize;
  return sizeof(params->fp16arith);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_qb4w_minmax_avx_params(
  union xnn_f16_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t output_min,
  uint16_t output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  const float min_f32 = fp16_ieee_to_fp32_value(output_min);
  const float max_f32 = fp16_ieee_to_fp32_value(output_max);
  params->avx.min = min_f32;
  params->avx.max = max_f32;
  params->avx.blocksize = blocksize;
  return sizeof(params->avx);
}

size_t xnn_init_f16_qb4w_minmax_avxvnni_params(
  union xnn_f16_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t output_min,
  uint16_t output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  const float min_f32 = fp16_ieee_to_fp32_value(output_min);
  const float max_f32 = fp16_ieee_to_fp32_value(output_max);
  params->avxvnni.min = min_f32;
  params->avxvnni.max = max_f32;
  params->avxvnni.blocksize = blocksize;
  return sizeof(params->avxvnni);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_minmax_sse_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  params->sse.min = output_min;
  params->sse.max = output_max;
  return sizeof(params->sse);
}

size_t xnn_init_f32_minmax_avx_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  params->avx.min = output_min;
  params->avx.max = output_max;
  return sizeof(params->avx);
}

size_t xnn_init_f32_minmax_avx512vnni_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max) {
  params->avx512vnni.min = output_min;
  params->avx512vnni.max = output_max;
  return sizeof(params->avx512vnni);
}

size_t xnn_init_f32_minmax_avxvnni_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max) {
  params->avxvnni.min = output_min;
  params->avxvnni.max = output_max;
  return sizeof(params->avxvnni);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_minmax_wasmsimd_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  params->wasmsimd.min = output_min;
  params->wasmsimd.max = output_max;
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_HEXAGON
size_t xnn_init_f32_minmax_hvx_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max)
{
  params->hvx.min = output_min;
  params->hvx.max = output_max;
  return sizeof(params->hvx);
}
#endif // XNN_ARCH_HEXAGON

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
  params->sse.min = output_min;
  params->sse.max = output_max;
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.magic_bias_c0[i] = 0x4B0000F0;
    params->sse.magic_bias_c1[i] = 0x4900000F;
    params->sse.magic_bias_plus_kernel_zero_point_c0[i] = 0x1.0001E0p+23f + (float) kernel_zero_point;
    params->sse.magic_bias_plus_kernel_zero_point_c1[i] = 0x1.00001Ep+19f + (float) kernel_zero_point;
  }
  return sizeof(params->sse);
}

size_t xnn_init_f32_qc4w_minmax_avx_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->avx.min = output_min;
  params->avx.max = output_max;
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.magic_bias_c0[i] = 0x4B0000F0;
    params->avx.magic_bias_c1[i] = 0x4900000F;
    params->avx.magic_bias_plus_kernel_zero_point_c0[i] = 0x1.0001E0p+23f + (float) kernel_zero_point;
    params->avx.magic_bias_plus_kernel_zero_point_c1[i] = 0x1.00001Ep+19f + (float) kernel_zero_point;
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
  return sizeof(params->avx512vnni);
}

size_t xnn_init_f32_qc4w_minmax_avx512vnni_madd_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->avx512vnni.min = output_min;
  params->avx512vnni.max = output_max;
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
  return sizeof(params->avxvnni);
}

size_t xnn_init_f32_qc4w_minmax_avxvnni_madd_params(
  union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point)
{
  assert(kernel_zero_point <= 15);
  params->avxvnni.min = output_min;
  params->avxvnni.max = output_max;
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
  params->wasmsimd.min = output_min;
  params->wasmsimd.max = output_max;
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.minus_kernel_zero_point[i] = -(int32_t) kernel_zero_point;
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
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f32_qb4w_minmax_sse_params(
  union xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  params->sse.min = output_min;
  params->sse.max = output_max;
  for (uint32_t i = 0; i < 4; i++) {
    params->sse.magic_bias_c0[i] = 0x4B0000F0;
    params->sse.magic_bias_c1[i] = 0x4900000F;
    params->sse.magic_bias_plus_kernel_zero_point_c0[i] = 0x1.0001E0p+23f + (float) kernel_zero_point;
    params->sse.magic_bias_plus_kernel_zero_point_c1[i] = 0x1.00001Ep+19f + (float) kernel_zero_point;
  }
  params->sse.blocksize = blocksize;
  return sizeof(params->sse);
}

size_t xnn_init_f32_qb4w_minmax_avx_params(
  union xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  params->avx.min = output_min;
  params->avx.max = output_max;
  for (uint32_t i = 0; i < 8; i++) {
    params->avx.magic_bias_c0[i] = 0x4B0000F0;
    params->avx.magic_bias_c1[i] = 0x4900000F;
    params->avx.magic_bias_plus_kernel_zero_point_c0[i] = 0x1.0001E0p+23f + (float) kernel_zero_point;
    params->avx.magic_bias_plus_kernel_zero_point_c1[i] = 0x1.00001Ep+19f + (float) kernel_zero_point;
  }
  params->avx.blocksize = blocksize;
  return sizeof(params->avx);
}

size_t xnn_init_f32_qb4w_minmax_avx512_params(
  union xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  params->avx512.min = output_min;
  params->avx512.max = output_max;
  params->avx512.magic_bias_c0 = 0x4B0000F0;
  params->avx512.magic_bias_c1 = 0x4900000F;
  params->avx512.magic_bias_plus_kernel_zero_point_c0 = 0x1.0001E0p+23f + (float) kernel_zero_point;
  params->avx512.magic_bias_plus_kernel_zero_point_c1 = 0x1.00001Ep+19f + (float) kernel_zero_point;
  params->avx512.blocksize = blocksize;
  return sizeof(params->avx512);
}

size_t xnn_init_f32_qb4w_minmax_avx512vnni_params(
  union xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  params->avx512vnni.min = output_min;
  params->avx512vnni.max = output_max;
  params->avx512vnni.blocksize = blocksize;
  return sizeof(params->avx512vnni);
}


size_t xnn_init_f32_qb4w_minmax_avxvnni_params(
  union xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  params->avxvnni.min = output_min;
  params->avxvnni.max = output_max;
  params->avxvnni.blocksize = blocksize;
  return sizeof(params->avxvnni);
}

#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
size_t xnn_init_f32_qb4w_minmax_wasmsimd_params(
  union xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  params->wasmsimd.min = output_min;
  params->wasmsimd.max = output_max;
  for (uint32_t i = 0; i < 4; i++) {
    params->wasmsimd.minus_kernel_zero_point[i] = -(int32_t) kernel_zero_point;
  }
  params->wasmsimd.blocksize = blocksize;
  return sizeof(params->wasmsimd);
}
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

size_t xnn_init_f32_qb4w_minmax_scalar_params(
  union xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point,
  size_t blocksize)
{
  assert(kernel_zero_point <= 15);
  params->scalar.min = output_min;
  params->scalar.max = output_max;
  params->scalar.minus_kernel_zero_point = -(int32_t) kernel_zero_point;
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

size_t xnn_init_f16_elu_scalar_params(
  union xnn_f16_elu_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t prescale,
  uint16_t alpha,
  uint16_t beta)
{
  params->scalar.prescale = prescale;
  params->scalar.minus_alpha = alpha ^ UINT16_C(0x8000);
  params->scalar.beta = beta;
  return sizeof(params->scalar);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
size_t xnn_init_f16_elu_avx2_params(
  union xnn_f16_elu_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t prescale,
  uint16_t alpha,
  uint16_t beta)
{
  params->avx2.prescale = fp16_ieee_to_fp32_value(prescale);
  params->avx2.alpha = fp16_ieee_to_fp32_value(alpha);
  params->avx2.beta = fp16_ieee_to_fp32_value(beta);
  return sizeof(params->avx2);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

size_t xnn_init_f32_elu_scalar_params(
  union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)],
  float prescale,
  float alpha,
  float beta)
{
  params->scalar.prescale = prescale;
  params->scalar.alpha = alpha;
  params->scalar.beta = beta;
  return sizeof(params->scalar);
}



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
  params->avx.slope = fp16_ieee_to_fp32_value(slope);
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

size_t xnn_init_qs8_lrelu_scalar_params(
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
  params->scalar.input_zero_point = input_zero_point;
  params->scalar.positive_multiplier = positive_multiplier;
  params->scalar.negative_multiplier = negative_multiplier;
  params->scalar.output_zero_point = output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_qu8_lrelu_scalar_params(
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
  params->scalar.input_zero_point = input_zero_point;
  params->scalar.positive_multiplier = positive_multiplier;
  params->scalar.negative_multiplier = negative_multiplier;
  params->scalar.output_zero_point = output_zero_point;
  return sizeof(params->scalar);
}

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
  params->sse_stride1.min = output_min;
  params->sse_stride1.max = output_max;

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
  params->sse_stride2.min = output_min;
  params->sse_stride2.max = output_max;

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
  params->wasmsimd_stride1.min = output_min;
  params->wasmsimd_stride1.max = output_max;

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
  params->wasmsimd_stride2.min = output_min;
  params->wasmsimd_stride2.max = output_max;

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
  params->scalar.a_zero_point = a_zero_point;
  params->scalar.b_zero_point = b_zero_point;
  params->scalar.a_multiplier = a_multiplier;
  params->scalar.b_multiplier = b_multiplier;
  params->scalar.shift = shift;
  params->scalar.output_min = (int32_t) (uint32_t) output_min;
  params->scalar.output_max = (int32_t) (uint32_t) output_max;
  params->scalar.output_zero_point = (int32_t) (uint32_t) output_zero_point;
  return sizeof(params->scalar);
}

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
  params->scalar.a_zero_point = a_zero_point;
  params->scalar.b_zero_point = b_zero_point;
  params->scalar.a_multiplier = a_multiplier;
  params->scalar.b_multiplier = b_multiplier;
  params->scalar.shift = shift;
  params->scalar.output_zero_point = (int32_t) output_zero_point;
  params->scalar.output_min = (int32_t) output_min;
  params->scalar.output_max = (int32_t) output_max;
  return sizeof(params->scalar);
}

size_t xnn_init_qu8_mul_minmax_scalar_params(
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

  params->scalar.a_zero_point = a_zero_point;
  params->scalar.b_zero_point = b_zero_point;
  params->scalar.scale = product_output_scale;
  params->scalar.output_zero_point = output_zero_point;
  params->scalar.output_min = output_min;
  params->scalar.output_max = output_max;
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
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

  params->rndnu_neon.a_zero_point = a_zero_point;
  params->rndnu_neon.b_zero_point = b_zero_point;
  params->rndnu_neon.left_pre_shift = -pre_shift;
  params->rndnu_neon.multiplier = multiplier;
  params->rndnu_neon.left_post_shift = -post_shift;
  params->rndnu_neon.output_zero_point = (int16_t) output_zero_point;
  params->rndnu_neon.output_min = output_min;
  params->rndnu_neon.output_max = output_max;
  return sizeof(params->rndnu_neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

size_t xnn_init_qs8_mul_minmax_scalar_params(
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

  params->scalar.a_zero_point = a_zero_point;
  params->scalar.b_zero_point = b_zero_point;
  params->scalar.scale = product_output_scale;
  params->scalar.output_zero_point = output_zero_point;
  params->scalar.output_min = output_min;
  params->scalar.output_max = output_max;
  return sizeof(params->scalar);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
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

  params->rndnu_neon.a_zero_point = a_zero_point;
  params->rndnu_neon.b_zero_point = b_zero_point;
  params->rndnu_neon.left_pre_shift = -pre_shift;
  params->rndnu_neon.multiplier = multiplier;
  params->rndnu_neon.left_post_shift = -post_shift;
  params->rndnu_neon.output_zero_point = (int16_t) output_zero_point;
  params->rndnu_neon.output_min = output_min;
  params->rndnu_neon.output_max = output_max;
  return sizeof(params->rndnu_neon);
}
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

size_t xnn_init_f16_qs8_cvt_scalar_params(
  union xnn_f16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max)
{
  params->scalar.scale = fp16_ieee_to_fp32_value(scale);
  params->scalar.output_min = output_min;
  params->scalar.output_max = output_max;
  params->scalar.output_zero_point = output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_f32_qs8_cvt_scalar_params(
  union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)],
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

size_t xnn_init_f32_qu8_cvt_scalar_params(
  union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)],
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
  params->scalar.input_zero_point = (int16_t) input_zero_point;
  params->scalar.multiplier = (int32_t) multiplier;
  params->scalar.output_zero_point = (int16_t) output_zero_point;
  return sizeof(params->scalar);
}

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
  params->scalar.output_zero_point = (int32_t) output_zero_point;
  return sizeof(params->scalar);
}

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
size_t xnn_init_qs8_f16_cvt_neonfp16arith_params(
  union xnn_qs8_f16_cvt_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t scale,
  int8_t zero_point)
{
  params->neon.zero_point = (int16_t) zero_point;
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
  params->avx.zero_point = (int32_t) zero_point;
  params->avx.scale = fp16_ieee_to_fp32_value(scale);
  return sizeof(params->avx);
}
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

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
  params->scalar.input_zero_point = (uint16_t) input_zero_point;
  params->scalar.multiplier = (int32_t) multiplier;
  params->scalar.output_zero_point = (int16_t) output_zero_point;
  return sizeof(params->scalar);
}

size_t xnn_init_qu8_f32_cvt_scalar_params(
  union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  uint8_t zero_point)
{
  params->scalar.zero_point = (int32_t) zero_point;
  params->scalar.scale = scale;
  return sizeof(params->scalar);
}
