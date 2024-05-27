// Auto-generated file. Do not edit!
//   Template: src/qs8-rdsum/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/reduce.h>


void xnn_qs8_rdsum_minmax_fp32_ukernel_7p7x__scalar_c4(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  const int32_t vinit_bias = params->fp32_scalar_fmagic.init_bias;
  const float vscale = params->fp32_scalar_fmagic.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
  for (; channels >= 4; channels -= 4) {
    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
    const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
    const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
    const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
    int32_t vacc0 = vinit_bias;
    int32_t vacc1 = vinit_bias;
    int32_t vacc2 = vinit_bias;
    int32_t vacc3 = vinit_bias;

    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      vacc0 += (int32_t) i0[0];
      vacc1 += (int32_t) i0[1];
      vacc2 += (int32_t) i0[2];
      vacc3 += (int32_t) i0[3];
      vacc0 += (int32_t) i1[0];
      vacc1 += (int32_t) i1[1];
      vacc2 += (int32_t) i1[2];
      vacc3 += (int32_t) i1[3];
      vacc0 += (int32_t) i2[0];
      vacc1 += (int32_t) i2[1];
      vacc2 += (int32_t) i2[2];
      vacc3 += (int32_t) i2[3];
      vacc0 += (int32_t) i3[0];
      vacc1 += (int32_t) i3[1];
      vacc2 += (int32_t) i3[2];
      vacc3 += (int32_t) i3[3];
      vacc0 += (int32_t) i4[0];
      vacc1 += (int32_t) i4[1];
      vacc2 += (int32_t) i4[2];
      vacc3 += (int32_t) i4[3];
      vacc0 += (int32_t) i5[0];
      vacc1 += (int32_t) i5[1];
      vacc2 += (int32_t) i5[2];
      vacc3 += (int32_t) i5[3];
      vacc0 += (int32_t) i6[0];
      vacc1 += (int32_t) i6[1];
      vacc2 += (int32_t) i6[2];
      vacc3 += (int32_t) i6[3];
      i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
    }

    float vfpacc0 = (float) vacc0 * vscale;
    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc0 += vmagic_bias;
    vacc0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
    float vfpacc1 = (float) vacc1 * vscale;
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);
    vfpacc1 += vmagic_bias;
    vacc1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;
    float vfpacc2 = (float) vacc2 * vscale;
    vfpacc2 = math_max_f32(vfpacc2, voutput_min_less_zero_point);
    vfpacc2 = math_min_f32(vfpacc2, voutput_max_less_zero_point);
    vfpacc2 += vmagic_bias;
    vacc2 = (int32_t) float_as_uint32(vfpacc2) - vmagic_bias_less_output_zero_point;
    float vfpacc3 = (float) vacc3 * vscale;
    vfpacc3 = math_max_f32(vfpacc3, voutput_min_less_zero_point);
    vfpacc3 = math_min_f32(vfpacc3, voutput_max_less_zero_point);
    vfpacc3 += vmagic_bias;
    vacc3 = (int32_t) float_as_uint32(vfpacc3) - vmagic_bias_less_output_zero_point;
    *output++ += (int8_t) vacc0;
    *output++ += (int8_t) vacc1;
    *output++ += (int8_t) vacc2;
    *output++ += (int8_t) vacc3;

    input = (const int8_t*) ((uintptr_t) input + 4);
  }
  if (channels != 0) {
    size_t input_increment = 7 * input_stride;
    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
    const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
    const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
    const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
    int32_t vacc0 = vinit_bias;
    int32_t vacc1 = vinit_bias;
    int32_t vacc2 = vinit_bias;

    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      vacc0 += (int32_t) i0[0];
      vacc1 += (int32_t) i0[1];
      vacc2 += (int32_t) i0[2];
      vacc0 += (int32_t) i1[0];
      vacc1 += (int32_t) i1[1];
      vacc2 += (int32_t) i1[2];
      vacc0 += (int32_t) i2[0];
      vacc1 += (int32_t) i2[1];
      vacc2 += (int32_t) i2[2];
      vacc0 += (int32_t) i3[0];
      vacc1 += (int32_t) i3[1];
      vacc2 += (int32_t) i3[2];
      vacc0 += (int32_t) i4[0];
      vacc1 += (int32_t) i4[1];
      vacc2 += (int32_t) i4[2];
      vacc0 += (int32_t) i5[0];
      vacc1 += (int32_t) i5[1];
      vacc2 += (int32_t) i5[2];
      vacc0 += (int32_t) i6[0];
      vacc1 += (int32_t) i6[1];
      vacc2 += (int32_t) i6[2];
      i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
    }
    float vfpacc0 = (float) vacc0 * vscale;
    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc0 += vmagic_bias;
    vacc0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
    float vfpacc1 = (float) vacc1 * vscale;
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);
    vfpacc1 += vmagic_bias;
    vacc1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;
    float vfpacc2 = (float) vacc2 * vscale;
    vfpacc2 = math_max_f32(vfpacc2, voutput_min_less_zero_point);
    vfpacc2 = math_min_f32(vfpacc2, voutput_max_less_zero_point);
    vfpacc2 += vmagic_bias;
    vacc2 = (int32_t) float_as_uint32(vfpacc2) - vmagic_bias_less_output_zero_point;

    if (channels & 2) {
      *output++ += (int8_t) vacc0;
      *output++ += (int8_t) vacc1;
      vacc0 = vacc2;
    }
    if (channels & 1) {
      *output += (int8_t) vacc0;
    }
  }
}
