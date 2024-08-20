// Auto-generated file. Do not edit!
//   Template: src/f32-rdsum/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"


void xnn_f32_rdsum_ukernel_7p7x__scalar_c4(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vscale = params->scalar.scale;
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  size_t input_increment = 7 * input_stride;
  for (; channels >= 4; channels -= 4) {
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
    const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
    const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
    const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
    const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
    const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
    float vacc0 = 0.f;
    float vacc1 = 0.f;
    float vacc2 = 0.f;
    float vacc3 = 0.f;

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
      vacc0 += i0[0];
      vacc1 += i0[1];
      vacc2 += i0[2];
      vacc3 += i0[3];
      vacc0 += i1[0];
      vacc1 += i1[1];
      vacc2 += i1[2];
      vacc3 += i1[3];
      vacc0 += i2[0];
      vacc1 += i2[1];
      vacc2 += i2[2];
      vacc3 += i2[3];
      vacc0 += i3[0];
      vacc1 += i3[1];
      vacc2 += i3[2];
      vacc3 += i3[3];
      vacc0 += i4[0];
      vacc1 += i4[1];
      vacc2 += i4[2];
      vacc3 += i4[3];
      vacc0 += i5[0];
      vacc1 += i5[1];
      vacc2 += i5[2];
      vacc3 += i5[3];
      vacc0 += i6[0];
      vacc1 += i6[1];
      vacc2 += i6[2];
      vacc3 += i6[3];
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = vacc0 * vscale;
    vacc0 = math_max_f32(vacc0, vmin);
    vacc0 = math_min_f32(vacc0, vmax);
    vacc1 = vacc1 * vscale;
    vacc1 = math_max_f32(vacc1, vmin);
    vacc1 = math_min_f32(vacc1, vmax);
    vacc2 = vacc2 * vscale;
    vacc2 = math_max_f32(vacc2, vmin);
    vacc2 = math_min_f32(vacc2, vmax);
    vacc3 = vacc3 * vscale;
    vacc3 = math_max_f32(vacc3, vmin);
    vacc3 = math_min_f32(vacc3, vmax);

    *output++ += vacc0;
    *output++ += vacc1;
    *output++ += vacc2;
    *output++ += vacc3;

    input = (const float*) ((uintptr_t) input + 4 * sizeof(float));
  }
  if (channels != 0) {
    size_t input_increment = 7 * input_stride;
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
    const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
    const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
    const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
    const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
    const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
    float vacc0 = 0.f;
    float vacc1 = 0.f;
    float vacc2 = 0.f;

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
      vacc0 += i0[0];
      vacc1 += i0[1];
      vacc2 += i0[2];
      vacc0 += i1[0];
      vacc1 += i1[1];
      vacc2 += i1[2];
      vacc0 += i2[0];
      vacc1 += i2[1];
      vacc2 += i2[2];
      vacc0 += i3[0];
      vacc1 += i3[1];
      vacc2 += i3[2];
      vacc0 += i4[0];
      vacc1 += i4[1];
      vacc2 += i4[2];
      vacc0 += i5[0];
      vacc1 += i5[1];
      vacc2 += i5[2];
      vacc0 += i6[0];
      vacc1 += i6[1];
      vacc2 += i6[2];
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = vacc0 * vscale;
    vacc0 = math_max_f32(vacc0, vmin);
    vacc0 = math_min_f32(vacc0, vmax);
    vacc1 = vacc1 * vscale;
    vacc1 = math_max_f32(vacc1, vmin);
    vacc1 = math_min_f32(vacc1, vmax);
    vacc2 = vacc2 * vscale;
    vacc2 = math_max_f32(vacc2, vmin);
    vacc2 = math_min_f32(vacc2, vmax);

    if (channels & 2) {
      *output++ += vacc0;
      *output++ += vacc1;
      vacc0 = vacc2;
    }
    if (channels & 1) {
      *output++ += vacc0;
    }
  }
}
