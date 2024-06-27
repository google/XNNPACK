// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv_minmax_ukernel_4p1c__scalar_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);

      const float vi1 = *i1++;
      const float vk1 = w[2];
      float vacc0p1 = vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);

      w += 5;

      vacc0p0 += vacc0p1;

      float vacc0 = math_max_f32(vacc0p0, vmin);
      vacc0 = math_min_f32(vacc0, vmax);
      *output++ = vacc0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
