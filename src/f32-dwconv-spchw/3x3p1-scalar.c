// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_spchw_ukernel_3x3p1__scalar(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    size_t input_tuple_stride,
    size_t output_tuple_stride,
    size_t input_width_stride,
    size_t output_width_stride,
    const union xnn_f32_spchw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_width != 0);
  assert(input_height != 0);
  assert(padding_top == 1);

  const size_t padded_input_height = input_height + padding_top + 1 /* padding_bottom */;
  const size_t output_height = padded_input_height - 3 + 1;

  const size_t input_width_decrement = input_width * input_tuple_stride;
  const size_t input_width_increment = input_width_stride - input_width_decrement;
  const size_t output_width_increment = output_width_stride - (input_width - 1) * output_tuple_stride;

  const float params_min = params->scalar.min;
  const float params_max = params->scalar.max;

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width_stride);

  if (input_height == 1) {
    i2 = zero;
  }

  float* output0 = output;

  const float vw0 = weights[0];
  const float vw1 = weights[1];
  const float vw2 = weights[2];
  const float vw3 = weights[3];
  const float vw4 = weights[4];
  const float vw5 = weights[5];
  const float vw6 = weights[6];
  const float vw7 = weights[7];
  const float vw8 = weights[8];
  const float vw9 = weights[9];

  size_t m = output_height;
  while (m > 0) {
    float vi0x0 = 0.0f;
    float vi1x0 = 0.0f;
    float vi2x0 = 0.0f;
    float vi0x1 = *i0; i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
    float vi1x1 = *i1; i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
    float vi2x1 = *i2; i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);

    size_t k = input_width;
    for (; k > 1; k--) {
      const float vi0x2 = *i0; i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const float vi1x2 = *i1; i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const float vi2x2 = *i2; i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);

      const float vrow0_accum = vw1 * vi0x0 + vw2 * vi0x1 + vw3 * vi0x2;
      vi0x0 = vi0x1;
      vi0x1 = vi0x2;
      const float vrow1_accum = vw4 * vi1x0 + vw5 * vi1x1 + vw6 * vi1x2;
      vi1x0 = vi1x1;
      vi1x1 = vi1x2;
      const float vrow2_accum = vw7 * vi2x0 + vw8 * vi2x1 + vw9 * vi2x2;
      vi2x0 = vi2x1;
      vi2x1 = vi2x2;

      float voutput = (vw0 + vrow0_accum) + (vrow1_accum + vrow2_accum);

      voutput = math_max_f32(voutput, params_min);
      voutput = math_min_f32(voutput, params_max);

      *output0 = voutput; output0 = (float *) ((uintptr_t) output0 + output_tuple_stride);
    }
    // Always process the last pixel separately to account for right edge.
    assert(k == 1);
    {
      const float vrow0_accum = vw1 * vi0x0 + vw2 * vi0x1;
      const float vrow1_accum = vw4 * vi1x0 + vw5 * vi1x1;
      const float vrow2_accum = vw7 * vi2x0 + vw8 * vi2x1;

      float voutput = (vw0 + vrow0_accum) + (vrow1_accum + vrow2_accum);

      voutput = math_max_f32(voutput, params_min);
      voutput = math_min_f32(voutput, params_max);

      *output0 = voutput;
    }

    i0 = (const float*) ((uintptr_t) i1 - input_width_decrement);
    i1 = (const float*) ((uintptr_t) i1 + input_width_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_width_increment);
    output0 = (float*) ((uintptr_t) output0 + output_width_increment);
    m--;
    if (m == 1) {
      i2 = zero;
    }
  }
}
