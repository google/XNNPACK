// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_spchw_ukernel_5x5p2__scalar(
    size_t m,
    size_t n,
    const float* input,
    const float* weights,
    float* output,
    size_t input_tuple_stride,
    size_t output_tuple_stride,
    size_t input_width_stride,
    size_t output_width_stride,
    const union xnn_f32_spchw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);

  const float params_max = params->scalar.max;
  const float params_min = params->scalar.min;

  const size_t input_width_increment_single = input_width_stride - n * input_tuple_stride;
  const size_t output_width_increment_single = output_width_stride - (n - 1) * output_tuple_stride;

  // No vertical padding.
  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width_stride);

  float* output0 = output;

  // this almost certainly will use too many scalar registers
  // hope the compiler is good at spilling...
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
  const float vw10 = weights[10];
  const float vw11 = weights[11];
  const float vw12 = weights[12];
  const float vw13 = weights[13];
  const float vw14 = weights[14];
  const float vw15 = weights[15];
  const float vw16 = weights[16];
  const float vw17 = weights[17];
  const float vw18 = weights[18];
  const float vw19 = weights[19];
  const float vw20 = weights[20];
  const float vw21 = weights[21];
  const float vw22 = weights[22];
  const float vw23 = weights[23];
  const float vw24 = weights[24];
  const float vw25 = weights[25];

  do {
    float vi0x0 = 0.0f;
    float vi1x0 = 0.0f;
    float vi2x0 = 0.0f;
    float vi3x0 = 0.0f;
    float vi4x0 = 0.0f;
    float vi0x1 = 0.0f;
    float vi1x1 = 0.0f;
    float vi2x1 = 0.0f;
    float vi3x1 = 0.0f;
    float vi4x1 = 0.0f;
    float vi0x2 = *i0; i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
    float vi1x2 = *i1; i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
    float vi2x2 = *i2; i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
    float vi3x2 = *i3; i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
    float vi4x2 = *i4; i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);

    float vi0x3;
    float vi1x3;
    float vi2x3;
    float vi3x3;
    float vi4x3;
    if XNN_LIKELY(n > 1) {
      vi0x3 = *i0; i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      vi1x3 = *i1; i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      vi2x3 = *i2; i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      vi3x3 = *i3; i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      vi4x3 = *i4; i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);
    }

    size_t k = n;
    for (; k > 2; k -= 1) {
      const float vi0x4 = *i0; i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const float vi1x4 = *i1; i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const float vi2x4 = *i2; i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);
      const float vi3x4 = *i3; i3 = (const float*) ((uintptr_t) i3 + input_tuple_stride);
      const float vi4x4 = *i4; i4 = (const float*) ((uintptr_t) i4 + input_tuple_stride);

      const float vrow0_accum = vw1  * vi0x0 + vw2  * vi0x1 + vw3  * vi0x2 + vw4  * vi0x3 + vw5  * vi0x4;
      vi0x0 = vi0x1;
      vi0x1 = vi0x2;
      vi0x2 = vi0x3;
      vi0x3 = vi0x4;
      const float vrow1_accum = vw6  * vi1x0 + vw7  * vi1x1 + vw8  * vi1x2 + vw9  * vi1x3 + vw10 * vi1x4;
      vi1x0 = vi1x1;
      vi1x1 = vi1x2;
      vi1x2 = vi1x3;
      vi1x3 = vi1x4;
      const float vrow2_accum = vw11 * vi2x0 + vw12 * vi2x1 + vw13 * vi2x2 + vw14 * vi2x3 + vw15 * vi2x4;
      vi2x0 = vi2x1;
      vi2x1 = vi2x2;
      vi2x2 = vi2x3;
      vi2x3 = vi2x4;
      const float vrow3_accum = vw16 * vi3x0 + vw17 * vi3x1 + vw18 * vi3x2 + vw19 * vi3x3 + vw20 * vi3x4;
      vi3x0 = vi3x1;
      vi3x1 = vi3x2;
      vi3x2 = vi3x3;
      vi3x3 = vi3x4;
      const float vrow4_accum = vw21 * vi4x0 + vw22 * vi4x1 + vw23 * vi4x2 + vw24 * vi4x3 + vw25 * vi4x4;
      vi4x0 = vi4x1;
      vi4x1 = vi4x2;
      vi4x2 = vi4x3;
      vi4x3 = vi4x4;

      float voutput = (vw0 + vrow0_accum) + (vrow1_accum + vrow2_accum) + (vrow3_accum + vrow4_accum);

      voutput = math_max_f32(voutput, params_min);
      voutput = math_min_f32(voutput, params_max);

      *output0 = voutput; output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
    }
    if XNN_LIKELY(k > 1) {
      const float vrow0_accum = vw1  * vi0x0 + vw2  * vi0x1 + vw3  * vi0x2 + vw4  * vi0x3;
      vi0x0 = vi0x1;
      vi0x1 = vi0x2;
      vi0x2 = vi0x3;
      const float vrow1_accum = vw6  * vi1x0 + vw7  * vi1x1 + vw8  * vi1x2 + vw9  * vi1x3;
      vi1x0 = vi1x1;
      vi1x1 = vi1x2;
      vi1x2 = vi1x3;
      const float vrow2_accum = vw11 * vi2x0 + vw12 * vi2x1 + vw13 * vi2x2 + vw14 * vi2x3;
      vi2x0 = vi2x1;
      vi2x1 = vi2x2;
      vi2x2 = vi2x3;
      const float vrow3_accum = vw16 * vi3x0 + vw17 * vi3x1 + vw18 * vi3x2 + vw19 * vi3x3;
      vi3x0 = vi3x1;
      vi3x1 = vi3x2;
      vi3x2 = vi3x3;
      const float vrow4_accum = vw21 * vi4x0 + vw22 * vi4x1 + vw23 * vi4x2 + vw24 * vi4x3;
      vi4x0 = vi4x1;
      vi4x1 = vi4x2;
      vi4x2 = vi4x3;

      float voutput = (vw0 + vrow0_accum) + (vrow1_accum + vrow2_accum) + (vrow3_accum + vrow4_accum);

      voutput = math_max_f32(voutput, params_min);
      voutput = math_min_f32(voutput, params_max);

      *output0 = voutput; output0 = (float*) ((uintptr_t) output0 + output_tuple_stride);
      k -= 1;
    }
    assert(k == 1);
    {
      const float vrow0_accum = vw1  * vi0x0 + vw2  * vi0x1 + vw3  * vi0x2;
      const float vrow1_accum = vw6  * vi1x0 + vw7  * vi1x1 + vw8  * vi1x2;
      const float vrow2_accum = vw11 * vi2x0 + vw12 * vi2x1 + vw13 * vi2x2;
      const float vrow3_accum = vw16 * vi3x0 + vw17 * vi3x1 + vw18 * vi3x2;
      const float vrow4_accum = vw21 * vi4x0 + vw22 * vi4x1 + vw23 * vi4x2;

      float voutput = (vw0 + vrow0_accum) + (vrow1_accum + vrow2_accum) + (vrow3_accum + vrow4_accum);

      voutput = math_max_f32(voutput, params_min);
      voutput = math_min_f32(voutput, params_max);

      *output0 = voutput;;
    }

    i0 = (const float*) ((uintptr_t) i0 + input_width_increment_single);
    i1 = (const float*) ((uintptr_t) i1 + input_width_increment_single);
    i2 = (const float*) ((uintptr_t) i2 + input_width_increment_single);
    i3 = (const float*) ((uintptr_t) i3 + input_width_increment_single);
    i4 = (const float*) ((uintptr_t) i4 + input_width_increment_single);
    output0 = (float*) ((uintptr_t) output0 + output_width_increment_single);
    m -= 1;
  } while (m > 0);
}
