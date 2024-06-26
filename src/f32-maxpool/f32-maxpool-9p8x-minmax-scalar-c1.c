// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/maxpool.h"
#include "xnnpack/math.h"


void xnn_f32_maxpool_minmax_ukernel_9p8x__scalar_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;
  do {
    float* o = output;
    {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *i8++;

        const float vmax01 = math_max_f32(vi0, vi1);
        const float vmax23 = math_max_f32(vi2, vi3);
        const float vmax45 = math_max_f32(vi4, vi5);
        const float vmax67 = math_max_f32(vi6, vi7);
        const float vmax018 = math_max_f32(vmax01, vi8);

        const float vmax2345 = math_max_f32(vmax23, vmax45);
        const float vmax01678 = math_max_f32(vmax018, vmax67);
        float vout = math_max_f32(vmax2345, vmax01678);
        vout = math_max_f32(vout, voutput_min);
        vout = math_min_f32(vout, voutput_max);

        *o++ = vout;
      } while (--c != 0);
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vi8 = *o;

        const float vmax01 = math_max_f32(vi0, vi1);
        const float vmax23 = math_max_f32(vi2, vi3);
        const float vmax45 = math_max_f32(vi4, vi5);
        const float vmax67 = math_max_f32(vi6, vi7);
        const float vmax018 = math_max_f32(vmax01, vi8);

        const float vmax2345 = math_max_f32(vmax23, vmax45);
        const float vmax01678 = math_max_f32(vmax018, vmax67);
        float vout = math_max_f32(vmax2345, vmax01678);
        vout = math_max_f32(vout, voutput_min);
        vout = math_min_f32(vout, voutput_max);

        *o++ = vout;
      } while (--c != 0);
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
