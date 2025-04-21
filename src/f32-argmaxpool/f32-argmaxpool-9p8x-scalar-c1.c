// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "src/xnnpack/argmaxpool.h"


void xnn_f32_argmaxpool_ukernel_9p8x__scalar_c1(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    size_t input_pixel_stride,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    size_t index_increment)
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(channels != 0);

  do {
    // Accumulators start out null, after each pass the accumulator is set to
    // the output.
    const float* ab = NULL;
    const uint32_t* ib = NULL;
    const float** id = input;

    uint32_t vidx0 = 0;
    ptrdiff_t k = pooling_elements;
    for (; k > 0; k -= 9) {
      const float* i0 = *id++;
      const float* i1 = 1 < k ? *id++ : i0;
      const float* i2 = 2 < k ? *id++ : i0;
      const float* i3 = 3 < k ? *id++ : i0;
      const float* i4 = 4 < k ? *id++ : i0;
      const float* i5 = 5 < k ? *id++ : i0;
      const float* i6 = 6 < k ? *id++ : i0;
      const float* i7 = 7 < k ? *id++ : i0;
      const float* i8 = 8 < k ? *id++ : i0;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);

      float* o = output;
      uint32_t* i = index;
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

        float vmax = ab ? *ab++ : -INFINITY;
        uint32_t vidx = ib ? *ib++ : 0;

        if (vi0 > vmax) {
          vmax = vi0;
          vidx = vidx0;
        }

        if (vi1 > vmax) {
          vmax = vi1;
          vidx = vidx0 + 1;
        }

        if (vi2 > vmax) {
          vmax = vi2;
          vidx = vidx0 + 2;
        }

        if (vi3 > vmax) {
          vmax = vi3;
          vidx = vidx0 + 3;
        }

        if (vi4 > vmax) {
          vmax = vi4;
          vidx = vidx0 + 4;
        }

        if (vi5 > vmax) {
          vmax = vi5;
          vidx = vidx0 + 5;
        }

        if (vi6 > vmax) {
          vmax = vi6;
          vidx = vidx0 + 6;
        }

        if (vi7 > vmax) {
          vmax = vi7;
          vidx = vidx0 + 7;
        }

        if (vi8 > vmax) {
          vmax = vi8;
          vidx = vidx0 + 8;
        }

        *o++ = vmax;
        *i++ = vidx;
      } while (--c != 0);
      vidx0 += 9;
      ab = output;
      ib = index;
    }

    input = (const float**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
    index = (uint32_t*) ((uintptr_t) index + index_increment);
  } while (--output_pixels != 0);
}
