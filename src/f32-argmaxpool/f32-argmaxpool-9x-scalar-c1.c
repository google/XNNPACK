// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/argmaxpool.h>
#include <xnnpack/math.h>


void xnn_f32_argmaxpool_ukernel_9x__scalar_c1(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment)
{
  assert(output_pixels != 0);
  assert(pooling_elements != 0);
  assert(pooling_elements <= 9);
  assert(channels != 0);

  do {
    const float* i0 = input[0];
    const float* i1 = input[1];
    const float* i2 = input[2];
    const float* i3 = input[3];
    const float* i4 = input[4];
    const float* i5 = input[5];
    const float* i6 = input[6];
    const float* i7 = input[7];
    const float* i8 = input[8];
    i0 = (const float*) ((uintptr_t) i0 + input_offset);
    i1 = (const float*) ((uintptr_t) i1 + input_offset);
    i2 = (const float*) ((uintptr_t) i2 + input_offset);
    i3 = (const float*) ((uintptr_t) i3 + input_offset);
    i4 = (const float*) ((uintptr_t) i4 + input_offset);
    i5 = (const float*) ((uintptr_t) i5 + input_offset);
    i6 = (const float*) ((uintptr_t) i6 + input_offset);
    i7 = (const float*) ((uintptr_t) i7 + input_offset);
    i8 = (const float*) ((uintptr_t) i8 + input_offset);
    if (pooling_elements < 2) {
      i1 = i0;
    }
    if (pooling_elements <= 2) {
      i2 = i0;
    }
    if (pooling_elements < 4) {
      i3 = i0;
    }
    if (pooling_elements <= 4) {
      i4 = i0;
    }
    if (pooling_elements < 6) {
      i5 = i0;
    }
    if (pooling_elements <= 6) {
      i6 = i0;
    }
    if (pooling_elements < 8) {
      i7 = i0;
    }
    if (pooling_elements <= 8) {
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

      float vmax = vi0;
      uint32_t vidx = 0;

      if (vi1 > vmax) {
        vmax = vi1;
        vidx = 1;
      }

      if (vi2 > vmax) {
        vmax = vi2;
        vidx = 2;
      }

      if (vi3 > vmax) {
        vmax = vi3;
        vidx = 3;
      }

      if (vi4 > vmax) {
        vmax = vi4;
        vidx = 4;
      }

      if (vi5 > vmax) {
        vmax = vi5;
        vidx = 5;
      }

      if (vi6 > vmax) {
        vmax = vi6;
        vidx = 6;
      }

      if (vi7 > vmax) {
        vmax = vi7;
        vidx = 7;
      }

      if (vi8 > vmax) {
        vmax = vi8;
        vidx = 8;
      }

      *output++ = vmax;
      *index++ = vidx;
    } while (--c != 0);
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
