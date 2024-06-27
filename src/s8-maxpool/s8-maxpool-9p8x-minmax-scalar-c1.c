// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/maxpool.h"


void xnn_s8_maxpool_minmax_ukernel_9p8x__scalar_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const int32_t voutput_max = params->scalar.max;
  const int32_t voutput_min = params->scalar.min;
  do {
    int8_t* o = output;
    {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      const int8_t* i8 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
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
        const int32_t vi0 = (int32_t) *i0++;
        const int32_t vi1 = (int32_t) *i1++;
        const int32_t vi2 = (int32_t) *i2++;
        const int32_t vi3 = (int32_t) *i3++;
        const int32_t vi4 = (int32_t) *i4++;
        const int32_t vi5 = (int32_t) *i5++;
        const int32_t vi6 = (int32_t) *i6++;
        const int32_t vi7 = (int32_t) *i7++;
        const int32_t vi8 = (int32_t) *i8++;

        const int32_t vmax01 = math_max_s32(vi0, vi1);
        const int32_t vmax23 = math_max_s32(vi2, vi3);
        const int32_t vmax45 = math_max_s32(vi4, vi5);
        const int32_t vmax67 = math_max_s32(vi6, vi7);
        const int32_t vmax018 = math_max_s32(vmax01, vi8);

        const int32_t vmax2345 = math_max_s32(vmax23, vmax45);
        const int32_t vmax01678 = math_max_s32(vmax018, vmax67);

        int32_t vout = math_max_s32(vmax2345, vmax01678);
        vout = math_min_s32(vout, voutput_max);
        vout = math_max_s32(vout, voutput_min);

        *o++ = (int8_t) vout;
      } while (--c != 0);
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
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
        const int32_t vi0 = (int32_t) *i0++;
        const int32_t vi1 = (int32_t) *i1++;
        const int32_t vi2 = (int32_t) *i2++;
        const int32_t vi3 = (int32_t) *i3++;
        const int32_t vi4 = (int32_t) *i4++;
        const int32_t vi5 = (int32_t) *i5++;
        const int32_t vi6 = (int32_t) *i6++;
        const int32_t vi7 = (int32_t) *i7++;
        const int32_t vi8 = (int32_t) *o;

        const int32_t vmax01 = math_max_s32(vi0, vi1);
        const int32_t vmax23 = math_max_s32(vi2, vi3);
        const int32_t vmax45 = math_max_s32(vi4, vi5);
        const int32_t vmax67 = math_max_s32(vi6, vi7);
        const int32_t vmax018 = math_max_s32(vmax01, vi8);

        const int32_t vmax2345 = math_max_s32(vmax23, vmax45);
        const int32_t vmax01678 = math_max_s32(vmax018, vmax67);

        int32_t vout = math_max_s32(vmax2345, vmax01678);
        vout = math_min_s32(vout, voutput_max);
        vout = math_max_s32(vout, voutput_min);

        *o++ = (int8_t) vout;
      } while (--c != 0);
    }
    input = (const int8_t**) ((uintptr_t) input + input_increment);
    output = (int8_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
