// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/pavgpool.h>
#include <xnnpack/math.h>


void xnn_f32_pavgpool_ukernel_9p8x__wasm_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    const float* zero,
    const float* multiplier,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  do {
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

      float* b = buffer;
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

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum018 = vsum01 + vi8;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum01678 = vsum018 + vsum67;
        const float vsum = vsum2345 + vsum01678;

        *b++ = vsum;
      } while (--c != 0);
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;

      float* b = buffer;
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
        const float vacc = *b;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum01a = vsum01 + vacc;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum0167a = vsum01a + vsum67;
        const float vsum = vsum2345 + vsum0167a;

        *b++ = vsum;
      } while (--c != 0);
    }

    {
      const float* i0 = input[0];
      const float* i1 = input[1];
      const float* i2 = input[2];
      const float* i3 = input[3];
      const float* i4 = input[4];
      const float* i5 = input[5];
      const float* i6 = input[6];
      const float* i7 = input[7];
      input = (const float**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      if (k <= 2) {
        i2 = zero;
      }
      if (k < 4) {
        i3 = zero;
      }
      if (k <= 4) {
        i4 = zero;
      }
      if (k < 6) {
        i5 = zero;
      }
      if (k <= 6) {
        i6 = zero;
      }
      if (k != 8) {
        i7 = zero;
      }

      const float vmultiplier = *multiplier++;

      size_t c = channels;
      float* b = buffer;
      do {
        const float vi0 = *i0++;
        const float vi1 = *i1++;
        const float vi2 = *i2++;
        const float vi3 = *i3++;
        const float vi4 = *i4++;
        const float vi5 = *i5++;
        const float vi6 = *i6++;
        const float vi7 = *i7++;
        const float vacc = *b++;

        const float vsum01 = vi0 + vi1;
        const float vsum23 = vi2 + vi3;
        const float vsum45 = vi4 + vi5;
        const float vsum67 = vi6 + vi7;
        const float vsum01a = vsum01 + vacc;
        const float vsum2345 = vsum23 + vsum45;
        const float vsum0167a = vsum01a + vsum67;
        const float vsum = vsum2345 + vsum0167a;

        float vout = vsum * vmultiplier;
        vout = __builtin_wasm_max_f32(vout, voutput_min);
        vout = __builtin_wasm_min_f32(vout, voutput_max);

        *output++ = vout;
      } while (--c != 0);
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
