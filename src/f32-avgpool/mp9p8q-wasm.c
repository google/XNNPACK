// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/avgpool.h>
#include <xnnpack/math.h>


void xnn_f32_avgpool_ukernel_mp9p8q__wasm(
    size_t n,
    size_t ks,
    size_t kc,
    const float** input,
    const float* zero,
    float* buffer,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_avgpool_params params[restrict static 1])
{
  assert(n != 0);
  assert(ks > 9);
  assert(kc != 0);

  const float vmultiplier = params->scalar.multiplier;
  const float voutput_min = params->scalar.output_min;
  const float voutput_max = params->scalar.output_max;

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
      size_t k = kc;
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
      } while (--k != 0);
    }

    size_t m = ks;
    for (m -= 9; m > 8; m -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;

      float* b = buffer;
      size_t k = kc;
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
      } while (--k != 0);
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
      if (m < 2) {
        i1 = zero;
      }
      if (m <= 2) {
        i2 = zero;
      }
      if (m < 4) {
        i3 = zero;
      }
      if (m <= 4) {
        i4 = zero;
      }
      if (m < 6) {
        i5 = zero;
      }
      if (m <= 6) {
        i6 = zero;
      }
      if (m != 8) {
        i7 = zero;
      }

      size_t k = kc;
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
      } while (--k != 0);
    }
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--n != 0);
}
