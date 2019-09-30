// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/scalar-utils.h>
#include <xnnpack/avgpool.h>


void xnn_q8_avgpool_ukernel_mp9p8q__scalar(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_q8_avgpool_params params[restrict static 1])
{
  assert(n != 0);
  assert(ks > 9);
  assert(kc != 0);

  const int32_t vbias = params->scalar.bias;
  const int32_t vmultiplier = params->scalar.multiplier;
  const int64_t vrounding = params->scalar.rounding;
  const uint32_t vshift = params->scalar.right_shift;
  const int32_t voutput_min = params->scalar.output_min_less_zero_point;
  const int32_t voutput_max = params->scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;
  do {
    // First pass.
    {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      const uint8_t* i8 = *input++;

      int32_t* b = buffer;
      size_t k = kc;
      do {
        const uint32_t vi0 = (uint32_t) *i0++;
        const uint32_t vi1 = (uint32_t) *i1++;
        const uint32_t vi2 = (uint32_t) *i2++;
        const uint32_t vi3 = (uint32_t) *i3++;
        const uint32_t vi4 = (uint32_t) *i4++;
        const uint32_t vi5 = (uint32_t) *i5++;
        const uint32_t vi6 = (uint32_t) *i6++;
        const uint32_t vi7 = (uint32_t) *i7++;
        const uint32_t vi8 = (uint32_t) *i8++;

        const uint32_t vsum01 = vi0 + vi1;
        const uint32_t vsum23 = vi2 + vi3;
        const uint32_t vsum45 = vi4 + vi5;
        const uint32_t vsum67 = vi6 + vi7;
        const uint32_t vsum018 = vsum01 + vi8;
        const uint32_t vsum2345 = vsum23 + vsum45;
        const uint32_t vsum01678 = vsum018 + vsum67;
        int32_t vacc = vbias + (int32_t) vsum2345;
        vacc += (int32_t) vsum01678;
        *b++ = vacc;
      } while (--k != 0);
    }

    size_t m = ks;
    // Intermediate passes.
    for (m -= 9; m > 8; m -= 8) {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;

      int32_t* b = buffer;
      size_t k = kc;
      do {
        int32_t vacc = *b;

        const uint32_t vi0 = (uint32_t) *i0++;
        const uint32_t vi1 = (uint32_t) *i1++;
        const uint32_t vi2 = (uint32_t) *i2++;
        const uint32_t vi3 = (uint32_t) *i3++;
        const uint32_t vi4 = (uint32_t) *i4++;
        const uint32_t vi5 = (uint32_t) *i5++;
        const uint32_t vi6 = (uint32_t) *i6++;
        const uint32_t vi7 = (uint32_t) *i7++;

        const uint32_t vsum01 = vi0 + vi1;
        const uint32_t vsum23 = vi2 + vi3;
        const uint32_t vsum45 = vi4 + vi5;
        const uint32_t vsum67 = vi6 + vi7;
        const uint32_t vsum0123 = vsum01 + vsum23;
        const uint32_t vsum4567 = vsum45 + vsum67;
        vacc += (int32_t) vsum0123;
        vacc += (int32_t) vsum4567;

        *b++ = vacc;
      } while (--k != 0);
    }

    // Last pass.
    {
      const uint8_t* i0 = input[0];
      const uint8_t* i1 = input[1];
      const uint8_t* i2 = input[2];
      const uint8_t* i3 = input[3];
      const uint8_t* i4 = input[4];
      const uint8_t* i5 = input[5];
      const uint8_t* i6 = input[6];
      const uint8_t* i7 = input[7];
      input = (const uint8_t**) ((uintptr_t) input + input_increment);
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
      int32_t* b = buffer;
      do {
        int32_t vacc = *b++;

        const uint32_t vi0 = (uint32_t) *i0++;
        const uint32_t vi1 = (uint32_t) *i1++;
        const uint32_t vi2 = (uint32_t) *i2++;
        const uint32_t vi3 = (uint32_t) *i3++;
        const uint32_t vi4 = (uint32_t) *i4++;
        const uint32_t vi5 = (uint32_t) *i5++;
        const uint32_t vi6 = (uint32_t) *i6++;
        const uint32_t vi7 = (uint32_t) *i7++;

        const uint32_t vsum01 = vi0 + vi1;
        const uint32_t vsum23 = vi2 + vi3;
        const uint32_t vsum45 = vi4 + vi5;
        const uint32_t vsum67 = vi6 + vi7;
        const uint32_t vsum0123 = vsum01 + vsum23;
        const uint32_t vsum4567 = vsum45 + vsum67;
        vacc += (int32_t) vsum0123;
        vacc += (int32_t) vsum4567;

        const int64_t vproduct = (int64_t) vacc * (int64_t) vmultiplier;
        const int64_t vadjusted_product = vproduct - (int64_t) (vacc < 0);
        int32_t vout = (int32_t) asr_s64(vadjusted_product + vrounding, vshift);
        vout = vout < voutput_min ? voutput_min : vout;
        vout = vout > voutput_max ? voutput_max : vout;
        vout += voutput_zero_point;

        *output++ = (uint8_t) vout;
      } while (--k != 0);
    }
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--n != 0);
}
