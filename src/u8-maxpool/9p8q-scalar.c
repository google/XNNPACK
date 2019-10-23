// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/maxpool.h>


void xnn_u8_maxpool_ukernel_9p8q__scalar(
    size_t n,
    size_t ks,
    size_t kc,
    const uint8_t** input,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_u8_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(ks != 0);
  assert(kc != 0);

  const uint8_t voutput_max = params->scalar.max;
  const uint8_t voutput_min = params->scalar.min;
  do {
    uint8_t* o = output;
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
      if (ks < 2) {
        i1 = i0;
      }
      if (ks <= 2) {
        i2 = i0;
      }
      if (ks < 4) {
        i3 = i0;
      }
      if (ks <= 4) {
        i4 = i0;
      }
      if (ks < 6) {
        i5 = i0;
      }
      if (ks <= 6) {
        i6 = i0;
      }
      if (ks < 8) {
        i7 = i0;
      }
      if (ks <= 8) {
        i8 = i0;
      }

      size_t k = kc;
      do {
        const uint8_t vi0 = *i0++;
        const uint8_t vi1 = *i1++;
        const uint8_t vi2 = *i2++;
        const uint8_t vi3 = *i3++;
        const uint8_t vi4 = *i4++;
        const uint8_t vi5 = *i5++;
        const uint8_t vi6 = *i6++;
        const uint8_t vi7 = *i7++;
        const uint8_t vi8 = *i8++;

        const uint8_t vmax01 = vi0 > vi1 ? vi0 : vi1;
        const uint8_t vmax23 = vi2 > vi3 ? vi2 : vi3;
        const uint8_t vmax45 = vi4 > vi5 ? vi4 : vi5;
        const uint8_t vmax67 = vi6 > vi7 ? vi6 : vi7;
        const uint8_t vmax018 = vmax01 > vi8 ? vmax01 : vi8;

        const uint8_t vmax2345 = vmax23 > vmax45 ? vmax23 : vmax45;
        const uint8_t vmax01678 = vmax018 > vmax67 ? vmax018 : vmax67;

        uint8_t vout = vmax2345 > vmax01678 ? vmax2345 : vmax01678;
        vout = vout > voutput_max ? voutput_max : vout;
        vout = vout < voutput_min ? voutput_min : vout;

        *o++ = vout;
      } while (--k != 0);
    }

    for (ptrdiff_t m = (ptrdiff_t) ks - 9; m > 0; m -= 8) {
      const uint8_t* i0 = *input++;
      const uint8_t* i1 = *input++;
      const uint8_t* i2 = *input++;
      const uint8_t* i3 = *input++;
      const uint8_t* i4 = *input++;
      const uint8_t* i5 = *input++;
      const uint8_t* i6 = *input++;
      const uint8_t* i7 = *input++;
      if (m < 2) {
        i1 = i0;
      }
      if (m <= 2) {
        i2 = i0;
      }
      if (m < 4) {
        i3 = i0;
      }
      if (m <= 4) {
        i4 = i0;
      }
      if (m < 6) {
        i5 = i0;
      }
      if (m <= 6) {
        i6 = i0;
      }
      if (m < 8) {
        i7 = i0;
      }

      o = output;
      size_t k = kc;
      do {
        const uint8_t vi0 = *i0++;
        const uint8_t vi1 = *i1++;
        const uint8_t vi2 = *i2++;
        const uint8_t vi3 = *i3++;
        const uint8_t vi4 = *i4++;
        const uint8_t vi5 = *i5++;
        const uint8_t vi6 = *i6++;
        const uint8_t vi7 = *i7++;
        const uint8_t vi8 = *o;

        const uint8_t vmax01 = vi0 > vi1 ? vi0 : vi1;
        const uint8_t vmax23 = vi2 > vi3 ? vi2 : vi3;
        const uint8_t vmax45 = vi4 > vi5 ? vi4 : vi5;
        const uint8_t vmax67 = vi6 > vi7 ? vi6 : vi7;
        const uint8_t vmax018 = vmax01 > vi8 ? vmax01 : vi8;

        const uint8_t vmax2345 = vmax23 > vmax45 ? vmax23 : vmax45;
        const uint8_t vmax01678 = vmax018 > vmax67 ? vmax018 : vmax67;

        uint8_t vout = vmax2345 > vmax01678 ? vmax2345 : vmax01678;
        vout = vout > voutput_max ? voutput_max : vout;
        vout = vout < voutput_min ? voutput_min : vout;

        *o++ = vout;
      } while (--k != 0);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    output = (uint8_t*) ((uintptr_t) o + output_increment);
  } while (--n != 0);
}
