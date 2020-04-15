// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/scalar-utils.h>
#include <xnnpack/gavgpool.h>


void xnn_q8_gavgpool_minmax_ukernel_7x__scalar_c1(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union xnn_q8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  if (rows < 2) {
    i1 = zero;
  }
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  if (rows <= 2) {
    i2 = zero;
  }
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  if (rows < 4) {
    i3 = zero;
  }
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  if (rows <= 4) {
    i4 = zero;
  }
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  if (rows < 6) {
    i5 = zero;
  }
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  if (rows <= 6) {
    i6 = zero;
  }

  const int32_t vbias = params->scalar.bias;
  const int32_t vmultiplier = params->scalar.multiplier;
  const int64_t vrounding = params->scalar.rounding;
  const uint32_t vshift = params->scalar.right_shift;
  const int32_t voutput_min = params->scalar.output_min_less_zero_point;
  const int32_t voutput_max = params->scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;
  do {
    const uint32_t vi0 = (uint32_t) *i0++;
    const uint32_t vi1 = (uint32_t) *i1++;
    const uint32_t vi2 = (uint32_t) *i2++;
    const uint32_t vi3 = (uint32_t) *i3++;
    const uint32_t vi4 = (uint32_t) *i4++;
    const uint32_t vi5 = (uint32_t) *i5++;
    const uint32_t vi6 = (uint32_t) *i6++;

    const uint32_t vsum01 = vi0 + vi1;
    const uint32_t vsum23 = vi2 + vi3;
    const uint32_t vsum45 = vi4 + vi5;

    const uint32_t vsum016 = vsum01 + vi6;
    const uint32_t vsum2345 = vsum23 + vsum45;

    const uint32_t vsum = vsum016 + vsum2345;
    const int32_t vacc = vbias + (int32_t) vsum;

    const int64_t vproduct = (int64_t) vacc * (int64_t) vmultiplier;
    const int64_t vadjusted_product = vproduct - (int64_t) (vacc < 0);
    int32_t vout = (int32_t) asr_s64(vadjusted_product + vrounding, vshift);
    vout = vout < voutput_min ? voutput_min : vout;
    vout = vout > voutput_max ? voutput_max : vout;
    vout += voutput_zero_point;

    *output++ = (uint8_t) vout;
  } while (--channels != 0);
}
