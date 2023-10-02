// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f32_gavgpool_cw_ukernel__scalar_u1(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(elements != 0);
  assert(elements % sizeof(float) == 0);
  assert(channels != 0);

  const float* i0 = input;

  const float vmultiplier = params->scalar.multiplier;
  const float voutput_max = params->scalar.output_max;
  const float voutput_min = params->scalar.output_min;

  while (channels != 0) {
    float vsum0 = 0.f;
    float vsum1 = 0.f;
    float vsum2 = 0.f;
    float vsum3 = 0.f;
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      vsum0 += i0[0];
      vsum1 += i0[1];
      vsum2 += i0[2];
      vsum3 += i0[3];

      i0 += 4;
      n -= 4 * sizeof(float);
    }

    while (n != 0) {
      vsum0 += *i0++;
      n -= sizeof(float);
    }

    float vout = ( (vsum0 + vsum1) + (vsum2 + vsum3) ) * vmultiplier;

    vout = math_min_f32(vout, voutput_max);
    vout = math_max_f32(vout, voutput_min);

    *output++ = vout;
    channels -= 1;
  }
}
