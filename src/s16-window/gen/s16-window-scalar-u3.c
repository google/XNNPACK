// Auto-generated file. Do not edit!
//   Template: src/s16-window/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/math.h"
#include "xnnpack/window.h"


void xnn_s16_window_ukernel__scalar_u3(
    size_t rows,
    size_t channels,
    const int16_t* input,
    const int16_t* weights,
    int16_t* output,
    uint32_t shift)
{
  assert(rows > 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(weights != NULL);
  assert(output != NULL);
  assert(shift < 32);

  do {
    size_t c = channels;
    const int16_t* w = weights;
    for (; c >= 3 * sizeof(int16_t); c -= 3 * sizeof(int16_t)) {
      const int16_t vi0 = input[0];
      const int16_t vi1 = input[1];
      const int16_t vi2 = input[2];
      input += 3;

      const int16_t w0 = w[0];
      const int16_t w1 = w[1];
      const int16_t w2 = w[2];
      w += 3;

      int32_t vout0 = (int32_t) vi0 * (int32_t) w0;
      int32_t vout1 = (int32_t) vi1 * (int32_t) w1;
      int32_t vout2 = (int32_t) vi2 * (int32_t) w2;

      vout0 = math_asr_s32(vout0, shift);
      vout1 = math_asr_s32(vout1, shift);
      vout2 = math_asr_s32(vout2, shift);

      vout0 = math_max_s32(vout0, INT16_MIN);
      vout1 = math_max_s32(vout1, INT16_MIN);
      vout2 = math_max_s32(vout2, INT16_MIN);

      vout0 = math_min_s32(vout0, INT16_MAX);
      vout1 = math_min_s32(vout1, INT16_MAX);
      vout2 = math_min_s32(vout2, INT16_MAX);

      output[0] = (int16_t) vout0;
      output[1] = (int16_t) vout1;
      output[2] = (int16_t) vout2;

      output += 3;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        const int32_t vi = (int32_t) *input++;
        const int32_t vw = (int32_t) *w++;
        int32_t vout = vi * vw;
        vout = math_asr_s32(vout, shift);
        vout = math_max_s32(vout, INT16_MIN);
        vout = math_min_s32(vout, INT16_MAX);
        *output++ = (int16_t) vout;
        c -= sizeof(int16_t);
      } while (c != 0);
    }
  } while (--rows != 0);
}
