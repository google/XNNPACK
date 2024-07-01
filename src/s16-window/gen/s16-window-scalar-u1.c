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


void xnn_s16_window_ukernel__scalar_u1(
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
  } while (--rows != 0);
}
