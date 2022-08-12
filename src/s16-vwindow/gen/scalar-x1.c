// Auto-generated file. Do not edit!
//   Template: src/s16-vwindow/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/vwindow.h>


void xnn_s16_vwindow_ukernel__scalar_x1(
    size_t rows,
    size_t batch_size,
    const int16_t* input,
    const int16_t* weights,
    uint32_t shift,
    int16_t* output) {

  assert(rows > 0);
  assert(batch_size != 0);
  assert(input != NULL);
  assert(weights != NULL);
  assert(shift < 32);
  assert(output != NULL);

  do {
    size_t n = batch_size;
    const int16_t* w = weights;

    if XNN_UNLIKELY(n != 0) {
      do {
        const int32_t vi = (int32_t) *input++;
        const int32_t vw = (int32_t) *w++;
        int32_t vout = vi * vw;
        vout = math_asr_s32(vout, shift);
        vout = math_max_s32(vout, INT16_MIN);
        vout = math_min_s32(vout, INT16_MAX);
        *output++ = (int16_t) vout;
      } while (--n != 0);
    }
  } while (--rows != 0);
}
