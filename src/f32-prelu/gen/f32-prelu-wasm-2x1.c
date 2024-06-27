// Auto-generated file. Do not edit!
//   Template: src/f32-prelu/wasm.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/prelu.h"


void xnn_f32_prelu_ukernel__wasm_2x1(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float vzero = 0.0f;
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    do {
      const float vw = *w++;

      float vi0 = *i0++;
      float vi1 = *i1++;

      float vacc0 = __builtin_wasm_max_f32(vi0, vzero);
      vi0 = __builtin_wasm_min_f32(vi0, vzero);
      float vacc1 = __builtin_wasm_max_f32(vi1, vzero);
      vi1 = __builtin_wasm_min_f32(vi1, vzero);

      vacc0 += vi0 * vw;
      vacc1 += vi1 * vw;

      *o0++ = vacc0;
      *o1++ = vacc1;

      c -= sizeof(float);
    } while (c != 0);
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}
