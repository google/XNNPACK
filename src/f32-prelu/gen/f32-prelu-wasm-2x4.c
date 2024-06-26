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


void xnn_f32_prelu_ukernel__wasm_2x4(
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
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float vw0 = w[0];
      const float vw1 = w[1];
      const float vw2 = w[2];
      const float vw3 = w[3];

      float vi0x0 = i0[0];
      float vi0x1 = i0[1];
      float vi0x2 = i0[2];
      float vi0x3 = i0[3];
      i0 += 4;
      float vi1x0 = i1[0];
      float vi1x1 = i1[1];
      float vi1x2 = i1[2];
      float vi1x3 = i1[3];
      i1 += 4;

      float vacc0x0 = __builtin_wasm_max_f32(vi0x0, vzero);
      vi0x0 = __builtin_wasm_min_f32(vi0x0, vzero);
      float vacc0x1 = __builtin_wasm_max_f32(vi0x1, vzero);
      vi0x1 = __builtin_wasm_min_f32(vi0x1, vzero);
      float vacc0x2 = __builtin_wasm_max_f32(vi0x2, vzero);
      vi0x2 = __builtin_wasm_min_f32(vi0x2, vzero);
      float vacc0x3 = __builtin_wasm_max_f32(vi0x3, vzero);
      vi0x3 = __builtin_wasm_min_f32(vi0x3, vzero);
      float vacc1x0 = __builtin_wasm_max_f32(vi1x0, vzero);
      vi1x0 = __builtin_wasm_min_f32(vi1x0, vzero);
      float vacc1x1 = __builtin_wasm_max_f32(vi1x1, vzero);
      vi1x1 = __builtin_wasm_min_f32(vi1x1, vzero);
      float vacc1x2 = __builtin_wasm_max_f32(vi1x2, vzero);
      vi1x2 = __builtin_wasm_min_f32(vi1x2, vzero);
      float vacc1x3 = __builtin_wasm_max_f32(vi1x3, vzero);
      vi1x3 = __builtin_wasm_min_f32(vi1x3, vzero);

      vacc0x0 += vi0x0 * vw0;
      vacc0x1 += vi0x1 * vw1;
      vacc0x2 += vi0x2 * vw2;
      vacc0x3 += vi0x3 * vw3;
      vacc1x0 += vi1x0 * vw0;
      vacc1x1 += vi1x1 * vw1;
      vacc1x2 += vi1x2 * vw2;
      vacc1x3 += vi1x3 * vw3;

      o0[0] = vacc0x0;
      o0[1] = vacc0x1;
      o0[2] = vacc0x2;
      o0[3] = vacc0x3;
      o0 += 4;
      o1[0] = vacc1x0;
      o1[1] = vacc1x1;
      o1[2] = vacc1x2;
      o1[3] = vacc1x3;
      o1 += 4;

      w += 4;
    }
    for (; c != 0; c -= sizeof(float)) {
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
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}
