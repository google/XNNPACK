// Auto-generated file. Do not edit!
//   Template: src/f32-prelu/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <math.h>

#include <xnnpack/math.h>
#include <xnnpack/prelu.h>


void xnn_f32_prelu_ukernel__wasm_2x4(
    size_t rows,
    size_t channels,
    const float*restrict input,
    size_t input_stride,
    const float*restrict weights,
    float*restrict output,
    size_t output_stride,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = i0;
    o1 = o0;
  }

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float vw0 = w[0];
      const float vw1 = w[1];
      const float vw2 = w[2];
      const float vw3 = w[3];

      const float vi0x0 = i0[0];
      const float vi0x1 = i0[1];
      const float vi0x2 = i0[2];
      const float vi0x3 = i0[3];
      i0 += 4;
      const float vi1x0 = i1[0];
      const float vi1x1 = i1[1];
      const float vi1x2 = i1[2];
      const float vi1x3 = i1[3];
      i1 += 4;

      float vacc0x0 = signbit(vi0x0) ? vi0x0 * vw0 : vi0x0;
      float vacc0x1 = signbit(vi0x1) ? vi0x1 * vw1 : vi0x1;
      float vacc0x2 = signbit(vi0x2) ? vi0x2 * vw2 : vi0x2;
      float vacc0x3 = signbit(vi0x3) ? vi0x3 * vw3 : vi0x3;
      float vacc1x0 = signbit(vi1x0) ? vi1x0 * vw0 : vi1x0;
      float vacc1x1 = signbit(vi1x1) ? vi1x1 * vw1 : vi1x1;
      float vacc1x2 = signbit(vi1x2) ? vi1x2 * vw2 : vi1x2;
      float vacc1x3 = signbit(vi1x3) ? vi1x3 * vw3 : vi1x3;

      vacc0x0 = __builtin_wasm_max_f32(vacc0x0, vmin);
      vacc0x1 = __builtin_wasm_max_f32(vacc0x1, vmin);
      vacc0x2 = __builtin_wasm_max_f32(vacc0x2, vmin);
      vacc0x3 = __builtin_wasm_max_f32(vacc0x3, vmin);
      vacc1x0 = __builtin_wasm_max_f32(vacc1x0, vmin);
      vacc1x1 = __builtin_wasm_max_f32(vacc1x1, vmin);
      vacc1x2 = __builtin_wasm_max_f32(vacc1x2, vmin);
      vacc1x3 = __builtin_wasm_max_f32(vacc1x3, vmin);

      vacc0x0 = __builtin_wasm_min_f32(vacc0x0, vmax);
      vacc0x1 = __builtin_wasm_min_f32(vacc0x1, vmax);
      vacc0x2 = __builtin_wasm_min_f32(vacc0x2, vmax);
      vacc0x3 = __builtin_wasm_min_f32(vacc0x3, vmax);
      vacc1x0 = __builtin_wasm_min_f32(vacc1x0, vmax);
      vacc1x1 = __builtin_wasm_min_f32(vacc1x1, vmax);
      vacc1x2 = __builtin_wasm_min_f32(vacc1x2, vmax);
      vacc1x3 = __builtin_wasm_min_f32(vacc1x3, vmax);

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

      const float vi0 = *i0++;
      const float vi1 = *i1++;

      float vacc0 = signbit(vi0) ? vi0 * vw : vi0;
      float vacc1 = signbit(vi1) ? vi1 * vw : vi1;

      vacc0 = __builtin_wasm_max_f32(vacc0, vmin);
      vacc1 = __builtin_wasm_max_f32(vacc1, vmin);

      vacc0 = __builtin_wasm_min_f32(vacc0, vmax);
      vacc1 = __builtin_wasm_min_f32(vacc1, vmax);

      *o0++ = vacc0;
      *o1++ = vacc1;
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    if XNN_UNPREDICTABLE(rows < 4) {
      i1 = i0;
      o1 = o0;
    }
    rows = doz(rows, 2);
  } while (rows != 0);
}
