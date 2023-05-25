// Auto-generated file. Do not edit!
//   Template: src/f32-vmulcaddc/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/vmulcaddc.h>


void xnn_f32_vmulcaddc_minmax_ukernel_c4__wasm_2x(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float vscale0 = w[0];
      const float vscale1 = w[1];
      const float vscale2 = w[2];
      const float vscale3 = w[3];

      float vacc0x0 = i0[0];
      float vacc0x1 = i0[1];
      float vacc0x2 = i0[2];
      float vacc0x3 = i0[3];
      i0 += 4;
      float vacc1x0 = i1[0];
      float vacc1x1 = i1[1];
      float vacc1x2 = i1[2];
      float vacc1x3 = i1[3];
      i1 += 4;

      const float vbias0 = w[4];
      const float vbias1 = w[5];
      const float vbias2 = w[6];
      const float vbias3 = w[7];

      vacc0x0 = vacc0x0 * vscale0 + vbias0;
      vacc0x1 = vacc0x1 * vscale1 + vbias1;
      vacc0x2 = vacc0x2 * vscale2 + vbias2;
      vacc0x3 = vacc0x3 * vscale3 + vbias3;
      vacc1x0 = vacc1x0 * vscale0 + vbias0;
      vacc1x1 = vacc1x1 * vscale1 + vbias1;
      vacc1x2 = vacc1x2 * vscale2 + vbias2;
      vacc1x3 = vacc1x3 * vscale3 + vbias3;

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

      w += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        const float vscale = *w++;

        float vacc0 = *i0++;
        float vacc1 = *i1++;

        const float vbias = w[3];

        vacc0 = vacc0 * vscale + vbias;
        vacc1 = vacc1 * vscale + vbias;

        vacc0 = __builtin_wasm_max_f32(vacc0, vmin);
        vacc1 = __builtin_wasm_max_f32(vacc1, vmin);

        vacc0 = __builtin_wasm_min_f32(vacc0, vmax);
        vacc1 = __builtin_wasm_min_f32(vacc1, vmax);

        *o0++ = vacc0;
        *o1++ = vacc1;

        c -= sizeof(float);
      } while (c != 0);
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}
