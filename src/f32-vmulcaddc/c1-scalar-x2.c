// Auto-generated file. Do not edit!
//   Template: src/f32-vmulcaddc/scalar.c.in
//   Generator: tools/xngen
//
/*
 * Copyright 2019 Google LLC
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/vmulcaddc.h>


void xnn_f32_vmulcaddc_ukernel_c1__scalar_x2(
    size_t m,
    size_t channels,
    const float*restrict x,
    size_t x_stride,
    const float*restrict weights,
    float*restrict y,
    size_t y_stride,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(m != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const size_t x_increment = x_stride * 2 - (channels & -(1 * sizeof(float)));
  const size_t y_increment = y_stride * 2 - channels;

  const float* x0 = x;
  float* y0 = y;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  float* y1 = (float*) ((uintptr_t) y0 + y_stride);
  if XNN_UNPREDICTABLE(m < 2) {
    x1 = x0;
    y1 = y0;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* w = weights;
    size_t c = channels;
    for (; c >= 1 * sizeof(float); c -= 1 * sizeof(float)) {
      const float vscale0 = w[0];

      const float vx0x0 = x0[0];
      x0 += 1;
      const float vx1x0 = x1[0];
      x1 += 1;

      const float vbias0 = w[1];

      float vacc0x0 = vx0x0 * vscale0 + vbias0;
      float vacc1x0 = vx1x0 * vscale0 + vbias0;

      vacc0x0 = math_max_f32(vacc0x0, vmin);
      vacc1x0 = math_max_f32(vacc1x0, vmin);

      vacc0x0 = math_min_f32(vacc0x0, vmax);
      vacc1x0 = math_min_f32(vacc1x0, vmax);

      y0[0] = vacc0x0;
      y0 += 1;
      y1[0] = vacc1x0;
      y1 += 1;

      w += 2;
    }
    if XNN_UNLIKELY(c != 0) {
      const float vscale0 = w[0];

      const float vx0x0 = x0[0];
      x0 += 1;
      const float vx1x0 = x1[0];
      x1 += 1;

      const float vbias0 = w[1];

      float vacc0x0 = vx0x0 * vscale0 + vbias0;
      float vacc1x0 = vx1x0 * vscale0 + vbias0;

      vacc0x0 = math_max_f32(vacc0x0, vmin);
      vacc1x0 = math_max_f32(vacc1x0, vmin);

      vacc0x0 = math_min_f32(vacc0x0, vmax);
      vacc1x0 = math_min_f32(vacc1x0, vmax);

      w += 2;

    }
    x0 = (const float*) ((uintptr_t) x0 + x_increment);
    y0 = (float*) ((uintptr_t) y0 + y_increment);
    x1 = (const float*) ((uintptr_t) x1 + x_increment);
    y1 = (float*) ((uintptr_t) y1 + y_increment);
    if XNN_UNPREDICTABLE(m < 4) {
      x1 = x0;
      y1 = y0;
    }
    m = doz(m, 2);
  } while (m != 0);
}
