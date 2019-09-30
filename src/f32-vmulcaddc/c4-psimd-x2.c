// Auto-generated file. Do not edit!
//   Template: src/f32-vmulcaddc/psimd.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/math.h>
#include <xnnpack/vmulcaddc.h>


void xnn_f32_vmulcaddc_ukernel_c4__psimd_x2(
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

  const size_t x_increment = x_stride * 2 - (channels & -(4 * sizeof(float)));
  const size_t y_increment = y_stride * 2 - channels;

  const float* x0 = x;
  float* y0 = y;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  float* y1 = (float*) ((uintptr_t) y0 + y_stride);
  if XNN_UNPREDICTABLE(m < 2) {
    x1 = x0;
    y1 = y0;
  }

  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  do {
    const float* w = weights;
    size_t c = channels;
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const psimd_f32 vscale0123 = psimd_load_f32(w);

      const psimd_f32 vx0x0123 = psimd_load_f32(x0);
      x0 += 4;
      const psimd_f32 vx1x0123 = psimd_load_f32(x1);
      x1 += 4;

      const psimd_f32 vbias0123 = psimd_load_f32(w + 4);

      psimd_f32 vacc0x0123 = psimd_qfma_f32(vbias0123, vx0x0123, vscale0123);
      psimd_f32 vacc1x0123 = psimd_qfma_f32(vbias0123, vx1x0123, vscale0123);

      vacc0x0123 = psimd_max_f32(vacc0x0123, vmin);
      vacc1x0123 = psimd_max_f32(vacc1x0123, vmin);

      vacc0x0123 = psimd_min_f32(vacc0x0123, vmax);
      vacc1x0123 = psimd_min_f32(vacc1x0123, vmax);

      psimd_store_f32(y0, vacc0x0123);
      y0 += 4;
      psimd_store_f32(y1, vacc1x0123);
      y1 += 4;

      w += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const psimd_f32 vscale0123 = psimd_load_f32(w);

      const psimd_f32 vx0x0123 = psimd_load_f32(x0);
      const psimd_f32 vx1x0123 = psimd_load_f32(x1);

      const psimd_f32 vbias0123 = psimd_load_f32(w + 4);

      psimd_f32 vacc0x0123 = psimd_qfma_f32(vbias0123, vx0x0123, vscale0123);
      psimd_f32 vacc1x0123 = psimd_qfma_f32(vbias0123, vx1x0123, vscale0123);

      vacc0x0123 = psimd_max_f32(vacc0x0123, vmin);
      vacc1x0123 = psimd_max_f32(vacc1x0123, vmin);

      vacc0x0123 = psimd_min_f32(vacc0x0123, vmax);
      vacc1x0123 = psimd_min_f32(vacc1x0123, vmax);

      w += 8;

      if (c & (2 * sizeof(float))) {
        psimd_store2_f32(y0, vacc0x0123);
        psimd_store2_f32(y1, vacc1x0123);

        y0 += 2;
        y1 += 2;

        vacc0x0123 = psimd_concat_hi_f32(vacc0x0123, vacc0x0123);
        vacc1x0123 = psimd_concat_hi_f32(vacc1x0123, vacc1x0123);
      }
      if (c & (1 * sizeof(float))) {
        psimd_store1_f32(y0, vacc0x0123);
        psimd_store1_f32(y1, vacc1x0123);

        y0 += 1;
        y1 += 1;
      }
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
