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


void xnn_f32_vmulcaddc_ukernel_c8__psimd_2x(
    size_t rows,
    size_t channels,
    const float*restrict input,
    size_t input_stride,
    const float*restrict weights,
    float*restrict output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
  do {
    const float* w = weights;
    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const psimd_f32 vscale0123 = psimd_load_f32(w);
      const psimd_f32 vscale4567 = psimd_load_f32(w + 4);

      psimd_f32 vacc0x0123 = psimd_load_f32(i0);
      psimd_f32 vacc0x4567 = psimd_load_f32(i0 + 4);
      i0 += 8;
      psimd_f32 vacc1x0123 = psimd_load_f32(i1);
      psimd_f32 vacc1x4567 = psimd_load_f32(i1 + 4);
      i1 += 8;

      const psimd_f32 vbias0123 = psimd_load_f32(w + 8);
      const psimd_f32 vbias4567 = psimd_load_f32(w + 12);

      vacc0x0123 = psimd_qfma_f32(vbias0123, vscale0123, vacc0x0123);
      vacc0x4567 = psimd_qfma_f32(vbias4567, vscale4567, vacc0x4567);
      vacc1x0123 = psimd_qfma_f32(vbias0123, vscale0123, vacc1x0123);
      vacc1x4567 = psimd_qfma_f32(vbias4567, vscale4567, vacc1x4567);

      vacc0x0123 = psimd_max_f32(vacc0x0123, vmin);
      vacc0x4567 = psimd_max_f32(vacc0x4567, vmin);
      vacc1x0123 = psimd_max_f32(vacc1x0123, vmin);
      vacc1x4567 = psimd_max_f32(vacc1x4567, vmin);

      vacc0x0123 = psimd_min_f32(vacc0x0123, vmax);
      vacc0x4567 = psimd_min_f32(vacc0x4567, vmax);
      vacc1x0123 = psimd_min_f32(vacc1x0123, vmax);
      vacc1x4567 = psimd_min_f32(vacc1x4567, vmax);

      psimd_store_f32(o0, vacc0x0123);
      psimd_store_f32(o0 + 4, vacc0x4567);
      o0 += 8;
      psimd_store_f32(o1, vacc1x0123);
      psimd_store_f32(o1 + 4, vacc1x4567);
      o1 += 8;

      w += 16;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const psimd_f32 vscale0123 = psimd_load_f32(w);

      psimd_f32 vacc0x0123 = psimd_load_f32(i0);
      i0 += 4;
      psimd_f32 vacc1x0123 = psimd_load_f32(i1);
      i1 += 4;

      const psimd_f32 vbias0123 = psimd_load_f32(w + 8);

      vacc0x0123 = psimd_qfma_f32(vbias0123, vscale0123, vacc0x0123);
      vacc1x0123 = psimd_qfma_f32(vbias0123, vscale0123, vacc1x0123);

      vacc0x0123 = psimd_max_f32(vacc0x0123, vmin);
      vacc1x0123 = psimd_max_f32(vacc1x0123, vmin);

      vacc0x0123 = psimd_min_f32(vacc0x0123, vmax);
      vacc1x0123 = psimd_min_f32(vacc1x0123, vmax);

      psimd_store_f32(o0, vacc0x0123);
      o0 += 4;
      psimd_store_f32(o1, vacc1x0123);
      o1 += 4;

      w += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const psimd_f32 vscale0123 = psimd_load_f32(w);

      psimd_f32 vacc0x0123 = psimd_load_f32(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      psimd_f32 vacc1x0123 = psimd_load_f32(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      const psimd_f32 vbias0123 = psimd_load_f32(w + 8);

      vacc0x0123 = psimd_qfma_f32(vbias0123, vscale0123, vacc0x0123);
      vacc1x0123 = psimd_qfma_f32(vbias0123, vscale0123, vacc1x0123);

      vacc0x0123 = psimd_max_f32(vacc0x0123, vmin);
      vacc1x0123 = psimd_max_f32(vacc1x0123, vmin);

      vacc0x0123 = psimd_min_f32(vacc0x0123, vmax);
      vacc1x0123 = psimd_min_f32(vacc1x0123, vmax);

      if (c & (2 * sizeof(float))) {
        psimd_store2_f32(o0, vacc0x0123);
        psimd_store2_f32(o1, vacc1x0123);

        vacc0x0123 = psimd_concat_hi_f32(vacc0x0123, vacc0x0123);
        vacc1x0123 = psimd_concat_hi_f32(vacc1x0123, vacc1x0123);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        psimd_store1_f32(o0, vacc0x0123);
        psimd_store1_f32(o1, vacc1x0123);

        o0 += 1;
        o1 += 1;
      }
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
