// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/MRx2c4-psimd.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/igemm.h>


void xnn_f32_igemm_minmax_ukernel_4x2c4__psimd(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    psimd_f32 vacc0x0c4 = psimd_load1_f32(w);
    psimd_f32 vacc0x1c4 = psimd_load1_f32(w + 1);
    psimd_f32 vacc1x0c4 = vacc0x0c4;
    psimd_f32 vacc1x1c4 = vacc0x1c4;
    psimd_f32 vacc2x0c4 = vacc0x0c4;
    psimd_f32 vacc2x1c4 = vacc0x1c4;
    psimd_f32 vacc3x0c4 = vacc0x0c4;
    psimd_f32 vacc3x1c4 = vacc0x1c4;
    w += 2;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
        const psimd_f32 va0 = psimd_load_f32(a0);
        a0 += 4;
        const psimd_f32 va1 = psimd_load_f32(a1);
        a1 += 4;
        const psimd_f32 va2 = psimd_load_f32(a2);
        a2 += 4;
        const psimd_f32 va3 = psimd_load_f32(a3);
        a3 += 4;

        const psimd_f32 vb0 = psimd_load_f32(w);
        const psimd_f32 vb1 = psimd_load_f32(w + 4);
        w += 8;

        vacc0x0c4 = psimd_qfma_f32(vacc0x0c4, va0, vb0);
        vacc0x1c4 = psimd_qfma_f32(vacc0x1c4, va0, vb1);
        vacc1x0c4 = psimd_qfma_f32(vacc1x0c4, va1, vb0);
        vacc1x1c4 = psimd_qfma_f32(vacc1x1c4, va1, vb1);
        vacc2x0c4 = psimd_qfma_f32(vacc2x0c4, va2, vb0);
        vacc2x1c4 = psimd_qfma_f32(vacc2x1c4, va2, vb1);
        vacc3x0c4 = psimd_qfma_f32(vacc3x0c4, va3, vb0);
        vacc3x1c4 = psimd_qfma_f32(vacc3x1c4, va3, vb1);
      }
      if XNN_UNLIKELY(k != 0) {
        const psimd_f32 va0 = psimd_load_f32(a0);
        const psimd_f32 va1 = psimd_load_f32(a1);
        const psimd_f32 va2 = psimd_load_f32(a2);
        const psimd_f32 va3 = psimd_load_f32(a3);

        const psimd_f32 vb0 = psimd_load_f32(w);
        const psimd_f32 vb1 = psimd_load_f32(w + 4);
        w += 8;

        const psimd_f32 vzero = psimd_splat_f32(0.0f);
        const psimd_s32 vmask0 = vb0 != vzero;
        const psimd_s32 vmask1 = vb1 != vzero;

        vacc0x0c4 = psimd_qfma_f32(vacc0x0c4, psimd_andmask_f32(vmask0, va0), vb0);
        vacc0x1c4 = psimd_qfma_f32(vacc0x1c4, psimd_andmask_f32(vmask1, va0), vb1);
        vacc1x0c4 = psimd_qfma_f32(vacc1x0c4, psimd_andmask_f32(vmask0, va1), vb0);
        vacc1x1c4 = psimd_qfma_f32(vacc1x1c4, psimd_andmask_f32(vmask1, va1), vb1);
        vacc2x0c4 = psimd_qfma_f32(vacc2x0c4, psimd_andmask_f32(vmask0, va2), vb0);
        vacc2x1c4 = psimd_qfma_f32(vacc2x1c4, psimd_andmask_f32(vmask1, va2), vb1);
        vacc3x0c4 = psimd_qfma_f32(vacc3x0c4, psimd_andmask_f32(vmask0, va3), vb0);
        vacc3x1c4 = psimd_qfma_f32(vacc3x1c4, psimd_andmask_f32(vmask1, va3), vb1);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const psimd_f32 vacc0x01c2 = psimd_add_f32(psimd_interleave_lo_f32(vacc0x0c4, vacc0x1c4), psimd_interleave_hi_f32(vacc0x0c4, vacc0x1c4));
    const psimd_f32 vacc1x01c2 = psimd_add_f32(psimd_interleave_lo_f32(vacc1x0c4, vacc1x1c4), psimd_interleave_hi_f32(vacc1x0c4, vacc1x1c4));
    const psimd_f32 vacc2x01c2 = psimd_add_f32(psimd_interleave_lo_f32(vacc2x0c4, vacc2x1c4), psimd_interleave_hi_f32(vacc2x0c4, vacc2x1c4));
    const psimd_f32 vacc3x01c2 = psimd_add_f32(psimd_interleave_lo_f32(vacc3x0c4, vacc3x1c4), psimd_interleave_hi_f32(vacc3x0c4, vacc3x1c4));

    psimd_f32 vacc01x01 = psimd_add_f32(psimd_concat_lo_f32(vacc0x01c2, vacc1x01c2), psimd_concat_hi_f32(vacc0x01c2, vacc1x01c2));
    psimd_f32 vacc23x01 = psimd_add_f32(psimd_concat_lo_f32(vacc2x01c2, vacc3x01c2), psimd_concat_hi_f32(vacc2x01c2, vacc3x01c2));

    const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
    vacc01x01 = psimd_min_f32(vacc01x01, vmax);
    vacc23x01 = psimd_min_f32(vacc23x01, vmax);

    const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);
    vacc01x01 = psimd_max_f32(vacc01x01, vmin);
    vacc23x01 = psimd_max_f32(vacc23x01, vmin);

    if XNN_LIKELY(nc >= 2) {
      psimd_store2_f32(c3, psimd_concat_hi_f32(vacc23x01, vacc23x01));
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      psimd_store2_f32(c2, vacc23x01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      psimd_store2_f32(c1, psimd_concat_hi_f32(vacc01x01, vacc01x01));
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      psimd_store2_f32(c0, vacc01x01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      assert(nc == 1);
      psimd_store1_f32(c3, psimd_concat_hi_f32(vacc23x01, vacc23x01));
      psimd_store1_f32(c2, vacc23x01);
      psimd_store1_f32(c1, psimd_concat_hi_f32(vacc01x01, vacc01x01));
      psimd_store1_f32(c0, vacc01x01);

      nc = 0;
    }
  } while (nc != 0);
}
