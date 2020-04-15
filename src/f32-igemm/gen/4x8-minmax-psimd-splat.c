// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/psimd-splat.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/igemm.h>


void xnn_f32_igemm_minmax_ukernel_4x8__psimd_splat(
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
    psimd_f32 vacc0x0123 = psimd_load_f32(w);
    psimd_f32 vacc0x4567 = psimd_load_f32(w + 4);
    psimd_f32 vacc1x0123 = vacc0x0123;
    psimd_f32 vacc1x4567 = vacc0x4567;
    psimd_f32 vacc2x0123 = vacc0x0123;
    psimd_f32 vacc2x4567 = vacc0x4567;
    psimd_f32 vacc3x0123 = vacc0x0123;
    psimd_f32 vacc3x4567 = vacc0x4567;
    w += 8;

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
      while (k >= 4 * sizeof(float)) {
        const psimd_f32 va0 = psimd_load_f32(a0);
        a0 += 4;
        const psimd_f32 va1 = psimd_load_f32(a1);
        a1 += 4;
        const psimd_f32 va2 = psimd_load_f32(a2);
        a2 += 4;
        const psimd_f32 va3 = psimd_load_f32(a3);
        a3 += 4;

        const psimd_f32 va0c0 = psimd_splat0_f32(va0);
        const psimd_f32 va1c0 = psimd_splat0_f32(va1);
        const psimd_f32 va2c0 = psimd_splat0_f32(va2);
        const psimd_f32 va3c0 = psimd_splat0_f32(va3);

        const psimd_f32 vb0123c0 = psimd_load_f32(w + 0);
        const psimd_f32 vb4567c0 = psimd_load_f32(w + 4);

        vacc0x0123 = psimd_qfma_f32(vacc0x0123, va0c0, vb0123c0);
        vacc1x0123 = psimd_qfma_f32(vacc1x0123, va1c0, vb0123c0);
        vacc2x0123 = psimd_qfma_f32(vacc2x0123, va2c0, vb0123c0);
        vacc3x0123 = psimd_qfma_f32(vacc3x0123, va3c0, vb0123c0);
        vacc0x4567 = psimd_qfma_f32(vacc0x4567, va0c0, vb4567c0);
        vacc1x4567 = psimd_qfma_f32(vacc1x4567, va1c0, vb4567c0);
        vacc2x4567 = psimd_qfma_f32(vacc2x4567, va2c0, vb4567c0);
        vacc3x4567 = psimd_qfma_f32(vacc3x4567, va3c0, vb4567c0);
        const psimd_f32 va0c1 = psimd_splat1_f32(va0);
        const psimd_f32 va1c1 = psimd_splat1_f32(va1);
        const psimd_f32 va2c1 = psimd_splat1_f32(va2);
        const psimd_f32 va3c1 = psimd_splat1_f32(va3);

        const psimd_f32 vb0123c1 = psimd_load_f32(w + 8);
        const psimd_f32 vb4567c1 = psimd_load_f32(w + 12);

        vacc0x0123 = psimd_qfma_f32(vacc0x0123, va0c1, vb0123c1);
        vacc1x0123 = psimd_qfma_f32(vacc1x0123, va1c1, vb0123c1);
        vacc2x0123 = psimd_qfma_f32(vacc2x0123, va2c1, vb0123c1);
        vacc3x0123 = psimd_qfma_f32(vacc3x0123, va3c1, vb0123c1);
        vacc0x4567 = psimd_qfma_f32(vacc0x4567, va0c1, vb4567c1);
        vacc1x4567 = psimd_qfma_f32(vacc1x4567, va1c1, vb4567c1);
        vacc2x4567 = psimd_qfma_f32(vacc2x4567, va2c1, vb4567c1);
        vacc3x4567 = psimd_qfma_f32(vacc3x4567, va3c1, vb4567c1);
        const psimd_f32 va0c2 = psimd_splat2_f32(va0);
        const psimd_f32 va1c2 = psimd_splat2_f32(va1);
        const psimd_f32 va2c2 = psimd_splat2_f32(va2);
        const psimd_f32 va3c2 = psimd_splat2_f32(va3);

        const psimd_f32 vb0123c2 = psimd_load_f32(w + 16);
        const psimd_f32 vb4567c2 = psimd_load_f32(w + 20);

        vacc0x0123 = psimd_qfma_f32(vacc0x0123, va0c2, vb0123c2);
        vacc1x0123 = psimd_qfma_f32(vacc1x0123, va1c2, vb0123c2);
        vacc2x0123 = psimd_qfma_f32(vacc2x0123, va2c2, vb0123c2);
        vacc3x0123 = psimd_qfma_f32(vacc3x0123, va3c2, vb0123c2);
        vacc0x4567 = psimd_qfma_f32(vacc0x4567, va0c2, vb4567c2);
        vacc1x4567 = psimd_qfma_f32(vacc1x4567, va1c2, vb4567c2);
        vacc2x4567 = psimd_qfma_f32(vacc2x4567, va2c2, vb4567c2);
        vacc3x4567 = psimd_qfma_f32(vacc3x4567, va3c2, vb4567c2);
        const psimd_f32 va0c3 = psimd_splat3_f32(va0);
        const psimd_f32 va1c3 = psimd_splat3_f32(va1);
        const psimd_f32 va2c3 = psimd_splat3_f32(va2);
        const psimd_f32 va3c3 = psimd_splat3_f32(va3);

        const psimd_f32 vb0123c3 = psimd_load_f32(w + 24);
        const psimd_f32 vb4567c3 = psimd_load_f32(w + 28);

        vacc0x0123 = psimd_qfma_f32(vacc0x0123, va0c3, vb0123c3);
        vacc1x0123 = psimd_qfma_f32(vacc1x0123, va1c3, vb0123c3);
        vacc2x0123 = psimd_qfma_f32(vacc2x0123, va2c3, vb0123c3);
        vacc3x0123 = psimd_qfma_f32(vacc3x0123, va3c3, vb0123c3);
        vacc0x4567 = psimd_qfma_f32(vacc0x4567, va0c3, vb4567c3);
        vacc1x4567 = psimd_qfma_f32(vacc1x4567, va1c3, vb4567c3);
        vacc2x4567 = psimd_qfma_f32(vacc2x4567, va2c3, vb4567c3);
        vacc3x4567 = psimd_qfma_f32(vacc3x4567, va3c3, vb4567c3);

        w += 32;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        do {
          const psimd_f32 vb0123 = psimd_load_f32(w);
          const psimd_f32 vb4567 = psimd_load_f32(w + 4);
          w += 8;

          const psimd_f32 va0 = psimd_load_splat_f32(a0);
          a0 += 1;
          const psimd_f32 va1 = psimd_load_splat_f32(a1);
          a1 += 1;
          const psimd_f32 va2 = psimd_load_splat_f32(a2);
          a2 += 1;
          const psimd_f32 va3 = psimd_load_splat_f32(a3);
          a3 += 1;

          vacc0x0123 = psimd_qfma_f32(vacc0x0123, va0, vb0123);
          vacc0x4567 = psimd_qfma_f32(vacc0x4567, va0, vb4567);
          vacc1x0123 = psimd_qfma_f32(vacc1x0123, va1, vb0123);
          vacc1x4567 = psimd_qfma_f32(vacc1x4567, va1, vb4567);
          vacc2x0123 = psimd_qfma_f32(vacc2x0123, va2, vb0123);
          vacc2x4567 = psimd_qfma_f32(vacc2x4567, va2, vb4567);
          vacc3x0123 = psimd_qfma_f32(vacc3x0123, va3, vb0123);
          vacc3x4567 = psimd_qfma_f32(vacc3x4567, va3, vb4567);
          k -= sizeof(float);
        } while (k != 0);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);
    vacc0x0123 = psimd_min_f32(vacc0x0123, vmax);
    vacc1x0123 = psimd_min_f32(vacc1x0123, vmax);
    vacc2x0123 = psimd_min_f32(vacc2x0123, vmax);
    vacc3x0123 = psimd_min_f32(vacc3x0123, vmax);
    vacc0x4567 = psimd_min_f32(vacc0x4567, vmax);
    vacc1x4567 = psimd_min_f32(vacc1x4567, vmax);
    vacc2x4567 = psimd_min_f32(vacc2x4567, vmax);
    vacc3x4567 = psimd_min_f32(vacc3x4567, vmax);

    const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);
    vacc0x0123 = psimd_max_f32(vacc0x0123, vmin);
    vacc1x0123 = psimd_max_f32(vacc1x0123, vmin);
    vacc2x0123 = psimd_max_f32(vacc2x0123, vmin);
    vacc3x0123 = psimd_max_f32(vacc3x0123, vmin);
    vacc0x4567 = psimd_max_f32(vacc0x4567, vmin);
    vacc1x4567 = psimd_max_f32(vacc1x4567, vmin);
    vacc2x4567 = psimd_max_f32(vacc2x4567, vmin);
    vacc3x4567 = psimd_max_f32(vacc3x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      psimd_store_f32(c3, vacc3x0123);
      psimd_store_f32(c3 + 4, vacc3x4567);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      psimd_store_f32(c2, vacc2x0123);
      psimd_store_f32(c2 + 4, vacc2x4567);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      psimd_store_f32(c1, vacc1x0123);
      psimd_store_f32(c1 + 4, vacc1x4567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      psimd_store_f32(c0, vacc0x0123);
      psimd_store_f32(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        psimd_store_f32(c3, vacc3x0123);
        psimd_store_f32(c2, vacc2x0123);
        psimd_store_f32(c1, vacc1x0123);
        psimd_store_f32(c0, vacc0x0123);

        vacc3x0123 = vacc3x4567;
        vacc2x0123 = vacc2x4567;
        vacc1x0123 = vacc1x4567;
        vacc0x0123 = vacc0x4567;

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        psimd_store2_f32(c3, vacc3x0123);
        psimd_store2_f32(c2, vacc2x0123);
        psimd_store2_f32(c1, vacc1x0123);
        psimd_store2_f32(c0, vacc0x0123);

        vacc3x0123 = psimd_concat_hi_f32(vacc3x0123, vacc3x0123);
        vacc2x0123 = psimd_concat_hi_f32(vacc2x0123, vacc2x0123);
        vacc1x0123 = psimd_concat_hi_f32(vacc1x0123, vacc1x0123);
        vacc0x0123 = psimd_concat_hi_f32(vacc0x0123, vacc0x0123);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        psimd_store1_f32(c3, vacc3x0123);
        psimd_store1_f32(c2, vacc2x0123);
        psimd_store1_f32(c1, vacc1x0123);
        psimd_store1_f32(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
