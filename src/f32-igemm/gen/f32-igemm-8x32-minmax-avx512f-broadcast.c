// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/avx512-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"


void xnn_f32_igemm_minmax_ukernel_8x32__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (8 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    c7 = c6;
  }

  do {
    __m512 vacc0x0 = _mm512_load_ps(w);
    __m512 vacc0x1 = _mm512_load_ps(w + 16);
    __m512 vacc1x0 = vacc0x0;
    __m512 vacc1x1 = vacc0x1;
    __m512 vacc2x0 = vacc0x0;
    __m512 vacc2x1 = vacc0x1;
    __m512 vacc3x0 = vacc0x0;
    __m512 vacc3x1 = vacc0x1;
    __m512 vacc4x0 = vacc0x0;
    __m512 vacc4x1 = vacc0x1;
    __m512 vacc5x0 = vacc0x0;
    __m512 vacc5x1 = vacc0x1;
    __m512 vacc6x0 = vacc0x0;
    __m512 vacc6x1 = vacc0x1;
    __m512 vacc7x0 = vacc0x0;
    __m512 vacc7x1 = vacc0x1;
    w += 32;

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
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      const float* restrict a6 = a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const float*) ((uintptr_t) a6 + a_offset);
      }
      const float* restrict a7 = a[7];
      assert(a7 != NULL);
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const float*) ((uintptr_t) a7 + a_offset);
      }
      a += 8;

      size_t k = kc;
      do {
        const __m512 vb0 = _mm512_load_ps(w);
        const __m512 vb1 = _mm512_load_ps(w + 16);
        w += 32;

        const __m512 va0 = _mm512_set1_ps(*a0);
        vacc0x0 = _mm512_fmadd_ps(va0, vb0, vacc0x0);
        vacc0x1 = _mm512_fmadd_ps(va0, vb1, vacc0x1);
        const __m512 va1 = _mm512_set1_ps(*a1);
        vacc1x0 = _mm512_fmadd_ps(va1, vb0, vacc1x0);
        vacc1x1 = _mm512_fmadd_ps(va1, vb1, vacc1x1);
        const __m512 va2 = _mm512_set1_ps(*a2);
        vacc2x0 = _mm512_fmadd_ps(va2, vb0, vacc2x0);
        vacc2x1 = _mm512_fmadd_ps(va2, vb1, vacc2x1);
        const __m512 va3 = _mm512_set1_ps(*a3);
        vacc3x0 = _mm512_fmadd_ps(va3, vb0, vacc3x0);
        vacc3x1 = _mm512_fmadd_ps(va3, vb1, vacc3x1);
        const __m512 va4 = _mm512_set1_ps(*a4);
        vacc4x0 = _mm512_fmadd_ps(va4, vb0, vacc4x0);
        vacc4x1 = _mm512_fmadd_ps(va4, vb1, vacc4x1);
        const __m512 va5 = _mm512_set1_ps(*a5);
        vacc5x0 = _mm512_fmadd_ps(va5, vb0, vacc5x0);
        vacc5x1 = _mm512_fmadd_ps(va5, vb1, vacc5x1);
        const __m512 va6 = _mm512_set1_ps(*a6);
        vacc6x0 = _mm512_fmadd_ps(va6, vb0, vacc6x0);
        vacc6x1 = _mm512_fmadd_ps(va6, vb1, vacc6x1);
        const __m512 va7 = _mm512_set1_ps(*a7);
        vacc7x0 = _mm512_fmadd_ps(va7, vb0, vacc7x0);
        vacc7x1 = _mm512_fmadd_ps(va7, vb1, vacc7x1);

        a0 += 1;
        a1 += 1;
        a2 += 1;
        a3 += 1;
        a4 += 1;
        a5 += 1;
        a6 += 1;
        a7 += 1;

        k -= sizeof(float);
      } while (k != 0);
      p -= 8 * sizeof(void*);
    } while (p != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0 = _mm512_max_ps(vmin, vacc0x0);
    vacc1x0 = _mm512_max_ps(vmin, vacc1x0);
    vacc2x0 = _mm512_max_ps(vmin, vacc2x0);
    vacc3x0 = _mm512_max_ps(vmin, vacc3x0);
    vacc4x0 = _mm512_max_ps(vmin, vacc4x0);
    vacc5x0 = _mm512_max_ps(vmin, vacc5x0);
    vacc6x0 = _mm512_max_ps(vmin, vacc6x0);
    vacc7x0 = _mm512_max_ps(vmin, vacc7x0);
    vacc0x1 = _mm512_max_ps(vmin, vacc0x1);
    vacc1x1 = _mm512_max_ps(vmin, vacc1x1);
    vacc2x1 = _mm512_max_ps(vmin, vacc2x1);
    vacc3x1 = _mm512_max_ps(vmin, vacc3x1);
    vacc4x1 = _mm512_max_ps(vmin, vacc4x1);
    vacc5x1 = _mm512_max_ps(vmin, vacc5x1);
    vacc6x1 = _mm512_max_ps(vmin, vacc6x1);
    vacc7x1 = _mm512_max_ps(vmin, vacc7x1);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0 = _mm512_min_ps(vmax, vacc0x0);
    vacc1x0 = _mm512_min_ps(vmax, vacc1x0);
    vacc2x0 = _mm512_min_ps(vmax, vacc2x0);
    vacc3x0 = _mm512_min_ps(vmax, vacc3x0);
    vacc4x0 = _mm512_min_ps(vmax, vacc4x0);
    vacc5x0 = _mm512_min_ps(vmax, vacc5x0);
    vacc6x0 = _mm512_min_ps(vmax, vacc6x0);
    vacc7x0 = _mm512_min_ps(vmax, vacc7x0);
    vacc0x1 = _mm512_min_ps(vmax, vacc0x1);
    vacc1x1 = _mm512_min_ps(vmax, vacc1x1);
    vacc2x1 = _mm512_min_ps(vmax, vacc2x1);
    vacc3x1 = _mm512_min_ps(vmax, vacc3x1);
    vacc4x1 = _mm512_min_ps(vmax, vacc4x1);
    vacc5x1 = _mm512_min_ps(vmax, vacc5x1);
    vacc6x1 = _mm512_min_ps(vmax, vacc6x1);
    vacc7x1 = _mm512_min_ps(vmax, vacc7x1);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c7, vacc7x0);
      _mm512_storeu_ps(c7 + 16, vacc7x1);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c6, vacc6x0);
      _mm512_storeu_ps(c6 + 16, vacc6x1);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5, vacc5x0);
      _mm512_storeu_ps(c5 + 16, vacc5x1);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4, vacc4x0);
      _mm512_storeu_ps(c4 + 16, vacc4x1);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3, vacc3x0);
      _mm512_storeu_ps(c3 + 16, vacc3x1);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2, vacc2x0);
      _mm512_storeu_ps(c2 + 16, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1, vacc1x0);
      _mm512_storeu_ps(c1 + 16, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0, vacc0x0);
      _mm512_storeu_ps(c0 + 16, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 32;
    } else {
      // NC remainder (1..31)
      assert(nc >= 1);
      assert(nc <= 31);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 0));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 16));

      _mm512_mask_storeu_ps(c7 + 0, vmask0, vacc7x0);
      _mm512_mask_storeu_ps(c7 + 16, vmask1, vacc7x1);
      _mm512_mask_storeu_ps(c6 + 0, vmask0, vacc6x0);
      _mm512_mask_storeu_ps(c6 + 16, vmask1, vacc6x1);
      _mm512_mask_storeu_ps(c5 + 0, vmask0, vacc5x0);
      _mm512_mask_storeu_ps(c5 + 16, vmask1, vacc5x1);
      _mm512_mask_storeu_ps(c4 + 0, vmask0, vacc4x0);
      _mm512_mask_storeu_ps(c4 + 16, vmask1, vacc4x1);
      _mm512_mask_storeu_ps(c3 + 0, vmask0, vacc3x0);
      _mm512_mask_storeu_ps(c3 + 16, vmask1, vacc3x1);
      _mm512_mask_storeu_ps(c2 + 0, vmask0, vacc2x0);
      _mm512_mask_storeu_ps(c2 + 16, vmask1, vacc2x1);
      _mm512_mask_storeu_ps(c1 + 0, vmask0, vacc1x0);
      _mm512_mask_storeu_ps(c1 + 16, vmask1, vacc1x1);
      _mm512_mask_storeu_ps(c0 + 0, vmask0, vacc0x0);
      _mm512_mask_storeu_ps(c0 + 16, vmask1, vacc0x1);
      nc = 0;
    }
  } while (nc != 0);
}
