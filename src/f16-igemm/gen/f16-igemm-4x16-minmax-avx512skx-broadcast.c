// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-igemm/avx512skx-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/igemm.h"


void xnn_f16_igemm_minmax_ukernel_4x16__avx512skx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const xnn_float16** restrict a,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const xnn_float16* zero,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const __m512 vmin = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m512 vmax = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.max));
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* c0 = (uint16_t*) c;
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    __m512 vacc0x0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) w));
    __m512 vacc1x0 = vacc0x0;
    __m512 vacc2x0 = vacc0x0;
    __m512 vacc3x0 = vacc0x0;
    w = (const xnn_float16*) w + 16;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != (const uint16_t*) zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint16_t* restrict a1 = (const uint16_t*) a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != (const uint16_t*) zero) {
        a1 = (const uint16_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint16_t* restrict a2 = (const uint16_t*) a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != (const uint16_t*) zero) {
        a2 = (const uint16_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint16_t* restrict a3 = (const uint16_t*) a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != (const uint16_t*) zero) {
        a3 = (const uint16_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const __m512 vb0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) w));
        w = (const xnn_float16*) w + 16;

        const __m512 va0 = _mm512_cvtph_ps(_mm256_set1_epi16((short) *a0));
        a0 += 1;
        const __m512 va1 = _mm512_cvtph_ps(_mm256_set1_epi16((short) *a1));
        a1 += 1;
        const __m512 va2 = _mm512_cvtph_ps(_mm256_set1_epi16((short) *a2));
        a2 += 1;
        const __m512 va3 = _mm512_cvtph_ps(_mm256_set1_epi16((short) *a3));
        a3 += 1;

        vacc0x0 = _mm512_cvtph_ps(_mm512_cvtps_ph(_mm512_fmadd_ps(va0, vb0, vacc0x0), _MM_FROUND_TO_NEAREST_INT));
        vacc1x0 = _mm512_cvtph_ps(_mm512_cvtps_ph(_mm512_fmadd_ps(va1, vb0, vacc1x0), _MM_FROUND_TO_NEAREST_INT));
        vacc2x0 = _mm512_cvtph_ps(_mm512_cvtps_ph(_mm512_fmadd_ps(va2, vb0, vacc2x0), _MM_FROUND_TO_NEAREST_INT));
        vacc3x0 = _mm512_cvtph_ps(_mm512_cvtps_ph(_mm512_fmadd_ps(va3, vb0, vacc3x0), _MM_FROUND_TO_NEAREST_INT));

        k -= sizeof(uint16_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc0x0 = _mm512_max_ps(vacc0x0, vmin);
    vacc1x0 = _mm512_max_ps(vacc1x0, vmin);
    vacc2x0 = _mm512_max_ps(vacc2x0, vmin);
    vacc3x0 = _mm512_max_ps(vacc3x0, vmin);

    vacc0x0 = _mm512_min_ps(vacc0x0, vmax);
    vacc1x0 = _mm512_min_ps(vacc1x0, vmax);
    vacc2x0 = _mm512_min_ps(vacc2x0, vmax);
    vacc3x0 = _mm512_min_ps(vacc3x0, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_si256((__m256i*) c3, _mm512_cvtps_ph(vacc3x0, _MM_FROUND_TO_NEAREST_INT));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_si256((__m256i*) c2, _mm512_cvtps_ph(vacc2x0, _MM_FROUND_TO_NEAREST_INT));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_si256((__m256i*) c1, _mm512_cvtps_ph(vacc1x0, _MM_FROUND_TO_NEAREST_INT));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_si256((__m256i*) c0, _mm512_cvtps_ph(vacc0x0, _MM_FROUND_TO_NEAREST_INT));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const xnn_float16**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 0));

      _mm256_mask_storeu_epi16(c3 + 0, vmask0, _mm512_cvtps_ph(vacc3x0, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c2 + 0, vmask0, _mm512_cvtps_ph(vacc2x0, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c1 + 0, vmask0, _mm512_cvtps_ph(vacc1x0, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c0 + 0, vmask0, _mm512_cvtps_ph(vacc0x0, _MM_FROUND_TO_NEAREST_INT));
      nc = 0;
    }
  } while (nc != 0);
}
