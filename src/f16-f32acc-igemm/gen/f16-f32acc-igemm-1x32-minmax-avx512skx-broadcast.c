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


void xnn_f16_f32acc_igemm_minmax_ukernel_1x32__avx512skx_broadcast(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const __m512 vmin = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m512 vmax = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.max));
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* c0 = (uint16_t*) c;

  do {
    __m512 vacc0x0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) w));
    __m512 vacc0x1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) ((const uint16_t*) w + 16)));
    w = (const xnn_float16*) w + 32;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != (const uint16_t*) zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const __m512 vb0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) w));
        const __m512 vb1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) ((const uint16_t*) w + 16)));
        w = (const xnn_float16*) w + 32;

        const __m512 va0 = _mm512_cvtph_ps(_mm256_set1_epi16((short) *a0));
        a0 += 1;

        vacc0x0 = _mm512_fmadd_ps(va0, vb0, vacc0x0);
        vacc0x1 = _mm512_fmadd_ps(va0, vb1, vacc0x1);

        k -= sizeof(uint16_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x0 = _mm512_max_ps(vacc0x0, vmin);
    vacc0x1 = _mm512_max_ps(vacc0x1, vmin);

    vacc0x0 = _mm512_min_ps(vacc0x0, vmax);
    vacc0x1 = _mm512_min_ps(vacc0x1, vmax);

    if XNN_LIKELY(nc >= 32) {
      _mm256_storeu_si256((__m256i*) c0, _mm512_cvtps_ph(vacc0x0, _MM_FROUND_TO_NEAREST_INT));
      _mm256_storeu_si256((__m256i*) (c0 + 16), _mm512_cvtps_ph(vacc0x1, _MM_FROUND_TO_NEAREST_INT));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const xnn_float16**restrict) ((uintptr_t) a - ks);
      nc -= 32;
    } else {
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 0));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 16));

      _mm256_mask_storeu_epi16(c0 + 0, vmask0, _mm512_cvtps_ph(vacc0x0, _MM_FROUND_TO_NEAREST_INT));
      _mm256_mask_storeu_epi16(c0 + 16, vmask1, _mm512_cvtps_ph(vacc0x1, _MM_FROUND_TO_NEAREST_INT));
      nc = 0;
    }
  } while (nc != 0);
}
