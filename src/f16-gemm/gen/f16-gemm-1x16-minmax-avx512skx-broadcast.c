// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/avx512skx-broadcast.c.in
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
#include "src/xnnpack/gemm.h"


void xnn_f16_gemm_minmax_ukernel_1x16__avx512skx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* restrict a,
    size_t a_stride,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;

  const __m512 vmin = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m512 vmax = _mm512_cvtph_ps(_mm256_set1_epi16(*(const uint16_t*) &params->scalar.max));
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);


  do {
    __m512 vacc0x0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) w));
    w = (const xnn_float16*) w + 16;

    size_t k = kc;
    do {
      const __m512 va0 = _mm512_cvtph_ps(_mm256_set1_epi16((short) *a0));
      a0 += 1;

      const __m512 vb0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) w));
      w = (const xnn_float16*) w + 16;

      vacc0x0 = _mm512_cvtph_ps(_mm512_cvtps_ph(_mm512_fmadd_ps(va0, vb0, vacc0x0), _MM_FROUND_TO_NEAREST_INT));

      k -= sizeof(uint16_t);
    } while (k != 0);

    vacc0x0 = _mm512_max_ps(vacc0x0, vmin);

    vacc0x0 = _mm512_min_ps(vacc0x0, vmax);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_si256((__m256i*) c0, _mm512_cvtps_ph(vacc0x0, _MM_FROUND_TO_NEAREST_INT));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 0));

      _mm256_mask_storeu_epi16(c0 + 0, vmask0, _mm512_cvtps_ph(vacc0x0, _MM_FROUND_TO_NEAREST_INT));
      nc = 0;
    }
  } while (nc != 0);
}
