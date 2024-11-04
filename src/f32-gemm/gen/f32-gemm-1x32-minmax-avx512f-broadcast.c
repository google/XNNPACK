// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/avx512-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f32_gemm_minmax_ukernel_1x32__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  do {
    __m512 vacc0x0 = _mm512_load_ps(w + 0);
    __m512 vacc0x1 = _mm512_load_ps(w + 16);
    w += 32;

    size_t k = kc;
    do {
      const __m512 vb0 = _mm512_load_ps(w + 0);
      const __m512 vb1 = _mm512_load_ps(w + 16);
      w += 32;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0 = _mm512_fmadd_ps(va0, vb0, vacc0x0);
      vacc0x1 = _mm512_fmadd_ps(va0, vb1, vacc0x1);

      a0 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0 = _mm512_max_ps(vmin, vacc0x0);
    vacc0x1 = _mm512_max_ps(vmin, vacc0x1);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0 = _mm512_min_ps(vmax, vacc0x0);
    vacc0x1 = _mm512_min_ps(vmax, vacc0x1);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c0 + 0, vacc0x0);
      _mm512_storeu_ps(c0 + 16, vacc0x1);
      a0 = (const float*) ((uintptr_t) a0 - kc);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      nc -= 32;
    } else {
      // NC remainder (1..31)
      assert(nc >= 1);
      assert(nc <= 31);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));

      _mm_mask_storeu_epi32(c0 + 0, vmask0, vacc0x0);
      _mm_mask_storeu_epi32(c0 + 16, vmask1, vacc0x1);

      nc = 0;
    }
  } while (nc != 0);
}
