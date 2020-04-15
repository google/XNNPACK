// Auto-generated file. Do not edit!
//   Template: src/f32-spmm/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/spmm.h>


void xnn_f32_spmm_minmax_ukernel_4x1__sse(
    uint32_t m,
    uint32_t n,
    const float*restrict a,
    const float*restrict weights,
    const int32_t*restrict widx_dmap,
    const uint32_t*restrict nidx_nnzmap,
    float*restrict c,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(m != 0);

  const __m128 vmin = _mm_load_ps(params->sse.min);
  const __m128 vmax = _mm_load_ps(params->sse.max);
  size_t i = m;
  while XNN_LIKELY(i >= 4) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t j = n;
    do {
      uint32_t nnz = *nnzmap++;
      __m128 vacc0123 = _mm_load1_ps(w); w += 1;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const __m128 va0123 = _mm_loadu_ps(a);
          a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
          const __m128 vb = _mm_load1_ps(w); w += 1;
          vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(va0123, vb));
        } while (--nnz != 0);
      }
      __m128 vout0123 = _mm_min_ps(vacc0123, vmax);
      vout0123 = _mm_max_ps(vout0123, vmin);
      _mm_storeu_ps(c, vout0123);
      c += 1 * m;
    } while (--j != 0);
    c -= m * n;
    c += 4;
    a += 4;
    i -= 4;
  }
  if XNN_UNLIKELY(i != 0) {
    if (i & 2) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t j = n;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc01 = _mm_load_ss(w); w += 1;
        vacc01 = _mm_unpacklo_ps(vacc01, vacc01);
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 va01 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) a);
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            __m128 vb = _mm_load_ss(w); w += 1;
            vb = _mm_unpacklo_ps(vb, vb);
            vacc01 = _mm_add_ps(vacc01, _mm_mul_ps(va01, vb));
          } while (--nnz != 0);
        }
        __m128 vout01 = _mm_min_ps(vacc01, vmax);
        vout01 = _mm_max_ps(vout01, vmin);
        _mm_storel_pi((__m64*) c, vout01);
        c += 1 * m;
      } while (--j != 0);
      c -= m * n;
      c += 2;
      a += 2;
    }
    if (i & 1) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t j = n;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc0 = _mm_load_ss(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 va0 = _mm_load_ss(a);
            a = (const float*restrict) ((uintptr_t) a + (uintptr_t) diff);
            const __m128 vb = _mm_load_ss(w); w += 1;
            vacc0 = _mm_add_ss(vacc0, _mm_mul_ss(va0, vb));
          } while (--nnz != 0);
        }
        __m128 vout0 = _mm_min_ss(vacc0, vmax);
        vout0 = _mm_max_ss(vout0, vmin);
        _mm_store_ss(c, vout0);
        c += 1 * m;
      } while (--j != 0);
      c -= m * n;
      c += 1;
      a += 1;
    }
  }
}
