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

#include "xnnpack/spmm.h"


void xnn_f32_spmm_minmax_ukernel_4x1__sse(
    size_t mc,
    size_t nc,
    const float* input,
    const float* weights,
    const int32_t* widx_dmap,
    const uint32_t* nidx_nnzmap,
    float* output,
    size_t output_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mc != 0);
  assert(mc % sizeof(float) == 0);
  assert(nc != 0);

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  size_t output_decrement = output_stride * nc - 4 * sizeof(float);
  while XNN_LIKELY(mc >= 4 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      __m128 vacc0123 = _mm_load1_ps(w); w += 1;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const __m128 vi0123 = _mm_loadu_ps(input);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const __m128 vw = _mm_load1_ps(w); w += 1;
          vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(vi0123, vw));
        } while (--nnz != 0);
      }
      __m128 vout0123 = _mm_min_ps(vacc0123, vmax);
      vout0123 = _mm_max_ps(vout0123, vmin);
      _mm_storeu_ps(output, vout0123);
      output = (float*restrict) ((uintptr_t) output + output_stride);
    } while (--n != 0);
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 4;
    mc -= 4 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 2 * sizeof(float);
    if (mc & (2 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc01 = _mm_load_ss(w); w += 1;
        vacc01 = _mm_unpacklo_ps(vacc01, vacc01);
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 vi01 = _mm_loadl_pi(_mm_undefined_ps(), (const __m64*) input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            __m128 vw = _mm_load_ss(w); w += 1;
            vw = _mm_unpacklo_ps(vw, vw);
            vacc01 = _mm_add_ps(vacc01, _mm_mul_ps(vi01, vw));
          } while (--nnz != 0);
        }
        __m128 vout01 = _mm_min_ps(vacc01, vmax);
        vout01 = _mm_max_ps(vout01, vmin);
        _mm_storel_pi((__m64*) output, vout01);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 2;
    }
    output_decrement += 1 * sizeof(float);
    if (mc & (1 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc0 = _mm_load_ss(w); w += 1;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 vi0 = _mm_load_ss(input);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const __m128 vw = _mm_load_ss(w); w += 1;
            vacc0 = _mm_add_ss(vacc0, _mm_mul_ss(vi0, vw));
          } while (--nnz != 0);
        }
        __m128 vout0 = _mm_min_ss(vacc0, vmax);
        vout0 = _mm_max_ss(vout0, vmin);
        _mm_store_ss(output, vout0);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 1;
    }
  }
}
