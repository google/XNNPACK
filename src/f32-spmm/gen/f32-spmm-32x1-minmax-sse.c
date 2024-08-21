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


void xnn_f32_spmm_minmax_ukernel_32x1__sse(
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

  size_t output_decrement = output_stride * nc - 32 * sizeof(float);
  while XNN_LIKELY(mc >= 32 * sizeof(float)) {
    const float* w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t n = nc;
    do {
      uint32_t nnz = *nnzmap++;
      __m128 vacc0123 = _mm_load1_ps(w); w += 1;
      __m128 vacc4567 = vacc0123;
      __m128 vacc89AB = vacc0123;
      __m128 vaccCDEF = vacc0123;
      __m128 vaccGHIJ = vacc0123;
      __m128 vaccKLMN = vacc0123;
      __m128 vaccOPQR = vacc0123;
      __m128 vaccSTUV = vacc0123;
      if XNN_LIKELY(nnz != 0) {
        do {
          const intptr_t diff = *dmap++;
          const __m128 vi0123 = _mm_loadu_ps(input);
          const __m128 vi4567 = _mm_loadu_ps(input + 4);
          const __m128 vi89AB = _mm_loadu_ps(input + 8);
          const __m128 viCDEF = _mm_loadu_ps(input + 12);
          const __m128 viGHIJ = _mm_loadu_ps(input + 16);
          const __m128 viKLMN = _mm_loadu_ps(input + 20);
          const __m128 viOPQR = _mm_loadu_ps(input + 24);
          const __m128 viSTUV = _mm_loadu_ps(input + 28);
          input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
          const __m128 vw = _mm_load1_ps(w); w += 1;
          vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(vi0123, vw));
          vacc4567 = _mm_add_ps(vacc4567, _mm_mul_ps(vi4567, vw));
          vacc89AB = _mm_add_ps(vacc89AB, _mm_mul_ps(vi89AB, vw));
          vaccCDEF = _mm_add_ps(vaccCDEF, _mm_mul_ps(viCDEF, vw));
          vaccGHIJ = _mm_add_ps(vaccGHIJ, _mm_mul_ps(viGHIJ, vw));
          vaccKLMN = _mm_add_ps(vaccKLMN, _mm_mul_ps(viKLMN, vw));
          vaccOPQR = _mm_add_ps(vaccOPQR, _mm_mul_ps(viOPQR, vw));
          vaccSTUV = _mm_add_ps(vaccSTUV, _mm_mul_ps(viSTUV, vw));
        } while (--nnz != 0);
      }
      __m128 vout0123 = _mm_min_ps(vacc0123, vmax);
      __m128 vout4567 = _mm_min_ps(vacc4567, vmax);
      __m128 vout89AB = _mm_min_ps(vacc89AB, vmax);
      __m128 voutCDEF = _mm_min_ps(vaccCDEF, vmax);
      __m128 voutGHIJ = _mm_min_ps(vaccGHIJ, vmax);
      __m128 voutKLMN = _mm_min_ps(vaccKLMN, vmax);
      __m128 voutOPQR = _mm_min_ps(vaccOPQR, vmax);
      __m128 voutSTUV = _mm_min_ps(vaccSTUV, vmax);
      vout0123 = _mm_max_ps(vout0123, vmin);
      vout4567 = _mm_max_ps(vout4567, vmin);
      vout89AB = _mm_max_ps(vout89AB, vmin);
      voutCDEF = _mm_max_ps(voutCDEF, vmin);
      voutGHIJ = _mm_max_ps(voutGHIJ, vmin);
      voutKLMN = _mm_max_ps(voutKLMN, vmin);
      voutOPQR = _mm_max_ps(voutOPQR, vmin);
      voutSTUV = _mm_max_ps(voutSTUV, vmin);
      _mm_storeu_ps(output, vout0123);
      _mm_storeu_ps(output + 4, vout4567);
      _mm_storeu_ps(output + 8, vout89AB);
      _mm_storeu_ps(output + 12, voutCDEF);
      _mm_storeu_ps(output + 16, voutGHIJ);
      _mm_storeu_ps(output + 20, voutKLMN);
      _mm_storeu_ps(output + 24, voutOPQR);
      _mm_storeu_ps(output + 28, voutSTUV);
      output = (float*restrict) ((uintptr_t) output + output_stride);
    } while (--n != 0);
    output = (float*restrict) ((uintptr_t) output - output_decrement);
    input += 32;
    mc -= 32 * sizeof(float);
  }
  if XNN_UNLIKELY(mc != 0) {
    output_decrement += 16 * sizeof(float);
    if (mc & (16 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc0123 = _mm_load1_ps(w); w += 1;
        __m128 vacc4567 = vacc0123;
        __m128 vacc89AB = vacc0123;
        __m128 vaccCDEF = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 vi0123 = _mm_loadu_ps(input);
            const __m128 vi4567 = _mm_loadu_ps(input + 4);
            const __m128 vi89AB = _mm_loadu_ps(input + 8);
            const __m128 viCDEF = _mm_loadu_ps(input + 12);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const __m128 vw = _mm_load1_ps(w); w += 1;
            vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(vi0123, vw));
            vacc4567 = _mm_add_ps(vacc4567, _mm_mul_ps(vi4567, vw));
            vacc89AB = _mm_add_ps(vacc89AB, _mm_mul_ps(vi89AB, vw));
            vaccCDEF = _mm_add_ps(vaccCDEF, _mm_mul_ps(viCDEF, vw));
          } while (--nnz != 0);
        }
        __m128 vout0123 = _mm_min_ps(vacc0123, vmax);
        __m128 vout4567 = _mm_min_ps(vacc4567, vmax);
        __m128 vout89AB = _mm_min_ps(vacc89AB, vmax);
        __m128 voutCDEF = _mm_min_ps(vaccCDEF, vmax);
        vout0123 = _mm_max_ps(vout0123, vmin);
        vout4567 = _mm_max_ps(vout4567, vmin);
        vout89AB = _mm_max_ps(vout89AB, vmin);
        voutCDEF = _mm_max_ps(voutCDEF, vmin);
        _mm_storeu_ps(output, vout0123);
        _mm_storeu_ps(output + 4, vout4567);
        _mm_storeu_ps(output + 8, vout89AB);
        _mm_storeu_ps(output + 12, voutCDEF);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 16;
    }
    output_decrement += 8 * sizeof(float);
    if (mc & (8 * sizeof(float))) {
      const float* w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t n = nc;
      do {
        uint32_t nnz = *nnzmap++;
        __m128 vacc0123 = _mm_load1_ps(w); w += 1;
        __m128 vacc4567 = vacc0123;
        if XNN_LIKELY(nnz != 0) {
          do {
            const intptr_t diff = *dmap++;
            const __m128 vi0123 = _mm_loadu_ps(input);
            const __m128 vi4567 = _mm_loadu_ps(input + 4);
            input = (const float*restrict) ((uintptr_t) input + (uintptr_t) diff);
            const __m128 vw = _mm_load1_ps(w); w += 1;
            vacc0123 = _mm_add_ps(vacc0123, _mm_mul_ps(vi0123, vw));
            vacc4567 = _mm_add_ps(vacc4567, _mm_mul_ps(vi4567, vw));
          } while (--nnz != 0);
        }
        __m128 vout0123 = _mm_min_ps(vacc0123, vmax);
        __m128 vout4567 = _mm_min_ps(vacc4567, vmax);
        vout0123 = _mm_max_ps(vout0123, vmin);
        vout4567 = _mm_max_ps(vout4567, vmin);
        _mm_storeu_ps(output, vout0123);
        _mm_storeu_ps(output + 4, vout4567);
        output = (float*restrict) ((uintptr_t) output + output_stride);
      } while (--n != 0);
      output = (float*restrict) ((uintptr_t) output - output_decrement);
      input += 8;
    }
    output_decrement += 4 * sizeof(float);
    if (mc & (4 * sizeof(float))) {
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
    }
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
