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


void xnn_f32_spmm_minmax_ukernel_8x1__sse(
    uint32_t batch_size,
    uint32_t output_channels,
    const float*restrict input,
    const float*restrict weights,
    const int32_t*restrict widx_dmap,
    const uint32_t*restrict nidx_nnzmap,
    float*restrict output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch_size != 0);

  const uintptr_t output_stride = 1 * batch_size * sizeof(float);
  const __m128 vmin = _mm_load_ps(params->sse.min);
  const __m128 vmax = _mm_load_ps(params->sse.max);
  size_t n = batch_size;
  while XNN_LIKELY(n >= 8) {
    const float*restrict w = weights;
    const int32_t* dmap = widx_dmap;
    const uint32_t* nnzmap = nidx_nnzmap;
    size_t c = output_channels;
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
    } while (--c != 0);
    output -= batch_size * output_channels;
    output += 8;
    input += 8;
    n -= 8;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 4) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
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
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 4;
      input += 4;
    }
    if (n & 2) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
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
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 2;
      input += 2;
    }
    if (n & 1) {
      const float*restrict w = weights;
      const int32_t* dmap = widx_dmap;
      const uint32_t* nnzmap = nidx_nnzmap;
      size_t c = output_channels;
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
      } while (--c != 0);
      output -= batch_size * output_channels;
      output += 1;
      input += 1;
    }
  }
}
