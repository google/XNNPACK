// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c8-ssevnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/prefetch.h"


void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_madd_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const __m128i vsign_mask = _mm_set1_epi8(0x80);
  XNN_FORCE_REALIZATION(vsign_mask);
  const __m128i vinput_zero_point0 = _mm_set1_epi32((int) quantization_params[0].zero_point + 128);
  const __m128i vinput_zero_point1 = _mm_set1_epi32((int) quantization_params[1].zero_point + 128);
  const __m128 voutput_min = _mm_set1_ps(params->scalar.min);
  const __m128 voutput_max = _mm_set1_ps(params->scalar.max);
  const __m128i vmask = _mm_set1_epi8(0x0F);
  XNN_FORCE_REALIZATION(vmask);
  do {
    const __m128i vksum0123 = _mm_load_si128(w);
    const __m128i vsum0x0123 = _mm_mullo_epi32(vksum0123, vinput_zero_point0);
    __m128i vacc0x01 = _mm_unpacklo_epi32(vsum0x0123, _mm_setzero_si128());
    __m128i vacc0x23 = _mm_unpackhi_epi32(vsum0x0123, _mm_setzero_si128());
    const __m128i vsum1x0123 = _mm_mullo_epi32(vksum0123, vinput_zero_point1);
    __m128i vacc1x01 = _mm_unpacklo_epi32(vsum1x0123, _mm_setzero_si128());
    __m128i vacc1x23 = _mm_unpackhi_epi32(vsum1x0123, _mm_setzero_si128());
    __m128i vacc1x0x01 = _mm_setzero_si128();
    __m128i vacc1x0x23 = _mm_setzero_si128();
    __m128i vacc1x1x01 = _mm_setzero_si128();
    __m128i vacc1x1x23 = _mm_setzero_si128();
    w = (const int32_t*) w + 4;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      const __m128i va0x01234567 = _mm_xor_si128(_mm_set1_epi64x((int64_t) unaligned_load_u64(a0)), vsign_mask);
      const __m128i va0x89ABCDEF = _mm_xor_si128(_mm_set1_epi64x((int64_t) unaligned_load_u64(a0 + 8)), vsign_mask);
      a0 += 16;
      const __m128i va1x01234567 = _mm_xor_si128(_mm_set1_epi64x((int64_t) unaligned_load_u64(a1)), vsign_mask);
      const __m128i va1x89ABCDEF = _mm_xor_si128(_mm_set1_epi64x((int64_t) unaligned_load_u64(a1 + 8)), vsign_mask);
      a1 += 16;

      const __m128i vbb01234567x0123 = _mm_load_si128(w);
      const __m128i vbb89ABCDEFx0123 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbs01234567x01 = _mm_srli_epi32(vbb01234567x0123, 4);
      const __m128i vbs89ABCDEFx01 = _mm_srli_epi32(vbb89ABCDEFx0123, 4);
      const __m128i vb01234567x01 = _mm_and_si128(vbb01234567x0123, vmask);
      const __m128i vb89ABCDEFx01 = _mm_and_si128(vbb89ABCDEFx0123, vmask);
      const __m128i vb01234567x23 = _mm_and_si128(vbs01234567x01, vmask);
      const __m128i vb89ABCDEFx23 = _mm_and_si128(vbs89ABCDEFx01, vmask);

      vacc0x01 = _mm_dpbusd_epi32_madd(vacc0x01, va0x01234567, vb01234567x01);
      vacc0x23 = _mm_dpbusd_epi32_madd(vacc0x23, va0x01234567, vb89ABCDEFx01);
      vacc1x01 = _mm_dpbusd_epi32_madd(vacc1x01, va1x01234567, vb01234567x01);
      vacc1x23 = _mm_dpbusd_epi32_madd(vacc1x23, va1x01234567, vb89ABCDEFx01);
      xnn_prefetch_to_l1((const int8_t*) w + 960);
      vacc1x0x01 = _mm_dpbusd_epi32_madd(vacc1x0x01, va0x89ABCDEF, vb01234567x23);
      vacc1x0x23 = _mm_dpbusd_epi32_madd(vacc1x0x23, va0x89ABCDEF, vb89ABCDEFx23);
      vacc1x1x01 = _mm_dpbusd_epi32_madd(vacc1x1x01, va1x89ABCDEF, vb01234567x23);
      vacc1x1x23 = _mm_dpbusd_epi32_madd(vacc1x1x23, va1x89ABCDEF, vb89ABCDEFx23);

      w = (const int8_t*) w + 32;
      k -= 16 * sizeof(int8_t);
    }

    if (k != 0) {
      const __m128i va0x01234567 = _mm_xor_si128(_mm_set1_epi64x((int64_t) unaligned_load_u64(a0)), vsign_mask);
      a0 += 8;
      const __m128i va1x01234567 = _mm_xor_si128(_mm_set1_epi64x((int64_t) unaligned_load_u64(a1)), vsign_mask);
      a1 += 8;

      const __m128i vbb01234567x0123 = _mm_load_si128(w);
      const __m128i vbb89ABCDEFx0123 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vb01234567x01 = _mm_and_si128(vbb01234567x0123, vmask);
      const __m128i vb89ABCDEFx01 = _mm_and_si128(vbb89ABCDEFx0123, vmask);

      vacc0x01 = _mm_dpbusd_epi32_madd(vacc0x01, va0x01234567, vb01234567x01);
      vacc0x23 = _mm_dpbusd_epi32_madd(vacc0x23, va0x01234567, vb89ABCDEFx01);
      vacc1x01 = _mm_dpbusd_epi32_madd(vacc1x01, va1x01234567, vb01234567x01);
      vacc1x23 = _mm_dpbusd_epi32_madd(vacc1x23, va1x01234567, vb89ABCDEFx01);
      xnn_prefetch_to_l1((const int8_t*) w + 960);

      w = (const int8_t*) w + 32;
      k -= 8 * sizeof(int8_t);
    }
    vacc0x01 = _mm_add_epi32(vacc0x01, vacc1x0x01);
    vacc0x23 = _mm_add_epi32(vacc0x23, vacc1x0x23);
    vacc1x01 = _mm_add_epi32(vacc1x01, vacc1x1x01);
    vacc1x23 = _mm_add_epi32(vacc1x23, vacc1x1x23);

    // Add adjacent pairs
    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
    __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);

    __m128 vout0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vout1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    vout0x0123 = _mm_mul_ps(vout0x0123, _mm_set1_ps(quantization_params[0].inv_scale));
    vout1x0123 = _mm_mul_ps(vout1x0123, _mm_set1_ps(quantization_params[1].inv_scale));

    const __m128 vfilter_output_scale0123 = _mm_load_ps((const float*) w);
    const __m128 vbias0123 = _mm_load_ps((const float*) w + 4);
    w = (const float*) w + 8;

    vout0x0123 = _mm_add_ps(_mm_mul_ps(vout0x0123, vfilter_output_scale0123), vbias0123);
    vout1x0123 = _mm_add_ps(_mm_mul_ps(vout1x0123, vfilter_output_scale0123), vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, voutput_min);
    vout1x0123 = _mm_max_ps(vout1x0123, voutput_min);

    vout0x0123 = _mm_min_ps(vout0x0123, voutput_max);
    vout1x0123 = _mm_min_ps(vout1x0123, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm_storeu_ps(c1, vout1x0123);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        c0 += 2;
        vout0x0123 = _mm_movehl_ps(vout0x0123, vout0x0123);
        _mm_storel_pi((__m64*) c1, vout1x0123);
        c1 += 2;
        vout1x0123 = _mm_movehl_ps(vout1x0123, vout1x0123);
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
        _mm_store_ss(c1, vout1x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}
