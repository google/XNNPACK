// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx16c8-avx512vnni.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>
#include <xnnpack/prefetch.h>


void xnn_qs8_igemm_minmax_fp32_ukernel_6x16c8__avx512vnni_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  int8_t* c4 = (int8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  int8_t* c5 = (int8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  const __m512i vsign_mask = _mm512_set1_epi8(params->fp32_avx512vnni.sign_mask);  // 0x80
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512vnni.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512vnni.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512vnni.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512vnni.output_min);
  const __m128i vshuffle_control_mask = _mm_loadu_si128((const __m128i*) params->fp32_avx512vnni.shuffle_control_mask);
  do {
    __m512i vacc0x01234567 = _mm512_cvtepu32_epi64(_mm256_load_si256((const __m256i*) w));
    __m512i vacc0x89ABCDEF = _mm512_cvtepu32_epi64(_mm256_load_si256((const __m256i*) ((const int32_t*) w + 8)));
    __m512i vacc1x01234567 = vacc0x01234567;
    __m512i vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc2x01234567 = vacc0x01234567;
    __m512i vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc3x01234567 = vacc0x01234567;
    __m512i vacc3x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc4x01234567 = vacc0x01234567;
    __m512i vacc4x89ABCDEF = vacc0x89ABCDEF;
    __m512i vacc5x01234567 = vacc0x01234567;
    __m512i vacc5x89ABCDEF = vacc0x89ABCDEF;
    w = (const int32_t*) w + 16;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      }
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      while (k >= 16 * sizeof(int8_t)) {
        const __m512i va0x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a0)), vsign_mask);
        const __m512i va0x89ABCDEF = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) (a0 + 8))), vsign_mask);
        a0 += 16;
        const __m512i va1x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a1)), vsign_mask);
        const __m512i va1x89ABCDEF = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) (a1 + 8))), vsign_mask);
        a1 += 16;
        const __m512i va2x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a2)), vsign_mask);
        const __m512i va2x89ABCDEF = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) (a2 + 8))), vsign_mask);
        a2 += 16;
        const __m512i va3x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a3)), vsign_mask);
        const __m512i va3x89ABCDEF = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) (a3 + 8))), vsign_mask);
        a3 += 16;
        const __m512i va4x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a4)), vsign_mask);
        const __m512i va4x89ABCDEF = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) (a4 + 8))), vsign_mask);
        a4 += 16;
        const __m512i va5x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a5)), vsign_mask);
        const __m512i va5x89ABCDEF = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) (a5 + 8))), vsign_mask);
        a5 += 16;

        const __m512i vb01234567x01234567 = _mm512_load_si512(w);
        const __m512i vb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);
        const __m512i vb01234567x89ABCDEF = _mm512_load_si512((const int8_t*) w + 128);
        const __m512i vb89ABCDEFx89ABCDEF = _mm512_load_si512((const int8_t*) w + 192);
        xnn_prefetch_to_l1((const int8_t*) w + 768);
        xnn_prefetch_to_l1((const int8_t*) w + 832);

        vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
        vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
        vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, va1x01234567, vb01234567x01234567);
        vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, va1x01234567, vb89ABCDEFx01234567);
        vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, va2x01234567, vb01234567x01234567);
        vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, va2x01234567, vb89ABCDEFx01234567);
        vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, va3x01234567, vb01234567x01234567);
        vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, va3x01234567, vb89ABCDEFx01234567);
        vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, va4x01234567, vb01234567x01234567);
        vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, va4x01234567, vb89ABCDEFx01234567);
        vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, va5x01234567, vb01234567x01234567);
        vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, va5x01234567, vb89ABCDEFx01234567);
        xnn_prefetch_to_l1((const int8_t*) w + 896);
        xnn_prefetch_to_l1((const int8_t*) w + 960);
        vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x89ABCDEF, vb01234567x89ABCDEF);
        vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, va1x89ABCDEF, vb01234567x89ABCDEF);
        vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, va1x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, va2x89ABCDEF, vb01234567x89ABCDEF);
        vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, va2x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, va3x89ABCDEF, vb01234567x89ABCDEF);
        vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, va3x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, va4x89ABCDEF, vb01234567x89ABCDEF);
        vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, va4x89ABCDEF, vb89ABCDEFx89ABCDEF);
        vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, va5x89ABCDEF, vb01234567x89ABCDEF);
        vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, va5x89ABCDEF, vb89ABCDEFx89ABCDEF);

        w = (const int8_t*) w + 256;
        k -= 16 * sizeof(int8_t);
      }

      if (k != 0) {
        const __m512i va0x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a0)), vsign_mask);
        a0 += 8;
        const __m512i va1x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a1)), vsign_mask);
        a1 += 8;
        const __m512i va2x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a2)), vsign_mask);
        a2 += 8;
        const __m512i va3x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a3)), vsign_mask);
        a3 += 8;
        const __m512i va4x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a4)), vsign_mask);
        a4 += 8;
        const __m512i va5x01234567 = _mm512_xor_epi64(_mm512_broadcast_i32x2(_mm_loadl_epi64((const __m128i*) a5)), vsign_mask);
        a5 += 8;

        const __m512i vb01234567x01234567 = _mm512_load_si512(w);
        const __m512i vb89ABCDEFx01234567 = _mm512_load_si512((const int8_t*) w + 64);

        vacc0x01234567 = _mm512_dpbusd_epi32(vacc0x01234567, va0x01234567, vb01234567x01234567);
        vacc0x89ABCDEF = _mm512_dpbusd_epi32(vacc0x89ABCDEF, va0x01234567, vb89ABCDEFx01234567);
        vacc1x01234567 = _mm512_dpbusd_epi32(vacc1x01234567, va1x01234567, vb01234567x01234567);
        vacc1x89ABCDEF = _mm512_dpbusd_epi32(vacc1x89ABCDEF, va1x01234567, vb89ABCDEFx01234567);
        vacc2x01234567 = _mm512_dpbusd_epi32(vacc2x01234567, va2x01234567, vb01234567x01234567);
        vacc2x89ABCDEF = _mm512_dpbusd_epi32(vacc2x89ABCDEF, va2x01234567, vb89ABCDEFx01234567);
        vacc3x01234567 = _mm512_dpbusd_epi32(vacc3x01234567, va3x01234567, vb01234567x01234567);
        vacc3x89ABCDEF = _mm512_dpbusd_epi32(vacc3x89ABCDEF, va3x01234567, vb89ABCDEFx01234567);
        vacc4x01234567 = _mm512_dpbusd_epi32(vacc4x01234567, va4x01234567, vb01234567x01234567);
        vacc4x89ABCDEF = _mm512_dpbusd_epi32(vacc4x89ABCDEF, va4x01234567, vb89ABCDEFx01234567);
        vacc5x01234567 = _mm512_dpbusd_epi32(vacc5x01234567, va5x01234567, vb01234567x01234567);
        vacc5x89ABCDEF = _mm512_dpbusd_epi32(vacc5x89ABCDEF, va5x01234567, vb89ABCDEFx01234567);
        xnn_prefetch_to_l1((const int8_t*) w + 896);
        xnn_prefetch_to_l1((const int8_t*) w + 960);

        w = (const int8_t*) w + 128;
        k -= 8 * sizeof(int8_t);
      }

      p -= 6 * sizeof(void*);
    } while (p != 0);

    // Add adjacent pairs
    const __m512i vidx = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i vsum0x01234567 = _mm512_add_epi32(vacc0x01234567, _mm512_srai_epi64(vacc0x01234567, 32));
    const __m512i vsum0x89ABCDEF = _mm512_add_epi32(vacc0x89ABCDEF, _mm512_srai_epi64(vacc0x89ABCDEF, 32));
    __m512i vacc0x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum0x01234567, vidx, vsum0x89ABCDEF);
    const __m512i vsum1x01234567 = _mm512_add_epi32(vacc1x01234567, _mm512_srai_epi64(vacc1x01234567, 32));
    const __m512i vsum1x89ABCDEF = _mm512_add_epi32(vacc1x89ABCDEF, _mm512_srai_epi64(vacc1x89ABCDEF, 32));
    __m512i vacc1x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum1x01234567, vidx, vsum1x89ABCDEF);
    const __m512i vsum2x01234567 = _mm512_add_epi32(vacc2x01234567, _mm512_srai_epi64(vacc2x01234567, 32));
    const __m512i vsum2x89ABCDEF = _mm512_add_epi32(vacc2x89ABCDEF, _mm512_srai_epi64(vacc2x89ABCDEF, 32));
    __m512i vacc2x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum2x01234567, vidx, vsum2x89ABCDEF);
    const __m512i vsum3x01234567 = _mm512_add_epi32(vacc3x01234567, _mm512_srai_epi64(vacc3x01234567, 32));
    const __m512i vsum3x89ABCDEF = _mm512_add_epi32(vacc3x89ABCDEF, _mm512_srai_epi64(vacc3x89ABCDEF, 32));
    __m512i vacc3x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum3x01234567, vidx, vsum3x89ABCDEF);
    const __m512i vsum4x01234567 = _mm512_add_epi32(vacc4x01234567, _mm512_srai_epi64(vacc4x01234567, 32));
    const __m512i vsum4x89ABCDEF = _mm512_add_epi32(vacc4x89ABCDEF, _mm512_srai_epi64(vacc4x89ABCDEF, 32));
    __m512i vacc4x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum4x01234567, vidx, vsum4x89ABCDEF);
    const __m512i vsum5x01234567 = _mm512_add_epi32(vacc5x01234567, _mm512_srai_epi64(vacc5x01234567, 32));
    const __m512i vsum5x89ABCDEF = _mm512_add_epi32(vacc5x89ABCDEF, _mm512_srai_epi64(vacc5x89ABCDEF, 32));
    __m512i vacc5x0123456789ABCDEF = _mm512_permutex2var_epi32(vsum5x01234567, vidx, vsum5x89ABCDEF);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
    __m512 vscaled5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vscale);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vscale);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vscale);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vscale);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vscale);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled5x0123456789ABCDEF);

    __m256i vacc0x0123456789AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0x0123456789ABCDEF, 1));
    __m256i vacc1x0123456789AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc1x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc1x0123456789ABCDEF, 1));
    __m256i vacc2x0123456789AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc2x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc2x0123456789ABCDEF, 1));
    __m256i vacc3x0123456789AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc3x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc3x0123456789ABCDEF, 1));
    __m256i vacc4x0123456789AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc4x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc4x0123456789ABCDEF, 1));
    __m256i vacc5x0123456789AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc5x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc5x0123456789ABCDEF, 1));

    vacc0x0123456789AB4567CDEF = _mm256_adds_epi16(vacc0x0123456789AB4567CDEF, voutput_zero_point);
    vacc1x0123456789AB4567CDEF = _mm256_adds_epi16(vacc1x0123456789AB4567CDEF, voutput_zero_point);
    vacc2x0123456789AB4567CDEF = _mm256_adds_epi16(vacc2x0123456789AB4567CDEF, voutput_zero_point);
    vacc3x0123456789AB4567CDEF = _mm256_adds_epi16(vacc3x0123456789AB4567CDEF, voutput_zero_point);
    vacc4x0123456789AB4567CDEF = _mm256_adds_epi16(vacc4x0123456789AB4567CDEF, voutput_zero_point);
    vacc5x0123456789AB4567CDEF = _mm256_adds_epi16(vacc5x0123456789AB4567CDEF, voutput_zero_point);

    const __m128i vout0x0123456789AB4567CDEF = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x0123456789AB4567CDEF), _mm256_extracti128_si256(vacc0x0123456789AB4567CDEF, 1));
    const __m128i vout1x0123456789AB4567CDEF = _mm_packs_epi16(_mm256_castsi256_si128(vacc1x0123456789AB4567CDEF), _mm256_extracti128_si256(vacc1x0123456789AB4567CDEF, 1));
    const __m128i vout2x0123456789AB4567CDEF = _mm_packs_epi16(_mm256_castsi256_si128(vacc2x0123456789AB4567CDEF), _mm256_extracti128_si256(vacc2x0123456789AB4567CDEF, 1));
    const __m128i vout3x0123456789AB4567CDEF = _mm_packs_epi16(_mm256_castsi256_si128(vacc3x0123456789AB4567CDEF), _mm256_extracti128_si256(vacc3x0123456789AB4567CDEF, 1));
    const __m128i vout4x0123456789AB4567CDEF = _mm_packs_epi16(_mm256_castsi256_si128(vacc4x0123456789AB4567CDEF), _mm256_extracti128_si256(vacc4x0123456789AB4567CDEF, 1));
    const __m128i vout5x0123456789AB4567CDEF = _mm_packs_epi16(_mm256_castsi256_si128(vacc5x0123456789AB4567CDEF), _mm256_extracti128_si256(vacc5x0123456789AB4567CDEF, 1));

    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x0123456789AB4567CDEF, vshuffle_control_mask);
    __m128i vout1x0123456789ABCDEF = _mm_shuffle_epi8(vout1x0123456789AB4567CDEF, vshuffle_control_mask);
    __m128i vout2x0123456789ABCDEF = _mm_shuffle_epi8(vout2x0123456789AB4567CDEF, vshuffle_control_mask);
    __m128i vout3x0123456789ABCDEF = _mm_shuffle_epi8(vout3x0123456789AB4567CDEF, vshuffle_control_mask);
    __m128i vout4x0123456789ABCDEF = _mm_shuffle_epi8(vout4x0123456789AB4567CDEF, vshuffle_control_mask);
    __m128i vout5x0123456789ABCDEF = _mm_shuffle_epi8(vout5x0123456789AB4567CDEF, vshuffle_control_mask);

    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = _mm_max_epi8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = _mm_max_epi8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = _mm_max_epi8(vout3x0123456789ABCDEF, voutput_min);
    vout4x0123456789ABCDEF = _mm_max_epi8(vout4x0123456789ABCDEF, voutput_min);
    vout5x0123456789ABCDEF = _mm_max_epi8(vout5x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c5, vout5x0123456789ABCDEF);
      c5 = (int8_t*) ((uintptr_t) c5 + cn_stride);
      _mm_storeu_si128((__m128i*) c4, vout4x0123456789ABCDEF);
      c4 = (int8_t*) ((uintptr_t) c4 + cn_stride);
      _mm_storeu_si128((__m128i*) c3, vout3x0123456789ABCDEF);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      _mm_storeu_si128((__m128i*) c2, vout2x0123456789ABCDEF);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      _mm_storeu_si128((__m128i*) c1, vout1x0123456789ABCDEF);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - UINT32_C(1));
      _mm_mask_storeu_epi8(c5, vmask, vout5x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c4, vmask, vout4x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c3, vmask, vout3x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c2, vmask, vout2x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c1, vmask, vout1x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
