// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx8c8-avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c2__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  size_t bl = params->avx.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);

  kc = round_up_po2(kc, 2 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  do {
    const __m256 vksum01234567 = _mm256_loadu_ps((const float*) w);
    const __m256 vinput_zero_point0 = _mm256_set1_ps((const float) quantization_params[0].zero_point);
    __m256 vout0x01234567 = _mm256_mul_ps(vksum01234567, vinput_zero_point0);
    w = (const float*) w + 8;

    size_t n_blocks = kc / bl;
    for (size_t nb=0; nb<n_blocks; ++nb) {
        __m256i vacc0x01234567 = _mm256_setzero_si256();
        size_t k = bl;

        for(; k >= 2 * sizeof(uint8_t); k -= 2 * sizeof(uint8_t)) {
            const __m256i va0c0 = _mm256_set1_epi32((int32_t) a0[0]);
            const __m256i va0c1 = _mm256_set1_epi32((int32_t) a0[1]);
            a0 += 2;

            const __m128i mask = _mm_set1_epi8(0xf0);
            const __m128i vbi = _mm_loadl_epi64((const __m128i*) w);
            const __m256i vb01234567c0 = _mm256_cvtepi8_epi32(_mm_and_si128(_mm_slli_epi32(vbi, 4), mask));
            const __m256i vb01234567c1 = _mm256_cvtepi8_epi32(_mm_and_si128(vbi, mask));

            w += 8;

            vacc0x01234567 = _mm256_add_epi32(vacc0x01234567, _mm256_mullo_epi32(va0c0, vb01234567c0));
            vacc0x01234567 = _mm256_add_epi32(vacc0x01234567, _mm256_mullo_epi32(va0c1, vb01234567c1));
        }

        __m256 vf0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
        const __m256 vfilter_output_scale01234567 = _mm256_loadu_ps((const float*) w);
        vf0x01234567 = _mm256_mul_ps(vf0x01234567, vfilter_output_scale01234567);
        vout0x01234567 = _mm256_add_ps(vout0x01234567, vf0x01234567);

        w = (const float*) w + 8;
    }

    const __m256 div16 = _mm256_set1_ps(1.0f/16);
    vout0x01234567 = _mm256_mul_ps(vout0x01234567, div16);
    
    const __m256 vinput_scale0 = _mm256_set1_ps(quantization_params[0].inv_scale);
    vout0x01234567 = _mm256_mul_ps(vout0x01234567, vinput_scale0);

    const __m256 vbias01234567 = _mm256_loadu_ps((const float*) w);
    vout0x01234567 = _mm256_add_ps(vout0x01234567, vbias01234567);

    w = (const float*) w + 8;

    const __m256 min = _mm256_load_ps(params->avx.min);
    const __m256 max = _mm256_load_ps(params->avx.max);

    vout0x01234567 = _mm256_max_ps(vout0x01234567, min);
    vout0x01234567 = _mm256_min_ps(vout0x01234567, max);

    if XNN_LIKELY(nc >= 8) {
        _mm256_storeu_ps(c0, vout0x01234567);
        a0 = (const int8_t*) ((uintptr_t) a0 - kc);
        c0 = (float*) ((uintptr_t) c0 + cn_stride);
        nc -= 8;
    } else {
        __m128 vout0x0123 = _mm256_castps256_ps128(vout0x01234567);
        if (nc & 4) {
            _mm_storeu_ps(c0, vout0x0123);
            vout0x0123 = _mm256_extractf128_ps(vout0x01234567, 1);
            c0 += 4;
        }
        if (nc & 2) {
            _mm_storel_pi((__m64*) c0, vout0x0123);
            vout0x0123 = _mm_movehl_ps(vout0x0123, vout0x0123);
            c0 += 2;
        }
        if (nc & 1) {
            _mm_store_ss(c0, vout0x0123);
        }
        nc = 0;
    }

  } while (nc != 0);
}

