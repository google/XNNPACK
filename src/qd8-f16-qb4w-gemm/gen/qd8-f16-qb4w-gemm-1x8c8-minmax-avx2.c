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

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__avx2(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint16_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  size_t bl = params->avx.blocksize;
  assert(bl <= round_up_po2(kc, 16));
  assert(bl != 0);
  const int8_t* a0 = a;
  uint16_t* c0 = (uint16_t*) c;

  const __m128i vmask = _mm_load_si128((const __m128i*) params->avx.mask);  // 0xF0
  do {
    const __m128 vinit0 = _mm_load_ss(&((const float*) w)[0]);
    const __m128 vinit1 = _mm_load_ss(&((const float*) w)[1]);
    const __m256 vinit01 = _mm256_insertf128_ps(_mm256_castps128_ps256(vinit0), vinit1, 1);
    const __m128 vinit2 = _mm_load_ss(&((const float*) w)[2]);
    const __m128 vinit3 = _mm_load_ss(&((const float*) w)[3]);
    const __m256 vinit23 = _mm256_insertf128_ps(_mm256_castps128_ps256(vinit2), vinit3, 1);
    const __m128 vinit4 = _mm_load_ss(&((const float*) w)[4]);
    const __m128 vinit5 = _mm_load_ss(&((const float*) w)[5]);
    const __m256 vinit45 = _mm256_insertf128_ps(_mm256_castps128_ps256(vinit4), vinit5, 1);
    const __m128 vinit6 = _mm_load_ss(&((const float*) w)[6]);
    const __m128 vinit7 = _mm_load_ss(&((const float*) w)[7]);
    const __m256 vinit67 = _mm256_insertf128_ps(_mm256_castps128_ps256(vinit6), vinit7, 1);
    const __m256 vinput_zero_point0 = _mm256_set1_ps((float) quantization_params[0].zero_point);
    __m256 vout0x01 = _mm256_mul_ps(vinit01, vinput_zero_point0);
    __m256 vout0x23 = _mm256_mul_ps(vinit23, vinput_zero_point0);
    __m256 vout0x45 = _mm256_mul_ps(vinit45, vinput_zero_point0);
    __m256 vout0x67 = _mm256_mul_ps(vinit67, vinput_zero_point0);
    w = (const int32_t*) w + 8;

    size_t n_blocks = kc / bl;
    for (size_t nb = 0; nb < n_blocks; ++nb) {
      __m256i vacc0x01 = _mm256_setzero_si256();
      __m256i vacc0x23 = _mm256_setzero_si256();
      __m256i vacc0x45 = _mm256_setzero_si256();
      __m256i vacc0x67 = _mm256_setzero_si256();

      size_t k = bl;
      while (k >= 16 * sizeof(int8_t)) {
        __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
        a0 += 8;

        __m128i vb01 = _mm_load_si128((const __m128i*) w);
        __m128i vbs01 = _mm_slli_epi32(vb01, 4);
        __m128i vbm01 = _mm_and_si128(vbs01, vmask);
        __m256i vxb01 = _mm256_cvtepi8_epi16(vbm01);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        __m128i vbs23 = _mm_slli_epi32(vb23, 4);
        __m128i vbm23 = _mm_and_si128(vbs23, vmask);
        __m256i vxb23 = _mm256_cvtepi8_epi16(vbm23);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
        __m128i vbs45 = _mm_slli_epi32(vb45, 4);
        __m128i vbm45 = _mm_and_si128(vbs45, vmask);
        __m256i vxb45 = _mm256_cvtepi8_epi16(vbm45);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
        __m128i vbs67 = _mm_slli_epi32(vb67, 4);
        __m128i vbm67 = _mm_and_si128(vbs67, vmask);
        __m256i vxb67 = _mm256_cvtepi8_epi16(vbm67);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));

        va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        vxa0 = _mm256_cvtepi8_epi16(va0);
        a0 += 8;

        vbm01 = _mm_and_si128(vb01, vmask);
        vxb01 = _mm256_cvtepi8_epi16(vbm01);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        vbm23 = _mm_and_si128(vb23, vmask);
        vxb23 = _mm256_cvtepi8_epi16(vbm23);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        vbm45 = _mm_and_si128(vb45, vmask);
        vxb45 = _mm256_cvtepi8_epi16(vbm45);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        vbm67 = _mm_and_si128(vb67, vmask);
        vxb67 = _mm256_cvtepi8_epi16(vbm67);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));

        w = (const int8_t*) w + 64;
        k -= 16 * sizeof(int8_t);
      }

      while (k >= 8 * sizeof(int8_t)) {
        const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
        const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
        a0 += 8;

        const __m128i vb01 = _mm_load_si128((const __m128i*) w);
        const __m128i vbs01 = _mm_slli_epi32(vb01, 4);
        const __m128i vbm01 = _mm_and_si128(vbs01, vmask);
        const __m256i vxb01 = _mm256_cvtepi8_epi16(vbm01);

        vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
        const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vbs23 = _mm_slli_epi32(vb23, 4);
        const __m128i vbm23 = _mm_and_si128(vbs23, vmask);
        const __m256i vxb23 = _mm256_cvtepi8_epi16(vbm23);

        vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
        const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
        const __m128i vbs45 = _mm_slli_epi32(vb45, 4);
        const __m128i vbm45 = _mm_and_si128(vbs45, vmask);
        const __m256i vxb45 = _mm256_cvtepi8_epi16(vbm45);

        vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
        const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
        const __m128i vbs67 = _mm_slli_epi32(vb67, 4);
        const __m128i vbm67 = _mm_and_si128(vbs67, vmask);
        const __m256i vxb67 = _mm256_cvtepi8_epi16(vbm67);

        vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));

        w = (const int8_t*) w + 64;
        k -= 8 * sizeof(int8_t);
      }

      const __m128 vfilter_output_scale0 = _mm_broadcast_ss(&((const float*) w)[0]);
      const __m128 vfilter_output_scale1 = _mm_broadcast_ss(&((const float*) w)[1]);
      const __m256 vfilter_output_scale01 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(vfilter_output_scale0), vfilter_output_scale1, 1);
      vout0x01 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc0x01), vfilter_output_scale01, vout0x01);
      const __m128 vfilter_output_scale2 = _mm_broadcast_ss(&((const float*) w)[2]);
      const __m128 vfilter_output_scale3 = _mm_broadcast_ss(&((const float*) w)[3]);
      const __m256 vfilter_output_scale23 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(vfilter_output_scale2), vfilter_output_scale3, 1);
      vout0x23 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc0x23), vfilter_output_scale23, vout0x23);
      const __m128 vfilter_output_scale4 = _mm_broadcast_ss(&((const float*) w)[4]);
      const __m128 vfilter_output_scale5 = _mm_broadcast_ss(&((const float*) w)[5]);
      const __m256 vfilter_output_scale45 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(vfilter_output_scale4), vfilter_output_scale5, 1);
      vout0x45 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc0x45), vfilter_output_scale45, vout0x45);
      const __m128 vfilter_output_scale6 = _mm_broadcast_ss(&((const float*) w)[6]);
      const __m128 vfilter_output_scale7 = _mm_broadcast_ss(&((const float*) w)[7]);
      const __m256 vfilter_output_scale67 = _mm256_insertf128_ps(
          _mm256_castps128_ps256(vfilter_output_scale6), vfilter_output_scale7, 1);
      vout0x67 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(vacc0x67), vfilter_output_scale67, vout0x67);

      w = (const float*) w + 8;
    }

    const __m256 vout0x0213 = _mm256_hadd_ps(vout0x01, vout0x23);
    const __m256 vout0x4657 = _mm256_hadd_ps(vout0x45, vout0x67);

    const __m256 vout0x02461357 = _mm256_hadd_ps(vout0x0213, vout0x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256 vout0x01234567 = _mm256_permutevar8x32_ps(vout0x02461357, vpermute_mask);

    vout0x01234567 = _mm256_mul_ps(vout0x01234567, _mm256_set1_ps(1.0f/16));
    const __m256 vinput_scale0 = _mm256_broadcast_ss(&quantization_params[0].inv_scale);

    const __m256 vbias01234567 = _mm256_load_ps((const float*) w);
    w = (const float*) w + 8;
    vout0x01234567 = _mm256_fmadd_ps(vout0x01234567, vinput_scale0, vbias01234567);

    

    const __m256 vmin = _mm256_load_ps(params->avx.min);
    vout0x01234567 = _mm256_max_ps(vout0x01234567, vmin);

    const __m256 vmax = _mm256_load_ps(params->avx.max);
    vout0x01234567 = _mm256_min_ps(vout0x01234567, vmax);
    __m128i vfp16out0x01234567 = _mm256_cvtps_ph(vout0x01234567, _MM_FROUND_TO_NEAREST_INT);
    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_si128((__m128i*) c0, vfp16out0x01234567);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storel_epi64((__m128i*) c0, vfp16out0x01234567);

        vfp16out0x01234567 = _mm_unpackhi_epi64(vfp16out0x01234567, vfp16out0x01234567);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storeu_si32(c0, vfp16out0x01234567);

        vfp16out0x01234567 = _mm_srli_epi64(vfp16out0x01234567, 32);

        c0 += 2;
      }
      if (nc & 1) {
        *c0 = (uint16_t) _mm_extract_epi16(vfp16out0x01234567, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
