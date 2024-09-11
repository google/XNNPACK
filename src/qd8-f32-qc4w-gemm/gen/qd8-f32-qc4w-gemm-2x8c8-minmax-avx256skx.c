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

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx256skx(
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

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);
  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  const __m256 vmax = _mm256_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    const __m128i vinit0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vinit1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    const __m256i vinit01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vinit0), vinit1, 1);
    const __m128i vinit2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vinit3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    const __m256i vinit23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vinit2), vinit3, 1);
    const __m128i vinit4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vinit5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    const __m256i vinit45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vinit4), vinit5, 1);
    const __m128i vinit6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vinit7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    const __m256i vinit67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vinit6), vinit7, 1);
    const __m256i vinput_zero_point0 = _mm256_set1_epi32((int) quantization_params[0].zero_point);
    __m256i vacc0x01 = _mm256_mullo_epi32(vinit01, vinput_zero_point0);
    __m256i vacc0x23 = _mm256_mullo_epi32(vinit23, vinput_zero_point0);
    __m256i vacc0x45 = _mm256_mullo_epi32(vinit45, vinput_zero_point0);
    __m256i vacc0x67 = _mm256_mullo_epi32(vinit67, vinput_zero_point0);
    const __m256i vinput_zero_point1 = _mm256_set1_epi32((int) quantization_params[1].zero_point);
    __m256i vacc1x01 = _mm256_mullo_epi32(vinit01, vinput_zero_point1);
    __m256i vacc1x23 = _mm256_mullo_epi32(vinit23, vinput_zero_point1);
    __m256i vacc1x45 = _mm256_mullo_epi32(vinit45, vinput_zero_point1);
    __m256i vacc1x67 = _mm256_mullo_epi32(vinit67, vinput_zero_point1);
    w = (const int32_t*) w + 8;

    size_t k = kc;
    while (k >= 16 * sizeof(int8_t)) {
      __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
      a0 += 8;
      __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
      __m256i vxa1 = _mm256_cvtepi8_epi16(va1);
      a1 += 8;

      __m128i vb01 = _mm_load_si128((const __m128i*) w);
      __m128i vbs01 = _mm_slli_epi32(vb01, 4);
      __m128i vbm01 = _mm_and_si128(vbs01, vmask);
      __m256i vxb01 = _mm256_cvtepi8_epi16(vbm01);

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
      __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      __m128i vbs23 = _mm_slli_epi32(vb23, 4);
      __m128i vbm23 = _mm_and_si128(vbs23, vmask);
      __m256i vxb23 = _mm256_cvtepi8_epi16(vbm23);

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
      __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
      __m128i vbs45 = _mm_slli_epi32(vb45, 4);
      __m128i vbm45 = _mm_and_si128(vbs45, vmask);
      __m256i vxb45 = _mm256_cvtepi8_epi16(vbm45);

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
      __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
      __m128i vbs67 = _mm_slli_epi32(vb67, 4);
      __m128i vbm67 = _mm_and_si128(vbs67, vmask);
      __m256i vxb67 = _mm256_cvtepi8_epi16(vbm67);

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
      vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));

      va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      vxa0 = _mm256_cvtepi8_epi16(va0);
      a0 += 8;
      va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
      vxa1 = _mm256_cvtepi8_epi16(va1);
      a1 += 8;

      vbm01 = _mm_and_si128(vb01, vmask);
      vxb01 = _mm256_cvtepi8_epi16(vbm01);

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
      vbm23 = _mm_and_si128(vb23, vmask);
      vxb23 = _mm256_cvtepi8_epi16(vbm23);

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
      vbm45 = _mm_and_si128(vb45, vmask);
      vxb45 = _mm256_cvtepi8_epi16(vbm45);

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
      vbm67 = _mm_and_si128(vb67, vmask);
      vxb67 = _mm256_cvtepi8_epi16(vbm67);

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
      vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));

      w = (const int8_t*) w + 64;
      k -= 16 * sizeof(int8_t);
    }

    while (k >= 8 * sizeof(int8_t)) {
      const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      const __m256i vxa0 = _mm256_cvtepi8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
      const __m256i vxa1 = _mm256_cvtepi8_epi16(va1);
      a1 += 8;

      const __m128i vb01 = _mm_load_si128((const __m128i*) w);
      const __m128i vbs01 = _mm_slli_epi32(vb01, 4);
      const __m128i vbm01 = _mm_and_si128(vbs01, vmask);
      const __m256i vxb01 = _mm256_cvtepi8_epi16(vbm01);

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
      const __m128i vb23 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 16));
      const __m128i vbs23 = _mm_slli_epi32(vb23, 4);
      const __m128i vbm23 = _mm_and_si128(vbs23, vmask);
      const __m256i vxb23 = _mm256_cvtepi8_epi16(vbm23);

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
      const __m128i vb45 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 32));
      const __m128i vbs45 = _mm_slli_epi32(vb45, 4);
      const __m128i vbm45 = _mm_and_si128(vbs45, vmask);
      const __m256i vxb45 = _mm256_cvtepi8_epi16(vbm45);

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
      const __m128i vb67 = _mm_load_si128((const __m128i*) ((const int8_t*) w + 48));
      const __m128i vbs67 = _mm_slli_epi32(vb67, 4);
      const __m128i vbm67 = _mm_and_si128(vbs67, vmask);
      const __m256i vxb67 = _mm256_cvtepi8_epi16(vbm67);

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
      vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));

      w = (const int8_t*) w + 64;
      k -= 8 * sizeof(int8_t);
    }

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);

    vacc0x01234567 = _mm256_srai_epi32(vacc0x01234567, 4);
    vacc1x01234567 = _mm256_srai_epi32(vacc1x01234567, 4);
    __m256 vout0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    const __m256 vinput_scale0 = _mm256_broadcast_ss(&quantization_params[0].inv_scale);
    __m256 vout1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    const __m256 vinput_scale1 = _mm256_broadcast_ss(&quantization_params[1].inv_scale);

    vout0x01234567 = _mm256_mul_ps(vout0x01234567, vinput_scale0);
    vout1x01234567 = _mm256_mul_ps(vout1x01234567, vinput_scale1);

    const __m256 vfilter_output_scale01234567 = _mm256_load_ps((const float*) w);
    const __m256 vbias01234567 = _mm256_load_ps((const float*) w + 8);
    w = (const float*) w + 16;
    vout0x01234567 = _mm256_fmadd_ps(vout0x01234567, vfilter_output_scale01234567, vbias01234567);
    vout1x01234567 = _mm256_fmadd_ps(vout1x01234567, vfilter_output_scale01234567, vbias01234567);

    vout0x01234567 = _mm256_max_ps(vout0x01234567, vmin);
    vout1x01234567 = _mm256_max_ps(vout1x01234567, vmin);

    vout0x01234567 = _mm256_min_ps(vout0x01234567, vmax);
    vout1x01234567 = _mm256_min_ps(vout1x01234567, vmax);

    if XNN_LIKELY(nc >= 8) {
      _mm256_storeu_ps(c0, vout0x01234567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm256_storeu_ps(c1, vout1x01234567);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 8;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask8 vmask = _cvtu32_mask8((UINT32_C(1) << nc) - 1);
      _mm256_mask_storeu_ps(c0, vmask, vout0x01234567);
      _mm256_mask_storeu_ps(c1, vmask, vout1x01234567);
      nc = 0;
    }
  } while (nc != 0);
}
