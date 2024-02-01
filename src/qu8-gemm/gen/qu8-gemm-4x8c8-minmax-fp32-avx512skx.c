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


void xnn_qu8_gemm_minmax_fp32_ukernel_4x8c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __m256i vb_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.kernel_zero_point);
  do {
    const __m128i vbias0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    const __m128i vbias0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m256i vacc0x01 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x0), vbias0x1, 1);
    const __m128i vbias0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    const __m128i vbias0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m256i vacc0x23 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x2), vbias0x3, 1);
    const __m128i vbias0x4 = _mm_cvtsi32_si128(((const int*) w)[4]);
    const __m128i vbias0x5 = _mm_cvtsi32_si128(((const int*) w)[5]);
    __m256i vacc0x45 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x4), vbias0x5, 1);
    const __m128i vbias0x6 = _mm_cvtsi32_si128(((const int*) w)[6]);
    const __m128i vbias0x7 = _mm_cvtsi32_si128(((const int*) w)[7]);
    __m256i vacc0x67 = _mm256_inserti128_si256(_mm256_castsi128_si256(vbias0x6), vbias0x7, 1);
    __m256i vacc1x01 = vacc0x01;
    __m256i vacc1x23 = vacc0x23;
    __m256i vacc1x45 = vacc0x45;
    __m256i vacc1x67 = vacc0x67;
    __m256i vacc2x01 = vacc0x01;
    __m256i vacc2x23 = vacc0x23;
    __m256i vacc2x45 = vacc0x45;
    __m256i vacc2x67 = vacc0x67;
    __m256i vacc3x01 = vacc0x01;
    __m256i vacc3x23 = vacc0x23;
    __m256i vacc3x45 = vacc0x45;
    __m256i vacc3x67 = vacc0x67;
    w = (const int32_t*) w + 8;

    size_t k = kc;

    while (k >= 8 * sizeof(uint8_t)) {
      const __m128i va0 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a0));
      const __m256i vxa0 = _mm256_cvtepu8_epi16(va0);
      a0 += 8;
      const __m128i va1 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a1));
      const __m256i vxa1 = _mm256_cvtepu8_epi16(va1);
      a1 += 8;
      const __m128i va2 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a2));
      const __m256i vxa2 = _mm256_cvtepu8_epi16(va2);
      a2 += 8;
      const __m128i va3 = _mm_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) a3));
      const __m256i vxa3 = _mm256_cvtepu8_epi16(va3);
      a3 += 8;

      const __m256i vxb01 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*) w)), vb_zero_point);

      vacc0x01 = _mm256_add_epi32(vacc0x01, _mm256_madd_epi16(vxa0, vxb01));
      vacc1x01 = _mm256_add_epi32(vacc1x01, _mm256_madd_epi16(vxa1, vxb01));
      vacc2x01 = _mm256_add_epi32(vacc2x01, _mm256_madd_epi16(vxa2, vxb01));
      vacc3x01 = _mm256_add_epi32(vacc3x01, _mm256_madd_epi16(vxa3, vxb01));
      const __m256i vxb23 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*) ((const uint8_t*) w + 16))), vb_zero_point);

      vacc0x23 = _mm256_add_epi32(vacc0x23, _mm256_madd_epi16(vxa0, vxb23));
      vacc1x23 = _mm256_add_epi32(vacc1x23, _mm256_madd_epi16(vxa1, vxb23));
      vacc2x23 = _mm256_add_epi32(vacc2x23, _mm256_madd_epi16(vxa2, vxb23));
      vacc3x23 = _mm256_add_epi32(vacc3x23, _mm256_madd_epi16(vxa3, vxb23));
      const __m256i vxb45 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*) ((const uint8_t*) w + 32))), vb_zero_point);

      vacc0x45 = _mm256_add_epi32(vacc0x45, _mm256_madd_epi16(vxa0, vxb45));
      vacc1x45 = _mm256_add_epi32(vacc1x45, _mm256_madd_epi16(vxa1, vxb45));
      vacc2x45 = _mm256_add_epi32(vacc2x45, _mm256_madd_epi16(vxa2, vxb45));
      vacc3x45 = _mm256_add_epi32(vacc3x45, _mm256_madd_epi16(vxa3, vxb45));
      const __m256i vxb67 = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_load_si128((const __m128i*) ((const uint8_t*) w + 48))), vb_zero_point);

      vacc0x67 = _mm256_add_epi32(vacc0x67, _mm256_madd_epi16(vxa0, vxb67));
      vacc1x67 = _mm256_add_epi32(vacc1x67, _mm256_madd_epi16(vxa1, vxb67));
      vacc2x67 = _mm256_add_epi32(vacc2x67, _mm256_madd_epi16(vxa2, vxb67));
      vacc3x67 = _mm256_add_epi32(vacc3x67, _mm256_madd_epi16(vxa3, vxb67));

      w = (const uint8_t*) w + 64;
      k -= 8 * sizeof(uint8_t);
    }

    const __m256i vacc0x0213 = _mm256_hadd_epi32(vacc0x01, vacc0x23);
    const __m256i vacc0x4657 = _mm256_hadd_epi32(vacc0x45, vacc0x67);
    const __m256i vacc1x0213 = _mm256_hadd_epi32(vacc1x01, vacc1x23);
    const __m256i vacc1x4657 = _mm256_hadd_epi32(vacc1x45, vacc1x67);
    const __m256i vacc2x0213 = _mm256_hadd_epi32(vacc2x01, vacc2x23);
    const __m256i vacc2x4657 = _mm256_hadd_epi32(vacc2x45, vacc2x67);
    const __m256i vacc3x0213 = _mm256_hadd_epi32(vacc3x01, vacc3x23);
    const __m256i vacc3x4657 = _mm256_hadd_epi32(vacc3x45, vacc3x67);

    const __m256i vacc0x02461357 = _mm256_hadd_epi32(vacc0x0213, vacc0x4657);
    const __m256i vacc1x02461357 = _mm256_hadd_epi32(vacc1x0213, vacc1x4657);
    const __m256i vacc2x02461357 = _mm256_hadd_epi32(vacc2x0213, vacc2x4657);
    const __m256i vacc3x02461357 = _mm256_hadd_epi32(vacc3x0213, vacc3x4657);

    const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i vacc0x01234567 = _mm256_permutevar8x32_epi32(vacc0x02461357, vpermute_mask);
    __m256i vacc1x01234567 = _mm256_permutevar8x32_epi32(vacc1x02461357, vpermute_mask);
    __m256i vacc2x01234567 = _mm256_permutevar8x32_epi32(vacc2x02461357, vpermute_mask);
    __m256i vacc3x01234567 = _mm256_permutevar8x32_epi32(vacc3x02461357, vpermute_mask);

    __m256 vfpacc0x01234567 = _mm256_cvtepi32_ps(vacc0x01234567);
    __m256 vfpacc1x01234567 = _mm256_cvtepi32_ps(vacc1x01234567);
    __m256 vfpacc2x01234567 = _mm256_cvtepi32_ps(vacc2x01234567);
    __m256 vfpacc3x01234567 = _mm256_cvtepi32_ps(vacc3x01234567);

    const __m256 vscale = _mm256_load_ps(params->fp32_avx2.scale);
    vfpacc0x01234567 = _mm256_mul_ps(vfpacc0x01234567, vscale);
    vfpacc1x01234567 = _mm256_mul_ps(vfpacc1x01234567, vscale);
    vfpacc2x01234567 = _mm256_mul_ps(vfpacc2x01234567, vscale);
    vfpacc3x01234567 = _mm256_mul_ps(vfpacc3x01234567, vscale);

    const __m256 voutput_max_less_zero_point = _mm256_load_ps(params->fp32_avx2.output_max_less_zero_point);
    vfpacc0x01234567 = _mm256_min_ps(vfpacc0x01234567, voutput_max_less_zero_point);
    vfpacc1x01234567 = _mm256_min_ps(vfpacc1x01234567, voutput_max_less_zero_point);
    vfpacc2x01234567 = _mm256_min_ps(vfpacc2x01234567, voutput_max_less_zero_point);
    vfpacc3x01234567 = _mm256_min_ps(vfpacc3x01234567, voutput_max_less_zero_point);

    vacc0x01234567 = _mm256_cvtps_epi32(vfpacc0x01234567);
    vacc1x01234567 = _mm256_cvtps_epi32(vfpacc1x01234567);
    vacc2x01234567 = _mm256_cvtps_epi32(vfpacc2x01234567);
    vacc3x01234567 = _mm256_cvtps_epi32(vfpacc3x01234567);

    const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx2.output_zero_point);
    __m256i vacc01x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0x01234567, vacc1x01234567), voutput_zero_point);
    __m256i vacc23x01234567 = _mm256_adds_epi16(_mm256_packs_epi32(vacc2x01234567, vacc3x01234567), voutput_zero_point);

    vacc01x01234567 = _mm256_permute4x64_epi64(vacc01x01234567, _MM_SHUFFLE(3, 1, 2, 0));
    vacc23x01234567 = _mm256_permute4x64_epi64(vacc23x01234567, _MM_SHUFFLE(3, 1, 2, 0));

    __m256i vout = _mm256_packus_epi16(vacc01x01234567, vacc23x01234567);

    vout = _mm256_max_epu8(vout, _mm256_load_si256((const __m256i*) params->fp32_avx2.output_min));

    __m128i vout_lo = _mm256_castsi256_si128(vout);
    __m128i vout_hi = _mm256_extracti128_si256(vout, 1);

    if (nc >= 8) {
      _mm_storel_epi64((__m128i*) c0, vout_lo);
      _mm_storel_epi64((__m128i*) c1, vout_hi);
      _mm_storeh_pi((__m64*) c2, _mm_castsi128_ps(vout_lo));
      _mm_storeh_pi((__m64*) c3, _mm_castsi128_ps(vout_hi));

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint8_t*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_si32(c0, vout_lo);
        _mm_storeu_si32(c1, vout_hi);
        unaligned_store_u32(c2, (uint32_t) _mm_extract_epi32(vout_lo, 2));
        unaligned_store_u32(c3, (uint32_t) _mm_extract_epi32(vout_hi, 2));

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;

        vout_lo = _mm_srli_epi64(vout_lo, 32);
        vout_hi = _mm_srli_epi64(vout_hi, 32);
      }
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout_lo, 0));
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout_hi, 0));
        unaligned_store_u16(c2, (uint16_t) _mm_extract_epi16(vout_lo, 4));
        unaligned_store_u16(c3, (uint16_t) _mm_extract_epi16(vout_hi, 4));

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;

        vout_lo = _mm_srli_epi32(vout_lo, 16);
        vout_hi = _mm_srli_epi32(vout_hi, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout_lo, 0);
        *c1 = (uint8_t) _mm_extract_epi8(vout_hi, 0);
        *c2 = (uint8_t) _mm_extract_epi8(vout_lo, 8);
        *c3 = (uint8_t) _mm_extract_epi8(vout_hi, 8);
      }

      nc = 0;
    }
  } while (nc != 0);
}
