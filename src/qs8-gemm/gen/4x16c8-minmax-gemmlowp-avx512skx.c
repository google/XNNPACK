// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c8-avx512skx.c.in
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


void xnn_qs8_gemm_minmax_gemmlowp_ukernel_4x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __mmask16 vblend_mask = _cvtu32_mask16(0xAAAA);
  const __m512i vmultiplier = _mm512_set1_epi64(params->gemmlowp_avx512.multiplier);
  const __m512i vrounding = _mm512_set1_epi64(params->gemmlowp_avx512.rounding);
  const __m512i vremainder_mask = _mm512_set1_epi32(params->gemmlowp_avx512.remainder_mask);
  const __m512i vremainder_threshold = _mm512_set1_epi32(params->gemmlowp_avx512.remainder_threshold);
  const __m128i vshift = _mm_loadl_epi64((const __m128i*) &params->gemmlowp_avx512.shift);
  const __m512i voutput_zero_point = _mm512_load_si512(params->gemmlowp_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->gemmlowp_avx512.output_min);
  const __m512i voutput_max = _mm512_load_si512(params->gemmlowp_avx512.output_max);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    __m512i vacc1x0123 = vacc0x0123;
    __m512i vacc1x4567 = vacc0x4567;
    __m512i vacc1x89AB = vacc0x89AB;
    __m512i vacc1xCDEF = vacc0xCDEF;
    __m512i vacc2x0123 = vacc0x0123;
    __m512i vacc2x4567 = vacc0x4567;
    __m512i vacc2x89AB = vacc0x89AB;
    __m512i vacc2xCDEF = vacc0xCDEF;
    __m512i vacc3x0123 = vacc0x0123;
    __m512i vacc3x4567 = vacc0x4567;
    __m512i vacc3x89AB = vacc0x89AB;
    __m512i vacc3xCDEF = vacc0xCDEF;
    w = (const void*) ((const int32_t*) w + 16);

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

      w = (const void*) ((const int8_t*) w + 128);
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    const __m512i vacc0x88CC99DDAAEEBBFF = _mm512_shuffle_epi32(vacc0x084C195D2A6E3B7F, _MM_SHUFFLE(3, 3, 1, 1));
    const __m512i vacc1x88CC99DDAAEEBBFF = _mm512_shuffle_epi32(vacc1x084C195D2A6E3B7F, _MM_SHUFFLE(3, 3, 1, 1));
    const __m512i vacc2x88CC99DDAAEEBBFF = _mm512_shuffle_epi32(vacc2x084C195D2A6E3B7F, _MM_SHUFFLE(3, 3, 1, 1));
    const __m512i vacc3x88CC99DDAAEEBBFF = _mm512_shuffle_epi32(vacc3x084C195D2A6E3B7F, _MM_SHUFFLE(3, 3, 1, 1));

    const __m512i vprod0x04152637 = _mm512_add_epi64(_mm512_mul_epi32(vacc0x084C195D2A6E3B7F, vmultiplier), vrounding);
    const __m512i vprod1x04152637 = _mm512_add_epi64(_mm512_mul_epi32(vacc1x084C195D2A6E3B7F, vmultiplier), vrounding);
    const __m512i vprod2x04152637 = _mm512_add_epi64(_mm512_mul_epi32(vacc2x084C195D2A6E3B7F, vmultiplier), vrounding);
    const __m512i vprod3x04152637 = _mm512_add_epi64(_mm512_mul_epi32(vacc3x084C195D2A6E3B7F, vmultiplier), vrounding);

    const __m512i vprod0x8C9DAEBF = _mm512_add_epi64(_mm512_mul_epi32(vacc0x88CC99DDAAEEBBFF, vmultiplier), vrounding);
    const __m512i vprod1x8C9DAEBF = _mm512_add_epi64(_mm512_mul_epi32(vacc1x88CC99DDAAEEBBFF, vmultiplier), vrounding);
    const __m512i vprod2x8C9DAEBF = _mm512_add_epi64(_mm512_mul_epi32(vacc2x88CC99DDAAEEBBFF, vmultiplier), vrounding);
    const __m512i vprod3x8C9DAEBF = _mm512_add_epi64(_mm512_mul_epi32(vacc3x88CC99DDAAEEBBFF, vmultiplier), vrounding);

    const __m512i vq31prod0x04152637 = _mm512_srli_epi64(vprod0x04152637, 31);
    const __m512i vq31prod0x8C9DAEBF = _mm512_add_epi64(vprod0x8C9DAEBF, vprod0x8C9DAEBF);
    const __m512i vq31prod1x04152637 = _mm512_srli_epi64(vprod1x04152637, 31);
    const __m512i vq31prod1x8C9DAEBF = _mm512_add_epi64(vprod1x8C9DAEBF, vprod1x8C9DAEBF);
    const __m512i vq31prod2x04152637 = _mm512_srli_epi64(vprod2x04152637, 31);
    const __m512i vq31prod2x8C9DAEBF = _mm512_add_epi64(vprod2x8C9DAEBF, vprod2x8C9DAEBF);
    const __m512i vq31prod3x04152637 = _mm512_srli_epi64(vprod3x04152637, 31);
    const __m512i vq31prod3x8C9DAEBF = _mm512_add_epi64(vprod3x8C9DAEBF, vprod3x8C9DAEBF);

    const __m512i vq31prod0x084C195D2A6E3B7F = _mm512_mask_blend_epi32(vblend_mask, vq31prod0x04152637, vq31prod0x8C9DAEBF);
    const __m512i vq31prod1x084C195D2A6E3B7F = _mm512_mask_blend_epi32(vblend_mask, vq31prod1x04152637, vq31prod1x8C9DAEBF);
    const __m512i vq31prod2x084C195D2A6E3B7F = _mm512_mask_blend_epi32(vblend_mask, vq31prod2x04152637, vq31prod2x8C9DAEBF);
    const __m512i vq31prod3x084C195D2A6E3B7F = _mm512_mask_blend_epi32(vblend_mask, vq31prod3x04152637, vq31prod3x8C9DAEBF);

    const __m512i vrem0x084C195D2A6E3B7F =
      _mm512_add_epi32(_mm512_and_si512(vq31prod0x084C195D2A6E3B7F, vremainder_mask), _mm512_srai_epi32(vq31prod0x084C195D2A6E3B7F, 31));
    const __m512i vrem1x084C195D2A6E3B7F =
      _mm512_add_epi32(_mm512_and_si512(vq31prod1x084C195D2A6E3B7F, vremainder_mask), _mm512_srai_epi32(vq31prod1x084C195D2A6E3B7F, 31));
    const __m512i vrem2x084C195D2A6E3B7F =
      _mm512_add_epi32(_mm512_and_si512(vq31prod2x084C195D2A6E3B7F, vremainder_mask), _mm512_srai_epi32(vq31prod2x084C195D2A6E3B7F, 31));
    const __m512i vrem3x084C195D2A6E3B7F =
      _mm512_add_epi32(_mm512_and_si512(vq31prod3x084C195D2A6E3B7F, vremainder_mask), _mm512_srai_epi32(vq31prod3x084C195D2A6E3B7F, 31));

    vacc0x084C195D2A6E3B7F = _mm512_sra_epi32(vq31prod0x084C195D2A6E3B7F, vshift);
    vacc1x084C195D2A6E3B7F = _mm512_sra_epi32(vq31prod1x084C195D2A6E3B7F, vshift);
    vacc2x084C195D2A6E3B7F = _mm512_sra_epi32(vq31prod2x084C195D2A6E3B7F, vshift);
    vacc3x084C195D2A6E3B7F = _mm512_sra_epi32(vq31prod3x084C195D2A6E3B7F, vshift);

    const __m512i vminus_one = _mm512_set1_epi32(-1);
    vacc0x084C195D2A6E3B7F =
      _mm512_mask_sub_epi32(vacc0x084C195D2A6E3B7F, _mm512_cmpgt_epi32_mask(vrem0x084C195D2A6E3B7F, vremainder_threshold), vacc0x084C195D2A6E3B7F, vminus_one);
    vacc1x084C195D2A6E3B7F =
      _mm512_mask_sub_epi32(vacc1x084C195D2A6E3B7F, _mm512_cmpgt_epi32_mask(vrem1x084C195D2A6E3B7F, vremainder_threshold), vacc1x084C195D2A6E3B7F, vminus_one);
    vacc2x084C195D2A6E3B7F =
      _mm512_mask_sub_epi32(vacc2x084C195D2A6E3B7F, _mm512_cmpgt_epi32_mask(vrem2x084C195D2A6E3B7F, vremainder_threshold), vacc2x084C195D2A6E3B7F, vminus_one);
    vacc3x084C195D2A6E3B7F =
      _mm512_mask_sub_epi32(vacc3x084C195D2A6E3B7F, _mm512_cmpgt_epi32_mask(vrem3x084C195D2A6E3B7F, vremainder_threshold), vacc3x084C195D2A6E3B7F, vminus_one);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packs_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epi8(vout0123x0123456789ABCDEF, voutput_min);
    vout0123x0123456789ABCDEF = _mm512_min_epi8(vout0123x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));

      a0 = (const int8_t*) ((uintptr_t) a0 - k);
      a1 = (const int8_t*) ((uintptr_t) a1 - k);
      a2 = (const int8_t*) ((uintptr_t) a2 - k);
      a3 = (const int8_t*) ((uintptr_t) a3 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}
