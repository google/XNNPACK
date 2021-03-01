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


void xnn_qs8_gemm_minmax_ukernel_1x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8);
  const int8_t* a0 = a;
  int8_t* c0 = c;

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __mmask16 vblend_mask = _cvtu32_mask16(0xAAAA);
  const __m512i vmultiplier = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) params->sse2.multiplier));
  const __m512i vrounding = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) params->sse2.rounding));
  const __m512i vremainder_mask = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) params->sse2.remainder_mask));
  const __m512i vremainder_threshold = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) params->sse2.remainder_threshold));
  const __m128i vshift = _mm_load_si128((const __m128i*) params->sse2.shift);
  const __m256i voutput_zero_point = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.output_zero_point));
  const __m256i voutput_min = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.output_min));
  const __m256i voutput_max = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) params->sse2.output_max));
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((uintptr_t) w + 4 * sizeof(int32_t)));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((uintptr_t) w + 8 * sizeof(int32_t)));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((uintptr_t) w + 12 * sizeof(int32_t)));
    w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((uintptr_t) w + 32 * sizeof(int8_t))));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((uintptr_t) w + 64 * sizeof(int8_t))));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((uintptr_t) w + 96 * sizeof(int8_t))));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

      w = (const void*) ((uintptr_t) w + 128 * sizeof(int8_t));
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    const __m512i vacc0x88CC99DDAAEEBBFF = _mm512_shuffle_epi32(vacc0x084C195D2A6E3B7F, _MM_SHUFFLE(3, 3, 1, 1));

    const __m512i vprod0x04152637 = _mm512_add_epi64(_mm512_mul_epi32(vacc0x084C195D2A6E3B7F, vmultiplier), vrounding);

    const __m512i vprod0x8C9DAEBF = _mm512_add_epi64(_mm512_mul_epi32(vacc0x88CC99DDAAEEBBFF, vmultiplier), vrounding);

    const __m512i vq31prod0x04152637 = _mm512_srli_epi64(vprod0x04152637, 31);
    const __m512i vq31prod0x8C9DAEBF = _mm512_add_epi64(vprod0x8C9DAEBF, vprod0x8C9DAEBF);

    const __m512i vq31prod0x084C195D2A6E3B7F = _mm512_mask_blend_epi32(vblend_mask, vq31prod0x04152637, vq31prod0x8C9DAEBF);

    const __m512i vrem0x084C195D2A6E3B7F =
      _mm512_add_epi32(_mm512_and_si512(vq31prod0x084C195D2A6E3B7F, vremainder_mask), _mm512_srai_epi32(vq31prod0x084C195D2A6E3B7F, 31));

    vacc0x084C195D2A6E3B7F = _mm512_sra_epi32(vq31prod0x084C195D2A6E3B7F, vshift);

    const __m512i vminus_one = _mm512_set1_epi32(-1);
    vacc0x084C195D2A6E3B7F =
      _mm512_mask_sub_epi32(vacc0x084C195D2A6E3B7F, _mm512_cmpgt_epi32_mask(vrem0x084C195D2A6E3B7F, vremainder_threshold), vacc0x084C195D2A6E3B7F, vminus_one);

    __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);
    vacc0x084C2A6E195D3B7F = _mm256_min_epi16(_mm256_max_epi16(vacc0x084C2A6E195D3B7F, voutput_min), voutput_max);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    const __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}
