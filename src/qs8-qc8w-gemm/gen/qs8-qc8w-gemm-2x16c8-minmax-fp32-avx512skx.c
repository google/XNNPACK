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


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    __m512i vacc1x0123 = vacc0x0123;
    __m512i vacc1x4567 = vacc0x4567;
    __m512i vacc1x89AB = vacc0x89AB;
    __m512i vacc1xCDEF = vacc0xCDEF;
    w = (const void*) ((const int32_t*) w + 16);

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));

      w = (const void*) ((const int8_t*) w + 128);
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const void*) ((const float*) w + 16);
    const __m512 vscale084C195D2A6E3B7F = _mm512_permutexvar_ps(_mm512_set_epi32(15, 7, 11, 3, 14, 6, 10, 2, 13, 5, 9, 1, 12, 4, 8, 0), vscale012345678ABCDEF);
    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);

    const __m256i vout01x084Cx2A6Ex195Dx3B7F = _mm256_packs_epi16(_mm512_castsi512_si256(vacc01x084Cx195Dx2A6Ex3B7F), _mm512_extracti32x8_epi32(vacc01x084Cx195Dx2A6Ex3B7F, 1));
    const __m256i vout01x084C2A6E195D3B7F = _mm256_permutevar8x32_epi32(vout01x084Cx2A6Ex195Dx3B7F, _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0));
    __m256i vout01x0123456789ABCDEF = _mm256_shuffle_epi8(vout01x084C2A6E195D3B7F, _mm256_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0, 15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout01x0123456789ABCDEF = _mm256_max_epi8(vout01x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm256_castsi256_si128(vout01x0123456789ABCDEF));
      _mm_storeu_si128((__m128i*) c1, _mm256_extracti128_si256(vout01x0123456789ABCDEF, 1));

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm256_mask_storeu_epi8(c0, vmask, vout01x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm256_mask_storeu_epi8(c1 - 16, vmask, vout01x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}
