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


void xnn_qd8_f32_qs8w_gemm_minmax_ukernel_4x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
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
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  // Load quantization parameters.
  const __m512i vszp01 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) &quantization_params[0]));
  const __m512i vszp23 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) &quantization_params[2]));
  const __m512i vzp0 = _mm512_shuffle_epi32(vszp01, _MM_SHUFFLE(0, 0, 0, 0));
  const __m512 vscale0 = _mm512_permute_ps(_mm512_castsi512_ps(vszp01), _MM_SHUFFLE(1, 1, 1, 1));
  const __m512i vzp1 = _mm512_shuffle_epi32(vszp01, _MM_SHUFFLE(2, 2, 2, 2));
  const __m512 vscale1 = _mm512_permute_ps(_mm512_castsi512_ps(vszp01), _MM_SHUFFLE(3, 3, 3, 3));
  const __m512i vzp2 = _mm512_shuffle_epi32(vszp23, _MM_SHUFFLE(0, 0, 0, 0));
  const __m512 vscale2 = _mm512_permute_ps(_mm512_castsi512_ps(vszp23), _MM_SHUFFLE(1, 1, 1, 1));
  const __m512i vzp3 = _mm512_shuffle_epi32(vszp23, _MM_SHUFFLE(2, 2, 2, 2));
  const __m512 vscale3 = _mm512_permute_ps(_mm512_castsi512_ps(vszp23), _MM_SHUFFLE(3, 3, 3, 3));
  // Load output min and max.
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);

    // Set weight sums for 4 input rows.

    __m512i vacc3x0123 = _mm512_mullo_epi32(vacc0x0123, vzp3);
    __m512i vacc3x4567 = _mm512_mullo_epi32(vacc0x4567, vzp3);
    __m512i vacc3x89AB = _mm512_mullo_epi32(vacc0x89AB, vzp3);
    __m512i vacc3xCDEF = _mm512_mullo_epi32(vacc0xCDEF, vzp3);

    __m512i vacc2x0123 = _mm512_mullo_epi32(vacc0x0123, vzp2);
    __m512i vacc2x4567 = _mm512_mullo_epi32(vacc0x4567, vzp2);
    __m512i vacc2x89AB = _mm512_mullo_epi32(vacc0x89AB, vzp2);
    __m512i vacc2xCDEF = _mm512_mullo_epi32(vacc0xCDEF, vzp2);

    __m512i vacc1x0123 = _mm512_mullo_epi32(vacc0x0123, vzp1);
    __m512i vacc1x4567 = _mm512_mullo_epi32(vacc0x4567, vzp1);
    __m512i vacc1x89AB = _mm512_mullo_epi32(vacc0x89AB, vzp1);
    __m512i vacc1xCDEF = _mm512_mullo_epi32(vacc0xCDEF, vzp1);

    vacc0x0123 = _mm512_mullo_epi32(vacc0x0123, vzp0);
    vacc0x4567 = _mm512_mullo_epi32(vacc0x4567, vzp0);
    vacc0x89AB = _mm512_mullo_epi32(vacc0x89AB, vzp0);
    vacc0xCDEF = _mm512_mullo_epi32(vacc0xCDEF, vzp0);

    // Advance weights pointer.
    w = (const int32_t*) w + 16;

    size_t k = 0;
    // Accumulate blocks multiplication for each row.
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

      w = (const int8_t*) w + 128;
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

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    // Scale.
    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale0);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale1);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale2);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale3);

    // Reshuffle elements in order.
    const __m512 vscaled0x0123456789ABCDEF = _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0), vscaled0x084C195D2A6E3B7F);
    const __m512 vscaled1x0123456789ABCDEF = _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0), vscaled1x084C195D2A6E3B7F);
    const __m512 vscaled2x0123456789ABCDEF = _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0), vscaled2x084C195D2A6E3B7F);
    const __m512 vscaled3x0123456789ABCDEF = _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0), vscaled3x084C195D2A6E3B7F);

    // Add bias.
    __m512 vbias = _mm512_load_ps((const float*) w);
    w = (const float*) w + 16;
    const __m512 vbiased0x0123456789ABCDEF = _mm512_add_ps(vscaled0x0123456789ABCDEF, vbias);
    const __m512 vbiased1x0123456789ABCDEF = _mm512_add_ps(vscaled1x0123456789ABCDEF, vbias);
    const __m512 vbiased2x0123456789ABCDEF = _mm512_add_ps(vscaled2x0123456789ABCDEF, vbias);
    const __m512 vbiased3x0123456789ABCDEF = _mm512_add_ps(vscaled3x0123456789ABCDEF, vbias);

    // Clamp between min and max.
    const __m512 vclamp_min0x0123456789ABCDEF = _mm512_max_ps(vbiased0x0123456789ABCDEF, voutput_min);
    const __m512 vclamp_min1x0123456789ABCDEF = _mm512_max_ps(vbiased1x0123456789ABCDEF, voutput_min);
    const __m512 vclamp_min2x0123456789ABCDEF = _mm512_max_ps(vbiased2x0123456789ABCDEF, voutput_min);
    const __m512 vclamp_min3x0123456789ABCDEF = _mm512_max_ps(vbiased3x0123456789ABCDEF, voutput_min);

    const __m512 vout0x0123456789ABCDEF = _mm512_min_ps(vclamp_min0x0123456789ABCDEF, voutput_max);
    const __m512 vout1x0123456789ABCDEF = _mm512_min_ps(vclamp_min1x0123456789ABCDEF, voutput_max);
    const __m512 vout2x0123456789ABCDEF = _mm512_min_ps(vclamp_min2x0123456789ABCDEF, voutput_max);
    const __m512 vout3x0123456789ABCDEF = _mm512_min_ps(vclamp_min3x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      _mm512_storeu_ps(c3, vout3x0123456789ABCDEF);
      _mm512_storeu_ps(c2, vout2x0123456789ABCDEF);
      _mm512_storeu_ps(c1, vout1x0123456789ABCDEF);
      _mm512_storeu_ps(c0, vout0x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - k);
      a1 = (const int8_t*) ((uintptr_t) a1 - k);
      a2 = (const int8_t*) ((uintptr_t) a2 - k);
      a3 = (const int8_t*) ((uintptr_t) a3 - k);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c3, vmask, vout3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2, vmask, vout2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1, vmask, vout1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}
