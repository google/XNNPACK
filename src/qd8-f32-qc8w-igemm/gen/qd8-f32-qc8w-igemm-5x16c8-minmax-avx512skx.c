// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx16c8-avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_5x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const int8_t* zero_data,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512i vinput_zero_point = _mm512_set1_epi32((int) quantization_params->zero_point);
  const __m512 vinput_inv_scale = _mm512_set1_ps(quantization_params->inv_scale);
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  do {
    const __m512i vksum0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    const __m512i vksum4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    const __m512i vksum89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    const __m512i vksumCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);
    __m512i vacc0x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point);
    __m512i vacc0x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point);
    __m512i vacc0x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point);
    __m512i vacc0xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point);
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
    __m512i vacc4x0123 = vacc0x0123;
    __m512i vacc4x4567 = vacc0x4567;
    __m512i vacc4x89AB = vacc0x89AB;
    __m512i vacc4xCDEF = vacc0xCDEF;
    w = (const int32_t*) w + 16;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      } else {
        a0 = zero_data;
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      } else {
        a1 = zero_data;
      }
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      } else {
        a2 = zero_data;
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      } else {
        a3 = zero_data;
      }
      const int8_t* restrict a4 = a[4];
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const int8_t*) ((uintptr_t) a4 + a_offset);
      } else {
        a4 = zero_data;
      }
      a += 5;

      size_t k = kc;

      while (k >= 8 * sizeof(int8_t)) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;
        const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
        a1 += 8;
        const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
        a2 += 8;
        const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
        a3 += 8;
        const __m512i va4 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a4)));
        a4 += 8;

        const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
        vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
        vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
        vacc4x0123 = _mm512_add_epi32(vacc4x0123, _mm512_madd_epi16(va4, vb0123));
        const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
        vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
        vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
        vacc4x4567 = _mm512_add_epi32(vacc4x4567, _mm512_madd_epi16(va4, vb4567));
        const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
        vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
        vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
        vacc4x89AB = _mm512_add_epi32(vacc4x89AB, _mm512_madd_epi16(va4, vb89AB));
        const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
        vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
        vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
        vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));
        vacc4xCDEF = _mm512_add_epi32(vacc4xCDEF, _mm512_madd_epi16(va4, vbCDEF));

        w = (const int8_t*) w + 128;
        k -= 8 * sizeof(int8_t);
      }
      p -= 5 * sizeof(void*);
    } while (p != 0);

    // Add 4 adjacent sums
    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));
    const __m512i vacc4x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x0123, vacc4x4567), _mm512_unpackhi_epi32(vacc4x0123, vacc4x4567));
    const __m512i vacc4x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x89AB, vacc4xCDEF), _mm512_unpackhi_epi32(vacc4x89AB, vacc4xCDEF));
    const __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    const __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    const __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    const __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));
    const __m512i vacc4x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x04152637, vacc4x8C9DAEBF), _mm512_unpackhi_epi32(vacc4x04152637, vacc4x8C9DAEBF));
    const __m512i vidx = _mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0);
    __m512i vacc0x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc0x084C195D2A6E3B7F);
    __m512i vacc1x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc1x084C195D2A6E3B7F);
    __m512i vacc2x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc2x084C195D2A6E3B7F);
    __m512i vacc3x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc3x084C195D2A6E3B7F);
    __m512i vacc4x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc4x084C195D2A6E3B7F);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vinput_inv_scale);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vinput_inv_scale);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vinput_inv_scale);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vinput_inv_scale);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vinput_inv_scale);

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(vscaled1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(vscaled2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(vscaled3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_fmadd_ps(vscaled4x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled1x0123456789ABCDEF = _mm512_max_ps(vscaled1x0123456789ABCDEF, voutput_min);
    vscaled2x0123456789ABCDEF = _mm512_max_ps(vscaled2x0123456789ABCDEF, voutput_min);
    vscaled3x0123456789ABCDEF = _mm512_max_ps(vscaled3x0123456789ABCDEF, voutput_min);
    vscaled4x0123456789ABCDEF = _mm512_max_ps(vscaled4x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c4, vscaled4x0123456789ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3, vscaled3x0123456789ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2, vscaled2x0123456789ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1, vscaled1x0123456789ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0, vscaled0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c4, vmask, vscaled4x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3, vmask, vscaled3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2, vmask, vscaled2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1, vmask, vscaled1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c0, vmask, vscaled0x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
