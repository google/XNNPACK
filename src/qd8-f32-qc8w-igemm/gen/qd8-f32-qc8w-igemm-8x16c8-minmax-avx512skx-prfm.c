// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx16c8-avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/prefetch.h"


void xnn_qd8_f32_qc8w_igemm_minmax_ukernel_8x16c8__avx512skx_prfm(
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
    const struct xnn_f32_minmax_params* restrict params,
    const struct xnn_qd8_quantization_params* restrict quantization_params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 8);
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
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    c7 = c6;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512i vinput_zero_point = _mm512_set1_epi32((int) quantization_params->zero_point);
  const __m512 vinput_inv_scale = _mm512_set1_ps(quantization_params->inv_scale);
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  // XNN_FORCE_REALIZATION(vinput_zero_point);
  // XNN_FORCE_REALIZATION(vinput_inv_scale);
  // XNN_FORCE_REALIZATION(voutput_min);
  // XNN_FORCE_REALIZATION(voutput_max);
  do {
    const __m512i vksum0 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 0);
    const __m512i vksum1 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    const __m512i vksum2 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    const __m512i vksum3 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);
    __m512i vacc0x0 = _mm512_mullo_epi32(vksum0, vinput_zero_point);
    __m512i vacc0x1 = _mm512_mullo_epi32(vksum1, vinput_zero_point);
    __m512i vacc0x2 = _mm512_mullo_epi32(vksum2, vinput_zero_point);
    __m512i vacc0x3 = _mm512_mullo_epi32(vksum3, vinput_zero_point);
    __m512i vacc1x0 = vacc0x0;
    __m512i vacc1x1 = vacc0x1;
    __m512i vacc1x2 = vacc0x2;
    __m512i vacc1x3 = vacc0x3;
    __m512i vacc2x0 = vacc0x0;
    __m512i vacc2x1 = vacc0x1;
    __m512i vacc2x2 = vacc0x2;
    __m512i vacc2x3 = vacc0x3;
    __m512i vacc3x0 = vacc0x0;
    __m512i vacc3x1 = vacc0x1;
    __m512i vacc3x2 = vacc0x2;
    __m512i vacc3x3 = vacc0x3;
    __m512i vacc4x0 = vacc0x0;
    __m512i vacc4x1 = vacc0x1;
    __m512i vacc4x2 = vacc0x2;
    __m512i vacc4x3 = vacc0x3;
    __m512i vacc5x0 = vacc0x0;
    __m512i vacc5x1 = vacc0x1;
    __m512i vacc5x2 = vacc0x2;
    __m512i vacc5x3 = vacc0x3;
    __m512i vacc6x0 = vacc0x0;
    __m512i vacc6x1 = vacc0x1;
    __m512i vacc6x2 = vacc0x2;
    __m512i vacc6x3 = vacc0x3;
    __m512i vacc7x0 = vacc0x0;
    __m512i vacc7x1 = vacc0x1;
    __m512i vacc7x2 = vacc0x2;
    __m512i vacc7x3 = vacc0x3;
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
      const int8_t* restrict a5 = a[5];
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const int8_t*) ((uintptr_t) a5 + a_offset);
      } else {
        a5 = zero_data;
      }
      const int8_t* restrict a6 = a[6];
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const int8_t*) ((uintptr_t) a6 + a_offset);
      } else {
        a6 = zero_data;
      }
      const int8_t* restrict a7 = a[7];
      if XNN_UNPREDICTABLE(a7 != zero) {
        a7 = (const int8_t*) ((uintptr_t) a7 + a_offset);
      } else {
        a7 = zero_data;
      }
      a += 8;

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
        const __m512i va5 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a5)));
        a5 += 8;
        const __m512i va6 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a6)));
        a6 += 8;
        const __m512i va7 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a7)));
        a7 += 8;

        const __m512i vb0 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));
        xnn_prefetch_to_l1((const int8_t*) w + 896);

        vacc0x0 = _mm512_add_epi32(vacc0x0, _mm512_madd_epi16(va0, vb0));
        vacc1x0 = _mm512_add_epi32(vacc1x0, _mm512_madd_epi16(va1, vb0));
        vacc2x0 = _mm512_add_epi32(vacc2x0, _mm512_madd_epi16(va2, vb0));
        vacc3x0 = _mm512_add_epi32(vacc3x0, _mm512_madd_epi16(va3, vb0));
        vacc4x0 = _mm512_add_epi32(vacc4x0, _mm512_madd_epi16(va4, vb0));
        vacc5x0 = _mm512_add_epi32(vacc5x0, _mm512_madd_epi16(va5, vb0));
        vacc6x0 = _mm512_add_epi32(vacc6x0, _mm512_madd_epi16(va6, vb0));
        vacc7x0 = _mm512_add_epi32(vacc7x0, _mm512_madd_epi16(va7, vb0));
        const __m512i vb1 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

        vacc0x1 = _mm512_add_epi32(vacc0x1, _mm512_madd_epi16(va0, vb1));
        vacc1x1 = _mm512_add_epi32(vacc1x1, _mm512_madd_epi16(va1, vb1));
        vacc2x1 = _mm512_add_epi32(vacc2x1, _mm512_madd_epi16(va2, vb1));
        vacc3x1 = _mm512_add_epi32(vacc3x1, _mm512_madd_epi16(va3, vb1));
        vacc4x1 = _mm512_add_epi32(vacc4x1, _mm512_madd_epi16(va4, vb1));
        vacc5x1 = _mm512_add_epi32(vacc5x1, _mm512_madd_epi16(va5, vb1));
        vacc6x1 = _mm512_add_epi32(vacc6x1, _mm512_madd_epi16(va6, vb1));
        vacc7x1 = _mm512_add_epi32(vacc7x1, _mm512_madd_epi16(va7, vb1));
        const __m512i vb2 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));
        xnn_prefetch_to_l1((const int8_t*) w + 960);

        vacc0x2 = _mm512_add_epi32(vacc0x2, _mm512_madd_epi16(va0, vb2));
        vacc1x2 = _mm512_add_epi32(vacc1x2, _mm512_madd_epi16(va1, vb2));
        vacc2x2 = _mm512_add_epi32(vacc2x2, _mm512_madd_epi16(va2, vb2));
        vacc3x2 = _mm512_add_epi32(vacc3x2, _mm512_madd_epi16(va3, vb2));
        vacc4x2 = _mm512_add_epi32(vacc4x2, _mm512_madd_epi16(va4, vb2));
        vacc5x2 = _mm512_add_epi32(vacc5x2, _mm512_madd_epi16(va5, vb2));
        vacc6x2 = _mm512_add_epi32(vacc6x2, _mm512_madd_epi16(va6, vb2));
        vacc7x2 = _mm512_add_epi32(vacc7x2, _mm512_madd_epi16(va7, vb2));
        const __m512i vb3 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

        vacc0x3 = _mm512_add_epi32(vacc0x3, _mm512_madd_epi16(va0, vb3));
        vacc1x3 = _mm512_add_epi32(vacc1x3, _mm512_madd_epi16(va1, vb3));
        vacc2x3 = _mm512_add_epi32(vacc2x3, _mm512_madd_epi16(va2, vb3));
        vacc3x3 = _mm512_add_epi32(vacc3x3, _mm512_madd_epi16(va3, vb3));
        vacc4x3 = _mm512_add_epi32(vacc4x3, _mm512_madd_epi16(va4, vb3));
        vacc5x3 = _mm512_add_epi32(vacc5x3, _mm512_madd_epi16(va5, vb3));
        vacc6x3 = _mm512_add_epi32(vacc6x3, _mm512_madd_epi16(va6, vb3));
        vacc7x3 = _mm512_add_epi32(vacc7x3, _mm512_madd_epi16(va7, vb3));

        w = (const int8_t*) w + 128;
        k -= 8 * sizeof(int8_t);
      }
      p -= 8 * sizeof(void*);
    } while (p != 0);

    // Add 4 adjacent sums
    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0, vacc0x1), _mm512_unpackhi_epi32(vacc0x0, vacc0x1));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x2, vacc0x3), _mm512_unpackhi_epi32(vacc0x2, vacc0x3));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0, vacc1x1), _mm512_unpackhi_epi32(vacc1x0, vacc1x1));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x2, vacc1x3), _mm512_unpackhi_epi32(vacc1x2, vacc1x3));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0, vacc2x1), _mm512_unpackhi_epi32(vacc2x0, vacc2x1));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x2, vacc2x3), _mm512_unpackhi_epi32(vacc2x2, vacc2x3));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0, vacc3x1), _mm512_unpackhi_epi32(vacc3x0, vacc3x1));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x2, vacc3x3), _mm512_unpackhi_epi32(vacc3x2, vacc3x3));
    const __m512i vacc4x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x0, vacc4x1), _mm512_unpackhi_epi32(vacc4x0, vacc4x1));
    const __m512i vacc4x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x2, vacc4x3), _mm512_unpackhi_epi32(vacc4x2, vacc4x3));
    const __m512i vacc5x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc5x0, vacc5x1), _mm512_unpackhi_epi32(vacc5x0, vacc5x1));
    const __m512i vacc5x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc5x2, vacc5x3), _mm512_unpackhi_epi32(vacc5x2, vacc5x3));
    const __m512i vacc6x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc6x0, vacc6x1), _mm512_unpackhi_epi32(vacc6x0, vacc6x1));
    const __m512i vacc6x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc6x2, vacc6x3), _mm512_unpackhi_epi32(vacc6x2, vacc6x3));
    const __m512i vacc7x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc7x0, vacc7x1), _mm512_unpackhi_epi32(vacc7x0, vacc7x1));
    const __m512i vacc7x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc7x2, vacc7x3), _mm512_unpackhi_epi32(vacc7x2, vacc7x3));
    const __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    const __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    const __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    const __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));
    const __m512i vacc4x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x04152637, vacc4x8C9DAEBF), _mm512_unpackhi_epi32(vacc4x04152637, vacc4x8C9DAEBF));
    const __m512i vacc5x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc5x04152637, vacc5x8C9DAEBF), _mm512_unpackhi_epi32(vacc5x04152637, vacc5x8C9DAEBF));
    const __m512i vacc6x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc6x04152637, vacc6x8C9DAEBF), _mm512_unpackhi_epi32(vacc6x04152637, vacc6x8C9DAEBF));
    const __m512i vacc7x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc7x04152637, vacc7x8C9DAEBF), _mm512_unpackhi_epi32(vacc7x04152637, vacc7x8C9DAEBF));
    const __m512i vidx = _mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0);
    __m512i vacc0x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc0x084C195D2A6E3B7F);
    __m512i vacc1x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc1x084C195D2A6E3B7F);
    __m512i vacc2x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc2x084C195D2A6E3B7F);
    __m512i vacc3x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc3x084C195D2A6E3B7F);
    __m512i vacc4x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc4x084C195D2A6E3B7F);
    __m512i vacc5x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc5x084C195D2A6E3B7F);
    __m512i vacc6x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc6x084C195D2A6E3B7F);
    __m512i vacc7x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc7x084C195D2A6E3B7F);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
    __m512 vscaled5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);
    __m512 vscaled6x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc6x0123456789ABCDEF);
    __m512 vscaled7x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc7x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vinput_inv_scale);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vinput_inv_scale);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vinput_inv_scale);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vinput_inv_scale);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vinput_inv_scale);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vinput_inv_scale);
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, vinput_inv_scale);
    vscaled7x0123456789ABCDEF = _mm512_mul_ps(vscaled7x0123456789ABCDEF, vinput_inv_scale);

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(vscaled1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(vscaled2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(vscaled3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled4x0123456789ABCDEF = _mm512_fmadd_ps(vscaled4x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled5x0123456789ABCDEF = _mm512_fmadd_ps(vscaled5x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled6x0123456789ABCDEF = _mm512_fmadd_ps(vscaled6x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled7x0123456789ABCDEF = _mm512_fmadd_ps(vscaled7x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled1x0123456789ABCDEF = _mm512_max_ps(vscaled1x0123456789ABCDEF, voutput_min);
    vscaled2x0123456789ABCDEF = _mm512_max_ps(vscaled2x0123456789ABCDEF, voutput_min);
    vscaled3x0123456789ABCDEF = _mm512_max_ps(vscaled3x0123456789ABCDEF, voutput_min);
    vscaled4x0123456789ABCDEF = _mm512_max_ps(vscaled4x0123456789ABCDEF, voutput_min);
    vscaled5x0123456789ABCDEF = _mm512_max_ps(vscaled5x0123456789ABCDEF, voutput_min);
    vscaled6x0123456789ABCDEF = _mm512_max_ps(vscaled6x0123456789ABCDEF, voutput_min);
    vscaled7x0123456789ABCDEF = _mm512_max_ps(vscaled7x0123456789ABCDEF, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max);
    vscaled7x0123456789ABCDEF = _mm512_min_ps(vscaled7x0123456789ABCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c7, vscaled7x0123456789ABCDEF);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c6, vscaled6x0123456789ABCDEF);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5, vscaled5x0123456789ABCDEF);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
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
      _mm512_mask_storeu_ps(c7, vmask, vscaled7x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c6, vmask, vscaled6x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c5, vmask, vscaled5x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c4, vmask, vscaled4x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3, vmask, vscaled3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2, vmask, vscaled2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1, vmask, vscaled1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c0, vmask, vscaled0x0123456789ABCDEF);
      nc = 0;
    }
  } while (nc != 0);
}
