// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c8-avx512skx.c.in
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
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/prefetch.h"

void xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params* restrict params) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 7);
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
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint8_t* a4 = (const uint8_t*) ((uintptr_t) a3 + a_stride);
  uint8_t* c4 = (uint8_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const uint8_t* a5 = (const uint8_t*) ((uintptr_t) a4 + a_stride);
  uint8_t* c5 = (uint8_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const uint8_t* a6 = (const uint8_t*) ((uintptr_t) a5 + a_stride);
  uint8_t* c6 = (uint8_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_set1_ps(params->fp32_scalar.scale);
  // XNN_FORCE_REALIZATION(vscale);
  const __m512 voutput_max_less_zero_point = _mm512_set1_ps((int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point);
  const __m512i voutput_zero_point = _mm512_set1_epi32(params->fp32_scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->fp32_scalar.output_min);
  // XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  // XNN_FORCE_REALIZATION(voutput_zero_point);
  // XNN_FORCE_REALIZATION(voutput_min);
  const __m512i vb_zero_point = _mm512_set1_epi16(params->fp32_scalar.kernel_zero_point);
  // XNN_FORCE_REALIZATION(vb_zero_point);

  do {
    __m512i vacc0x0 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 0);
    __m512i vacc0x1 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    __m512i vacc0x2 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    __m512i vacc0x3 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);
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
    w = (const int32_t*) w + 16;

    size_t k = kc;

    while (k >= 8 * sizeof(uint8_t)) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;
      const __m512i va4 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a4)));
      a4 += 8;
      const __m512i va5 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a5)));
      a5 += 8;
      const __m512i va6 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a6)));
      a6 += 8;

      const __m512i vb0 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) w)), vb_zero_point);
      xnn_prefetch_to_l1((const uint8_t*) w + 896);

      vacc0x0 = _mm512_add_epi32(vacc0x0, _mm512_madd_epi16(va0, vb0));
      vacc1x0 = _mm512_add_epi32(vacc1x0, _mm512_madd_epi16(va1, vb0));
      vacc2x0 = _mm512_add_epi32(vacc2x0, _mm512_madd_epi16(va2, vb0));
      vacc3x0 = _mm512_add_epi32(vacc3x0, _mm512_madd_epi16(va3, vb0));
      vacc4x0 = _mm512_add_epi32(vacc4x0, _mm512_madd_epi16(va4, vb0));
      vacc5x0 = _mm512_add_epi32(vacc5x0, _mm512_madd_epi16(va5, vb0));
      vacc6x0 = _mm512_add_epi32(vacc6x0, _mm512_madd_epi16(va6, vb0));
      const __m512i vb1 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 32))), vb_zero_point);

      vacc0x1 = _mm512_add_epi32(vacc0x1, _mm512_madd_epi16(va0, vb1));
      vacc1x1 = _mm512_add_epi32(vacc1x1, _mm512_madd_epi16(va1, vb1));
      vacc2x1 = _mm512_add_epi32(vacc2x1, _mm512_madd_epi16(va2, vb1));
      vacc3x1 = _mm512_add_epi32(vacc3x1, _mm512_madd_epi16(va3, vb1));
      vacc4x1 = _mm512_add_epi32(vacc4x1, _mm512_madd_epi16(va4, vb1));
      vacc5x1 = _mm512_add_epi32(vacc5x1, _mm512_madd_epi16(va5, vb1));
      vacc6x1 = _mm512_add_epi32(vacc6x1, _mm512_madd_epi16(va6, vb1));
      const __m512i vb2 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 64))), vb_zero_point);
      xnn_prefetch_to_l1((const uint8_t*) w + 960);

      vacc0x2 = _mm512_add_epi32(vacc0x2, _mm512_madd_epi16(va0, vb2));
      vacc1x2 = _mm512_add_epi32(vacc1x2, _mm512_madd_epi16(va1, vb2));
      vacc2x2 = _mm512_add_epi32(vacc2x2, _mm512_madd_epi16(va2, vb2));
      vacc3x2 = _mm512_add_epi32(vacc3x2, _mm512_madd_epi16(va3, vb2));
      vacc4x2 = _mm512_add_epi32(vacc4x2, _mm512_madd_epi16(va4, vb2));
      vacc5x2 = _mm512_add_epi32(vacc5x2, _mm512_madd_epi16(va5, vb2));
      vacc6x2 = _mm512_add_epi32(vacc6x2, _mm512_madd_epi16(va6, vb2));
      const __m512i vb3 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 96))), vb_zero_point);

      vacc0x3 = _mm512_add_epi32(vacc0x3, _mm512_madd_epi16(va0, vb3));
      vacc1x3 = _mm512_add_epi32(vacc1x3, _mm512_madd_epi16(va1, vb3));
      vacc2x3 = _mm512_add_epi32(vacc2x3, _mm512_madd_epi16(va2, vb3));
      vacc3x3 = _mm512_add_epi32(vacc3x3, _mm512_madd_epi16(va3, vb3));
      vacc4x3 = _mm512_add_epi32(vacc4x3, _mm512_madd_epi16(va4, vb3));
      vacc5x3 = _mm512_add_epi32(vacc5x3, _mm512_madd_epi16(va5, vb3));
      vacc6x3 = _mm512_add_epi32(vacc6x3, _mm512_madd_epi16(va6, vb3));

      w = (const uint8_t*) w + 128;
      k -= 8 * sizeof(uint8_t);
    }

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

    const __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    const __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    const __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    const __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));
    const __m512i vacc4x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc4x04152637, vacc4x8C9DAEBF), _mm512_unpackhi_epi32(vacc4x04152637, vacc4x8C9DAEBF));
    const __m512i vacc5x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc5x04152637, vacc5x8C9DAEBF), _mm512_unpackhi_epi32(vacc5x04152637, vacc5x8C9DAEBF));
    const __m512i vacc6x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc6x04152637, vacc6x8C9DAEBF), _mm512_unpackhi_epi32(vacc6x04152637, vacc6x8C9DAEBF));

    const __m512i vidx = _mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0);
    __m512i vacc0x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc0x084C195D2A6E3B7F);
    __m512i vacc1x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc1x084C195D2A6E3B7F);
    __m512i vacc2x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc2x084C195D2A6E3B7F);
    __m512i vacc3x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc3x084C195D2A6E3B7F);
    __m512i vacc4x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc4x084C195D2A6E3B7F);
    __m512i vacc5x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc5x084C195D2A6E3B7F);
    __m512i vacc6x0123456789ABCDEF = _mm512_permutexvar_epi32(vidx, vacc6x084C195D2A6E3B7F);

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled4x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc4x0123456789ABCDEF);
    __m512 vscaled5x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc5x0123456789ABCDEF);
    __m512 vscaled6x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc6x0123456789ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vscale);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vscale);
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, vscale);
    vscaled4x0123456789ABCDEF = _mm512_mul_ps(vscaled4x0123456789ABCDEF, vscale);
    vscaled5x0123456789ABCDEF = _mm512_mul_ps(vscaled5x0123456789ABCDEF, vscale);
    vscaled6x0123456789ABCDEF = _mm512_mul_ps(vscaled6x0123456789ABCDEF, vscale);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled4x0123456789ABCDEF = _mm512_min_ps(vscaled4x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled5x0123456789ABCDEF = _mm512_min_ps(vscaled5x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled6x0123456789ABCDEF = _mm512_min_ps(vscaled6x0123456789ABCDEF, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled6x0123456789ABCDEF);

    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, voutput_zero_point);
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, voutput_zero_point);
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, voutput_zero_point);
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, voutput_zero_point);
    vacc4x0123456789ABCDEF = _mm512_add_epi32(vacc4x0123456789ABCDEF, voutput_zero_point);
    vacc5x0123456789ABCDEF = _mm512_add_epi32(vacc5x0123456789ABCDEF, voutput_zero_point);
    vacc6x0123456789ABCDEF = _mm512_add_epi32(vacc6x0123456789ABCDEF, voutput_zero_point);

    __m256i vaccph0x0123456789ABCDEF = _mm512_cvtsepi32_epi16(vacc0x0123456789ABCDEF);
    __m256i vaccph1x0123456789ABCDEF = _mm512_cvtsepi32_epi16(vacc1x0123456789ABCDEF);
    __m256i vaccph2x0123456789ABCDEF = _mm512_cvtsepi32_epi16(vacc2x0123456789ABCDEF);
    __m256i vaccph3x0123456789ABCDEF = _mm512_cvtsepi32_epi16(vacc3x0123456789ABCDEF);
    __m256i vaccph4x0123456789ABCDEF = _mm512_cvtsepi32_epi16(vacc4x0123456789ABCDEF);
    __m256i vaccph5x0123456789ABCDEF = _mm512_cvtsepi32_epi16(vacc5x0123456789ABCDEF);
    __m256i vaccph6x0123456789ABCDEF = _mm512_cvtsepi32_epi16(vacc6x0123456789ABCDEF);

    __m128i vout0x0123456789ABCDEF = _mm_packus_epi16(_mm256_castsi256_si128(vaccph0x0123456789ABCDEF), _mm256_extracti128_si256(vaccph0x0123456789ABCDEF, 1));
    __m128i vout1x0123456789ABCDEF = _mm_packus_epi16(_mm256_castsi256_si128(vaccph1x0123456789ABCDEF), _mm256_extracti128_si256(vaccph1x0123456789ABCDEF, 1));
    __m128i vout2x0123456789ABCDEF = _mm_packus_epi16(_mm256_castsi256_si128(vaccph2x0123456789ABCDEF), _mm256_extracti128_si256(vaccph2x0123456789ABCDEF, 1));
    __m128i vout3x0123456789ABCDEF = _mm_packus_epi16(_mm256_castsi256_si128(vaccph3x0123456789ABCDEF), _mm256_extracti128_si256(vaccph3x0123456789ABCDEF, 1));
    __m128i vout4x0123456789ABCDEF = _mm_packus_epi16(_mm256_castsi256_si128(vaccph4x0123456789ABCDEF), _mm256_extracti128_si256(vaccph4x0123456789ABCDEF, 1));
    __m128i vout5x0123456789ABCDEF = _mm_packus_epi16(_mm256_castsi256_si128(vaccph5x0123456789ABCDEF), _mm256_extracti128_si256(vaccph5x0123456789ABCDEF, 1));
    __m128i vout6x0123456789ABCDEF = _mm_packus_epi16(_mm256_castsi256_si128(vaccph6x0123456789ABCDEF), _mm256_extracti128_si256(vaccph6x0123456789ABCDEF, 1));

    vout0x0123456789ABCDEF = _mm_max_epu8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = _mm_max_epu8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = _mm_max_epu8(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = _mm_max_epu8(vout3x0123456789ABCDEF, voutput_min);
    vout4x0123456789ABCDEF = _mm_max_epu8(vout4x0123456789ABCDEF, voutput_min);
    vout5x0123456789ABCDEF = _mm_max_epu8(vout5x0123456789ABCDEF, voutput_min);
    vout6x0123456789ABCDEF = _mm_max_epu8(vout6x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      _mm_storeu_si128((__m128i*) c1, vout1x0123456789ABCDEF);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      _mm_storeu_si128((__m128i*) c2, vout2x0123456789ABCDEF);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);
      _mm_storeu_si128((__m128i*) c3, vout3x0123456789ABCDEF);
      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);
      a3 = (const uint8_t*) ((uintptr_t) a3 - kc);
      _mm_storeu_si128((__m128i*) c4, vout4x0123456789ABCDEF);
      c4 = (uint8_t*) ((uintptr_t) c4 + cn_stride);
      a4 = (const uint8_t*) ((uintptr_t) a4 - kc);
      _mm_storeu_si128((__m128i*) c5, vout5x0123456789ABCDEF);
      c5 = (uint8_t*) ((uintptr_t) c5 + cn_stride);
      a5 = (const uint8_t*) ((uintptr_t) a5 - kc);
      _mm_storeu_si128((__m128i*) c6, vout6x0123456789ABCDEF);
      c6 = (uint8_t*) ((uintptr_t) c6 + cn_stride);
      a6 = (const uint8_t*) ((uintptr_t) a6 - kc);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - UINT32_C(1));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c1, vmask, vout1x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c2, vmask, vout2x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c3, vmask, vout3x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c4, vmask, vout4x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c5, vmask, vout5x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c6, vmask, vout6x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}
