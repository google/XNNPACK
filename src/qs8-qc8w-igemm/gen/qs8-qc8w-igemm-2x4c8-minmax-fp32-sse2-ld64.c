// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/MRx4c8-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/igemm.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    __m128i vacc1x0 = vacc0x0;
    __m128i vacc1x1 = vacc0x1;
    __m128i vacc1x2 = vacc0x2;
    __m128i vacc1x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = 0;
      while (k < kc) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_srai_epi16(_mm_unpacklo_epi8(va0, va0), 8);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_srai_epi16(_mm_unpacklo_epi8(va1, va1), 8);
        a1 += 8;

        const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vxb0 = _mm_srai_epi16(_mm_unpacklo_epi8(vb0, vb0), 8);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vxb1 = _mm_srai_epi16(_mm_unpacklo_epi8(vb1, vb1), 8);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vxb2 = _mm_srai_epi16(_mm_unpacklo_epi8(vb2, vb2), 8);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vxb3 = _mm_srai_epi16(_mm_unpacklo_epi8(vb3, vb3), 8);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));

        w = (const void*) ((const int8_t*) w + 32);
        k += 8 * sizeof(int8_t);
      }
      p -= 2 * sizeof(void*);
    } while (p != 0);

    const __m128i vacc0x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x0, vacc0x2), _mm_unpackhi_epi32(vacc0x0, vacc0x2));
    const __m128i vacc0x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x1, vacc0x3), _mm_unpackhi_epi32(vacc0x1, vacc0x3));
    const __m128i vacc1x02 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x0, vacc1x2), _mm_unpackhi_epi32(vacc1x0, vacc1x2));
    const __m128i vacc1x13 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x1, vacc1x3), _mm_unpackhi_epi32(vacc1x1, vacc1x3));

    __m128i vacc0x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc0x02, vacc0x13), _mm_unpackhi_epi32(vacc0x02, vacc0x13));
    __m128i vacc1x0123 = _mm_add_epi32(_mm_unpacklo_epi32(vacc1x02, vacc1x13), _mm_unpackhi_epi32(vacc1x02, vacc1x13));

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);
    __m128 vscaled1x0123 = _mm_cvtepi32_ps(vacc1x0123);

    const __m128 vscale0123 = _mm_load_ps((const float*) w);
    w = (const float*) w + 4;
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale0123);
    vscaled1x0123 = _mm_mul_ps(vscaled1x0123, vscale0123);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);
    vscaled1x0123 = _mm_min_ps(vscaled1x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);
    vacc1x0123 = _mm_cvtps_epi32(vscaled1x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);

    const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
    vacc01x0123 = _mm_max_epi16(vacc01x0123, voutput_min);

    __m128i vout = _mm_packs_epi16(vacc01x0123, vacc01x0123);


    if (nc >= 4) {
      unaligned_store_u32(c1, (uint32_t) _mm_cvtsi128_si32(_mm_shuffle_epi32(vout, _MM_SHUFFLE(1, 1, 1, 1))));
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c1, (uint16_t) _mm_extract_epi16(vout, 2));
        c1 += 2;
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c1 = (int8_t) _mm_extract_epi16(vout, 2);
        *c0 = (int8_t) _mm_cvtsi128_si32(vout);
      }

      nc = 0;
    }
  } while (nc != 0);
}
