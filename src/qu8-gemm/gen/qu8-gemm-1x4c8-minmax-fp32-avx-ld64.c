// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c8-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#ifdef _MSC_VER
  #include <intrin.h>
#else
  #include <x86intrin.h>
#endif

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  const uint8_t* a0 = a;
  uint8_t* c0 = c;


  do {
    __m128i vacc0x0 = _mm_cvtsi32_si128(((const int*) w)[0]);
    __m128i vacc0x1 = _mm_cvtsi32_si128(((const int*) w)[1]);
    __m128i vacc0x2 = _mm_cvtsi32_si128(((const int*) w)[2]);
    __m128i vacc0x3 = _mm_cvtsi32_si128(((const int*) w)[3]);
    w = (const int32_t*) w + 4;

    const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.kernel_zero_point);
    size_t k = kc;


    while (k >= 8 * sizeof(uint8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_cvtepu8_epi16(va0);
      a0 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);

      const __m128i vxb0 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb0), vb_zero_point);

      vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8));

      const __m128i vxb1 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb1), vb_zero_point);

      vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 16));

      const __m128i vxb2 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb2), vb_zero_point);

      vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 24));

      const __m128i vxb3 = _mm_sub_epi16(_mm_cvtepu8_epi16(vb3), vb_zero_point);

      vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));

      w = (const uint8_t*) w + 32;
      k -= 8 * sizeof(uint8_t);
    }

    const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
    const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);

    __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);

    __m128 vscaled0x0123 = _mm_cvtepi32_ps(vacc0x0123);

    const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
    vscaled0x0123 = _mm_mul_ps(vscaled0x0123, vscale);

    const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
    vscaled0x0123 = _mm_min_ps(vscaled0x0123, voutput_max_less_zero_point);

    vacc0x0123 = _mm_cvtps_epi32(vscaled0x0123);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
    __m128i vacc00x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc0x0123), voutput_zero_point);

    __m128i vout = _mm_packus_epi16(vacc00x0123, vacc00x0123);

    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->fp32_sse2.output_min));

    if (nc >= 4) {
      unaligned_store_u32(c0, (uint32_t) _mm_cvtsi128_si32(vout));

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        unaligned_store_u16(c0, (uint16_t) _mm_extract_epi16(vout, 0));
        c0 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *c0 = (uint8_t) _mm_extract_epi8(vout, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
