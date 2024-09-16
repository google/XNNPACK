// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx4c8-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_4x4c8__sse41_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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

  size_t bl = params->scalar.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  assert(bl % 32 == 0);
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

  const __m128 vmin = _mm_set1_ps(params->scalar.min);
  const __m128 vmax = _mm_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128i vmask = _mm_set1_epi8(0xF0);
  XNN_FORCE_REALIZATION(vmask);

  do {
    const __m128 vksum = _mm_load_ps((const float*) w);
    const __m128i vinput_zero_point01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128i vinput_zero_point0 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point1 = _mm_shuffle_epi32(vinput_zero_point01, _MM_SHUFFLE(2, 2, 2, 2));
    const __m128i vinput_zero_point23 = _mm_loadu_si128((const __m128i*) &quantization_params[2]);
    const __m128i vinput_zero_point2 = _mm_shuffle_epi32(vinput_zero_point23, _MM_SHUFFLE(0, 0, 0, 0));
    const __m128i vinput_zero_point3 = _mm_shuffle_epi32(vinput_zero_point23, _MM_SHUFFLE(2, 2, 2, 2));

    __m128 vinput_zero_point0_float = _mm_cvtepi32_ps(vinput_zero_point0);
    __m128 vout0x0123 = _mm_mul_ps(vksum, vinput_zero_point0_float);
    __m128 vinput_zero_point1_float = _mm_cvtepi32_ps(vinput_zero_point1);
    __m128 vout1x0123 = _mm_mul_ps(vksum, vinput_zero_point1_float);
    __m128 vinput_zero_point2_float = _mm_cvtepi32_ps(vinput_zero_point2);
    __m128 vout2x0123 = _mm_mul_ps(vksum, vinput_zero_point2_float);
    __m128 vinput_zero_point3_float = _mm_cvtepi32_ps(vinput_zero_point3);
    __m128 vout3x0123 = _mm_mul_ps(vksum, vinput_zero_point3_float);
    w = (const int32_t*) w + 4;

    for (size_t kb=0; kb < kc; kb += bl) {
      __m128i vacc0x0 = _mm_setzero_si128();
      __m128i vacc0x1 = _mm_setzero_si128();
      __m128i vacc0x2 = _mm_setzero_si128();
      __m128i vacc0x3 = _mm_setzero_si128();
      __m128i vacc1x0 = _mm_setzero_si128();
      __m128i vacc1x1 = _mm_setzero_si128();
      __m128i vacc1x2 = _mm_setzero_si128();
      __m128i vacc1x3 = _mm_setzero_si128();
      __m128i vacc2x0 = _mm_setzero_si128();
      __m128i vacc2x1 = _mm_setzero_si128();
      __m128i vacc2x2 = _mm_setzero_si128();
      __m128i vacc2x3 = _mm_setzero_si128();
      __m128i vacc3x0 = _mm_setzero_si128();
      __m128i vacc3x1 = _mm_setzero_si128();
      __m128i vacc3x2 = _mm_setzero_si128();
      __m128i vacc3x3 = _mm_setzero_si128();
      size_t k = bl;

      while (k >= 16 * sizeof(int8_t)) {
        const __m128i va0c0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0c0 = _mm_cvtepi8_epi16(va0c0);
        a0 += 8;
        const __m128i va1c0 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1c0 = _mm_cvtepi8_epi16(va1c0);
        a1 += 8;
        const __m128i va2c0 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2c0 = _mm_cvtepi8_epi16(va2c0);
        a2 += 8;
        const __m128i va3c0 = _mm_loadl_epi64((const __m128i*) a3);
        const __m128i vxa3c0 = _mm_cvtepi8_epi16(va3c0);
        a3 += 8;

        const __m128i vb0c01 = _mm_loadl_epi64((const __m128i*) w);
        const __m128i vbs0c0 = _mm_slli_epi32(vb0c01, 4);
        const __m128i vb0c0 = _mm_and_si128(vbs0c0, vmask);
        const __m128i vxb0c0 = _mm_cvtepi8_epi16(vb0c0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c0, vxb0c0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1c0, vxb0c0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2c0, vxb0c0));
        vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3c0, vxb0c0));
        const __m128i vb1c01 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        const __m128i vbs1c0 = _mm_slli_epi32(vb1c01, 4);
        const __m128i vb1c0 = _mm_and_si128(vbs1c0, vmask);
        const __m128i vxb1c0 = _mm_cvtepi8_epi16(vb1c0);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c0, vxb1c0));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1c0, vxb1c0));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2c0, vxb1c0));
        vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3c0, vxb1c0));
        const __m128i vb2c01 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        const __m128i vbs2c0 = _mm_slli_epi32(vb2c01, 4);
        const __m128i vb2c0 = _mm_and_si128(vbs2c0, vmask);
        const __m128i vxb2c0 = _mm_cvtepi8_epi16(vb2c0);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c0, vxb2c0));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1c0, vxb2c0));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2c0, vxb2c0));
        vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3c0, vxb2c0));
        const __m128i vb3c01 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        const __m128i vbs3c0 = _mm_slli_epi32(vb3c01, 4);
        const __m128i vb3c0 = _mm_and_si128(vbs3c0, vmask);
        const __m128i vxb3c0 = _mm_cvtepi8_epi16(vb3c0);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c0, vxb3c0));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1c0, vxb3c0));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2c0, vxb3c0));
        vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3c0, vxb3c0));

        const __m128i va0c1 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0c1 = _mm_cvtepi8_epi16(va0c1);
        a0 += 8;
        const __m128i va1c1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1c1 = _mm_cvtepi8_epi16(va1c1);
        a1 += 8;
        const __m128i va2c1 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2c1 = _mm_cvtepi8_epi16(va2c1);
        a2 += 8;
        const __m128i va3c1 = _mm_loadl_epi64((const __m128i*) a3);
        const __m128i vxa3c1 = _mm_cvtepi8_epi16(va3c1);
        a3 += 8;

        const __m128i vb0c1 = _mm_and_si128(vb0c01, vmask);
        const __m128i vxb0c1 = _mm_cvtepi8_epi16(vb0c1);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0c1, vxb0c1));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1c1, vxb0c1));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2c1, vxb0c1));
        vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3c1, vxb0c1));
        const __m128i vb1c1 = _mm_and_si128(vb1c01, vmask);
        const __m128i vxb1c1 = _mm_cvtepi8_epi16(vb1c1);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0c1, vxb1c1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1c1, vxb1c1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2c1, vxb1c1));
        vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3c1, vxb1c1));
        const __m128i vb2c1 = _mm_and_si128(vb2c01, vmask);
        const __m128i vxb2c1 = _mm_cvtepi8_epi16(vb2c1);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0c1, vxb2c1));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1c1, vxb2c1));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2c1, vxb2c1));
        vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3c1, vxb2c1));
        const __m128i vb3c1 = _mm_and_si128(vb3c01, vmask);
        const __m128i vxb3c1 = _mm_cvtepi8_epi16(vb3c1);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0c1, vxb3c1));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1c1, vxb3c1));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2c1, vxb3c1));
        vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3c1, vxb3c1));


        w = (const int8_t*) w + 32;
        k -= 16 * sizeof(int8_t);
      }

      while (k >= 8 * sizeof(int8_t)) {
        const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
        const __m128i vxa0 = _mm_cvtepi8_epi16(va0);
        a0 += 8;
        const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
        const __m128i vxa1 = _mm_cvtepi8_epi16(va1);
        a1 += 8;
        const __m128i va2 = _mm_loadl_epi64((const __m128i*) a2);
        const __m128i vxa2 = _mm_cvtepi8_epi16(va2);
        a2 += 8;
        const __m128i va3 = _mm_loadl_epi64((const __m128i*) a3);
        const __m128i vxa3 = _mm_cvtepi8_epi16(va3);
        a3 += 8;

        __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
        vb0 = _mm_slli_epi32(vb0, 4);
        vb0 = _mm_and_si128(vb0, vmask);

        const __m128i vxb0 = _mm_cvtepi8_epi16(vb0);

        vacc0x0 = _mm_add_epi32(vacc0x0, _mm_madd_epi16(vxa0, vxb0));
        vacc1x0 = _mm_add_epi32(vacc1x0, _mm_madd_epi16(vxa1, vxb0));
        vacc2x0 = _mm_add_epi32(vacc2x0, _mm_madd_epi16(vxa2, vxb0));
        vacc3x0 = _mm_add_epi32(vacc3x0, _mm_madd_epi16(vxa3, vxb0));
        __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 8));
        vb1 = _mm_slli_epi32(vb1, 4);
        vb1 = _mm_and_si128(vb1, vmask);

        const __m128i vxb1 = _mm_cvtepi8_epi16(vb1);

        vacc0x1 = _mm_add_epi32(vacc0x1, _mm_madd_epi16(vxa0, vxb1));
        vacc1x1 = _mm_add_epi32(vacc1x1, _mm_madd_epi16(vxa1, vxb1));
        vacc2x1 = _mm_add_epi32(vacc2x1, _mm_madd_epi16(vxa2, vxb1));
        vacc3x1 = _mm_add_epi32(vacc3x1, _mm_madd_epi16(vxa3, vxb1));
        __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 16));
        vb2 = _mm_slli_epi32(vb2, 4);
        vb2 = _mm_and_si128(vb2, vmask);

        const __m128i vxb2 = _mm_cvtepi8_epi16(vb2);

        vacc0x2 = _mm_add_epi32(vacc0x2, _mm_madd_epi16(vxa0, vxb2));
        vacc1x2 = _mm_add_epi32(vacc1x2, _mm_madd_epi16(vxa1, vxb2));
        vacc2x2 = _mm_add_epi32(vacc2x2, _mm_madd_epi16(vxa2, vxb2));
        vacc3x2 = _mm_add_epi32(vacc3x2, _mm_madd_epi16(vxa3, vxb2));
        __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((const int8_t*) w + 24));
        vb3 = _mm_slli_epi32(vb3, 4);
        vb3 = _mm_and_si128(vb3, vmask);

        const __m128i vxb3 = _mm_cvtepi8_epi16(vb3);

        vacc0x3 = _mm_add_epi32(vacc0x3, _mm_madd_epi16(vxa0, vxb3));
        vacc1x3 = _mm_add_epi32(vacc1x3, _mm_madd_epi16(vxa1, vxb3));
        vacc2x3 = _mm_add_epi32(vacc2x3, _mm_madd_epi16(vxa2, vxb3));
        vacc3x3 = _mm_add_epi32(vacc3x3, _mm_madd_epi16(vxa3, vxb3));

        w = (const int8_t*) w + 32;
        k -= 8 * sizeof(int8_t);
      }
      // accumulate float
      const __m128 vfilter_output_scale0123 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadl_epi64((const __m128i*) w)), 16));
      w = (const uint16_t*) w + 4;

      const __m128i vacc0x01 = _mm_hadd_epi32(vacc0x0, vacc0x1);
      const __m128i vacc0x23 = _mm_hadd_epi32(vacc0x2, vacc0x3);
      const __m128i vacc1x01 = _mm_hadd_epi32(vacc1x0, vacc1x1);
      const __m128i vacc1x23 = _mm_hadd_epi32(vacc1x2, vacc1x3);
      const __m128i vacc2x01 = _mm_hadd_epi32(vacc2x0, vacc2x1);
      const __m128i vacc2x23 = _mm_hadd_epi32(vacc2x2, vacc2x3);
      const __m128i vacc3x01 = _mm_hadd_epi32(vacc3x0, vacc3x1);
      const __m128i vacc3x23 = _mm_hadd_epi32(vacc3x2, vacc3x3);

        __m128i vacc0x0123 = _mm_hadd_epi32(vacc0x01, vacc0x23);
        __m128i vacc1x0123 = _mm_hadd_epi32(vacc1x01, vacc1x23);
        __m128i vacc2x0123 = _mm_hadd_epi32(vacc2x01, vacc2x23);
        __m128i vacc3x0123 = _mm_hadd_epi32(vacc3x01, vacc3x23);

      vout0x0123 = _mm_add_ps(vout0x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc0x0123), vfilter_output_scale0123));
      vout1x0123 = _mm_add_ps(vout1x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc1x0123), vfilter_output_scale0123));
      vout2x0123 = _mm_add_ps(vout2x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc2x0123), vfilter_output_scale0123));
      vout3x0123 = _mm_add_ps(vout3x0123, _mm_mul_ps(_mm_cvtepi32_ps(vacc3x0123), vfilter_output_scale0123));
    }

    const __m128i vinput_scale01 = _mm_loadu_si128((const __m128i*) &quantization_params[0]);
    const __m128 vinput_scale0 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale1 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale01, _MM_SHUFFLE(3, 3, 3, 3)));
    const __m128i vinput_scale23 = _mm_loadu_si128((const __m128i*) &quantization_params[2]);
    const __m128 vinput_scale2 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale23, _MM_SHUFFLE(1, 1, 1, 1)));
    const __m128 vinput_scale3 = _mm_castsi128_ps(_mm_shuffle_epi32(vinput_scale23, _MM_SHUFFLE(3, 3, 3, 3)));

    vout0x0123 = _mm_mul_ps(vout0x0123, vinput_scale0);
    vout1x0123 = _mm_mul_ps(vout1x0123, vinput_scale1);
    vout2x0123 = _mm_mul_ps(vout2x0123, vinput_scale2);
    vout3x0123 = _mm_mul_ps(vout3x0123, vinput_scale3);


    const __m128 vbias0123 = _mm_loadu_ps((const float*) w);
    w = (const float*) w + 4;
    vout0x0123 = _mm_add_ps(vout0x0123, vbias0123);
    vout1x0123 = _mm_add_ps(vout1x0123, vbias0123);
    vout2x0123 = _mm_add_ps(vout2x0123, vbias0123);
    vout3x0123 = _mm_add_ps(vout3x0123, vbias0123);

    vout0x0123 = _mm_max_ps(vout0x0123, vmin);
    vout1x0123 = _mm_max_ps(vout1x0123, vmin);
    vout2x0123 = _mm_max_ps(vout2x0123, vmin);
    vout3x0123 = _mm_max_ps(vout3x0123, vmin);

    vout0x0123 = _mm_min_ps(vout0x0123, vmax);
    vout1x0123 = _mm_min_ps(vout1x0123, vmax);
    vout2x0123 = _mm_min_ps(vout2x0123, vmax);
    vout3x0123 = _mm_min_ps(vout3x0123, vmax);

    if XNN_LIKELY(nc >= 4) {
      _mm_storeu_ps(c0, vout0x0123);
      _mm_storeu_ps(c1, vout1x0123);
      _mm_storeu_ps(c2, vout2x0123);
      _mm_storeu_ps(c3, vout3x0123);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vout0x0123);
        vout0x0123 = _mm_unpackhi_ps(vout0x0123, vout0x0123);
        c0 += 2;
        _mm_storel_pi((__m64*) c1, vout1x0123);
        vout1x0123 = _mm_unpackhi_ps(vout1x0123, vout1x0123);
        c1 += 2;
        _mm_storel_pi((__m64*) c2, vout2x0123);
        vout2x0123 = _mm_unpackhi_ps(vout2x0123, vout2x0123);
        c2 += 2;
        _mm_storel_pi((__m64*) c3, vout3x0123);
        vout3x0123 = _mm_unpackhi_ps(vout3x0123, vout3x0123);
        c3 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vout0x0123);
        _mm_store_ss(c1, vout1x0123);
        _mm_store_ss(c2, vout2x0123);
        _mm_store_ss(c3, vout3x0123);
      }
      nc = 0;
    }
  } while (nc != 0);
}
