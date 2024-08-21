// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/sse-dup.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include "xnnpack/gemm.h"
#include "xnnpack/unaligned.h"


void xnn_f32_qc4w_gemm_minmax_ukernel_1x8__sse41_dup(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const __m128i vmagic_bias_c0 = _mm_load_si128((const __m128i*) params->sse.magic_bias_c0);
  const __m128i vmagic_bias_c1 = _mm_load_si128((const __m128i*) params->sse.magic_bias_c1);
  const __m128 vmagic_bias_plus_kernel_zero_point_c0 = _mm_load_ps(params->sse.magic_bias_plus_kernel_zero_point_c0);
  const __m128 vmagic_bias_plus_kernel_zero_point_c1 = _mm_load_ps(params->sse.magic_bias_plus_kernel_zero_point_c1);

  const __m128 vmax = _mm_set1_ps(params->sse.max);
  const __m128 vmin = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m128 vacc0x0123 = _mm_loadu_ps((const float*) w + 0);
    __m128 vacc0x4567 = _mm_loadu_ps((const float*) w + 4);
    w = (const float*) w + 8;

    size_t k = kc;
    for (; k >= 4 * sizeof(float); k -= 4 * sizeof(float)) {
      const __m128 va0 = _mm_loadu_ps(a0);
      a0 += 4;


      const __m128 va0c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(0, 0, 0, 0)));

      const __m128i vbi0123c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 0)));
      const __m128i vbi4567c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128i vbi0123c23 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 8)));
      const __m128i vbi4567c23 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 12)));
      const __m128 vbm0123c0 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c0));
      const __m128 vbm0123c1 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c1));
      const __m128 vbm0123c2 = _mm_castsi128_ps(_mm_or_si128(vbi0123c23, vmagic_bias_c0));
      const __m128 vbm0123c3 = _mm_castsi128_ps(_mm_or_si128(vbi0123c23, vmagic_bias_c1));
      const __m128 vbm4567c0 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c0));
      const __m128 vbm4567c1 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c1));
      const __m128 vbm4567c2 = _mm_castsi128_ps(_mm_or_si128(vbi4567c23, vmagic_bias_c0));
      const __m128 vbm4567c3 = _mm_castsi128_ps(_mm_or_si128(vbi4567c23, vmagic_bias_c1));
      const __m128 vb0123c0 = _mm_sub_ps(vbm0123c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb0123c1 = _mm_sub_ps(vbm0123c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb0123c2 = _mm_sub_ps(vbm0123c2, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb0123c3 = _mm_sub_ps(vbm0123c3, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb4567c0 = _mm_sub_ps(vbm4567c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567c1 = _mm_sub_ps(vbm4567c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb4567c2 = _mm_sub_ps(vbm4567c2, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567c3 = _mm_sub_ps(vbm4567c3, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));

      const __m128 va0c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(1, 1, 1, 1)));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));

      const __m128 va0c2222 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(2, 2, 2, 2)));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c2222, vb0123c2));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c2222, vb4567c2));

      const __m128 va0c3333 = _mm_shuffle_ps(va0, va0, _MM_SHUFFLE(3, 3, 3, 3));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c3333, vb0123c3));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c3333, vb4567c3));

      w = (const int8_t*) w + 16;
    }
    if XNN_UNLIKELY(k >= 2 * sizeof(float)) {
      const __m128 va0 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i *) a0));
      a0 += 2;

      const __m128 va0c0000 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(0, 0, 0, 0)));

      const __m128i vbi0123c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 0)));
      const __m128i vbi4567c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128 vbm0123c0 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c0));
      const __m128 vbm0123c1 = _mm_castsi128_ps(_mm_or_si128(vbi0123c01, vmagic_bias_c1));
      const __m128 vbm4567c0 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c0));
      const __m128 vbm4567c1 = _mm_castsi128_ps(_mm_or_si128(vbi4567c01, vmagic_bias_c1));
      const __m128 vb0123c0 = _mm_sub_ps(vbm0123c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb0123c1 = _mm_sub_ps(vbm0123c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m128 vb4567c0 = _mm_sub_ps(vbm4567c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567c1 = _mm_sub_ps(vbm4567c1, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c0000, vb0123c0));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c0000, vb4567c0));

      const __m128 va0c1111 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(va0), _MM_SHUFFLE(1, 1, 1, 1)));


      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0c1111, vb0123c1));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0c1111, vb4567c1));

      w = (const int8_t*) w + 8;
      k -= 2 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      const __m128 va0 = _mm_load1_ps(a0);
      a0 += 1;

      const __m128i vbi0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 0)));
      const __m128i vbi4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((uint32_t) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128 vbm0123 = _mm_castsi128_ps(_mm_or_si128(vbi0123, vmagic_bias_c0));
      const __m128 vbm4567 = _mm_castsi128_ps(_mm_or_si128(vbi4567, vmagic_bias_c0));
      const __m128 vb0123 = _mm_sub_ps(vbm0123, vmagic_bias_plus_kernel_zero_point_c0);
      const __m128 vb4567 = _mm_sub_ps(vbm4567, vmagic_bias_plus_kernel_zero_point_c0);
      w = (const int8_t*) w + 8;

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(va0, vb0123));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(va0, vb4567));

      k -= sizeof(float);
    }

    const __m128 vscale0123 = _mm_loadu_ps((const float*) w + 0);
    vacc0x0123 = _mm_mul_ps(vacc0x0123, vscale0123);
    const __m128 vscale4567 = _mm_loadu_ps((const float*) w + 4);
    vacc0x4567 = _mm_mul_ps(vacc0x4567, vscale4567);
    w = (const float*) w + 8;
    vacc0x0123 = _mm_min_ps(vacc0x0123, vmax);
    vacc0x4567 = _mm_min_ps(vacc0x4567, vmax);

    vacc0x0123 = _mm_max_ps(vacc0x0123, vmin);
    vacc0x4567 = _mm_max_ps(vacc0x4567, vmin);

    if XNN_LIKELY(nc >= 8) {
      _mm_storeu_ps(c0, vacc0x0123);
      _mm_storeu_ps(c0 + 4, vacc0x4567);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = vacc0x4567;

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
