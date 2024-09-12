// Auto-generated file. Do not edit!
//   Template: src/f32-qc4w-gemm/avx-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/gemm.h"
#include "xnnpack/unaligned.h"


void xnn_f32_qc4w_gemm_minmax_ukernel_1x16__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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
  const __m256i vmagic_bias_c0 = _mm256_set1_epi32(0x4B0000F0);
  const __m256i vmagic_bias_c1 = _mm256_set1_epi32(0x4900000F);
  const __m256 vmagic_bias_plus_kernel_zero_point_c0 = _mm256_set1_ps(0x1.0001E0p+23f + (float) params->scalar.kernel_zero_point);
  const __m256 vmagic_bias_plus_kernel_zero_point_c1 = _mm256_set1_ps(0x1.00001Ep+19f + (float) params->scalar.kernel_zero_point);
  XNN_FORCE_REALIZATION(vmagic_bias_c0);
  XNN_FORCE_REALIZATION(vmagic_bias_c1);
  XNN_FORCE_REALIZATION(vmagic_bias_plus_kernel_zero_point_c0);
  XNN_FORCE_REALIZATION(vmagic_bias_plus_kernel_zero_point_c1);

  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  const __m256 vmax = _mm256_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_loadu_ps((const float*) w + 0);
    __m256 vacc0x89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    w = (const float*) w + 16;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= sizeof(float) * 2) {
      const __m256 va0c0 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va0c1 = _mm256_broadcast_ss(a0);
      a0 += 1;

      const __m128i vbi0123c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w)));
      const __m128i vbi4567c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128i vbi89ABc01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 8)));
      const __m128i vbiCDEFc01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 12)));
      const __m256 vbi01234567c01 = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi0123c01)), _mm_castsi128_ps(vbi4567c01), 1);
      const __m256 vbi89ABCDEFc01 = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi89ABc01)), _mm_castsi128_ps(vbiCDEFc01), 1);
      w = (const uint8_t*) w + 16;

      const __m256 vbm01234567c0 = _mm256_or_ps(vbi01234567c01, _mm256_castsi256_ps(vmagic_bias_c0));
      const __m256 vbm89ABCDEFc0 = _mm256_or_ps(vbi89ABCDEFc01, _mm256_castsi256_ps(vmagic_bias_c0));
      const __m256 vbm01234567c1 = _mm256_or_ps(vbi01234567c01, _mm256_castsi256_ps(vmagic_bias_c1));
      const __m256 vbm89ABCDEFc1 = _mm256_or_ps(vbi89ABCDEFc01, _mm256_castsi256_ps(vmagic_bias_c1));

      const __m256 vb01234567c0 = _mm256_sub_ps(vbm01234567c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb89ABCDEFc0 = _mm256_sub_ps(vbm89ABCDEFc0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb01234567c1 = _mm256_sub_ps(vbm01234567c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m256 vb89ABCDEFc1 = _mm256_sub_ps(vbm89ABCDEFc1, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x01234567 = _mm256_fmadd_ps(va0c0, vb01234567c0, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0c0, vb89ABCDEFc0, vacc0x89ABCDEF);
      vacc0x01234567 = _mm256_fmadd_ps(va0c1, vb01234567c1, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0c1, vb89ABCDEFc1, vacc0x89ABCDEF);
    }

    if XNN_UNLIKELY(k != 0) {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;

      const __m128i vbi0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w)));
      const __m128i vbi4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128i vbi89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 8)));
      const __m128i vbiCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 12)));
      const __m256 vbi01234567 = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi0123)), _mm_castsi128_ps(vbi4567), 1);
      const __m256 vbi89ABCDEF = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi89AB)), _mm_castsi128_ps(vbiCDEF), 1);
      w = (const uint8_t*) w + 16;

      const __m256 vbm01234567 = _mm256_or_ps(vbi01234567, _mm256_castsi256_ps(vmagic_bias_c0));
      const __m256 vbm89ABCDEF = _mm256_or_ps(vbi89ABCDEF, _mm256_castsi256_ps(vmagic_bias_c0));

      const __m256 vb01234567 = _mm256_sub_ps(vbm01234567, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb89ABCDEF = _mm256_sub_ps(vbm89ABCDEF, vmagic_bias_plus_kernel_zero_point_c0);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
    }

    const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w + 0);
    vacc0x01234567 = _mm256_mul_ps(vacc0x01234567, vscale01234567);
    const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    vacc0x89ABCDEF = _mm256_mul_ps(vacc0x89ABCDEF, vscale89ABCDEF);
    w = (const float*) w + 16;
    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

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
