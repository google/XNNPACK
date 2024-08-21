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


void xnn_f32_qc4w_gemm_minmax_ukernel_2x16__avx2_broadcast(
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
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }
  const __m256i vmagic_bias_c0 = _mm256_load_si256((const __m256i*) params->avx.magic_bias_c0);
  const __m256i vmagic_bias_c1 = _mm256_load_si256((const __m256i*) params->avx.magic_bias_c1);
  const __m256 vmagic_bias_plus_kernel_zero_point_c0 = _mm256_load_ps(params->avx.magic_bias_plus_kernel_zero_point_c0);
  const __m256 vmagic_bias_plus_kernel_zero_point_c1 = _mm256_load_ps(params->avx.magic_bias_plus_kernel_zero_point_c1);

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_loadu_ps((const float*) w + 0);
    __m256 vacc0x89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    w = (const float*) w + 16;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= sizeof(float) * 2) {
      const __m256 va0c0 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va1c0 = _mm256_broadcast_ss(a1);
      a1 += 1;
      const __m256 va0c1 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va1c1 = _mm256_broadcast_ss(a1);
      a1 += 1;

      const __m256i vbi01234567c01 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) w));
      const __m256i vbi89ABCDEFc01 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8)));
      w = (const uint8_t*) w + 16;

      const __m256 vbm01234567c0 = _mm256_castsi256_ps(_mm256_or_si256(vbi01234567c01, vmagic_bias_c0));
      const __m256 vbm89ABCDEFc0 = _mm256_castsi256_ps(_mm256_or_si256(vbi89ABCDEFc01, vmagic_bias_c0));
      const __m256 vbm01234567c1 = _mm256_castsi256_ps(_mm256_or_si256(vbi01234567c01, vmagic_bias_c1));
      const __m256 vbm89ABCDEFc1 = _mm256_castsi256_ps(_mm256_or_si256(vbi89ABCDEFc01, vmagic_bias_c1));

      const __m256 vb01234567c0 = _mm256_sub_ps(vbm01234567c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb89ABCDEFc0 = _mm256_sub_ps(vbm89ABCDEFc0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb01234567c1 = _mm256_sub_ps(vbm01234567c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m256 vb89ABCDEFc1 = _mm256_sub_ps(vbm89ABCDEFc1, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x01234567 = _mm256_fmadd_ps(va0c0, vb01234567c0, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1c0, vb01234567c0, vacc1x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0c0, vb89ABCDEFc0, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1c0, vb89ABCDEFc0, vacc1x89ABCDEF);
      vacc0x01234567 = _mm256_fmadd_ps(va0c1, vb01234567c1, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1c1, vb01234567c1, vacc1x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0c1, vb89ABCDEFc1, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1c1, vb89ABCDEFc1, vacc1x89ABCDEF);
    }

    if XNN_UNLIKELY(k != 0) {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va1 = _mm256_broadcast_ss(a1);
      a1 += 1;

      const __m256i vbi01234567 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) w));
      const __m256i vbi89ABCDEF = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*) ((const uint8_t*) w + 8)));
      w = (const uint8_t*) w + 16;

      const __m256 vbm01234567 = _mm256_castsi256_ps(_mm256_or_si256(vbi01234567, vmagic_bias_c0));
      const __m256 vbm89ABCDEF = _mm256_castsi256_ps(_mm256_or_si256(vbi89ABCDEF, vmagic_bias_c0));

      const __m256 vb01234567 = _mm256_sub_ps(vbm01234567, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb89ABCDEF = _mm256_sub_ps(vbm89ABCDEF, vmagic_bias_plus_kernel_zero_point_c0);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567, vacc1x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF);
    }

    const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w + 0);
    vacc0x01234567 = _mm256_mul_ps(vacc0x01234567, vscale01234567);
    vacc1x01234567 = _mm256_mul_ps(vacc1x01234567, vscale01234567);
    const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    vacc0x89ABCDEF = _mm256_mul_ps(vacc0x89ABCDEF, vscale89ABCDEF);
    vacc1x89ABCDEF = _mm256_mul_ps(vacc1x89ABCDEF, vscale89ABCDEF);
    w = (const float*) w + 16;
    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_max_ps(vmin, vacc1x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_min_ps(vmax, vacc1x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;

        c1 += 8;
        c0 += 8;
      }
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}
