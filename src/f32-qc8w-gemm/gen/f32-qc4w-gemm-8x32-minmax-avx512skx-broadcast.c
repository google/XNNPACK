// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-qc4w-gemm/avx512-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"


void xnn_f32_qc4w_gemm_minmax_ukernel_8x32__avx512skx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 8);
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
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const float* a7 = (const float*) ((uintptr_t) a6 + a_stride);
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
  }
  const __m512i vmagic_bias_c0 = _mm512_set1_epi32(0x4B0000F0);
  const __m512i vmagic_bias_c1 = _mm512_set1_epi32(0x4900000F);
  const __m512 vmagic_bias_plus_kernel_zero_point_c0 = _mm512_set1_ps(0x1.0001E0p+23f + (float) params->scalar.kernel_zero_point);
  const __m512 vmagic_bias_plus_kernel_zero_point_c1 = _mm512_set1_ps(0x1.00001Ep+19f + (float) params->scalar.kernel_zero_point);
  XNN_FORCE_REALIZATION(vmagic_bias_c0);
  XNN_FORCE_REALIZATION(vmagic_bias_c1);
  XNN_FORCE_REALIZATION(vmagic_bias_plus_kernel_zero_point_c0);
  XNN_FORCE_REALIZATION(vmagic_bias_plus_kernel_zero_point_c1);

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    __m512 vacc0xGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    __m512 vacc1x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc1xGHIJKLMNOPQRSTUV = vacc0xGHIJKLMNOPQRSTUV;
    __m512 vacc2x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc2xGHIJKLMNOPQRSTUV = vacc0xGHIJKLMNOPQRSTUV;
    __m512 vacc3x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc3xGHIJKLMNOPQRSTUV = vacc0xGHIJKLMNOPQRSTUV;
    __m512 vacc4x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc4xGHIJKLMNOPQRSTUV = vacc0xGHIJKLMNOPQRSTUV;
    __m512 vacc5x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc5xGHIJKLMNOPQRSTUV = vacc0xGHIJKLMNOPQRSTUV;
    __m512 vacc6x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc6xGHIJKLMNOPQRSTUV = vacc0xGHIJKLMNOPQRSTUV;
    __m512 vacc7x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc7xGHIJKLMNOPQRSTUV = vacc0xGHIJKLMNOPQRSTUV;
    w = (const float*) w + 32;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= sizeof(float) * 2) {
      const __m512 va0c0 = _mm512_set1_ps(*a0);
      a0 += 1;
      const __m512 va1c0 = _mm512_set1_ps(*a1);
      a1 += 1;
      const __m512 va2c0 = _mm512_set1_ps(*a2);
      a2 += 1;
      const __m512 va3c0 = _mm512_set1_ps(*a3);
      a3 += 1;
      const __m512 va4c0 = _mm512_set1_ps(*a4);
      a4 += 1;
      const __m512 va5c0 = _mm512_set1_ps(*a5);
      a5 += 1;
      const __m512 va6c0 = _mm512_set1_ps(*a6);
      a6 += 1;
      const __m512 va7c0 = _mm512_set1_ps(*a7);
      a7 += 1;
      const __m512 va0c1 = _mm512_set1_ps(*a0);
      a0 += 1;
      const __m512 va1c1 = _mm512_set1_ps(*a1);
      a1 += 1;
      const __m512 va2c1 = _mm512_set1_ps(*a2);
      a2 += 1;
      const __m512 va3c1 = _mm512_set1_ps(*a3);
      a3 += 1;
      const __m512 va4c1 = _mm512_set1_ps(*a4);
      a4 += 1;
      const __m512 va5c1 = _mm512_set1_ps(*a5);
      a5 += 1;
      const __m512 va6c1 = _mm512_set1_ps(*a6);
      a6 += 1;
      const __m512 va7c1 = _mm512_set1_ps(*a7);
      a7 += 1;

      const __m512i vbi0123456789ABCDEFc01 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) w));
      const __m512i vbiGHIJKLMNOPQRSTUVc01 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16)));
      w = (const int8_t*) w + 32;

      const __m512 vbm0123456789ABCDEFc0 =  _mm512_castsi512_ps(_mm512_or_si512(vbi0123456789ABCDEFc01, vmagic_bias_c0));
      const __m512 vbmGHIJKLMNOPQRSTUVc0 =  _mm512_castsi512_ps(_mm512_or_si512(vbiGHIJKLMNOPQRSTUVc01, vmagic_bias_c0));
      const __m512 vbm0123456789ABCDEFc1 =  _mm512_castsi512_ps(_mm512_or_si512(vbi0123456789ABCDEFc01, vmagic_bias_c1));
      const __m512 vbmGHIJKLMNOPQRSTUVc1 =  _mm512_castsi512_ps(_mm512_or_si512(vbiGHIJKLMNOPQRSTUVc01, vmagic_bias_c1));

      const __m512 vb0123456789ABCDEFc0 = _mm512_sub_ps(vbm0123456789ABCDEFc0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m512 vbGHIJKLMNOPQRSTUVc0 = _mm512_sub_ps(vbmGHIJKLMNOPQRSTUVc0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m512 vb0123456789ABCDEFc1 = _mm512_sub_ps(vbm0123456789ABCDEFc1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m512 vbGHIJKLMNOPQRSTUVc1 = _mm512_sub_ps(vbmGHIJKLMNOPQRSTUVc1, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0c0, vb0123456789ABCDEFc0, vacc0x0123456789ABCDEF);
      vacc1x0123456789ABCDEF = _mm512_fmadd_ps(va1c0, vb0123456789ABCDEFc0, vacc1x0123456789ABCDEF);
      vacc2x0123456789ABCDEF = _mm512_fmadd_ps(va2c0, vb0123456789ABCDEFc0, vacc2x0123456789ABCDEF);
      vacc3x0123456789ABCDEF = _mm512_fmadd_ps(va3c0, vb0123456789ABCDEFc0, vacc3x0123456789ABCDEF);
      vacc4x0123456789ABCDEF = _mm512_fmadd_ps(va4c0, vb0123456789ABCDEFc0, vacc4x0123456789ABCDEF);
      vacc5x0123456789ABCDEF = _mm512_fmadd_ps(va5c0, vb0123456789ABCDEFc0, vacc5x0123456789ABCDEF);
      vacc6x0123456789ABCDEF = _mm512_fmadd_ps(va6c0, vb0123456789ABCDEFc0, vacc6x0123456789ABCDEF);
      vacc7x0123456789ABCDEF = _mm512_fmadd_ps(va7c0, vb0123456789ABCDEFc0, vacc7x0123456789ABCDEF);
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0c0, vbGHIJKLMNOPQRSTUVc0, vacc0xGHIJKLMNOPQRSTUV);
      vacc1xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va1c0, vbGHIJKLMNOPQRSTUVc0, vacc1xGHIJKLMNOPQRSTUV);
      vacc2xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va2c0, vbGHIJKLMNOPQRSTUVc0, vacc2xGHIJKLMNOPQRSTUV);
      vacc3xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va3c0, vbGHIJKLMNOPQRSTUVc0, vacc3xGHIJKLMNOPQRSTUV);
      vacc4xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va4c0, vbGHIJKLMNOPQRSTUVc0, vacc4xGHIJKLMNOPQRSTUV);
      vacc5xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va5c0, vbGHIJKLMNOPQRSTUVc0, vacc5xGHIJKLMNOPQRSTUV);
      vacc6xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va6c0, vbGHIJKLMNOPQRSTUVc0, vacc6xGHIJKLMNOPQRSTUV);
      vacc7xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va7c0, vbGHIJKLMNOPQRSTUVc0, vacc7xGHIJKLMNOPQRSTUV);
      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0c1, vb0123456789ABCDEFc1, vacc0x0123456789ABCDEF);
      vacc1x0123456789ABCDEF = _mm512_fmadd_ps(va1c1, vb0123456789ABCDEFc1, vacc1x0123456789ABCDEF);
      vacc2x0123456789ABCDEF = _mm512_fmadd_ps(va2c1, vb0123456789ABCDEFc1, vacc2x0123456789ABCDEF);
      vacc3x0123456789ABCDEF = _mm512_fmadd_ps(va3c1, vb0123456789ABCDEFc1, vacc3x0123456789ABCDEF);
      vacc4x0123456789ABCDEF = _mm512_fmadd_ps(va4c1, vb0123456789ABCDEFc1, vacc4x0123456789ABCDEF);
      vacc5x0123456789ABCDEF = _mm512_fmadd_ps(va5c1, vb0123456789ABCDEFc1, vacc5x0123456789ABCDEF);
      vacc6x0123456789ABCDEF = _mm512_fmadd_ps(va6c1, vb0123456789ABCDEFc1, vacc6x0123456789ABCDEF);
      vacc7x0123456789ABCDEF = _mm512_fmadd_ps(va7c1, vb0123456789ABCDEFc1, vacc7x0123456789ABCDEF);
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0c1, vbGHIJKLMNOPQRSTUVc1, vacc0xGHIJKLMNOPQRSTUV);
      vacc1xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va1c1, vbGHIJKLMNOPQRSTUVc1, vacc1xGHIJKLMNOPQRSTUV);
      vacc2xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va2c1, vbGHIJKLMNOPQRSTUVc1, vacc2xGHIJKLMNOPQRSTUV);
      vacc3xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va3c1, vbGHIJKLMNOPQRSTUVc1, vacc3xGHIJKLMNOPQRSTUV);
      vacc4xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va4c1, vbGHIJKLMNOPQRSTUVc1, vacc4xGHIJKLMNOPQRSTUV);
      vacc5xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va5c1, vbGHIJKLMNOPQRSTUVc1, vacc5xGHIJKLMNOPQRSTUV);
      vacc6xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va6c1, vbGHIJKLMNOPQRSTUVc1, vacc6xGHIJKLMNOPQRSTUV);
      vacc7xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va7c1, vbGHIJKLMNOPQRSTUVc1, vacc7xGHIJKLMNOPQRSTUV);
    }

    if XNN_UNLIKELY(k != 0) {
      const __m512 va0 = _mm512_set1_ps(*a0);
      a0 += 1;
      const __m512 va1 = _mm512_set1_ps(*a1);
      a1 += 1;
      const __m512 va2 = _mm512_set1_ps(*a2);
      a2 += 1;
      const __m512 va3 = _mm512_set1_ps(*a3);
      a3 += 1;
      const __m512 va4 = _mm512_set1_ps(*a4);
      a4 += 1;
      const __m512 va5 = _mm512_set1_ps(*a5);
      a5 += 1;
      const __m512 va6 = _mm512_set1_ps(*a6);
      a6 += 1;
      const __m512 va7 = _mm512_set1_ps(*a7);
      a7 += 1;

      const __m512i vbi0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) w));
      const __m512i vbiGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16)));
      w = (const int8_t*) w + 32;

      const __m512 vbm0123456789ABCDEF =  _mm512_castsi512_ps(_mm512_or_si512(vbi0123456789ABCDEF, vmagic_bias_c0));
      const __m512 vbmGHIJKLMNOPQRSTUV =  _mm512_castsi512_ps(_mm512_or_si512(vbiGHIJKLMNOPQRSTUV, vmagic_bias_c0));

      const __m512 vb0123456789ABCDEF = _mm512_sub_ps(vbm0123456789ABCDEF, vmagic_bias_plus_kernel_zero_point_c0);
      const __m512 vbGHIJKLMNOPQRSTUV = _mm512_sub_ps(vbmGHIJKLMNOPQRSTUV, vmagic_bias_plus_kernel_zero_point_c0);

      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);
      vacc1x0123456789ABCDEF = _mm512_fmadd_ps(va1, vb0123456789ABCDEF, vacc1x0123456789ABCDEF);
      vacc2x0123456789ABCDEF = _mm512_fmadd_ps(va2, vb0123456789ABCDEF, vacc2x0123456789ABCDEF);
      vacc3x0123456789ABCDEF = _mm512_fmadd_ps(va3, vb0123456789ABCDEF, vacc3x0123456789ABCDEF);
      vacc4x0123456789ABCDEF = _mm512_fmadd_ps(va4, vb0123456789ABCDEF, vacc4x0123456789ABCDEF);
      vacc5x0123456789ABCDEF = _mm512_fmadd_ps(va5, vb0123456789ABCDEF, vacc5x0123456789ABCDEF);
      vacc6x0123456789ABCDEF = _mm512_fmadd_ps(va6, vb0123456789ABCDEF, vacc6x0123456789ABCDEF);
      vacc7x0123456789ABCDEF = _mm512_fmadd_ps(va7, vb0123456789ABCDEF, vacc7x0123456789ABCDEF);
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0, vbGHIJKLMNOPQRSTUV, vacc0xGHIJKLMNOPQRSTUV);
      vacc1xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va1, vbGHIJKLMNOPQRSTUV, vacc1xGHIJKLMNOPQRSTUV);
      vacc2xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va2, vbGHIJKLMNOPQRSTUV, vacc2xGHIJKLMNOPQRSTUV);
      vacc3xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va3, vbGHIJKLMNOPQRSTUV, vacc3xGHIJKLMNOPQRSTUV);
      vacc4xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va4, vbGHIJKLMNOPQRSTUV, vacc4xGHIJKLMNOPQRSTUV);
      vacc5xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va5, vbGHIJKLMNOPQRSTUV, vacc5xGHIJKLMNOPQRSTUV);
      vacc6xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va6, vbGHIJKLMNOPQRSTUV, vacc6xGHIJKLMNOPQRSTUV);
      vacc7xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va7, vbGHIJKLMNOPQRSTUV, vacc7xGHIJKLMNOPQRSTUV);
    }

    const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    vacc0x0123456789ABCDEF = _mm512_mul_ps(vacc0x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_mul_ps(vacc1x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_mul_ps(vacc2x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_mul_ps(vacc3x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_mul_ps(vacc4x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_mul_ps(vacc5x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_mul_ps(vacc6x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc7x0123456789ABCDEF = _mm512_mul_ps(vacc7x0123456789ABCDEF, vscale0123456789ABCDEF);
    const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc0xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc1xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc2xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc3xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc4xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc5xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc6xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc7xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc7xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    w = (const float*) w + 32;
    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vmin, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vmin, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vmin, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vmin, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_max_ps(vmin, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_max_ps(vmin, vacc6x0123456789ABCDEF);
    vacc7x0123456789ABCDEF = _mm512_max_ps(vmin, vacc7x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc0xGHIJKLMNOPQRSTUV);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc1xGHIJKLMNOPQRSTUV);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc2xGHIJKLMNOPQRSTUV);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc3xGHIJKLMNOPQRSTUV);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc4xGHIJKLMNOPQRSTUV);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc5xGHIJKLMNOPQRSTUV);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc6xGHIJKLMNOPQRSTUV);
    vacc7xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc7xGHIJKLMNOPQRSTUV);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vmax, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vmax, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vmax, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vmax, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_min_ps(vmax, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_min_ps(vmax, vacc6x0123456789ABCDEF);
    vacc7x0123456789ABCDEF = _mm512_min_ps(vmax, vacc7x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc0xGHIJKLMNOPQRSTUV);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc1xGHIJKLMNOPQRSTUV);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc2xGHIJKLMNOPQRSTUV);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc3xGHIJKLMNOPQRSTUV);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc4xGHIJKLMNOPQRSTUV);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc5xGHIJKLMNOPQRSTUV);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc6xGHIJKLMNOPQRSTUV);
    vacc7xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc7xGHIJKLMNOPQRSTUV);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c7, vacc7x0123456789ABCDEF);
      _mm512_storeu_ps(c7 + 16, vacc7xGHIJKLMNOPQRSTUV);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c6, vacc6x0123456789ABCDEF);
      _mm512_storeu_ps(c6 + 16, vacc6xGHIJKLMNOPQRSTUV);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c5, vacc5x0123456789ABCDEF);
      _mm512_storeu_ps(c5 + 16, vacc5xGHIJKLMNOPQRSTUV);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c4, vacc4x0123456789ABCDEF);
      _mm512_storeu_ps(c4 + 16, vacc4xGHIJKLMNOPQRSTUV);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3, vacc3x0123456789ABCDEF);
      _mm512_storeu_ps(c3 + 16, vacc3xGHIJKLMNOPQRSTUV);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2, vacc2x0123456789ABCDEF);
      _mm512_storeu_ps(c2 + 16, vacc2xGHIJKLMNOPQRSTUV);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1, vacc1x0123456789ABCDEF);
      _mm512_storeu_ps(c1 + 16, vacc1xGHIJKLMNOPQRSTUV);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      _mm512_storeu_ps(c0 + 16, vacc0xGHIJKLMNOPQRSTUV);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a7 = (const float*) ((uintptr_t) a7 - kc);
      a6 = (const float*) ((uintptr_t) a6 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 32;
    } else {
      if (nc & 16) {
        _mm512_storeu_ps(c7, vacc7x0123456789ABCDEF);
        _mm512_storeu_ps(c6, vacc6x0123456789ABCDEF);
        _mm512_storeu_ps(c5, vacc5x0123456789ABCDEF);
        _mm512_storeu_ps(c4, vacc4x0123456789ABCDEF);
        _mm512_storeu_ps(c3, vacc3x0123456789ABCDEF);
        _mm512_storeu_ps(c2, vacc2x0123456789ABCDEF);
        _mm512_storeu_ps(c1, vacc1x0123456789ABCDEF);
        _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);

        vacc7x0123456789ABCDEF = vacc7xGHIJKLMNOPQRSTUV;
        vacc6x0123456789ABCDEF = vacc6xGHIJKLMNOPQRSTUV;
        vacc5x0123456789ABCDEF = vacc5xGHIJKLMNOPQRSTUV;
        vacc4x0123456789ABCDEF = vacc4xGHIJKLMNOPQRSTUV;
        vacc3x0123456789ABCDEF = vacc3xGHIJKLMNOPQRSTUV;
        vacc2x0123456789ABCDEF = vacc2xGHIJKLMNOPQRSTUV;
        vacc1x0123456789ABCDEF = vacc1xGHIJKLMNOPQRSTUV;
        vacc0x0123456789ABCDEF = vacc0xGHIJKLMNOPQRSTUV;

        c7 += 16;
        c6 += 16;
        c5 += 16;
        c4 += 16;
        c3 += 16;
        c2 += 16;
        c1 += 16;
        c0 += 16;
      }
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) (UINT32_C(1) << (nc & 15)) - UINT32_C(1));
        _mm512_mask_storeu_ps(c7, vmask, vacc7x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c6, vmask, vacc6x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c5, vmask, vacc5x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c4, vmask, vacc4x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c3, vmask, vacc3x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c2, vmask, vacc2x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c1, vmask, vacc1x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }
      nc = 0;
    }
  } while (nc != 0);
}
