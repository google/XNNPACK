// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/lut.h>
#include <xnnpack/math.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vadd.h>
#include <xnnpack/vcvt.h>
#include <xnnpack/vunary.h>


void xnn_f16_f32_vcvt_ukernel__avx512skx_u16(
    size_t batch,
    const void* input,
    float* output,
    const union xnn_f16_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m512 vacc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) i));
    i += 16;

    _mm512_storeu_ps(output, vacc);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 15 * sizeof(uint16_t));

    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vacc = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, i));

    _mm512_mask_storeu_ps(output, vmask, vacc);
  }
}

void xnn_f32_f16_vcvt_ukernel__avx512skx_u16(
    size_t batch,
    const float* input,
    void* output,
    const union xnn_f32_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vf = _mm512_loadu_ps(input);
    input += 16;

    _mm256_storeu_si256((__m256i*) o, _mm512_cvtps_ph(vf, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vf = _mm512_maskz_loadu_ps(vmask, input);
    const __m256i vh = _mm512_cvtps_ph(vf, _MM_FROUND_NO_EXC | _MM_FROUND_TO_NEAREST_INT);
    _mm256_mask_storeu_epi16(o, vmask, vh);
  }
}

void xnn_f32_qc4w_gemm_minmax_ukernel_1x32__avx512skx_broadcast(
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
  const __m512i vmagic_bias_c0 = _mm512_set1_epi32(params->avx512.magic_bias_c0);
  const __m512i vmagic_bias_c1 = _mm512_set1_epi32(params->avx512.magic_bias_c1);
  const __m512 vmagic_bias_plus_kernel_zero_point_c0 = _mm512_set1_ps(params->avx512.magic_bias_plus_kernel_zero_point_c0);
  const __m512 vmagic_bias_plus_kernel_zero_point_c1 = _mm512_set1_ps(params->avx512.magic_bias_plus_kernel_zero_point_c1);

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    __m512 vacc0xGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    w = (const float*) w + 32;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= sizeof(float) * 2) {
      const __m512 va0c0 = _mm512_set1_ps(*a0);
      a0 += 1;
      const __m512 va0c1 = _mm512_set1_ps(*a0);
      a0 += 1;

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
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0c0, vbGHIJKLMNOPQRSTUVc0, vacc0xGHIJKLMNOPQRSTUV);
      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0c1, vb0123456789ABCDEFc1, vacc0x0123456789ABCDEF);
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0c1, vbGHIJKLMNOPQRSTUVc1, vacc0xGHIJKLMNOPQRSTUV);
    }

    if XNN_UNLIKELY(k != 0) {
      const __m512 va0 = _mm512_set1_ps(*a0);
      a0 += 1;

      const __m512i vbi0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) w));
      const __m512i vbiGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16)));
      w = (const int8_t*) w + 32;

      const __m512 vbm0123456789ABCDEF =  _mm512_castsi512_ps(_mm512_or_si512(vbi0123456789ABCDEF, vmagic_bias_c0));
      const __m512 vbmGHIJKLMNOPQRSTUV =  _mm512_castsi512_ps(_mm512_or_si512(vbiGHIJKLMNOPQRSTUV, vmagic_bias_c0));

      const __m512 vb0123456789ABCDEF = _mm512_sub_ps(vbm0123456789ABCDEF, vmagic_bias_plus_kernel_zero_point_c0);
      const __m512 vbGHIJKLMNOPQRSTUV = _mm512_sub_ps(vbmGHIJKLMNOPQRSTUV, vmagic_bias_plus_kernel_zero_point_c0);

      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0, vbGHIJKLMNOPQRSTUV, vacc0xGHIJKLMNOPQRSTUV);
    }

    const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    vacc0x0123456789ABCDEF = _mm512_mul_ps(vacc0x0123456789ABCDEF, vscale0123456789ABCDEF);
    const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc0xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    w = (const float*) w + 32;
    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc0xGHIJKLMNOPQRSTUV);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc0xGHIJKLMNOPQRSTUV);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      _mm512_storeu_ps(c0 + 16, vacc0xGHIJKLMNOPQRSTUV);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 32;
    } else {
      if (nc & 16) {
        _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);

        vacc0x0123456789ABCDEF = vacc0xGHIJKLMNOPQRSTUV;

        c0 += 16;
      }
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << (nc & 15)) - UINT32_C(1)));
        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc4w_gemm_minmax_ukernel_7x32__avx512skx_broadcast(
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
  assert(mr <= 7);
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
  const __m512i vmagic_bias_c0 = _mm512_set1_epi32(params->avx512.magic_bias_c0);
  const __m512i vmagic_bias_c1 = _mm512_set1_epi32(params->avx512.magic_bias_c1);
  const __m512 vmagic_bias_plus_kernel_zero_point_c0 = _mm512_set1_ps(params->avx512.magic_bias_plus_kernel_zero_point_c0);
  const __m512 vmagic_bias_plus_kernel_zero_point_c1 = _mm512_set1_ps(params->avx512.magic_bias_plus_kernel_zero_point_c1);

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
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0c0, vbGHIJKLMNOPQRSTUVc0, vacc0xGHIJKLMNOPQRSTUV);
      vacc1xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va1c0, vbGHIJKLMNOPQRSTUVc0, vacc1xGHIJKLMNOPQRSTUV);
      vacc2xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va2c0, vbGHIJKLMNOPQRSTUVc0, vacc2xGHIJKLMNOPQRSTUV);
      vacc3xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va3c0, vbGHIJKLMNOPQRSTUVc0, vacc3xGHIJKLMNOPQRSTUV);
      vacc4xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va4c0, vbGHIJKLMNOPQRSTUVc0, vacc4xGHIJKLMNOPQRSTUV);
      vacc5xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va5c0, vbGHIJKLMNOPQRSTUVc0, vacc5xGHIJKLMNOPQRSTUV);
      vacc6xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va6c0, vbGHIJKLMNOPQRSTUVc0, vacc6xGHIJKLMNOPQRSTUV);
      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0c1, vb0123456789ABCDEFc1, vacc0x0123456789ABCDEF);
      vacc1x0123456789ABCDEF = _mm512_fmadd_ps(va1c1, vb0123456789ABCDEFc1, vacc1x0123456789ABCDEF);
      vacc2x0123456789ABCDEF = _mm512_fmadd_ps(va2c1, vb0123456789ABCDEFc1, vacc2x0123456789ABCDEF);
      vacc3x0123456789ABCDEF = _mm512_fmadd_ps(va3c1, vb0123456789ABCDEFc1, vacc3x0123456789ABCDEF);
      vacc4x0123456789ABCDEF = _mm512_fmadd_ps(va4c1, vb0123456789ABCDEFc1, vacc4x0123456789ABCDEF);
      vacc5x0123456789ABCDEF = _mm512_fmadd_ps(va5c1, vb0123456789ABCDEFc1, vacc5x0123456789ABCDEF);
      vacc6x0123456789ABCDEF = _mm512_fmadd_ps(va6c1, vb0123456789ABCDEFc1, vacc6x0123456789ABCDEF);
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0c1, vbGHIJKLMNOPQRSTUVc1, vacc0xGHIJKLMNOPQRSTUV);
      vacc1xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va1c1, vbGHIJKLMNOPQRSTUVc1, vacc1xGHIJKLMNOPQRSTUV);
      vacc2xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va2c1, vbGHIJKLMNOPQRSTUVc1, vacc2xGHIJKLMNOPQRSTUV);
      vacc3xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va3c1, vbGHIJKLMNOPQRSTUVc1, vacc3xGHIJKLMNOPQRSTUV);
      vacc4xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va4c1, vbGHIJKLMNOPQRSTUVc1, vacc4xGHIJKLMNOPQRSTUV);
      vacc5xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va5c1, vbGHIJKLMNOPQRSTUVc1, vacc5xGHIJKLMNOPQRSTUV);
      vacc6xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va6c1, vbGHIJKLMNOPQRSTUVc1, vacc6xGHIJKLMNOPQRSTUV);
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
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0, vbGHIJKLMNOPQRSTUV, vacc0xGHIJKLMNOPQRSTUV);
      vacc1xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va1, vbGHIJKLMNOPQRSTUV, vacc1xGHIJKLMNOPQRSTUV);
      vacc2xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va2, vbGHIJKLMNOPQRSTUV, vacc2xGHIJKLMNOPQRSTUV);
      vacc3xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va3, vbGHIJKLMNOPQRSTUV, vacc3xGHIJKLMNOPQRSTUV);
      vacc4xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va4, vbGHIJKLMNOPQRSTUV, vacc4xGHIJKLMNOPQRSTUV);
      vacc5xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va5, vbGHIJKLMNOPQRSTUV, vacc5xGHIJKLMNOPQRSTUV);
      vacc6xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va6, vbGHIJKLMNOPQRSTUV, vacc6xGHIJKLMNOPQRSTUV);
    }

    const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    vacc0x0123456789ABCDEF = _mm512_mul_ps(vacc0x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_mul_ps(vacc1x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_mul_ps(vacc2x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_mul_ps(vacc3x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_mul_ps(vacc4x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_mul_ps(vacc5x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_mul_ps(vacc6x0123456789ABCDEF, vscale0123456789ABCDEF);
    const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc0xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc1xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc2xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc3xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc4xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc5xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc6xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    w = (const float*) w + 32;
    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vmin, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vmin, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vmin, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vmin, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_max_ps(vmin, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_max_ps(vmin, vacc6x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc0xGHIJKLMNOPQRSTUV);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc1xGHIJKLMNOPQRSTUV);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc2xGHIJKLMNOPQRSTUV);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc3xGHIJKLMNOPQRSTUV);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc4xGHIJKLMNOPQRSTUV);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc5xGHIJKLMNOPQRSTUV);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc6xGHIJKLMNOPQRSTUV);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vmax, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vmax, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vmax, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vmax, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_min_ps(vmax, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_min_ps(vmax, vacc6x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc0xGHIJKLMNOPQRSTUV);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc1xGHIJKLMNOPQRSTUV);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc2xGHIJKLMNOPQRSTUV);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc3xGHIJKLMNOPQRSTUV);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc4xGHIJKLMNOPQRSTUV);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc5xGHIJKLMNOPQRSTUV);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc6xGHIJKLMNOPQRSTUV);

    if XNN_LIKELY(nc >= 32) {
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
        _mm512_storeu_ps(c6, vacc6x0123456789ABCDEF);
        _mm512_storeu_ps(c5, vacc5x0123456789ABCDEF);
        _mm512_storeu_ps(c4, vacc4x0123456789ABCDEF);
        _mm512_storeu_ps(c3, vacc3x0123456789ABCDEF);
        _mm512_storeu_ps(c2, vacc2x0123456789ABCDEF);
        _mm512_storeu_ps(c1, vacc1x0123456789ABCDEF);
        _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);

        vacc6x0123456789ABCDEF = vacc6xGHIJKLMNOPQRSTUV;
        vacc5x0123456789ABCDEF = vacc5xGHIJKLMNOPQRSTUV;
        vacc4x0123456789ABCDEF = vacc4xGHIJKLMNOPQRSTUV;
        vacc3x0123456789ABCDEF = vacc3xGHIJKLMNOPQRSTUV;
        vacc2x0123456789ABCDEF = vacc2xGHIJKLMNOPQRSTUV;
        vacc1x0123456789ABCDEF = vacc1xGHIJKLMNOPQRSTUV;
        vacc0x0123456789ABCDEF = vacc0xGHIJKLMNOPQRSTUV;

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
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << (nc & 15)) - UINT32_C(1)));
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

void xnn_f32_qc8w_gemm_minmax_ukernel_1x32__avx512skx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_loadu_ps(w);
    __m512 vacc0xGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    w = (const float*) w + 32;

    size_t k = kc;
    do {
      const __m512i vbi0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) w));
      const __m512i vbiGHIJKLMNOPQRSTUV  = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16)));
      const __m512 vb0123456789ABCDEF  = _mm512_cvtepi32_ps(vbi0123456789ABCDEF);
      const __m512 vbGHIJKLMNOPQRSTUV  = _mm512_cvtepi32_ps(vbiGHIJKLMNOPQRSTUV);
      w = (const int8_t*) w + 32;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0, vbGHIJKLMNOPQRSTUV, vacc0xGHIJKLMNOPQRSTUV);

      a0 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    vacc0x0123456789ABCDEF = _mm512_mul_ps(vacc0x0123456789ABCDEF, vscale0123456789ABCDEF);
    const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc0xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    w = (const float*) w + 32;
    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc0xGHIJKLMNOPQRSTUV);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc0xGHIJKLMNOPQRSTUV);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      _mm512_storeu_ps(c0 + 16, vacc0xGHIJKLMNOPQRSTUV);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 32;
    } else {
      if (nc & 16) {
        _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);

        vacc0x0123456789ABCDEF = vacc0xGHIJKLMNOPQRSTUV;

        c0 += 16;
      }
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << (nc & 15)) - UINT32_C(1)));
        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }
      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_7x32__avx512skx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
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

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_loadu_ps(w);
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
    w = (const float*) w + 32;

    size_t k = kc;
    do {
      const __m512i vbi0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) w));
      const __m512i vbiGHIJKLMNOPQRSTUV  = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16)));
      const __m512 vb0123456789ABCDEF  = _mm512_cvtepi32_ps(vbi0123456789ABCDEF);
      const __m512 vbGHIJKLMNOPQRSTUV  = _mm512_cvtepi32_ps(vbiGHIJKLMNOPQRSTUV);
      w = (const int8_t*) w + 32;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(va0, vb0123456789ABCDEF, vacc0x0123456789ABCDEF);
      vacc0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va0, vbGHIJKLMNOPQRSTUV, vacc0xGHIJKLMNOPQRSTUV);
      const __m512 va1 = _mm512_set1_ps(*a1);
      vacc1x0123456789ABCDEF = _mm512_fmadd_ps(va1, vb0123456789ABCDEF, vacc1x0123456789ABCDEF);
      vacc1xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va1, vbGHIJKLMNOPQRSTUV, vacc1xGHIJKLMNOPQRSTUV);
      const __m512 va2 = _mm512_set1_ps(*a2);
      vacc2x0123456789ABCDEF = _mm512_fmadd_ps(va2, vb0123456789ABCDEF, vacc2x0123456789ABCDEF);
      vacc2xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va2, vbGHIJKLMNOPQRSTUV, vacc2xGHIJKLMNOPQRSTUV);
      const __m512 va3 = _mm512_set1_ps(*a3);
      vacc3x0123456789ABCDEF = _mm512_fmadd_ps(va3, vb0123456789ABCDEF, vacc3x0123456789ABCDEF);
      vacc3xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va3, vbGHIJKLMNOPQRSTUV, vacc3xGHIJKLMNOPQRSTUV);
      const __m512 va4 = _mm512_set1_ps(*a4);
      vacc4x0123456789ABCDEF = _mm512_fmadd_ps(va4, vb0123456789ABCDEF, vacc4x0123456789ABCDEF);
      vacc4xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va4, vbGHIJKLMNOPQRSTUV, vacc4xGHIJKLMNOPQRSTUV);
      const __m512 va5 = _mm512_set1_ps(*a5);
      vacc5x0123456789ABCDEF = _mm512_fmadd_ps(va5, vb0123456789ABCDEF, vacc5x0123456789ABCDEF);
      vacc5xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va5, vbGHIJKLMNOPQRSTUV, vacc5xGHIJKLMNOPQRSTUV);
      const __m512 va6 = _mm512_set1_ps(*a6);
      vacc6x0123456789ABCDEF = _mm512_fmadd_ps(va6, vb0123456789ABCDEF, vacc6x0123456789ABCDEF);
      vacc6xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(va6, vbGHIJKLMNOPQRSTUV, vacc6xGHIJKLMNOPQRSTUV);

      a0 += 1;
      a1 += 1;
      a2 += 1;
      a3 += 1;
      a4 += 1;
      a5 += 1;
      a6 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const float*) w + 0);
    vacc0x0123456789ABCDEF = _mm512_mul_ps(vacc0x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_mul_ps(vacc1x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_mul_ps(vacc2x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_mul_ps(vacc3x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_mul_ps(vacc4x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_mul_ps(vacc5x0123456789ABCDEF, vscale0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_mul_ps(vacc6x0123456789ABCDEF, vscale0123456789ABCDEF);
    const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const float*) w + 16);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc0xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc1xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc2xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc3xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc4xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc5xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vacc6xGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);
    w = (const float*) w + 32;
    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0123456789ABCDEF = _mm512_max_ps(vmin, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vmin, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vmin, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vmin, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vmin, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_max_ps(vmin, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_max_ps(vmin, vacc6x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc0xGHIJKLMNOPQRSTUV);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc1xGHIJKLMNOPQRSTUV);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc2xGHIJKLMNOPQRSTUV);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc3xGHIJKLMNOPQRSTUV);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc4xGHIJKLMNOPQRSTUV);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc5xGHIJKLMNOPQRSTUV);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_max_ps(vmin, vacc6xGHIJKLMNOPQRSTUV);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0123456789ABCDEF = _mm512_min_ps(vmax, vacc0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vmax, vacc1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vmax, vacc2x0123456789ABCDEF);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vmax, vacc3x0123456789ABCDEF);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vmax, vacc4x0123456789ABCDEF);
    vacc5x0123456789ABCDEF = _mm512_min_ps(vmax, vacc5x0123456789ABCDEF);
    vacc6x0123456789ABCDEF = _mm512_min_ps(vmax, vacc6x0123456789ABCDEF);
    vacc0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc0xGHIJKLMNOPQRSTUV);
    vacc1xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc1xGHIJKLMNOPQRSTUV);
    vacc2xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc2xGHIJKLMNOPQRSTUV);
    vacc3xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc3xGHIJKLMNOPQRSTUV);
    vacc4xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc4xGHIJKLMNOPQRSTUV);
    vacc5xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc5xGHIJKLMNOPQRSTUV);
    vacc6xGHIJKLMNOPQRSTUV = _mm512_min_ps(vmax, vacc6xGHIJKLMNOPQRSTUV);

    if XNN_LIKELY(nc >= 32) {
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
        _mm512_storeu_ps(c6, vacc6x0123456789ABCDEF);
        _mm512_storeu_ps(c5, vacc5x0123456789ABCDEF);
        _mm512_storeu_ps(c4, vacc4x0123456789ABCDEF);
        _mm512_storeu_ps(c3, vacc3x0123456789ABCDEF);
        _mm512_storeu_ps(c2, vacc2x0123456789ABCDEF);
        _mm512_storeu_ps(c1, vacc1x0123456789ABCDEF);
        _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);

        vacc6x0123456789ABCDEF = vacc6xGHIJKLMNOPQRSTUV;
        vacc5x0123456789ABCDEF = vacc5xGHIJKLMNOPQRSTUV;
        vacc4x0123456789ABCDEF = vacc4xGHIJKLMNOPQRSTUV;
        vacc3x0123456789ABCDEF = vacc3xGHIJKLMNOPQRSTUV;
        vacc2x0123456789ABCDEF = vacc2xGHIJKLMNOPQRSTUV;
        vacc1x0123456789ABCDEF = vacc1xGHIJKLMNOPQRSTUV;
        vacc0x0123456789ABCDEF = vacc0xGHIJKLMNOPQRSTUV;

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
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << (nc & 15)) - UINT32_C(1)));
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

void xnn_f32_qs8_vcvt_ukernel__avx512skx_u128(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_load_ps(params->avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->avx512.output_zero_point);
  const __m512i vshuffle512_mask = _mm512_load_si512(params->avx512.shuffle512_mask);
  const __m512i voutput_min = _mm512_load_si512(params->avx512.output_min);
  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    __m512 vx4567 = _mm512_loadu_ps(input + 16);
    __m512 vx89AB = _mm512_loadu_ps(input + 32);
    __m512 vxCDEF = _mm512_loadu_ps(input + 48);
    __m512 vxGHIJ = _mm512_loadu_ps(input + 64);
    __m512 vxKLMN = _mm512_loadu_ps(input + 80);
    __m512 vxOPQR = _mm512_loadu_ps(input + 96);
    __m512 vxSTUV = _mm512_loadu_ps(input + 112);
    input += 128;

    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx4567 = _mm512_mul_ps(vx4567, vscale);
    vx89AB = _mm512_mul_ps(vx89AB, vscale);
    vxCDEF = _mm512_mul_ps(vxCDEF, vscale);
    vxGHIJ = _mm512_mul_ps(vxGHIJ, vscale);
    vxKLMN = _mm512_mul_ps(vxKLMN, vscale);
    vxOPQR = _mm512_mul_ps(vxOPQR, vscale);
    vxSTUV = _mm512_mul_ps(vxSTUV, vscale);

    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    vx4567 = _mm512_min_ps(vx4567, voutput_max_less_zero_point);
    vx89AB = _mm512_min_ps(vx89AB, voutput_max_less_zero_point);
    vxCDEF = _mm512_min_ps(vxCDEF, voutput_max_less_zero_point);
    vxGHIJ = _mm512_min_ps(vxGHIJ, voutput_max_less_zero_point);
    vxKLMN = _mm512_min_ps(vxKLMN, voutput_max_less_zero_point);
    vxOPQR = _mm512_min_ps(vxOPQR, voutput_max_less_zero_point);
    vxSTUV = _mm512_min_ps(vxSTUV, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);
    const __m512i vacc4567 = _mm512_cvtps_epi32(vx4567);
    const __m512i vacc89AB = _mm512_cvtps_epi32(vx89AB);
    const __m512i vaccCDEF = _mm512_cvtps_epi32(vxCDEF);
    const __m512i vaccGHIJ = _mm512_cvtps_epi32(vxGHIJ);
    const __m512i vaccKLMN = _mm512_cvtps_epi32(vxKLMN);
    const __m512i vaccOPQR = _mm512_cvtps_epi32(vxOPQR);
    const __m512i vaccSTUV = _mm512_cvtps_epi32(vxSTUV);

    __m512i vacc04152637 = _mm512_packs_epi32(vacc0123, vacc4567);
    __m512i vacc8C9DAEBF = _mm512_packs_epi32(vacc89AB, vaccCDEF);
    __m512i vaccGKHLIMJN = _mm512_packs_epi32(vaccGHIJ, vaccKLMN);
    __m512i vaccOSPTQURV = _mm512_packs_epi32(vaccOPQR, vaccSTUV);

    vacc04152637 = _mm512_adds_epi16(vacc04152637, voutput_zero_point);
    vacc8C9DAEBF = _mm512_adds_epi16(vacc8C9DAEBF, voutput_zero_point);
    vaccGKHLIMJN = _mm512_adds_epi16(vaccGKHLIMJN, voutput_zero_point);
    vaccOSPTQURV = _mm512_adds_epi16(vaccOSPTQURV, voutput_zero_point);

    __m512i vy048C159D26AE37BF = _mm512_packs_epi16(vacc04152637, vacc8C9DAEBF);
    __m512i vyGKOSHLPTIMQUJNRV = _mm512_packs_epi16(vaccGKHLIMJN, vaccOSPTQURV);

    vy048C159D26AE37BF = _mm512_max_epi8(vy048C159D26AE37BF, voutput_min);
    vyGKOSHLPTIMQUJNRV = _mm512_max_epi8(vyGKOSHLPTIMQUJNRV, voutput_min);

    const __m512i vy0123456789ABCDEF = _mm512_permutexvar_epi32(vshuffle512_mask, vy048C159D26AE37BF);
    const __m512i vyGHIJKLMNOPQRSTUV = _mm512_permutexvar_epi32(vshuffle512_mask, vyGKOSHLPTIMQUJNRV);

    _mm512_storeu_si512(output, vy0123456789ABCDEF);
    _mm512_storeu_si512(output + 64, vyGHIJKLMNOPQRSTUV);
    output += 128;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    input += 16;

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packs_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epi8(vy0123, _mm512_castsi512_si128(voutput_min));

    _mm_storeu_si128((__m128i*) output, vy0123);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vx0123 = _mm512_maskz_loadu_ps(vmask, input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packs_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epi8(vy0123, _mm512_castsi512_si128(voutput_min));

    _mm_mask_storeu_epi8(output, vmask, vy0123);
  }
}

void xnn_f32_qu8_vcvt_ukernel__avx512skx_u128(
    size_t batch,
    const float* input,
    uint8_t* output,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_load_ps(params->avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->avx512.output_zero_point);
  const __m512i vshuffle512_mask = _mm512_load_si512(params->avx512.shuffle512_mask);
  const __m512i voutput_min = _mm512_load_si512(params->avx512.output_min);
  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    __m512 vx4567 = _mm512_loadu_ps(input + 16);
    __m512 vx89AB = _mm512_loadu_ps(input + 32);
    __m512 vxCDEF = _mm512_loadu_ps(input + 48);
    __m512 vxGHIJ = _mm512_loadu_ps(input + 64);
    __m512 vxKLMN = _mm512_loadu_ps(input + 80);
    __m512 vxOPQR = _mm512_loadu_ps(input + 96);
    __m512 vxSTUV = _mm512_loadu_ps(input + 112);
    input += 128;

    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx4567 = _mm512_mul_ps(vx4567, vscale);
    vx89AB = _mm512_mul_ps(vx89AB, vscale);
    vxCDEF = _mm512_mul_ps(vxCDEF, vscale);
    vxGHIJ = _mm512_mul_ps(vxGHIJ, vscale);
    vxKLMN = _mm512_mul_ps(vxKLMN, vscale);
    vxOPQR = _mm512_mul_ps(vxOPQR, vscale);
    vxSTUV = _mm512_mul_ps(vxSTUV, vscale);

    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    vx4567 = _mm512_min_ps(vx4567, voutput_max_less_zero_point);
    vx89AB = _mm512_min_ps(vx89AB, voutput_max_less_zero_point);
    vxCDEF = _mm512_min_ps(vxCDEF, voutput_max_less_zero_point);
    vxGHIJ = _mm512_min_ps(vxGHIJ, voutput_max_less_zero_point);
    vxKLMN = _mm512_min_ps(vxKLMN, voutput_max_less_zero_point);
    vxOPQR = _mm512_min_ps(vxOPQR, voutput_max_less_zero_point);
    vxSTUV = _mm512_min_ps(vxSTUV, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);
    const __m512i vacc4567 = _mm512_cvtps_epi32(vx4567);
    const __m512i vacc89AB = _mm512_cvtps_epi32(vx89AB);
    const __m512i vaccCDEF = _mm512_cvtps_epi32(vxCDEF);
    const __m512i vaccGHIJ = _mm512_cvtps_epi32(vxGHIJ);
    const __m512i vaccKLMN = _mm512_cvtps_epi32(vxKLMN);
    const __m512i vaccOPQR = _mm512_cvtps_epi32(vxOPQR);
    const __m512i vaccSTUV = _mm512_cvtps_epi32(vxSTUV);

    __m512i vacc04152637 = _mm512_packs_epi32(vacc0123, vacc4567);
    __m512i vacc8C9DAEBF = _mm512_packs_epi32(vacc89AB, vaccCDEF);
    __m512i vaccGKHLIMJN = _mm512_packs_epi32(vaccGHIJ, vaccKLMN);
    __m512i vaccOSPTQURV = _mm512_packs_epi32(vaccOPQR, vaccSTUV);

    vacc04152637 = _mm512_adds_epi16(vacc04152637, voutput_zero_point);
    vacc8C9DAEBF = _mm512_adds_epi16(vacc8C9DAEBF, voutput_zero_point);
    vaccGKHLIMJN = _mm512_adds_epi16(vaccGKHLIMJN, voutput_zero_point);
    vaccOSPTQURV = _mm512_adds_epi16(vaccOSPTQURV, voutput_zero_point);

    __m512i vy048C159D26AE37BF = _mm512_packus_epi16(vacc04152637, vacc8C9DAEBF);
    __m512i vyGKOSHLPTIMQUJNRV = _mm512_packus_epi16(vaccGKHLIMJN, vaccOSPTQURV);

    vy048C159D26AE37BF = _mm512_max_epu8(vy048C159D26AE37BF, voutput_min);
    vyGKOSHLPTIMQUJNRV = _mm512_max_epu8(vyGKOSHLPTIMQUJNRV, voutput_min);

    const __m512i vy0123456789ABCDEF = _mm512_permutexvar_epi32(vshuffle512_mask, vy048C159D26AE37BF);
    const __m512i vyGHIJKLMNOPQRSTUV = _mm512_permutexvar_epi32(vshuffle512_mask, vyGKOSHLPTIMQUJNRV);

    _mm512_storeu_si512(output, vy0123456789ABCDEF);
    _mm512_storeu_si512(output + 64, vyGHIJKLMNOPQRSTUV);
    output += 128;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    input += 16;

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packus_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epu8(vy0123, _mm512_castsi512_si128(voutput_min));

    _mm_storeu_si128((__m128i*) output, vy0123);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vx0123 = _mm512_maskz_loadu_ps(vmask, input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packus_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epu8(vy0123, _mm512_castsi512_si128(voutput_min));

    _mm_mask_storeu_epi8(output, vmask, vy0123);
  }
}

void xnn_f32_vtanh_ukernel__avx512skx_expm1minus_rr1_lut4_p4h3ts_perm_div_u64(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vsat_cutoff = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut4_p4h3_perm.sat_cutoff);
  const __m512 vminus_log2e = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut4_p4h3_perm.minus_log2e);
  const __m512 vmagic_bias = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut4_p4h3_perm.magic_bias);
  const __m512 vtable = _mm512_load_ps(params->avx512_expm1minus_rr1_lut4_p4h3_perm.table);
  const __m512 vln2 = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut4_p4h3_perm.ln2);
  const __m512 vc4 = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut4_p4h3_perm.c4);
  const __m512 vc3 = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut4_p4h3_perm.c3);
  const __m512 vc2 = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut4_p4h3_perm.c2);
  const __m512 vminus_two = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut4_p4h3_perm.minus_two);
  const __m512 vone = _mm512_set1_ps(params->avx512_expm1minus_rr1_lut4_p4h3_perm.one);
  const __m512i vsign_mask = _mm512_set1_epi32((int) params->avx512_expm1minus_rr1_lut4_p4h3_perm.sign_mask);

  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 vx0 = _mm512_loadu_ps(input);
    const __m512 vx1 = _mm512_loadu_ps(input + 16);
    const __m512 vx2 = _mm512_loadu_ps(input + 32);
    const __m512 vx3 = _mm512_loadu_ps(input + 48);
    input += 64;

    const __m512 vz0 = _mm512_range_ps(vsat_cutoff, vx0, 0xA);
    const __m512 vz1 = _mm512_range_ps(vsat_cutoff, vx1, 0xA);
    const __m512 vz2 = _mm512_range_ps(vsat_cutoff, vx2, 0xA);
    const __m512 vz3 = _mm512_range_ps(vsat_cutoff, vx3, 0xA);
    __m512 vn0 = _mm512_fmadd_ps(vz0, vminus_log2e, vmagic_bias);
    __m512 vn1 = _mm512_fmadd_ps(vz1, vminus_log2e, vmagic_bias);
    __m512 vn2 = _mm512_fmadd_ps(vz2, vminus_log2e, vmagic_bias);
    __m512 vn3 = _mm512_fmadd_ps(vz3, vminus_log2e, vmagic_bias);

    const __m512i ve0 = _mm512_slli_epi32(_mm512_castps_si512(vn0), 21);
    const __m512i ve1 = _mm512_slli_epi32(_mm512_castps_si512(vn1), 21);
    const __m512i ve2 = _mm512_slli_epi32(_mm512_castps_si512(vn2), 21);
    const __m512i ve3 = _mm512_slli_epi32(_mm512_castps_si512(vn3), 21);

    const __m512i vl0 = _mm512_castps_si512(_mm512_permutevar_ps(vtable, _mm512_castps_si512(vn0)));
    const __m512i vl1 = _mm512_castps_si512(_mm512_permutevar_ps(vtable, _mm512_castps_si512(vn1)));
    const __m512i vl2 = _mm512_castps_si512(_mm512_permutevar_ps(vtable, _mm512_castps_si512(vn2)));
    const __m512i vl3 = _mm512_castps_si512(_mm512_permutevar_ps(vtable, _mm512_castps_si512(vn3)));

    const __m512 vs0 = _mm512_castsi512_ps(_mm512_add_epi32(vl0, ve0));
    vn0 = _mm512_sub_ps(vn0, vmagic_bias);
    const __m512 vs1 = _mm512_castsi512_ps(_mm512_add_epi32(vl1, ve1));
    vn1 = _mm512_sub_ps(vn1, vmagic_bias);
    const __m512 vs2 = _mm512_castsi512_ps(_mm512_add_epi32(vl2, ve2));
    vn2 = _mm512_sub_ps(vn2, vmagic_bias);
    const __m512 vs3 = _mm512_castsi512_ps(_mm512_add_epi32(vl3, ve3));
    vn3 = _mm512_sub_ps(vn3, vmagic_bias);

    const __m512 vt0 = _mm512_fmadd_ps(vn0, vln2, vz0);
    const __m512 vt1 = _mm512_fmadd_ps(vn1, vln2, vz1);
    const __m512 vt2 = _mm512_fmadd_ps(vn2, vln2, vz2);
    const __m512 vt3 = _mm512_fmadd_ps(vn3, vln2, vz3);

    __m512 vp0 = vc4;
    __m512 vp1 = vc4;
    __m512 vp2 = vc4;
    __m512 vp3 = vc4;
    vp0 = _mm512_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc3);
    vp0 = _mm512_fmadd_ps(vp0, vt0, vc2);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vc2);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vc2);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vc2);
    vp0 = _mm512_fmadd_ps(vp0, vt0, vminus_two);
    vp1 = _mm512_fmadd_ps(vp1, vt1, vminus_two);
    vp2 = _mm512_fmadd_ps(vp2, vt2, vminus_two);
    vp3 = _mm512_fmadd_ps(vp3, vt3, vminus_two);

    const __m512 vts0 = _mm512_mul_ps(vt0, vs0);
    const __m512 vsmo0 = _mm512_sub_ps(vs0, vone);
    const __m512 vts1 = _mm512_mul_ps(vt1, vs1);
    const __m512 vsmo1 = _mm512_sub_ps(vs1, vone);
    const __m512 vts2 = _mm512_mul_ps(vt2, vs2);
    const __m512 vsmo2 = _mm512_sub_ps(vs2, vone);
    const __m512 vts3 = _mm512_mul_ps(vt3, vs3);
    const __m512 vsmo3 = _mm512_sub_ps(vs3, vone);
    const __m512 vemo0 = _mm512_fmadd_ps(vp0, vts0, vsmo0);
    const __m512 vemo1 = _mm512_fmadd_ps(vp1, vts1, vsmo1);
    const __m512 vemo2 = _mm512_fmadd_ps(vp2, vts2, vsmo2);
    const __m512 vemo3 = _mm512_fmadd_ps(vp3, vts3, vsmo3);
    const __m512 vepo0 = _mm512_sub_ps(vemo0, vminus_two);
    const __m512 vepo1 = _mm512_sub_ps(vemo1, vminus_two);
    const __m512 vepo2 = _mm512_sub_ps(vemo2, vminus_two);
    const __m512 vepo3 = _mm512_sub_ps(vemo3, vminus_two);

    __m512 vy0 = _mm512_div_ps(vemo0, vepo0);
    __m512 vy1 = _mm512_div_ps(vemo1, vepo1);
    __m512 vy2 = _mm512_div_ps(vemo2, vepo2);
    __m512 vy3 = _mm512_div_ps(vemo3, vepo3);
    vy0 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy0), _mm512_castps_si512(vx0), vsign_mask, 0xD8));
    vy1 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy1), _mm512_castps_si512(vx1), vsign_mask, 0xD8));
    vy2 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy2), _mm512_castps_si512(vx2), vsign_mask, 0xD8));
    vy3 = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy3), _mm512_castps_si512(vx3), vsign_mask, 0xD8));

    _mm512_storeu_ps(output, vy0);
    _mm512_storeu_ps(output + 16, vy1);
    _mm512_storeu_ps(output + 32, vy2);
    _mm512_storeu_ps(output + 48, vy3);
    output += 64;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m512 vx = _mm512_loadu_ps(input);
    input += 16;

    const __m512 vz = _mm512_range_ps(vsat_cutoff, vx, 0xA);
    __m512 vn = _mm512_fmadd_ps(vz, vminus_log2e, vmagic_bias);

    const __m512i ve = _mm512_slli_epi32(_mm512_castps_si512(vn), 21);

    const __m512i vl = _mm512_castps_si512(_mm512_permutevar_ps(vtable, _mm512_castps_si512(vn)));

    const __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ve));

    vn = _mm512_sub_ps(vn, vmagic_bias);

    const __m512 vt = _mm512_fmadd_ps(vn, vln2, vz);

    __m512 vp = vc4;
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vminus_two);

    const __m512 vts = _mm512_mul_ps(vt, vs);
    const __m512 vsmo = _mm512_sub_ps(vs, vone);
    const __m512 vemo = _mm512_fmadd_ps(vp, vts, vsmo);
    const __m512 vepo = _mm512_sub_ps(vemo, vminus_two);

    __m512 vy = _mm512_div_ps(vemo, vepo);
    vy = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy), _mm512_castps_si512(vx), vsign_mask, 0xD8));

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid 32-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512 vx = _mm512_maskz_loadu_ps(vmask, input);

    const __m512 vz = _mm512_range_ps(vsat_cutoff, vx, 0xA);
    __m512 vn = _mm512_fmadd_ps(vz, vminus_log2e, vmagic_bias);

    const __m512i ve = _mm512_slli_epi32(_mm512_castps_si512(vn), 21);

    const __m512i vl = _mm512_castps_si512(_mm512_permutevar_ps(vtable, _mm512_castps_si512(vn)));

    const __m512 vs = _mm512_castsi512_ps(_mm512_add_epi32(vl, ve));

    vn = _mm512_sub_ps(vn, vmagic_bias);

    const __m512 vt = _mm512_fmadd_ps(vn, vln2, vz);

    __m512 vp = vc4;
    vp = _mm512_fmadd_ps(vp, vt, vc3);
    vp = _mm512_fmadd_ps(vp, vt, vc2);
    vp = _mm512_fmadd_ps(vp, vt, vminus_two);

    const __m512 vts = _mm512_mul_ps(vt, vs);
    const __m512 vsmo = _mm512_sub_ps(vs, vone);
    const __m512 vemo = _mm512_fmadd_ps(vp, vts, vsmo);
    const __m512 vepo = _mm512_sub_ps(vemo, vminus_two);

    __m512 vy = _mm512_div_ps(vemo, vepo);
    vy = _mm512_castsi512_ps(_mm512_ternarylogic_epi32(_mm512_castps_si512(vy), _mm512_castps_si512(vx), vsign_mask, 0xD8));

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point);
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  do {
    const __m512i vksum0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    const __m512i vksum4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    const __m512i vksum89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    const __m512i vksumCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);

    __m512i vacc0x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point0);
    __m512i vacc0x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point0);
    __m512i vacc0x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point0);
    __m512i vacc0xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point0);
    w = (const int32_t*) w + 16;

    size_t k = 0;
    // Accumulate blocks multiplication for each row.
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

      w = (const int8_t*) w + 128;
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    __m512 vout0x0123456789ABCDEF = _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0), vscaled0x084C195D2A6E3B7F);

    vout0x0123456789ABCDEF = _mm512_mul_ps(vout0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;
    vout0x0123456789ABCDEF = _mm512_fmadd_ps(vout0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vout0x0123456789ABCDEF = _mm512_max_ps(vout0x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = _mm512_min_ps(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      _mm512_storeu_ps(c0, vout0x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - k);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512i vinput_zero_point0 = _mm512_set1_epi32((int) quantization_params[0].zero_point);
  const __m512i vinput_zero_point1 = _mm512_set1_epi32((int) quantization_params[1].zero_point);
  const __m512i vinput_zero_point2 = _mm512_set1_epi32((int) quantization_params[2].zero_point);
  const __m512i vinput_zero_point3 = _mm512_set1_epi32((int) quantization_params[3].zero_point);
  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);
  do {
    const __m512i vksum0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    const __m512i vksum4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    const __m512i vksum89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    const __m512i vksumCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);

    __m512i vacc0x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point0);
    __m512i vacc0x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point0);
    __m512i vacc0x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point0);
    __m512i vacc0xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point0);
    __m512i vacc1x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point1);
    __m512i vacc1x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point1);
    __m512i vacc1x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point1);
    __m512i vacc1xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point1);
    __m512i vacc2x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point2);
    __m512i vacc2x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point2);
    __m512i vacc2x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point2);
    __m512i vacc2xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point2);
    __m512i vacc3x0123 = _mm512_mullo_epi32(vksum0123, vinput_zero_point3);
    __m512i vacc3x4567 = _mm512_mullo_epi32(vksum4567, vinput_zero_point3);
    __m512i vacc3x89AB = _mm512_mullo_epi32(vksum89AB, vinput_zero_point3);
    __m512i vacc3xCDEF = _mm512_mullo_epi32(vksumCDEF, vinput_zero_point3);
    w = (const int32_t*) w + 16;

    size_t k = 0;
    // Accumulate blocks multiplication for each row.
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

      w = (const int8_t*) w + 128;
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    __m512 vout0x0123456789ABCDEF = _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0), vscaled0x084C195D2A6E3B7F);
    __m512 vout1x0123456789ABCDEF = _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0), vscaled1x084C195D2A6E3B7F);
    __m512 vout2x0123456789ABCDEF = _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0), vscaled2x084C195D2A6E3B7F);
    __m512 vout3x0123456789ABCDEF = _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0), vscaled3x084C195D2A6E3B7F);

    vout0x0123456789ABCDEF = _mm512_mul_ps(vout0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));
    vout1x0123456789ABCDEF = _mm512_mul_ps(vout1x0123456789ABCDEF, _mm512_set1_ps(quantization_params[1].inv_scale));
    vout2x0123456789ABCDEF = _mm512_mul_ps(vout2x0123456789ABCDEF, _mm512_set1_ps(quantization_params[2].inv_scale));
    vout3x0123456789ABCDEF = _mm512_mul_ps(vout3x0123456789ABCDEF, _mm512_set1_ps(quantization_params[3].inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w);
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 16);
    w = (const float*) w + 32;
    vout0x0123456789ABCDEF = _mm512_fmadd_ps(vout0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vout1x0123456789ABCDEF = _mm512_fmadd_ps(vout1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vout2x0123456789ABCDEF = _mm512_fmadd_ps(vout2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vout3x0123456789ABCDEF = _mm512_fmadd_ps(vout3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);

    vout0x0123456789ABCDEF = _mm512_max_ps(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = _mm512_max_ps(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = _mm512_max_ps(vout2x0123456789ABCDEF, voutput_min);
    vout3x0123456789ABCDEF = _mm512_max_ps(vout3x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = _mm512_min_ps(vout0x0123456789ABCDEF, voutput_max);
    vout1x0123456789ABCDEF = _mm512_min_ps(vout1x0123456789ABCDEF, voutput_max);
    vout2x0123456789ABCDEF = _mm512_min_ps(vout2x0123456789ABCDEF, voutput_max);
    vout3x0123456789ABCDEF = _mm512_min_ps(vout3x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      _mm512_storeu_ps(c3, vout3x0123456789ABCDEF);
      _mm512_storeu_ps(c2, vout2x0123456789ABCDEF);
      _mm512_storeu_ps(c1, vout1x0123456789ABCDEF);
      _mm512_storeu_ps(c0, vout0x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - k);
      a1 = (const int8_t*) ((uintptr_t) a1 - k);
      a2 = (const int8_t*) ((uintptr_t) a2 - k);
      a3 = (const int8_t*) ((uintptr_t) a3 - k);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - 1);
      _mm512_mask_storeu_ps(c3, vmask, vout3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2, vmask, vout2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1, vmask, vout1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      const __m512i vi9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i9));
      const __m512i vk9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t))));
      const __m512i vi9xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i9 + 16)));
      const __m512i vk9xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 304 * sizeof(int8_t))));
      i9 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi9xGHIJKLMNOPQRSTUV, vk9xGHIJKLMNOPQRSTUV));

      const __m512i vi10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i10));
      const __m512i vk10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(int8_t))));
      const __m512i vi10xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i10 + 16)));
      const __m512i vk10xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 336 * sizeof(int8_t))));
      i10 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi10xGHIJKLMNOPQRSTUV, vk10xGHIJKLMNOPQRSTUV));

      const __m512i vi11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i11));
      const __m512i vk11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(int8_t))));
      const __m512i vi11xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i11 + 16)));
      const __m512i vk11xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 368 * sizeof(int8_t))));
      i11 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi11xGHIJKLMNOPQRSTUV, vk11xGHIJKLMNOPQRSTUV));

      const __m512i vi12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i12));
      const __m512i vk12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(int8_t))));
      const __m512i vi12xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i12 + 16)));
      const __m512i vk12xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 400 * sizeof(int8_t))));
      i12 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi12xGHIJKLMNOPQRSTUV, vk12xGHIJKLMNOPQRSTUV));

      const __m512i vi13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i13));
      const __m512i vk13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(int8_t))));
      const __m512i vi13xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i13 + 16)));
      const __m512i vk13xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 432 * sizeof(int8_t))));
      i13 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi13xGHIJKLMNOPQRSTUV, vk13xGHIJKLMNOPQRSTUV));

      const __m512i vi14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i14));
      const __m512i vk14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(int8_t))));
      const __m512i vi14xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i14 + 16)));
      const __m512i vk14xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 464 * sizeof(int8_t))));
      i14 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi14xGHIJKLMNOPQRSTUV, vk14xGHIJKLMNOPQRSTUV));

      const __m512i vi15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i15));
      const __m512i vk15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(int8_t))));
      const __m512i vi15xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i15 + 16)));
      const __m512i vk15xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 496 * sizeof(int8_t))));
      i15 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi15xGHIJKLMNOPQRSTUV, vk15xGHIJKLMNOPQRSTUV));

      const __m512i vi16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i16));
      const __m512i vk16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(int8_t))));
      const __m512i vi16xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i16 + 16)));
      const __m512i vk16xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 528 * sizeof(int8_t))));
      i16 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi16xGHIJKLMNOPQRSTUV, vk16xGHIJKLMNOPQRSTUV));

      const __m512i vi17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i17));
      const __m512i vk17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(int8_t))));
      const __m512i vi17xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i17 + 16)));
      const __m512i vk17xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 560 * sizeof(int8_t))));
      i17 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi17xGHIJKLMNOPQRSTUV, vk17xGHIJKLMNOPQRSTUV));

      const __m512i vi18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i18));
      const __m512i vk18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(int8_t))));
      const __m512i vi18xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i18 + 16)));
      const __m512i vk18xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 592 * sizeof(int8_t))));
      i18 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi18xGHIJKLMNOPQRSTUV, vk18xGHIJKLMNOPQRSTUV));

      const __m512i vi19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i19));
      const __m512i vk19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(int8_t))));
      const __m512i vi19xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i19 + 16)));
      const __m512i vk19xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 624 * sizeof(int8_t))));
      i19 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi19xGHIJKLMNOPQRSTUV, vk19xGHIJKLMNOPQRSTUV));

      const __m512i vi20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i20));
      const __m512i vk20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(int8_t))));
      const __m512i vi20xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i20 + 16)));
      const __m512i vk20xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 656 * sizeof(int8_t))));
      i20 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi20xGHIJKLMNOPQRSTUV, vk20xGHIJKLMNOPQRSTUV));

      const __m512i vi21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i21));
      const __m512i vk21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(int8_t))));
      const __m512i vi21xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i21 + 16)));
      const __m512i vk21xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 688 * sizeof(int8_t))));
      i21 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi21xGHIJKLMNOPQRSTUV, vk21xGHIJKLMNOPQRSTUV));

      const __m512i vi22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i22));
      const __m512i vk22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(int8_t))));
      const __m512i vi22xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i22 + 16)));
      const __m512i vk22xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 720 * sizeof(int8_t))));
      i22 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi22xGHIJKLMNOPQRSTUV, vk22xGHIJKLMNOPQRSTUV));

      const __m512i vi23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i23));
      const __m512i vk23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(int8_t))));
      const __m512i vi23xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i23 + 16)));
      const __m512i vk23xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 752 * sizeof(int8_t))));
      i23 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi23xGHIJKLMNOPQRSTUV, vk23xGHIJKLMNOPQRSTUV));

      const __m512i vi24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i24));
      const __m512i vk24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(int8_t))));
      const __m512i vi24xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i24 + 16)));
      const __m512i vk24xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 784 * sizeof(int8_t))));
      i24 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi24xGHIJKLMNOPQRSTUV, vk24xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscale);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packs_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epi8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 96)));
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 128)));
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 160)));
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 192)));
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 224)));
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 256)));
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        const __m512i vi9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i9));
        const __m512i vk9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 288)));
        i9 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));

        const __m512i vi10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i10));
        const __m512i vk10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 320)));
        i10 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));

        const __m512i vi11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i11));
        const __m512i vk11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 352)));
        i11 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));

        const __m512i vi12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i12));
        const __m512i vk12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 384)));
        i12 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));

        const __m512i vi13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i13));
        const __m512i vk13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 416)));
        i13 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));

        const __m512i vi14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i14));
        const __m512i vk14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 448)));
        i14 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));

        const __m512i vi15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i15));
        const __m512i vk15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 480)));
        i15 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));

        const __m512i vi16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i16));
        const __m512i vk16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 512)));
        i16 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));

        const __m512i vi17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i17));
        const __m512i vk17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 544)));
        i17 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));

        const __m512i vi18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i18));
        const __m512i vk18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 576)));
        i18 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));

        const __m512i vi19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i19));
        const __m512i vk19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 608)));
        i19 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));

        const __m512i vi20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i20));
        const __m512i vk20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 640)));
        i20 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));

        const __m512i vi21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i21));
        const __m512i vk21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 672)));
        i21 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));

        const __m512i vi22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i22));
        const __m512i vk22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 704)));
        i22 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));

        const __m512i vi23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i23));
        const __m512i vk23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 736)));
        i23 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));

        const __m512i vi24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i24));
        const __m512i vk24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 768)));
        i24 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (int8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscale);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packs_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epi8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 96)));
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 128)));
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 160)));
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 192)));
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 224)));
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 256)));
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (int8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_f32_vcvt_ukernel__avx512skx_u32(
    size_t batch,
    const int8_t* input,
    float* output,
    const union xnn_qs8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vminus_zero_point = _mm512_load_si512(params->avx512.minus_zero_point);
  const __m512 vscale = _mm512_load_ps(params->avx512.scale);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    __m512i vx0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input));
    __m512i vxGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (input + 16)));
    input += 32;

    vx0123456789ABCDEF = _mm512_add_epi32(vx0123456789ABCDEF, vminus_zero_point);
    vxGHIJKLMNOPQRSTUV = _mm512_add_epi32(vxGHIJKLMNOPQRSTUV, vminus_zero_point);

    __m512 vy0123456789ABCDEF = _mm512_cvtepi32_ps(vx0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vxGHIJKLMNOPQRSTUV);

    vy0123456789ABCDEF = _mm512_mul_ps(vy0123456789ABCDEF, vscale);
    vyGHIJKLMNOPQRSTUV = _mm512_mul_ps(vyGHIJKLMNOPQRSTUV, vscale);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    _mm512_storeu_ps(output + 16, vyGHIJKLMNOPQRSTUV);
    output += 32;
  }
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    __m512i vx = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input));
    vx = _mm512_add_epi32(vx, vminus_zero_point);
    input += 16;

    __m512 vy = _mm512_cvtepi32_ps(vx);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    // Prepare mask for valid elements (depends on batch).
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512i vx = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(vmask, input));
    vx = _mm512_add_epi32(vx, vminus_zero_point);

    __m512 vy = _mm512_cvtepi32_ps(vx);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_qs8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);
    w = (const int32_t*) w + 16;

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

      w = (const int8_t*) w + 128;
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);
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
    w = (const int32_t*) w + 16;

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

      w = (const int8_t*) w + 128;
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packs_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epi8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));

      a0 = (const int8_t*) ((uintptr_t) a0 - k);
      a1 = (const int8_t*) ((uintptr_t) a1 - k);
      a2 = (const int8_t*) ((uintptr_t) a2 - k);
      a3 = (const int8_t*) ((uintptr_t) a3 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx(
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
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    w = (const void*) ((const int32_t*) w + 16);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;

        const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

        w = (const void*) ((const int8_t*) w + 128);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx(
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
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
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
    w = (const void*) ((const int32_t*) w + 16);

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
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = 0;
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;
        const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
        a1 += 8;
        const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
        a2 += 8;
        const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
        a3 += 8;

        const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
        vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
        vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
        const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
        vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
        vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
        const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
        vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
        vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
        const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
        vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
        vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
        vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

        w = (const void*) ((const int8_t*) w + 128);
        k += 8 * sizeof(int8_t);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packs_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epi8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << (nc + 48)) - (UINT64_C(1) << 48)));

      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      const __m512i vi9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i9));
      const __m512i vk9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t))));
      const __m512i vi9xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i9 + 16)));
      const __m512i vk9xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 304 * sizeof(int8_t))));
      i9 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi9xGHIJKLMNOPQRSTUV, vk9xGHIJKLMNOPQRSTUV));

      const __m512i vi10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i10));
      const __m512i vk10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(int8_t))));
      const __m512i vi10xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i10 + 16)));
      const __m512i vk10xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 336 * sizeof(int8_t))));
      i10 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi10xGHIJKLMNOPQRSTUV, vk10xGHIJKLMNOPQRSTUV));

      const __m512i vi11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i11));
      const __m512i vk11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(int8_t))));
      const __m512i vi11xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i11 + 16)));
      const __m512i vk11xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 368 * sizeof(int8_t))));
      i11 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi11xGHIJKLMNOPQRSTUV, vk11xGHIJKLMNOPQRSTUV));

      const __m512i vi12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i12));
      const __m512i vk12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(int8_t))));
      const __m512i vi12xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i12 + 16)));
      const __m512i vk12xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 400 * sizeof(int8_t))));
      i12 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi12xGHIJKLMNOPQRSTUV, vk12xGHIJKLMNOPQRSTUV));

      const __m512i vi13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i13));
      const __m512i vk13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(int8_t))));
      const __m512i vi13xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i13 + 16)));
      const __m512i vk13xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 432 * sizeof(int8_t))));
      i13 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi13xGHIJKLMNOPQRSTUV, vk13xGHIJKLMNOPQRSTUV));

      const __m512i vi14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i14));
      const __m512i vk14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(int8_t))));
      const __m512i vi14xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i14 + 16)));
      const __m512i vk14xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 464 * sizeof(int8_t))));
      i14 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi14xGHIJKLMNOPQRSTUV, vk14xGHIJKLMNOPQRSTUV));

      const __m512i vi15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i15));
      const __m512i vk15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(int8_t))));
      const __m512i vi15xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i15 + 16)));
      const __m512i vk15xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 496 * sizeof(int8_t))));
      i15 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi15xGHIJKLMNOPQRSTUV, vk15xGHIJKLMNOPQRSTUV));

      const __m512i vi16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i16));
      const __m512i vk16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(int8_t))));
      const __m512i vi16xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i16 + 16)));
      const __m512i vk16xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 528 * sizeof(int8_t))));
      i16 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi16xGHIJKLMNOPQRSTUV, vk16xGHIJKLMNOPQRSTUV));

      const __m512i vi17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i17));
      const __m512i vk17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(int8_t))));
      const __m512i vi17xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i17 + 16)));
      const __m512i vk17xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 560 * sizeof(int8_t))));
      i17 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi17xGHIJKLMNOPQRSTUV, vk17xGHIJKLMNOPQRSTUV));

      const __m512i vi18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i18));
      const __m512i vk18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(int8_t))));
      const __m512i vi18xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i18 + 16)));
      const __m512i vk18xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 592 * sizeof(int8_t))));
      i18 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi18xGHIJKLMNOPQRSTUV, vk18xGHIJKLMNOPQRSTUV));

      const __m512i vi19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i19));
      const __m512i vk19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(int8_t))));
      const __m512i vi19xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i19 + 16)));
      const __m512i vk19xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 624 * sizeof(int8_t))));
      i19 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi19xGHIJKLMNOPQRSTUV, vk19xGHIJKLMNOPQRSTUV));

      const __m512i vi20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i20));
      const __m512i vk20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(int8_t))));
      const __m512i vi20xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i20 + 16)));
      const __m512i vk20xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 656 * sizeof(int8_t))));
      i20 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi20xGHIJKLMNOPQRSTUV, vk20xGHIJKLMNOPQRSTUV));

      const __m512i vi21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i21));
      const __m512i vk21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(int8_t))));
      const __m512i vi21xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i21 + 16)));
      const __m512i vk21xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 688 * sizeof(int8_t))));
      i21 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi21xGHIJKLMNOPQRSTUV, vk21xGHIJKLMNOPQRSTUV));

      const __m512i vi22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i22));
      const __m512i vk22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(int8_t))));
      const __m512i vi22xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i22 + 16)));
      const __m512i vk22xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 720 * sizeof(int8_t))));
      i22 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi22xGHIJKLMNOPQRSTUV, vk22xGHIJKLMNOPQRSTUV));

      const __m512i vi23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i23));
      const __m512i vk23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(int8_t))));
      const __m512i vi23xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i23 + 16)));
      const __m512i vk23xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 752 * sizeof(int8_t))));
      i23 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi23xGHIJKLMNOPQRSTUV, vk23xGHIJKLMNOPQRSTUV));

      const __m512i vi24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i24));
      const __m512i vk24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(int8_t))));
      const __m512i vi24xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i24 + 16)));
      const __m512i vk24xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 784 * sizeof(int8_t))));
      i24 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi24xGHIJKLMNOPQRSTUV, vk24xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps(w);
      const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const void*) ((uintptr_t) w + 16 * sizeof(float)));
      w = (const void*) ((uintptr_t) w + 32 * sizeof(float));
      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packs_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epi8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 96)));
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 128)));
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 160)));
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 192)));
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 224)));
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 256)));
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        const __m512i vi9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i9));
        const __m512i vk9x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 288)));
        i9 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));

        const __m512i vi10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i10));
        const __m512i vk10x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 320)));
        i10 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));

        const __m512i vi11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i11));
        const __m512i vk11x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 352)));
        i11 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));

        const __m512i vi12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i12));
        const __m512i vk12x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 384)));
        i12 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));

        const __m512i vi13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i13));
        const __m512i vk13x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 416)));
        i13 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));

        const __m512i vi14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i14));
        const __m512i vk14x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 448)));
        i14 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));

        const __m512i vi15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i15));
        const __m512i vk15x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 480)));
        i15 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));

        const __m512i vi16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i16));
        const __m512i vk16x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 512)));
        i16 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));

        const __m512i vi17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i17));
        const __m512i vk17x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 544)));
        i17 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));

        const __m512i vi18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i18));
        const __m512i vk18x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 576)));
        i18 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));

        const __m512i vi19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i19));
        const __m512i vk19x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 608)));
        i19 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));

        const __m512i vi20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i20));
        const __m512i vk20x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 640)));
        i20 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));

        const __m512i vi21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i21));
        const __m512i vk21x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 672)));
        i21 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));

        const __m512i vi22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i22));
        const __m512i vk22x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 704)));
        i22 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));

        const __m512i vi23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i23));
        const __m512i vk23x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 736)));
        i23 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));

        const __m512i vi24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i24));
        const __m512i vk24x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 768)));
        i24 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(int8_t)));
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (int8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps(w);
      const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const void*) ((uintptr_t) w + 16 * sizeof(float)));
      w = (const void*) ((uintptr_t) w + 32 * sizeof(float));
      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packs_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epi8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t)));
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (int8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(int8_t))));
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(int8_t))));
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(int8_t))));
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(int8_t))));
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(int8_t))));
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(int8_t))));
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(int8_t))));
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(int8_t))));
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(int8_t))));
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(int8_t))));
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(int8_t))));
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(int8_t))));
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_cvtepi8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(int8_t))));
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps(w);
      const __m512 vscaleGHIJKLMNOPQRSTUV = _mm512_loadu_ps((const void*) ((uintptr_t) w + 16 * sizeof(float)));
      w = (const void*) ((uintptr_t) w + 32 * sizeof(float));
      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscaleGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packs_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packs_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epi8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epi8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const int8_t* k = (const int8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) k));
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 32)));
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 64)));
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 96)));
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 128)));
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 160)));
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 192)));
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 224)));
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) (k + 256)));
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        const __m512 vscale0123456789ABCDEF = _mm512_loadu_ps((const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(int8_t)));
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (int8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);
    w = (const int32_t*) w + 16;

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

      w = (const int8_t*) w + 128;
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const float*) w + 16;
    const __m512 vscale084C195D2A6E3B7F = _mm512_permutexvar_ps(_mm512_set_epi32(15, 7, 11, 3, 14, 6, 10, 2, 13, 5, 9, 1, 12, 4, 8, 0), vscale012345678ABCDEF);
    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);
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
    w = (const int32_t*) w + 16;

    size_t k = 0;
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;

      const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

      w = (const int8_t*) w + 128;
      k += 8 * sizeof(int8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const float*) w + 16;
    const __m512 vscale084C195D2A6E3B7F = _mm512_permutexvar_ps(_mm512_set_epi32(15, 7, 11, 3, 14, 6, 10, 2, 13, 5, 9, 1, 12, 4, 8, 0), vscale012345678ABCDEF);
    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packs_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epi8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));

      a0 = (const int8_t*) ((uintptr_t) a0 - k);
      a1 = (const int8_t*) ((uintptr_t) a1 - k);
      a2 = (const int8_t*) ((uintptr_t) a2 - k);
      a3 = (const int8_t*) ((uintptr_t) a3 - k);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    w = (const void*) ((const int32_t*) w + 16);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;

        const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

        w = (const void*) ((const int8_t*) w + 128);
        k += 8 * sizeof(int8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const void*) ((const float*) w + 16);
    const __m512 vscale084C195D2A6E3B7F = _mm512_permutexvar_ps(_mm512_set_epi32(15, 7, 11, 3, 14, 6, 10, 2, 13, 5, 9, 1, 12, 4, 8, 0), vscale012345678ABCDEF);
    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__avx512skx(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
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
    w = (const void*) ((const int32_t*) w + 16);

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
      const int8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = 0;
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;
        const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
        a1 += 8;
        const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
        a2 += 8;
        const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
        a3 += 8;

        const __m512i vb0123 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) w));

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
        vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
        vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
        const __m512i vb4567 = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 32)));

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
        vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
        vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
        const __m512i vb89AB = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 64)));

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
        vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
        vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
        const __m512i vbCDEF = _mm512_cvtepi8_epi16(_mm256_load_si256((const __m256i*) ((const int8_t*) w + 96)));

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
        vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
        vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
        vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

        w = (const void*) ((const int8_t*) w + 128);
        k += 8 * sizeof(int8_t);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const void*) ((const float*) w + 16);
    const __m512 vscale084C195D2A6E3B7F = _mm512_permutexvar_ps(_mm512_set_epi32(15, 7, 11, 3, 14, 6, 10, 2, 13, 5, 9, 1, 12, 4, 8, 0), vscale012345678ABCDEF);
    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packs_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epi8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << (nc + 48)) - (UINT64_C(1) << 48)));

      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qs8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512i vbias = _mm512_load_si512(params->avx512.bias);
  const __m512i va_multiplier = _mm512_load_si512(params->avx512.a_multiplier);
  const __m512i vb_multiplier = _mm512_load_si512(params->avx512.b_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx512.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx512.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx512.output_max);

  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m512i va0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input_a));
    const __m512i vb0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input_b));
    input_a += 16;
    input_b += 16;

    __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

    vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vb0123456789ABCDEF, vb_multiplier));

    vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
      const __m512i va0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(vmask, input_a));
      const __m512i vb0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(vmask, input_b));

      __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vb0123456789ABCDEF, vb_multiplier));

      vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);
      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);
      vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

      _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
    }
  }
}

void xnn_qs8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512i va_multiplier = _mm512_load_si512(params->avx512.a_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx512.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx512.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx512.output_max);

  const __m512i vbias = _mm512_add_epi32(
    _mm512_broadcastd_epi32(_mm_cvtsi32_si128(params->avx512.b_multiplier[0] * (int32_t) *input_b)),
    _mm512_load_si512(params->avx512.bias));
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m512i va0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input_a));
    input_a += 16;

    __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

    vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
      const __m512i va0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(vmask, input_a));

      __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

      vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);
      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);
      vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

      _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
    }
  }
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_25p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  const __m512i vk_zero_point = _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i*) params->fp32_avx512.kernel_zero_point));
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint8_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint8_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint8_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint8_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint8_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint8_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint8_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint8_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint8_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint8_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint8_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint8_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint8_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint8_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint8_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(uint8_t)))), vk_zero_point);
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(uint8_t)))), vk_zero_point);
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(uint8_t)))), vk_zero_point);
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(uint8_t)))), vk_zero_point);
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(uint8_t)))), vk_zero_point);
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(uint8_t)))), vk_zero_point);
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(uint8_t)))), vk_zero_point);
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      const __m512i vi9x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i9));
      const __m512i vk9x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi9xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i9 + 16)));
      const __m512i vk9xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 304 * sizeof(uint8_t)))), vk_zero_point);
      i9 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi9xGHIJKLMNOPQRSTUV, vk9xGHIJKLMNOPQRSTUV));

      const __m512i vi10x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i10));
      const __m512i vk10x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 320 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi10xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i10 + 16)));
      const __m512i vk10xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 336 * sizeof(uint8_t)))), vk_zero_point);
      i10 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi10xGHIJKLMNOPQRSTUV, vk10xGHIJKLMNOPQRSTUV));

      const __m512i vi11x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i11));
      const __m512i vk11x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 352 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi11xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i11 + 16)));
      const __m512i vk11xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 368 * sizeof(uint8_t)))), vk_zero_point);
      i11 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi11xGHIJKLMNOPQRSTUV, vk11xGHIJKLMNOPQRSTUV));

      const __m512i vi12x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i12));
      const __m512i vk12x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 384 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi12xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i12 + 16)));
      const __m512i vk12xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 400 * sizeof(uint8_t)))), vk_zero_point);
      i12 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi12xGHIJKLMNOPQRSTUV, vk12xGHIJKLMNOPQRSTUV));

      const __m512i vi13x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i13));
      const __m512i vk13x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 416 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi13xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i13 + 16)));
      const __m512i vk13xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 432 * sizeof(uint8_t)))), vk_zero_point);
      i13 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi13xGHIJKLMNOPQRSTUV, vk13xGHIJKLMNOPQRSTUV));

      const __m512i vi14x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i14));
      const __m512i vk14x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 448 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi14xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i14 + 16)));
      const __m512i vk14xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 464 * sizeof(uint8_t)))), vk_zero_point);
      i14 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi14xGHIJKLMNOPQRSTUV, vk14xGHIJKLMNOPQRSTUV));

      const __m512i vi15x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i15));
      const __m512i vk15x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 480 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi15xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i15 + 16)));
      const __m512i vk15xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 496 * sizeof(uint8_t)))), vk_zero_point);
      i15 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi15xGHIJKLMNOPQRSTUV, vk15xGHIJKLMNOPQRSTUV));

      const __m512i vi16x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i16));
      const __m512i vk16x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 512 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi16xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i16 + 16)));
      const __m512i vk16xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 528 * sizeof(uint8_t)))), vk_zero_point);
      i16 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi16xGHIJKLMNOPQRSTUV, vk16xGHIJKLMNOPQRSTUV));

      const __m512i vi17x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i17));
      const __m512i vk17x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 544 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi17xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i17 + 16)));
      const __m512i vk17xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 560 * sizeof(uint8_t)))), vk_zero_point);
      i17 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi17xGHIJKLMNOPQRSTUV, vk17xGHIJKLMNOPQRSTUV));

      const __m512i vi18x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i18));
      const __m512i vk18x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 576 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi18xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i18 + 16)));
      const __m512i vk18xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 592 * sizeof(uint8_t)))), vk_zero_point);
      i18 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi18xGHIJKLMNOPQRSTUV, vk18xGHIJKLMNOPQRSTUV));

      const __m512i vi19x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i19));
      const __m512i vk19x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 608 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi19xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i19 + 16)));
      const __m512i vk19xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 624 * sizeof(uint8_t)))), vk_zero_point);
      i19 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi19xGHIJKLMNOPQRSTUV, vk19xGHIJKLMNOPQRSTUV));

      const __m512i vi20x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i20));
      const __m512i vk20x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 640 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi20xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i20 + 16)));
      const __m512i vk20xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 656 * sizeof(uint8_t)))), vk_zero_point);
      i20 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi20xGHIJKLMNOPQRSTUV, vk20xGHIJKLMNOPQRSTUV));

      const __m512i vi21x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i21));
      const __m512i vk21x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 672 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi21xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i21 + 16)));
      const __m512i vk21xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 688 * sizeof(uint8_t)))), vk_zero_point);
      i21 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi21xGHIJKLMNOPQRSTUV, vk21xGHIJKLMNOPQRSTUV));

      const __m512i vi22x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i22));
      const __m512i vk22x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 704 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi22xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i22 + 16)));
      const __m512i vk22xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 720 * sizeof(uint8_t)))), vk_zero_point);
      i22 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi22xGHIJKLMNOPQRSTUV, vk22xGHIJKLMNOPQRSTUV));

      const __m512i vi23x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i23));
      const __m512i vk23x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 736 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi23xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i23 + 16)));
      const __m512i vk23xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 752 * sizeof(uint8_t)))), vk_zero_point);
      i23 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi23xGHIJKLMNOPQRSTUV, vk23xGHIJKLMNOPQRSTUV));

      const __m512i vi24x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i24));
      const __m512i vk24x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 768 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi24xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i24 + 16)));
      const __m512i vk24xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 784 * sizeof(uint8_t)))), vk_zero_point);
      i24 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi24xGHIJKLMNOPQRSTUV, vk24xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 800 * sizeof(uint8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscale);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packus_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packus_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epu8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epu8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const uint8_t* k = (const uint8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) k)), vk_zero_point);
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 32))), vk_zero_point);
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 64))), vk_zero_point);
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 96))), vk_zero_point);
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 128))), vk_zero_point);
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 160))), vk_zero_point);
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 192))), vk_zero_point);
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 224))), vk_zero_point);
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 256))), vk_zero_point);
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        const __m512i vi9x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i9));
        const __m512i vk9x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 288))), vk_zero_point);
        i9 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi9x0123456789ABCDEF, vk9x0123456789ABCDEF));

        const __m512i vi10x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i10));
        const __m512i vk10x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 320))), vk_zero_point);
        i10 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi10x0123456789ABCDEF, vk10x0123456789ABCDEF));

        const __m512i vi11x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i11));
        const __m512i vk11x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 352))), vk_zero_point);
        i11 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi11x0123456789ABCDEF, vk11x0123456789ABCDEF));

        const __m512i vi12x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i12));
        const __m512i vk12x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 384))), vk_zero_point);
        i12 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi12x0123456789ABCDEF, vk12x0123456789ABCDEF));

        const __m512i vi13x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i13));
        const __m512i vk13x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 416))), vk_zero_point);
        i13 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi13x0123456789ABCDEF, vk13x0123456789ABCDEF));

        const __m512i vi14x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i14));
        const __m512i vk14x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 448))), vk_zero_point);
        i14 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi14x0123456789ABCDEF, vk14x0123456789ABCDEF));

        const __m512i vi15x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i15));
        const __m512i vk15x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 480))), vk_zero_point);
        i15 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi15x0123456789ABCDEF, vk15x0123456789ABCDEF));

        const __m512i vi16x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i16));
        const __m512i vk16x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 512))), vk_zero_point);
        i16 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi16x0123456789ABCDEF, vk16x0123456789ABCDEF));

        const __m512i vi17x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i17));
        const __m512i vk17x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 544))), vk_zero_point);
        i17 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi17x0123456789ABCDEF, vk17x0123456789ABCDEF));

        const __m512i vi18x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i18));
        const __m512i vk18x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 576))), vk_zero_point);
        i18 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi18x0123456789ABCDEF, vk18x0123456789ABCDEF));

        const __m512i vi19x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i19));
        const __m512i vk19x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 608))), vk_zero_point);
        i19 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi19x0123456789ABCDEF, vk19x0123456789ABCDEF));

        const __m512i vi20x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i20));
        const __m512i vk20x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 640))), vk_zero_point);
        i20 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi20x0123456789ABCDEF, vk20x0123456789ABCDEF));

        const __m512i vi21x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i21));
        const __m512i vk21x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 672))), vk_zero_point);
        i21 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi21x0123456789ABCDEF, vk21x0123456789ABCDEF));

        const __m512i vi22x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i22));
        const __m512i vk22x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 704))), vk_zero_point);
        i22 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi22x0123456789ABCDEF, vk22x0123456789ABCDEF));

        const __m512i vi23x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i23));
        const __m512i vk23x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 736))), vk_zero_point);
        i23 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi23x0123456789ABCDEF, vk23x0123456789ABCDEF));

        const __m512i vi24x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i24));
        const __m512i vk24x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 768))), vk_zero_point);
        i24 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi24x0123456789ABCDEF, vk24x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (uint8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  const __m512i vk_zero_point = _mm512_cvtepu16_epi32(_mm256_load_si256((const __m256i*) params->fp32_avx512.kernel_zero_point));
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 32; c -= 32) {
      __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);
      __m512i vaccGHIJKLMNOPQRSTUV = _mm512_loadu_si512((const void*) ((uintptr_t) w + 16 * sizeof(int32_t)));


      const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
      const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 0 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi0xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i0 + 16)));
      const __m512i vk0xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 16 * sizeof(uint8_t)))), vk_zero_point);
      i0 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi0xGHIJKLMNOPQRSTUV, vk0xGHIJKLMNOPQRSTUV));

      const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
      const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 32 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi1xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i1 + 16)));
      const __m512i vk1xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 48 * sizeof(uint8_t)))), vk_zero_point);
      i1 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi1xGHIJKLMNOPQRSTUV, vk1xGHIJKLMNOPQRSTUV));

      const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
      const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 64 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi2xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i2 + 16)));
      const __m512i vk2xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 80 * sizeof(uint8_t)))), vk_zero_point);
      i2 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi2xGHIJKLMNOPQRSTUV, vk2xGHIJKLMNOPQRSTUV));

      const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
      const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 96 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi3xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i3 + 16)));
      const __m512i vk3xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 112 * sizeof(uint8_t)))), vk_zero_point);
      i3 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi3xGHIJKLMNOPQRSTUV, vk3xGHIJKLMNOPQRSTUV));

      const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
      const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 128 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi4xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i4 + 16)));
      const __m512i vk4xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 144 * sizeof(uint8_t)))), vk_zero_point);
      i4 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi4xGHIJKLMNOPQRSTUV, vk4xGHIJKLMNOPQRSTUV));

      const __m512i vi5x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i5));
      const __m512i vk5x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 160 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi5xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i5 + 16)));
      const __m512i vk5xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 176 * sizeof(uint8_t)))), vk_zero_point);
      i5 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi5xGHIJKLMNOPQRSTUV, vk5xGHIJKLMNOPQRSTUV));

      const __m512i vi6x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i6));
      const __m512i vk6x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 192 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi6xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i6 + 16)));
      const __m512i vk6xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 208 * sizeof(uint8_t)))), vk_zero_point);
      i6 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi6xGHIJKLMNOPQRSTUV, vk6xGHIJKLMNOPQRSTUV));

      const __m512i vi7x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i7));
      const __m512i vk7x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 224 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi7xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i7 + 16)));
      const __m512i vk7xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 240 * sizeof(uint8_t)))), vk_zero_point);
      i7 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi7xGHIJKLMNOPQRSTUV, vk7xGHIJKLMNOPQRSTUV));

      const __m512i vi8x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i8));
      const __m512i vk8x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 256 * sizeof(uint8_t)))), vk_zero_point);
      const __m512i vi8xGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (i8 + 16)));
      const __m512i vk8xGHIJKLMNOPQRSTUV = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_load_si128((const __m128i*) ((uintptr_t) w + 32 * sizeof(int32_t) + 272 * sizeof(uint8_t)))), vk_zero_point);
      i8 += 32;

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));
      vaccGHIJKLMNOPQRSTUV = _mm512_add_epi32(vaccGHIJKLMNOPQRSTUV, _mm512_mullo_epi32(vi8xGHIJKLMNOPQRSTUV, vk8xGHIJKLMNOPQRSTUV));

      w = (const void*) ((uintptr_t) w + 32 * sizeof(int32_t) + 288 * sizeof(uint8_t));

      __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
      __m512 vscaledGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vaccGHIJKLMNOPQRSTUV);

      vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
      vscaledGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaledGHIJKLMNOPQRSTUV, vscale);

      vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
      vscaledGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaledGHIJKLMNOPQRSTUV, voutput_max_less_zero_point);

      vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);
      vaccGHIJKLMNOPQRSTUV = _mm512_cvtps_epi32(vscaledGHIJKLMNOPQRSTUV);

      __m512i vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV = _mm512_adds_epi16(_mm512_packs_epi32(vacc0123456789ABCDEF, vaccGHIJKLMNOPQRSTUV), voutput_zero_point);
      __m256i voutGHIJOPQRKLMNSTUV = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vaccGHIJKLMNOPQRSTUV), _mm512_extracti32x8_epi32(vaccGHIJKLMNOPQRSTUV, 1)), _mm512_castsi512_si256(voutput_zero_point));

      const __m256i vout0123GHIJ4567KLMN = _mm512_castsi512_si256(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV);
      const __m256i vout89ABOPQRCDEFSTUV = _mm512_extracti32x8_epi32(vout0123GHIJ4567KLMN89ABOPQRCDEFSTUV, 1);
      const __m256i vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV = _mm256_packus_epi16(vout0123GHIJ4567KLMN, vout89ABOPQRCDEFSTUV);
      __m256i vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_permutevar8x32_epi32(vout0123GHIJ89ABOPQR4567KLMNCDEFSTUV, vpermute_mask);
      const __m128i voutGHIJOPQR = _mm256_castsi256_si128(voutGHIJOPQRKLMNSTUV);
      const __m128i voutKLMNSTUV = _mm256_extracti128_si256(voutGHIJOPQRKLMNSTUV, 1);
      __m128i voutGHIJKLMNOPQRSTUV = _mm_shuffle_epi32(_mm_packus_epi16(voutGHIJOPQR, voutKLMNSTUV), _MM_SHUFFLE(3, 1, 2, 0));

      vout0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm256_max_epu8(vout0123456789ABCDEFGHIJKLMNOPQRSTUV, voutput_min);
      voutGHIJKLMNOPQRSTUV = _mm_max_epu8(voutGHIJKLMNOPQRSTUV, _mm256_castsi256_si128(voutput_min));

      _mm256_storeu_si256((__m256i*) output, vout0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm_storeu_si128((__m128i*) (output + 16), voutGHIJKLMNOPQRSTUV);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      const uint8_t* k = (const uint8_t*) ((uintptr_t) w + 32 * sizeof(int32_t));
      do {
        __m512i vacc0123456789ABCDEF = _mm512_loadu_si512(w);


        const __m512i vi0x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i0));
        const __m512i vk0x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) k)), vk_zero_point);
        i0 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi0x0123456789ABCDEF, vk0x0123456789ABCDEF));

        const __m512i vi1x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i1));
        const __m512i vk1x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 32))), vk_zero_point);
        i1 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi1x0123456789ABCDEF, vk1x0123456789ABCDEF));

        const __m512i vi2x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i2));
        const __m512i vk2x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 64))), vk_zero_point);
        i2 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi2x0123456789ABCDEF, vk2x0123456789ABCDEF));

        const __m512i vi3x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i3));
        const __m512i vk3x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 96))), vk_zero_point);
        i3 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi3x0123456789ABCDEF, vk3x0123456789ABCDEF));

        const __m512i vi4x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i4));
        const __m512i vk4x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 128))), vk_zero_point);
        i4 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi4x0123456789ABCDEF, vk4x0123456789ABCDEF));

        const __m512i vi5x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i5));
        const __m512i vk5x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 160))), vk_zero_point);
        i5 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi5x0123456789ABCDEF, vk5x0123456789ABCDEF));

        const __m512i vi6x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i6));
        const __m512i vk6x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 192))), vk_zero_point);
        i6 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi6x0123456789ABCDEF, vk6x0123456789ABCDEF));

        const __m512i vi7x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i7));
        const __m512i vk7x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 224))), vk_zero_point);
        i7 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi7x0123456789ABCDEF, vk7x0123456789ABCDEF));

        const __m512i vi8x0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) i8));
        const __m512i vk8x0123456789ABCDEF = _mm512_sub_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (k + 256))), vk_zero_point);
        i8 += 16;

        vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vi8x0123456789ABCDEF, vk8x0123456789ABCDEF));

        k += 16;

        __m512 vscaled0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0123456789ABCDEF);
        vscaled0123456789ABCDEF = _mm512_mul_ps(vscaled0123456789ABCDEF, vscale);
        vscaled0123456789ABCDEF = _mm512_min_ps(vscaled0123456789ABCDEF, voutput_max_less_zero_point);
        vacc0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0123456789ABCDEF);

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int32_t));

        __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), _mm512_castsi512_si256(voutput_zero_point));

        const __m128i vout012389AB = _mm256_castsi256_si128(vout012389AB4567CDEF);
        const __m128i vout4567CDEF = _mm256_extracti128_si256(vout012389AB4567CDEF, 1);
        __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(vout012389AB, vout4567CDEF), _MM_SHUFFLE(3, 1, 2, 0));
        vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
          output += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
          output = (uint8_t*) ((uintptr_t) output + c);
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_qu8_f32_vcvt_ukernel__avx512skx_u32(
    size_t batch,
    const uint8_t* input,
    float* output,
    const union xnn_qu8_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vminus_zero_point = _mm512_load_si512(params->avx512.minus_zero_point);
  const __m512 vscale = _mm512_load_ps(params->avx512.scale);
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    __m512i vx0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input));
    __m512i vxGHIJKLMNOPQRSTUV = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) (input + 16)));
    input += 32;

    vx0123456789ABCDEF = _mm512_add_epi32(vx0123456789ABCDEF, vminus_zero_point);
    vxGHIJKLMNOPQRSTUV = _mm512_add_epi32(vxGHIJKLMNOPQRSTUV, vminus_zero_point);

    __m512 vy0123456789ABCDEF = _mm512_cvtepi32_ps(vx0123456789ABCDEF);
    __m512 vyGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vxGHIJKLMNOPQRSTUV);

    vy0123456789ABCDEF = _mm512_mul_ps(vy0123456789ABCDEF, vscale);
    vyGHIJKLMNOPQRSTUV = _mm512_mul_ps(vyGHIJKLMNOPQRSTUV, vscale);

    _mm512_storeu_ps(output, vy0123456789ABCDEF);
    _mm512_storeu_ps(output + 16, vyGHIJKLMNOPQRSTUV);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m512i vx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input));
    vx = _mm512_add_epi32(vx, vminus_zero_point);
    input += 16;

    __m512 vy = _mm512_cvtepi32_ps(vx);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_storeu_ps(output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    // Prepare mask for valid elements (depends on batch).
    const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << batch) - UINT32_C(1)));

    __m512i vx = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(vmask, input));
    vx = _mm512_add_epi32(vx, vminus_zero_point);

    __m512 vy = _mm512_cvtepi32_ps(vx);
    vy = _mm512_mul_ps(vy, vscale);

    _mm512_mask_storeu_ps(output, vmask, vy);
  }
}

void xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx(
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

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);
    w = (const int32_t*) w + 16;

    size_t k = 0;
    const __m512i vb_zero_point = _mm512_load_si512(params->fp32_avx512.kernel_zero_point);
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;

      const __m512i vb0123 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) w)), vb_zero_point);

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      const __m512i vb4567 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 32))), vb_zero_point);

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      const __m512i vb89AB = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 64))), vb_zero_point);

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      const __m512i vbCDEF = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 96))), vb_zero_point);

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

      w = (const uint8_t*) w + 128;
      k += 8 * sizeof(uint8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packus_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epu8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      a0 = (const uint8_t*) ((uintptr_t) a0 - k);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx(
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
  assert(mr <= 4);
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
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 4);
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 8);
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const int32_t*) w + 12);
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
    w = (const int32_t*) w + 16;

    size_t k = 0;
    const __m512i vb_zero_point = _mm512_load_si512(params->fp32_avx512.kernel_zero_point);
    while (k < kc) {
      const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
      a0 += 8;
      const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
      a1 += 8;
      const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
      a2 += 8;
      const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
      a3 += 8;

      const __m512i vb0123 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) w)), vb_zero_point);

      vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
      vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
      vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
      vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
      const __m512i vb4567 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 32))), vb_zero_point);

      vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
      vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
      vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
      vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
      const __m512i vb89AB = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 64))), vb_zero_point);

      vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
      vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
      vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
      vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
      const __m512i vbCDEF = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 96))), vb_zero_point);

      vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
      vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
      vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
      vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

      w = (const uint8_t*) w + 128;
      k += 8 * sizeof(uint8_t);
    }

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packus_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epu8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));

      a0 = (const uint8_t*) ((uintptr_t) a0 - k);
      a1 = (const uint8_t*) ((uintptr_t) a1 - k);
      a2 = (const uint8_t*) ((uintptr_t) a2 - k);
      a3 = (const uint8_t*) ((uintptr_t) a3 - k);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftli_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
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
  uint8_t* c0 = c;

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
    w = (const void*) ((const int32_t*) w + 16);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = 0;
      const __m512i vb_zero_point = _mm512_load_si512(params->fp32_avx512.kernel_zero_point);
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;

        const __m512i vb0123 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) w)), vb_zero_point);

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        const __m512i vb4567 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 32))), vb_zero_point);

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        const __m512i vb89AB = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 64))), vb_zero_point);

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        const __m512i vbCDEF = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 96))), vb_zero_point);

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));

        w = (const void*) ((const uint8_t*) w + 128);
        k += 8 * sizeof(uint8_t);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);

    const __m256i vacc0x084C2A6E195D3B7F = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0x084C195D2A6E3B7F), _mm512_extracti32x8_epi32(vacc0x084C195D2A6E3B7F, 1)), voutput_zero_point);

    const __m128i vout0x084C2A6E195D3B7F = _mm_packus_epi16(_mm256_castsi256_si128(vacc0x084C2A6E195D3B7F), _mm256_extracti128_si256(vacc0x084C2A6E195D3B7F, 1));
    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi8(vout0x084C2A6E195D3B7F, _mm_set_epi8(15, 7, 11, 3, 13, 5, 9, 1, 14, 6, 10, 2, 12, 4, 8, 0));
    vout0x0123456789ABCDEF = _mm_max_epu8(vout0x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT32_C(1) << nc) - UINT32_C(1)));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(uint8_t));
  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const __mmask16 vbias_mask = _cvtu32_mask16(0x1111);
  const __m512 vscale = _mm512_load_ps(params->fp32_avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->fp32_avx512.output_zero_point);
  const __m512i voutput_min = _mm512_load_si512(params->fp32_avx512.output_min);
  do {
    __m512i vacc0x0123 = _mm512_maskz_expandloadu_epi32(vbias_mask, w);
    __m512i vacc0x4567 = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 4));
    __m512i vacc0x89AB = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 8));
    __m512i vacc0xCDEF = _mm512_maskz_expandloadu_epi32(vbias_mask, (const void*) ((const int32_t*) w + 12));
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
    w = (const void*) ((const int32_t*) w + 16);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint8_t* restrict a2 = a[2];
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint8_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint8_t* restrict a3 = a[3];
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const uint8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = 0;
      const __m512i vb_zero_point = _mm512_load_si512(params->fp32_avx512.kernel_zero_point);
      while (k < kc) {
        const __m512i va0 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a0)));
        a0 += 8;
        const __m512i va1 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a1)));
        a1 += 8;
        const __m512i va2 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a2)));
        a2 += 8;
        const __m512i va3 = _mm512_broadcast_i32x4(_mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) a3)));
        a3 += 8;

        const __m512i vb0123 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) w)), vb_zero_point);

        vacc0x0123 = _mm512_add_epi32(vacc0x0123, _mm512_madd_epi16(va0, vb0123));
        vacc1x0123 = _mm512_add_epi32(vacc1x0123, _mm512_madd_epi16(va1, vb0123));
        vacc2x0123 = _mm512_add_epi32(vacc2x0123, _mm512_madd_epi16(va2, vb0123));
        vacc3x0123 = _mm512_add_epi32(vacc3x0123, _mm512_madd_epi16(va3, vb0123));
        const __m512i vb4567 = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 32))), vb_zero_point);

        vacc0x4567 = _mm512_add_epi32(vacc0x4567, _mm512_madd_epi16(va0, vb4567));
        vacc1x4567 = _mm512_add_epi32(vacc1x4567, _mm512_madd_epi16(va1, vb4567));
        vacc2x4567 = _mm512_add_epi32(vacc2x4567, _mm512_madd_epi16(va2, vb4567));
        vacc3x4567 = _mm512_add_epi32(vacc3x4567, _mm512_madd_epi16(va3, vb4567));
        const __m512i vb89AB = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 64))), vb_zero_point);

        vacc0x89AB = _mm512_add_epi32(vacc0x89AB, _mm512_madd_epi16(va0, vb89AB));
        vacc1x89AB = _mm512_add_epi32(vacc1x89AB, _mm512_madd_epi16(va1, vb89AB));
        vacc2x89AB = _mm512_add_epi32(vacc2x89AB, _mm512_madd_epi16(va2, vb89AB));
        vacc3x89AB = _mm512_add_epi32(vacc3x89AB, _mm512_madd_epi16(va3, vb89AB));
        const __m512i vbCDEF = _mm512_sub_epi16(_mm512_cvtepu8_epi16(_mm256_load_si256((const __m256i*) ((const uint8_t*) w + 96))), vb_zero_point);

        vacc0xCDEF = _mm512_add_epi32(vacc0xCDEF, _mm512_madd_epi16(va0, vbCDEF));
        vacc1xCDEF = _mm512_add_epi32(vacc1xCDEF, _mm512_madd_epi16(va1, vbCDEF));
        vacc2xCDEF = _mm512_add_epi32(vacc2xCDEF, _mm512_madd_epi16(va2, vbCDEF));
        vacc3xCDEF = _mm512_add_epi32(vacc3xCDEF, _mm512_madd_epi16(va3, vbCDEF));

        w = (const void*) ((const uint8_t*) w + 128);
        k += 8 * sizeof(uint8_t);
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m512i vacc0x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x0123, vacc0x4567), _mm512_unpackhi_epi32(vacc0x0123, vacc0x4567));
    const __m512i vacc0x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x89AB, vacc0xCDEF), _mm512_unpackhi_epi32(vacc0x89AB, vacc0xCDEF));
    const __m512i vacc1x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x0123, vacc1x4567), _mm512_unpackhi_epi32(vacc1x0123, vacc1x4567));
    const __m512i vacc1x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x89AB, vacc1xCDEF), _mm512_unpackhi_epi32(vacc1x89AB, vacc1xCDEF));
    const __m512i vacc2x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x0123, vacc2x4567), _mm512_unpackhi_epi32(vacc2x0123, vacc2x4567));
    const __m512i vacc2x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x89AB, vacc2xCDEF), _mm512_unpackhi_epi32(vacc2x89AB, vacc2xCDEF));
    const __m512i vacc3x04152637 = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x0123, vacc3x4567), _mm512_unpackhi_epi32(vacc3x0123, vacc3x4567));
    const __m512i vacc3x8C9DAEBF = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x89AB, vacc3xCDEF), _mm512_unpackhi_epi32(vacc3x89AB, vacc3xCDEF));

    __m512i vacc0x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc0x04152637, vacc0x8C9DAEBF), _mm512_unpackhi_epi32(vacc0x04152637, vacc0x8C9DAEBF));
    __m512i vacc1x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc1x04152637, vacc1x8C9DAEBF), _mm512_unpackhi_epi32(vacc1x04152637, vacc1x8C9DAEBF));
    __m512i vacc2x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc2x04152637, vacc2x8C9DAEBF), _mm512_unpackhi_epi32(vacc2x04152637, vacc2x8C9DAEBF));
    __m512i vacc3x084C195D2A6E3B7F = _mm512_add_epi32(_mm512_unpacklo_epi32(vacc3x04152637, vacc3x8C9DAEBF), _mm512_unpackhi_epi32(vacc3x04152637, vacc3x8C9DAEBF));

    __m512 vscaled0x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc0x084C195D2A6E3B7F);
    __m512 vscaled1x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc1x084C195D2A6E3B7F);
    __m512 vscaled2x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc2x084C195D2A6E3B7F);
    __m512 vscaled3x084C195D2A6E3B7F = _mm512_cvtepi32_ps(vacc3x084C195D2A6E3B7F);

    vscaled0x084C195D2A6E3B7F = _mm512_mul_ps(vscaled0x084C195D2A6E3B7F, vscale);
    vscaled1x084C195D2A6E3B7F = _mm512_mul_ps(vscaled1x084C195D2A6E3B7F, vscale);
    vscaled2x084C195D2A6E3B7F = _mm512_mul_ps(vscaled2x084C195D2A6E3B7F, vscale);
    vscaled3x084C195D2A6E3B7F = _mm512_mul_ps(vscaled3x084C195D2A6E3B7F, vscale);

    vscaled0x084C195D2A6E3B7F = _mm512_min_ps(vscaled0x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled1x084C195D2A6E3B7F = _mm512_min_ps(vscaled1x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled2x084C195D2A6E3B7F = _mm512_min_ps(vscaled2x084C195D2A6E3B7F, voutput_max_less_zero_point);
    vscaled3x084C195D2A6E3B7F = _mm512_min_ps(vscaled3x084C195D2A6E3B7F, voutput_max_less_zero_point);

    vacc0x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled0x084C195D2A6E3B7F);
    vacc1x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled1x084C195D2A6E3B7F);
    vacc2x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled2x084C195D2A6E3B7F);
    vacc3x084C195D2A6E3B7F = _mm512_cvtps_epi32(vscaled3x084C195D2A6E3B7F);

    const __m512i vacc01x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc0x084C195D2A6E3B7F, vacc1x084C195D2A6E3B7F), voutput_zero_point);
    const __m512i vacc23x084Cx195Dx2A6Ex3B7F = _mm512_adds_epi16(_mm512_packs_epi32(vacc2x084C195D2A6E3B7F, vacc3x084C195D2A6E3B7F), voutput_zero_point);

    __m512i vout0123x084Cx195Dx2A6Ex3B7F = _mm512_packus_epi16(vacc01x084Cx195Dx2A6Ex3B7F, vacc23x084Cx195Dx2A6Ex3B7F);
    vout0123x084Cx195Dx2A6Ex3B7F = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), vout0123x084Cx195Dx2A6Ex3B7F);
    __m512i vout0123x0123456789ABCDEF = _mm512_shuffle_epi8(vout0123x084Cx195Dx2A6Ex3B7F, _mm512_set_epi8(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0, 15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0));
    vout0123x0123456789ABCDEF = _mm512_max_epu8(vout0123x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c3, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 3));
      _mm_storeu_si128((__m128i*) c2, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 2));
      _mm_storeu_si128((__m128i*) c1, _mm512_extracti32x4_epi32(vout0123x0123456789ABCDEF, 1));
      _mm_storeu_si128((__m128i*) c0, _mm512_castsi512_si128(vout0123x0123456789ABCDEF));

      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << (nc + 48)) - (UINT64_C(1) << 48)));

      _mm512_mask_storeu_epi8(c3 - 48, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c2 - 32, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c1 - 16, vmask, vout0123x0123456789ABCDEF);
      vmask = _kshiftri_mask64(vmask, 16);
      _mm512_mask_storeu_epi8(c0, vmask, vout0123x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qu8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512i vbias = _mm512_load_si512(params->avx512.bias);
  const __m512i va_multiplier = _mm512_load_si512(params->avx512.a_multiplier);
  const __m512i vb_multiplier = _mm512_load_si512(params->avx512.b_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx512.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx512.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx512.output_max);

  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m512i va0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input_a));
    const __m512i vb0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input_b));
    input_a += 16;
    input_b += 16;

    __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

    vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vb0123456789ABCDEF, vb_multiplier));

    vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
      const __m512i va0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(vmask, input_a));
      const __m512i vb0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(vmask, input_b));

      __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vb0123456789ABCDEF, vb_multiplier));

      vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);
      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);
      vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

      _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
    }
  }
}

void xnn_qu8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512i va_multiplier = _mm512_load_si512(params->avx512.a_multiplier);
  const __m128i vshift = _mm_load_si128((const __m128i*) params->avx512.shift);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->avx512.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->avx512.output_min);
  const __m128i voutput_max = _mm_load_si128((const __m128i*) params->avx512.output_max);

  const __m512i vbias = _mm512_add_epi32(
    _mm512_broadcastd_epi32(_mm_cvtsi32_si128(params->avx512.b_multiplier[0] * (int32_t) *input_b)),
    _mm512_load_si512(params->avx512.bias));
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m512i va0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) input_a));
    input_a += 16;

    __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

    vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
      const __m512i va0123456789ABCDEF = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(vmask, input_a));

      __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

      vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);
      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packus_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
      vout0123456789ABCDEF = _mm_max_epu8(vout0123456789ABCDEF, voutput_min);
      vout0123456789ABCDEF = _mm_min_epu8(vout0123456789ABCDEF, voutput_max);

      _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
    }
  }
}

void xnn_x8_lut_ukernel__avx512skx_vpshufb_u64(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512i vt0 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) table));
  const __m512i vt1 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 16)));
  const __m512i vt2 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 32)));
  const __m512i vt3 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 48)));
  const __m512i vt4 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 64)));
  const __m512i vt5 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 80)));
  const __m512i vt6 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 96)));
  const __m512i vt7 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 112)));
  const __m512i vt8 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 128)));
  const __m512i vt9 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 144)));
  const __m512i vtA = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 160)));
  const __m512i vtB = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 176)));
  const __m512i vtC = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 192)));
  const __m512i vtD = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 208)));
  const __m512i vtE = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 224)));
  const __m512i vtF = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) (table + 240)));

  const __m512i vtable0 = vt0;
  const __m512i vtable1 = _mm512_xor_si512(vt0, vt1);
  const __m512i vtable2 = _mm512_xor_si512(vt1, vt2);
  const __m512i vtable3 = _mm512_xor_si512(vt2, vt3);
  const __m512i vtable4 = _mm512_xor_si512(vt3, vt4);
  const __m512i vtable5 = _mm512_xor_si512(vt4, vt5);
  const __m512i vtable6 = _mm512_xor_si512(vt5, vt6);
  const __m512i vtable7 = _mm512_xor_si512(vt6, vt7);
  const __m512i vtable8 = _mm512_xor_si512(_mm512_xor_si512(vt7, vt8), vtable0);
  const __m512i vtable9 = _mm512_xor_si512(_mm512_xor_si512(vt8, vt9), vtable1);
  const __m512i vtableA = _mm512_xor_si512(_mm512_xor_si512(vt9, vtA), vtable2);
  const __m512i vtableB = _mm512_xor_si512(_mm512_xor_si512(vtA, vtB), vtable3);
  const __m512i vtableC = _mm512_xor_si512(_mm512_xor_si512(vtB, vtC), vtable4);
  const __m512i vtableD = _mm512_xor_si512(_mm512_xor_si512(vtC, vtD), vtable5);
  const __m512i vtableE = _mm512_xor_si512(_mm512_xor_si512(vtD, vtE), vtable6);
  const __m512i vtableF = _mm512_xor_si512(_mm512_xor_si512(vtE, vtF), vtable7);

  const __m512i voffset = _mm512_set1_epi8(16);
  for (; batch >= 64 * sizeof(uint8_t); batch -= 64 * sizeof(uint8_t)) {
    __m512i vx = _mm512_loadu_si512(input);
    input += 64;

    __m512i vy = _mm512_shuffle_epi8(vtable0, vx);

    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable1, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable2, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable3, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable4, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable5, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable6, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable7, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable8, vx));

    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable9, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableA, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableB, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableC, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableD, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableE, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableF, vx));

    _mm512_storeu_si512(output, vy);
    output += 64;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch < 64);
    const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << batch) - UINT64_C(1)));

    __m512i vx = _mm512_maskz_loadu_epi8(vmask, input);

    __m512i vy = _mm512_shuffle_epi8(vtable0, vx);

    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable1, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable2, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable3, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable4, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable5, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable6, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable7, vx));
    vx = _mm512_sub_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable8, vx));

    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtable9, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableA, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableB, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableC, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableD, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableE, vx));
    vx = _mm512_subs_epi8(vx, voffset);
    vy = _mm512_xor_si512(vy, _mm512_shuffle_epi8(vtableF, vx));

    _mm512_mask_storeu_epi8(output, vmask, vy);
  }
}
