// Auto-generated file. Do not edit!
//   Template: src/f32-prelu/avx512f.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/prelu.h>


void xnn_f32_prelu_ukernel__avx512f_2x16(
    size_t rows,
    size_t channels,
    const float*restrict input,
    size_t input_stride,
    const float*restrict weights,
    float*restrict output,
    size_t output_stride)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = i0;
    o1 = o0;
  }

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const __m512 vzero = _mm512_setzero_ps();
  do {
    const float* w = weights;
    size_t c = channels;
    for (; c >= 16 * sizeof(float); c -= 16 * sizeof(float)) {
      const __m512 vw0123456789ABCDEF = _mm512_load_ps(w);
      w += 16;

      const __m512 vi0x0123456789ABCDEF = _mm512_loadu_ps(i0);
      i0 += 16;
      const __m512 vi1x0123456789ABCDEF = _mm512_loadu_ps(i1);
      i1 += 16;

      const __mmask16 vsign0x0123456789ABCDEF = _mm512_cmp_ps_mask(vi0x0123456789ABCDEF, vzero, _CMP_LT_OQ);
      const __m512 vacc0x0123456789ABCDEF = _mm512_mask_mul_ps(vi0x0123456789ABCDEF, vsign0x0123456789ABCDEF, vi0x0123456789ABCDEF, vw0123456789ABCDEF);
      const __mmask16 vsign1x0123456789ABCDEF = _mm512_cmp_ps_mask(vi1x0123456789ABCDEF, vzero, _CMP_LT_OQ);
      const __m512 vacc1x0123456789ABCDEF = _mm512_mask_mul_ps(vi1x0123456789ABCDEF, vsign1x0123456789ABCDEF, vi1x0123456789ABCDEF, vw0123456789ABCDEF);

      _mm512_storeu_ps(o0, vacc0x0123456789ABCDEF);
      o0 += 16;
      _mm512_storeu_ps(o1, vacc1x0123456789ABCDEF);
      o1 += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1 * sizeof(float));
      assert(c <= 15 * sizeof(float));
      // Prepare mask for valid 32-bit elements (depends on c).
      const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << (c >> 2 /* log2(sizeof(float))*/)) - UINT32_C(1)));

      const __m512 vw = _mm512_maskz_loadu_ps(vmask, w);

      const __m512 vi0 = _mm512_maskz_loadu_ps(vmask, i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      const __m512 vi1 = _mm512_maskz_loadu_ps(vmask, i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      const __mmask16 vsign0 = _mm512_cmp_ps_mask(vi0, vzero, _CMP_LT_OQ);
      const __m512 vacc0 = _mm512_mask_mul_ps(vi0, vsign0, vi0, vw);
      const __mmask16 vsign1 = _mm512_cmp_ps_mask(vi1, vzero, _CMP_LT_OQ);
      const __m512 vacc1 = _mm512_mask_mul_ps(vi1, vsign1, vi1, vw);

      _mm512_mask_storeu_ps(o0, vmask, vacc0);
      o0 = (float*) ((uintptr_t) o0 + c);
      _mm512_mask_storeu_ps(o1, vmask, vacc1);
      o1 = (float*) ((uintptr_t) o1 + c);
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    if XNN_UNPREDICTABLE(rows < 4) {
      i1 = i0;
      o1 = o0;
    }
    rows = doz(rows, 2);
  } while (rows != 0);
}
