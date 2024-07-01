// Auto-generated file. Do not edit!
//   Template: src/f32-vcmul/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vcmul_ukernel__sse_u12(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float* ar = input_a;
  const float* ai = (const float*) ((uintptr_t) input_a + batch);
  const float* br = input_b;
  const float* bi = (const float*) ((uintptr_t) input_b + batch);
  float* or = output;
  float* oi = (float*) ((uintptr_t) output + batch);
  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    const __m128 va0r = _mm_loadu_ps(ar);
    const __m128 va0i = _mm_loadu_ps(ai);
    const __m128 vb0r = _mm_loadu_ps(br);
    const __m128 vb0i = _mm_loadu_ps(bi);
    const __m128 va1r = _mm_loadu_ps(ar + 4);
    const __m128 va1i = _mm_loadu_ps(ai + 4);
    const __m128 vb1r = _mm_loadu_ps(br + 4);
    const __m128 vb1i = _mm_loadu_ps(bi + 4);
    const __m128 va2r = _mm_loadu_ps(ar + 8);
    const __m128 va2i = _mm_loadu_ps(ai + 8);
    const __m128 vb2r = _mm_loadu_ps(br + 8);
    const __m128 vb2i = _mm_loadu_ps(bi + 8);
    ar += 12;
    ai += 12;
    br += 12;
    bi += 12;

    __m128 vacc0r = _mm_mul_ps(va0r, vb0r);
    __m128 vacc0i = _mm_mul_ps(va0r, vb0i);
    __m128 vacc1r = _mm_mul_ps(va1r, vb1r);
    __m128 vacc1i = _mm_mul_ps(va1r, vb1i);
    __m128 vacc2r = _mm_mul_ps(va2r, vb2r);
    __m128 vacc2i = _mm_mul_ps(va2r, vb2i);

    vacc0r = _mm_sub_ps(vacc0r, _mm_mul_ps(va0i, vb0i));
    vacc0i = _mm_add_ps(vacc0i, _mm_mul_ps(va0i, vb0r));
    vacc1r = _mm_sub_ps(vacc1r, _mm_mul_ps(va1i, vb1i));
    vacc1i = _mm_add_ps(vacc1i, _mm_mul_ps(va1i, vb1r));
    vacc2r = _mm_sub_ps(vacc2r, _mm_mul_ps(va2i, vb2i));
    vacc2i = _mm_add_ps(vacc2i, _mm_mul_ps(va2i, vb2r));

    _mm_storeu_ps(or, vacc0r);
    _mm_storeu_ps(oi, vacc0i);
    _mm_storeu_ps(or + 4, vacc1r);
    _mm_storeu_ps(oi + 4, vacc1i);
    _mm_storeu_ps(or + 8, vacc2r);
    _mm_storeu_ps(oi + 8, vacc2i);
    or += 12;
    oi += 12;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const __m128 var = _mm_loadu_ps(ar);
    ar += 4;
    const __m128 vai = _mm_loadu_ps(ai);
    ai += 4;
    const __m128 vbr = _mm_loadu_ps(br);
    br += 4;
    const __m128 vbi = _mm_loadu_ps(bi);
    bi += 4;

    __m128 vaccr = _mm_mul_ps(var, vbr);
    __m128 vacci = _mm_mul_ps(var, vbi);

    vaccr = _mm_sub_ps(vaccr, _mm_mul_ps(vai, vbi));
    vacci = _mm_add_ps(vacci, _mm_mul_ps(vai, vbr));

    _mm_storeu_ps(or, vaccr);
    or += 4;
    _mm_storeu_ps(oi, vacci);
    oi += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m128 var = _mm_loadu_ps(ar);
    ar += 4;
    const __m128 vai = _mm_loadu_ps(ai);
    ai += 4;
    const __m128 vbr = _mm_loadu_ps(br);
    br += 4;
    const __m128 vbi = _mm_loadu_ps(bi);
    bi += 4;

    __m128 vaccr = _mm_mul_ps(var, vbr);
    __m128 vacci = _mm_mul_ps(var, vbi);

    vaccr = _mm_sub_ps(vaccr, _mm_mul_ps(vai, vbi));
    vacci = _mm_add_ps(vacci, _mm_mul_ps(vai, vbr));

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) or, vaccr);
      or += 2;
      _mm_storel_pi((__m64*) oi, vacci);
      oi += 2;
      vaccr = _mm_movehl_ps(vaccr, vaccr);
      vacci = _mm_movehl_ps(vacci, vacci);
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(or, vaccr);
      _mm_store_ss(oi, vacci);
    }
  }
}
