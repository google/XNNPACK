// Auto-generated file. Do not edit!
//   Template: src/f32-vcmul/avx512f.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"

void xnn_f32_vcmul_ukernel__avx512f_u64(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const __m512 va0r = _mm512_loadu_ps(ar);
    const __m512 va0i = _mm512_loadu_ps(ai);
    const __m512 vb0r = _mm512_loadu_ps(br);
    const __m512 vb0i = _mm512_loadu_ps(bi);
    const __m512 va1r = _mm512_loadu_ps(ar + 16);
    const __m512 va1i = _mm512_loadu_ps(ai + 16);
    const __m512 vb1r = _mm512_loadu_ps(br + 16);
    const __m512 vb1i = _mm512_loadu_ps(bi + 16);
    const __m512 va2r = _mm512_loadu_ps(ar + 32);
    const __m512 va2i = _mm512_loadu_ps(ai + 32);
    const __m512 vb2r = _mm512_loadu_ps(br + 32);
    const __m512 vb2i = _mm512_loadu_ps(bi + 32);
    const __m512 va3r = _mm512_loadu_ps(ar + 48);
    const __m512 va3i = _mm512_loadu_ps(ai + 48);
    const __m512 vb3r = _mm512_loadu_ps(br + 48);
    const __m512 vb3i = _mm512_loadu_ps(bi + 48);
    ar += 64;
    ai += 64;
    br += 64;
    bi += 64;

    __m512 vacc0r = _mm512_mul_ps(va0r, vb0r);
    __m512 vacc0i = _mm512_mul_ps(va0r, vb0i);
    __m512 vacc1r = _mm512_mul_ps(va1r, vb1r);
    __m512 vacc1i = _mm512_mul_ps(va1r, vb1i);
    __m512 vacc2r = _mm512_mul_ps(va2r, vb2r);
    __m512 vacc2i = _mm512_mul_ps(va2r, vb2i);
    __m512 vacc3r = _mm512_mul_ps(va3r, vb3r);
    __m512 vacc3i = _mm512_mul_ps(va3r, vb3i);

    vacc0r = _mm512_fnmadd_ps(va0i, vb0i, vacc0r);
    vacc0i = _mm512_fmadd_ps(va0i, vb0r, vacc0i);
    vacc1r = _mm512_fnmadd_ps(va1i, vb1i, vacc1r);
    vacc1i = _mm512_fmadd_ps(va1i, vb1r, vacc1i);
    vacc2r = _mm512_fnmadd_ps(va2i, vb2i, vacc2r);
    vacc2i = _mm512_fmadd_ps(va2i, vb2r, vacc2i);
    vacc3r = _mm512_fnmadd_ps(va3i, vb3i, vacc3r);
    vacc3i = _mm512_fmadd_ps(va3i, vb3r, vacc3i);

    _mm512_storeu_ps(or, vacc0r);
    _mm512_storeu_ps(oi, vacc0i);
    _mm512_storeu_ps(or + 16, vacc1r);
    _mm512_storeu_ps(oi + 16, vacc1i);
    _mm512_storeu_ps(or + 32, vacc2r);
    _mm512_storeu_ps(oi + 32, vacc2i);
    _mm512_storeu_ps(or + 48, vacc3r);
    _mm512_storeu_ps(oi + 48, vacc3i);
    or += 64;
    oi += 64;
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
