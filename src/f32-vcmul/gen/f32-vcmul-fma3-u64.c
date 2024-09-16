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

void xnn_f32_vcmul_ukernel__fma3_u64(
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
    const __m256 va0r = _mm256_loadu_ps(ar);
    const __m256 va0i = _mm256_loadu_ps(ai);
    const __m256 vb0r = _mm256_loadu_ps(br);
    const __m256 vb0i = _mm256_loadu_ps(bi);
    const __m256 va1r = _mm256_loadu_ps(ar + 8);
    const __m256 va1i = _mm256_loadu_ps(ai + 8);
    const __m256 vb1r = _mm256_loadu_ps(br + 8);
    const __m256 vb1i = _mm256_loadu_ps(bi + 8);
    const __m256 va2r = _mm256_loadu_ps(ar + 16);
    const __m256 va2i = _mm256_loadu_ps(ai + 16);
    const __m256 vb2r = _mm256_loadu_ps(br + 16);
    const __m256 vb2i = _mm256_loadu_ps(bi + 16);
    const __m256 va3r = _mm256_loadu_ps(ar + 24);
    const __m256 va3i = _mm256_loadu_ps(ai + 24);
    const __m256 vb3r = _mm256_loadu_ps(br + 24);
    const __m256 vb3i = _mm256_loadu_ps(bi + 24);
    const __m256 va4r = _mm256_loadu_ps(ar + 32);
    const __m256 va4i = _mm256_loadu_ps(ai + 32);
    const __m256 vb4r = _mm256_loadu_ps(br + 32);
    const __m256 vb4i = _mm256_loadu_ps(bi + 32);
    const __m256 va5r = _mm256_loadu_ps(ar + 40);
    const __m256 va5i = _mm256_loadu_ps(ai + 40);
    const __m256 vb5r = _mm256_loadu_ps(br + 40);
    const __m256 vb5i = _mm256_loadu_ps(bi + 40);
    const __m256 va6r = _mm256_loadu_ps(ar + 48);
    const __m256 va6i = _mm256_loadu_ps(ai + 48);
    const __m256 vb6r = _mm256_loadu_ps(br + 48);
    const __m256 vb6i = _mm256_loadu_ps(bi + 48);
    const __m256 va7r = _mm256_loadu_ps(ar + 56);
    const __m256 va7i = _mm256_loadu_ps(ai + 56);
    const __m256 vb7r = _mm256_loadu_ps(br + 56);
    const __m256 vb7i = _mm256_loadu_ps(bi + 56);
    ar += 64;
    ai += 64;
    br += 64;
    bi += 64;

    __m256 vacc0r = _mm256_mul_ps(va0r, vb0r);
    __m256 vacc0i = _mm256_mul_ps(va0r, vb0i);
    __m256 vacc1r = _mm256_mul_ps(va1r, vb1r);
    __m256 vacc1i = _mm256_mul_ps(va1r, vb1i);
    __m256 vacc2r = _mm256_mul_ps(va2r, vb2r);
    __m256 vacc2i = _mm256_mul_ps(va2r, vb2i);
    __m256 vacc3r = _mm256_mul_ps(va3r, vb3r);
    __m256 vacc3i = _mm256_mul_ps(va3r, vb3i);
    __m256 vacc4r = _mm256_mul_ps(va4r, vb4r);
    __m256 vacc4i = _mm256_mul_ps(va4r, vb4i);
    __m256 vacc5r = _mm256_mul_ps(va5r, vb5r);
    __m256 vacc5i = _mm256_mul_ps(va5r, vb5i);
    __m256 vacc6r = _mm256_mul_ps(va6r, vb6r);
    __m256 vacc6i = _mm256_mul_ps(va6r, vb6i);
    __m256 vacc7r = _mm256_mul_ps(va7r, vb7r);
    __m256 vacc7i = _mm256_mul_ps(va7r, vb7i);

    vacc0r = _mm256_fnmadd_ps(va0i, vb0i, vacc0r);
    vacc0i = _mm256_fmadd_ps(va0i, vb0r, vacc0i);
    vacc1r = _mm256_fnmadd_ps(va1i, vb1i, vacc1r);
    vacc1i = _mm256_fmadd_ps(va1i, vb1r, vacc1i);
    vacc2r = _mm256_fnmadd_ps(va2i, vb2i, vacc2r);
    vacc2i = _mm256_fmadd_ps(va2i, vb2r, vacc2i);
    vacc3r = _mm256_fnmadd_ps(va3i, vb3i, vacc3r);
    vacc3i = _mm256_fmadd_ps(va3i, vb3r, vacc3i);
    vacc4r = _mm256_fnmadd_ps(va4i, vb4i, vacc4r);
    vacc4i = _mm256_fmadd_ps(va4i, vb4r, vacc4i);
    vacc5r = _mm256_fnmadd_ps(va5i, vb5i, vacc5r);
    vacc5i = _mm256_fmadd_ps(va5i, vb5r, vacc5i);
    vacc6r = _mm256_fnmadd_ps(va6i, vb6i, vacc6r);
    vacc6i = _mm256_fmadd_ps(va6i, vb6r, vacc6i);
    vacc7r = _mm256_fnmadd_ps(va7i, vb7i, vacc7r);
    vacc7i = _mm256_fmadd_ps(va7i, vb7r, vacc7i);

    _mm256_storeu_ps(or, vacc0r);
    _mm256_storeu_ps(oi, vacc0i);
    _mm256_storeu_ps(or + 8, vacc1r);
    _mm256_storeu_ps(oi + 8, vacc1i);
    _mm256_storeu_ps(or + 16, vacc2r);
    _mm256_storeu_ps(oi + 16, vacc2i);
    _mm256_storeu_ps(or + 24, vacc3r);
    _mm256_storeu_ps(oi + 24, vacc3i);
    _mm256_storeu_ps(or + 32, vacc4r);
    _mm256_storeu_ps(oi + 32, vacc4i);
    _mm256_storeu_ps(or + 40, vacc5r);
    _mm256_storeu_ps(oi + 40, vacc5i);
    _mm256_storeu_ps(or + 48, vacc6r);
    _mm256_storeu_ps(oi + 48, vacc6i);
    _mm256_storeu_ps(or + 56, vacc7r);
    _mm256_storeu_ps(oi + 56, vacc7i);
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
