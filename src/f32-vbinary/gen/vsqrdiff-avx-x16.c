// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-avx.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


void xnn_f32_vsqrdiff_ukernel__avx_x16(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m256 va01234567 = _mm256_loadu_ps(input_a);
    const __m256 va89ABCDEF = _mm256_loadu_ps(input_a + 8);
    input_a += 16;

    const __m256 vb01234567 = _mm256_loadu_ps(input_b);
    const __m256 vb89ABCDEF = _mm256_loadu_ps(input_b + 8);
    input_b += 16;

    __m256 vy01234567 = _mm256_sub_ps(va01234567, vb01234567);
    __m256 vy89ABCDEF = _mm256_sub_ps(va89ABCDEF, vb89ABCDEF);

    vy01234567 = _mm256_mul_ps(vy01234567, vy01234567);
    vy89ABCDEF = _mm256_mul_ps(vy89ABCDEF, vy89ABCDEF);


    _mm256_storeu_ps(output, vy01234567);
    _mm256_storeu_ps(output + 8, vy89ABCDEF);
    output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 va = _mm256_loadu_ps(input_a);
    input_a += 8;

    const __m256 vb = _mm256_loadu_ps(input_b);
    input_b += 8;

    __m256 vy = _mm256_sub_ps(va, vb);
    vy = _mm256_mul_ps(vy, vy);
    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->avx.mask_table[7] - batch));

    const __m256 va = _mm256_maskload_ps(input_a, vmask);
    const __m256 vb = _mm256_maskload_ps(input_b, vmask);

    __m256 vy = _mm256_sub_ps(va, vb);
    vy = _mm256_mul_ps(vy, vy);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}
