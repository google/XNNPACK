// Auto-generated file. Do not edit!
//   Template: src/f32-prelu/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/math.h>
#include <xnnpack/prelu.h>


static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

void xnn_f32_prelu_ukernel__avx_2x16(
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

  do {
    const float* w = weights;
    size_t c = channels;
    for (; c >= 16 * sizeof(float); c -= 16 * sizeof(float)) {
      const __m256 vw01234567 = _mm256_load_ps(w);
      const __m256 vw89ABCDEF = _mm256_load_ps(w + 8);
      w += 16;

      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
      i0 += 16;
      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vprod0x01234567 = _mm256_mul_ps(vi0x01234567, vw01234567);
      const __m256 vprod0x89ABCDEF = _mm256_mul_ps(vi0x89ABCDEF, vw89ABCDEF);
      const __m256 vprod1x01234567 = _mm256_mul_ps(vi1x01234567, vw01234567);
      const __m256 vprod1x89ABCDEF = _mm256_mul_ps(vi1x89ABCDEF, vw89ABCDEF);

      const __m256 vacc0x01234567 = _mm256_blendv_ps(vi0x01234567, vprod0x01234567, vi0x01234567);
      const __m256 vacc0x89ABCDEF = _mm256_blendv_ps(vi0x89ABCDEF, vprod0x89ABCDEF, vi0x89ABCDEF);
      const __m256 vacc1x01234567 = _mm256_blendv_ps(vi1x01234567, vprod1x01234567, vi1x01234567);
      const __m256 vacc1x89ABCDEF = _mm256_blendv_ps(vi1x89ABCDEF, vprod1x89ABCDEF, vi1x89ABCDEF);

      _mm256_storeu_ps(o0, vacc0x01234567);
      _mm256_storeu_ps(o0 + 8, vacc0x89ABCDEF);
      o0 += 16;
      _mm256_storeu_ps(o1, vacc1x01234567);
      _mm256_storeu_ps(o1 + 8, vacc1x89ABCDEF);
      o1 += 16;
    }
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const __m256 vw = _mm256_load_ps(w);
      w += 8;

      const __m256 vi0 = _mm256_loadu_ps(i0);
      i0 += 8;
      const __m256 vi1 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vprod0 = _mm256_mul_ps(vi0, vw);
      const __m256 vprod1 = _mm256_mul_ps(vi1, vw);

      const __m256 vacc0 = _mm256_blendv_ps(vi0, vprod0, vi0);
      const __m256 vacc1 = _mm256_blendv_ps(vi1, vprod1, vi1);

      _mm256_storeu_ps(o0, vacc0);
      o0 += 8;
      _mm256_storeu_ps(o1, vacc1);
      o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1 * sizeof(float));
      assert(c <= 7 * sizeof(float));
      __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - c));

      const __m256 vw = _mm256_maskload_ps(w, vmask);

      const __m256 vi0 = _mm256_maskload_ps(i0, vmask);
      i0 = (const float*) ((uintptr_t) i0 + c);
      const __m256 vi1 = _mm256_maskload_ps(i1, vmask);
      i1 = (const float*) ((uintptr_t) i1 + c);

      const __m256 vprod0 = _mm256_mul_ps(vi0, vw);
      const __m256 vprod1 = _mm256_mul_ps(vi1, vw);

      __m256 vacc0 = _mm256_blendv_ps(vi0, vprod0, vi0);
      __m256 vacc1 = _mm256_blendv_ps(vi1, vprod1, vi1);

      // _mm256_maskstore_ps(o1, vmask, vacc1) could be used here, but triggers msan failures (probably an msan bug).
      __m128 vacc0_lo = _mm256_castps256_ps128(vacc0);
      __m128 vacc1_lo = _mm256_castps256_ps128(vacc1);
      if (c & (4 * sizeof(float))) {
        _mm_storeu_ps(o0, vacc0_lo);
        _mm_storeu_ps(o1, vacc1_lo);

        vacc0_lo = _mm256_extractf128_ps(vacc0, 1);
        vacc1_lo = _mm256_extractf128_ps(vacc1, 1);

        o0 += 4;
        o1 += 4;
      }
      if (c & (2 * sizeof(float))) {
        _mm_storel_pi((__m64*) o0, vacc0_lo);
        _mm_storel_pi((__m64*) o1, vacc1_lo);

        vacc0_lo = _mm_movehl_ps(vacc0_lo, vacc0_lo);
        vacc1_lo = _mm_movehl_ps(vacc1_lo, vacc1_lo);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        _mm_store_ss(o0, vacc0_lo);
        _mm_store_ss(o1, vacc1_lo);

        o0 += 1;
        o1 += 1;
      }
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
