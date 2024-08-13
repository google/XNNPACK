// Auto-generated file. Do not edit!
//   Template: src/f32-rdsum/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"


void xnn_f32_rdsum_ukernel_7p7x__avx_c32(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vscale = _mm256_set1_ps(params->scalar.scale);
  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  const __m256 vmax = _mm256_set1_ps(params->scalar.max);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 32; channels -= 32) {
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);

    __m256 vacc0 = _mm256_setzero_ps();
    __m256 vacc1 = _mm256_setzero_ps();
    __m256 vacc2 = _mm256_setzero_ps();
    __m256 vacc3 = _mm256_setzero_ps();

    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      __m256 vin0;
      __m256 vin1;
      __m256 vin2;
      __m256 vin3;
      vin0 = _mm256_loadu_ps(&i0[0]);
      vin1 = _mm256_loadu_ps(&i0[8]);
      vin2 = _mm256_loadu_ps(&i0[16]);
      vin3 = _mm256_loadu_ps(&i0[24]);
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_loadu_ps(&i1[0]);
      vin1 = _mm256_loadu_ps(&i1[8]);
      vin2 = _mm256_loadu_ps(&i1[16]);
      vin3 = _mm256_loadu_ps(&i1[24]);
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_loadu_ps(&i2[0]);
      vin1 = _mm256_loadu_ps(&i2[8]);
      vin2 = _mm256_loadu_ps(&i2[16]);
      vin3 = _mm256_loadu_ps(&i2[24]);
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_loadu_ps(&i3[0]);
      vin1 = _mm256_loadu_ps(&i3[8]);
      vin2 = _mm256_loadu_ps(&i3[16]);
      vin3 = _mm256_loadu_ps(&i3[24]);
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_loadu_ps(&i4[0]);
      vin1 = _mm256_loadu_ps(&i4[8]);
      vin2 = _mm256_loadu_ps(&i4[16]);
      vin3 = _mm256_loadu_ps(&i4[24]);
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_loadu_ps(&i5[0]);
      vin1 = _mm256_loadu_ps(&i5[8]);
      vin2 = _mm256_loadu_ps(&i5[16]);
      vin3 = _mm256_loadu_ps(&i5[24]);
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_loadu_ps(&i6[0]);
      vin1 = _mm256_loadu_ps(&i6[8]);
      vin2 = _mm256_loadu_ps(&i6[16]);
      vin3 = _mm256_loadu_ps(&i6[24]);
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = _mm256_mul_ps(vacc0, vscale);
    vacc0 = _mm256_max_ps(vacc0, vmin);
    vacc0 = _mm256_min_ps(vacc0, vmax);
    vacc1 = _mm256_mul_ps(vacc1, vscale);
    vacc1 = _mm256_max_ps(vacc1, vmin);
    vacc1 = _mm256_min_ps(vacc1, vmax);
    vacc2 = _mm256_mul_ps(vacc2, vscale);
    vacc2 = _mm256_max_ps(vacc2, vmin);
    vacc2 = _mm256_min_ps(vacc2, vmax);
    vacc3 = _mm256_mul_ps(vacc3, vscale);
    vacc3 = _mm256_max_ps(vacc3, vmin);
    vacc3 = _mm256_min_ps(vacc3, vmax);

    const float* o = output;
    __m256 vo0 = _mm256_loadu_ps(o); o += 8;
    __m256 vo1 = _mm256_loadu_ps(o); o += 8;
    __m256 vo2 = _mm256_loadu_ps(o); o += 8;
    __m256 vo3 = _mm256_loadu_ps(o); o += 8;
    vacc0 = _mm256_add_ps(vo0, vacc0);
    vacc1 = _mm256_add_ps(vo1, vacc1);
    vacc2 = _mm256_add_ps(vo2, vacc2);
    vacc3 = _mm256_add_ps(vo3, vacc3);
    _mm256_storeu_ps(output, vacc0); output += 8;
    _mm256_storeu_ps(output, vacc1); output += 8;
    _mm256_storeu_ps(output, vacc2); output += 8;
    _mm256_storeu_ps(output, vacc3); output += 8;

    input = (const float*) ((uintptr_t) input + 32 * sizeof(float));
  }
  __m256i vmask;
  if (channels != 0) {
    input_increment = 7 * input_stride;
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);
    __m256 vacc[4];
    vacc[0] = _mm256_setzero_ps();
    vacc[1] = _mm256_setzero_ps();
    vacc[2] = _mm256_setzero_ps();
    vacc[3] = _mm256_setzero_ps();

    const size_t num_full_chunks = channels >> 3;
    const size_t num_chunks = round_up_po2(channels, 8) >> 3;
    const size_t remainder = channels & 0x7;
    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      for (int i = 0; i < num_full_chunks; ++i) {
        vacc[i] = _mm256_add_ps(_mm256_loadu_ps(&i0[i*8]), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_loadu_ps(&i1[i*8]), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_loadu_ps(&i2[i*8]), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_loadu_ps(&i3[i*8]), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_loadu_ps(&i4[i*8]), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_loadu_ps(&i5[i*8]), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_loadu_ps(&i6[i*8]), vacc[i]);
      }

      if (remainder) {
        vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - (channels & 0x7) * sizeof(float)));
        vacc[num_full_chunks] = _mm256_add_ps(_mm256_maskload_ps(&i0[num_full_chunks*8], vmask), vacc[num_full_chunks]);
        vacc[num_full_chunks] = _mm256_add_ps(_mm256_maskload_ps(&i1[num_full_chunks*8], vmask), vacc[num_full_chunks]);
        vacc[num_full_chunks] = _mm256_add_ps(_mm256_maskload_ps(&i2[num_full_chunks*8], vmask), vacc[num_full_chunks]);
        vacc[num_full_chunks] = _mm256_add_ps(_mm256_maskload_ps(&i3[num_full_chunks*8], vmask), vacc[num_full_chunks]);
        vacc[num_full_chunks] = _mm256_add_ps(_mm256_maskload_ps(&i4[num_full_chunks*8], vmask), vacc[num_full_chunks]);
        vacc[num_full_chunks] = _mm256_add_ps(_mm256_maskload_ps(&i5[num_full_chunks*8], vmask), vacc[num_full_chunks]);
        vacc[num_full_chunks] = _mm256_add_ps(_mm256_maskload_ps(&i6[num_full_chunks*8], vmask), vacc[num_full_chunks]);
      }
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    for (size_t i = 0; i < num_chunks; ++i) {
      vacc[i] = _mm256_mul_ps(vacc[i], vscale);
      vacc[i] = _mm256_max_ps(vacc[i], vmin);
      vacc[i] = _mm256_min_ps(vacc[i], vmax);
    }

    __m256 vo[4];
    const float* o = output;
    for (int i = 0; i < channels >> 3; ++i) {
      vo[i] = _mm256_loadu_ps(o); o += 8;
    }
    for (int i = 0; i < channels >> 3; ++i) {
      vacc[i] = _mm256_add_ps(vo[i], vacc[i]);
    }
    for (int i = 0; i < channels >> 3; ++i) {
      _mm256_storeu_ps(output, vacc[i]); output += 8;
    }
    if (remainder) {
      const size_t pos = num_full_chunks;
      __m256 vout = vacc[pos];
      const __m256 vdata = _mm256_maskload_ps(output, vmask);
      vout = _mm256_add_ps(vout, vdata);
      __m128 vout_lo = _mm256_castps256_ps128(vout);
      if (channels & 4) {
        _mm_storeu_ps(output, vout_lo);
        vout_lo = _mm256_extractf128_ps(vout, 1);
        output += 4;
      }
      if (channels & 2) {
        _mm_storel_pi((__m64*) output, vout_lo);
        vout_lo = _mm_movehl_ps(vout_lo, vout_lo);
        output += 2;
      }
      if (channels & 1) {
        _mm_store_ss(output, vout_lo);
      }
    }
  }
}
