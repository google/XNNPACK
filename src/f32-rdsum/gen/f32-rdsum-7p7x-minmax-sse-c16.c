// Auto-generated file. Do not edit!
//   Template: src/f32-rdsum/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"


void xnn_f32_rdsum_ukernel_7p7x__sse_c16(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128 vmin = _mm_set1_ps(params->scalar.min);
  const __m128 vmax = _mm_set1_ps(params->scalar.max);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 16; channels -= 16) {
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);

    __m128 vacc0 = _mm_setzero_ps();
    __m128 vacc1 = _mm_setzero_ps();
    __m128 vacc2 = _mm_setzero_ps();
    __m128 vacc3 = _mm_setzero_ps();

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
      __m128 vin0;
      __m128 vin1;
      __m128 vin2;
      __m128 vin3;
      vin0 = _mm_loadu_ps(&i0[0]);
      vin1 = _mm_loadu_ps(&i0[4]);
      vin2 = _mm_loadu_ps(&i0[8]);
      vin3 = _mm_loadu_ps(&i0[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i1[0]);
      vin1 = _mm_loadu_ps(&i1[4]);
      vin2 = _mm_loadu_ps(&i1[8]);
      vin3 = _mm_loadu_ps(&i1[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i2[0]);
      vin1 = _mm_loadu_ps(&i2[4]);
      vin2 = _mm_loadu_ps(&i2[8]);
      vin3 = _mm_loadu_ps(&i2[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i3[0]);
      vin1 = _mm_loadu_ps(&i3[4]);
      vin2 = _mm_loadu_ps(&i3[8]);
      vin3 = _mm_loadu_ps(&i3[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i4[0]);
      vin1 = _mm_loadu_ps(&i4[4]);
      vin2 = _mm_loadu_ps(&i4[8]);
      vin3 = _mm_loadu_ps(&i4[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i5[0]);
      vin1 = _mm_loadu_ps(&i5[4]);
      vin2 = _mm_loadu_ps(&i5[8]);
      vin3 = _mm_loadu_ps(&i5[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vin0 = _mm_loadu_ps(&i6[0]);
      vin1 = _mm_loadu_ps(&i6[4]);
      vin2 = _mm_loadu_ps(&i6[8]);
      vin3 = _mm_loadu_ps(&i6[12]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = _mm_mul_ps(vacc0, vscale);
    vacc0 = _mm_max_ps(vacc0, vmin);
    vacc0 = _mm_min_ps(vacc0, vmax);
    vacc1 = _mm_mul_ps(vacc1, vscale);
    vacc1 = _mm_max_ps(vacc1, vmin);
    vacc1 = _mm_min_ps(vacc1, vmax);
    vacc2 = _mm_mul_ps(vacc2, vscale);
    vacc2 = _mm_max_ps(vacc2, vmin);
    vacc2 = _mm_min_ps(vacc2, vmax);
    vacc3 = _mm_mul_ps(vacc3, vscale);
    vacc3 = _mm_max_ps(vacc3, vmin);
    vacc3 = _mm_min_ps(vacc3, vmax);

    const float* o = output;
    __m128 vo0 = _mm_loadu_ps(o); o += 4;
    __m128 vo1 = _mm_loadu_ps(o); o += 4;
    __m128 vo2 = _mm_loadu_ps(o); o += 4;
    __m128 vo3 = _mm_loadu_ps(o); o += 4;
    vacc0 = _mm_add_ps(vo0, vacc0);
    vacc1 = _mm_add_ps(vo1, vacc1);
    vacc2 = _mm_add_ps(vo2, vacc2);
    vacc3 = _mm_add_ps(vo3, vacc3);
    _mm_storeu_ps(output, vacc0); output += 4;
    _mm_storeu_ps(output, vacc1); output += 4;
    _mm_storeu_ps(output, vacc2); output += 4;
    _mm_storeu_ps(output, vacc3); output += 4;

    input = (const float*) ((uintptr_t) input + 16 * sizeof(float));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);
    __m128 vacc[4];
    vacc[0] = _mm_setzero_ps();
    vacc[1] = _mm_setzero_ps();
    vacc[2] = _mm_setzero_ps();
    vacc[3] = _mm_setzero_ps();

    size_t num_chunks = round_up_po2(channels, 4) >> 2;
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
      for (int i = 0; i < num_chunks; ++i) {
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i0[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i1[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i2[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i3[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i4[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i5[i*4]), vacc[i]);
        vacc[i] = _mm_add_ps(_mm_loadu_ps(&i6[i*4]), vacc[i]);
      }
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    for (int i = 0; i < num_chunks; ++i) {
      vacc[i] = _mm_mul_ps(vacc[i], vscale);
      vacc[i] = _mm_max_ps(vacc[i], vmin);
      vacc[i] = _mm_min_ps(vacc[i], vmax);
    }

    __m128 vo[4];
    const float* o = output;
    for (int i = 0; i < channels >> 2; ++i) {
      vo[i] = _mm_loadu_ps(o); o += 4;
    }
    for (int i = 0; i < channels >> 2; ++i) {
      vacc[i] = _mm_add_ps(vo[i], vacc[i]);
    }
    for (int i = 0; i < channels >> 2; ++i) {
      _mm_storeu_ps(output, vacc[i]); output += 4;
    }
    const size_t pos = channels >> 2;
    channels &= 0x3;
    __m128 vout = vacc[pos];
    if (channels & 2) {
      __m128 vo = _mm_loadl_pi(vscale, (__m64*) output);
      _mm_storel_pi((__m64*) output, _mm_add_ps(vo, vout));
      vout = _mm_movehl_ps(vout, vout);
      output += 2;
    }
    if (channels & 1) {
      __m128 vo = _mm_load_ss(output);
      _mm_store_ss(output, _mm_add_ps(vo, vout));
    }
  }
}
