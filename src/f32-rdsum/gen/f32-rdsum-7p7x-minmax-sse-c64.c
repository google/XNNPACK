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


void xnn_f32_rdsum_ukernel_7p7x__sse_c64(
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
  for (; channels >= 64; channels -= 64) {
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
    __m128 vacc4 = _mm_setzero_ps();
    __m128 vacc5 = _mm_setzero_ps();
    __m128 vacc6 = _mm_setzero_ps();
    __m128 vacc7 = _mm_setzero_ps();
    __m128 vacc8 = _mm_setzero_ps();
    __m128 vacc9 = _mm_setzero_ps();
    __m128 vacc10 = _mm_setzero_ps();
    __m128 vacc11 = _mm_setzero_ps();
    __m128 vacc12 = _mm_setzero_ps();
    __m128 vacc13 = _mm_setzero_ps();
    __m128 vacc14 = _mm_setzero_ps();
    __m128 vacc15 = _mm_setzero_ps();

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
      __m128 vin4;
      __m128 vin5;
      __m128 vin6;
      __m128 vin7;
      __m128 vin8;
      __m128 vin9;
      __m128 vin10;
      __m128 vin11;
      __m128 vin12;
      __m128 vin13;
      __m128 vin14;
      __m128 vin15;
      vin0 = _mm_loadu_ps(&i0[0]);
      vin1 = _mm_loadu_ps(&i0[4]);
      vin2 = _mm_loadu_ps(&i0[8]);
      vin3 = _mm_loadu_ps(&i0[12]);
      vin4 = _mm_loadu_ps(&i0[16]);
      vin5 = _mm_loadu_ps(&i0[20]);
      vin6 = _mm_loadu_ps(&i0[24]);
      vin7 = _mm_loadu_ps(&i0[28]);
      vin8 = _mm_loadu_ps(&i0[32]);
      vin9 = _mm_loadu_ps(&i0[36]);
      vin10 = _mm_loadu_ps(&i0[40]);
      vin11 = _mm_loadu_ps(&i0[44]);
      vin12 = _mm_loadu_ps(&i0[48]);
      vin13 = _mm_loadu_ps(&i0[52]);
      vin14 = _mm_loadu_ps(&i0[56]);
      vin15 = _mm_loadu_ps(&i0[60]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vacc8 = _mm_add_ps(vin8, vacc8);
      vacc9 = _mm_add_ps(vin9, vacc9);
      vacc10 = _mm_add_ps(vin10, vacc10);
      vacc11 = _mm_add_ps(vin11, vacc11);
      vacc12 = _mm_add_ps(vin12, vacc12);
      vacc13 = _mm_add_ps(vin13, vacc13);
      vacc14 = _mm_add_ps(vin14, vacc14);
      vacc15 = _mm_add_ps(vin15, vacc15);
      vin0 = _mm_loadu_ps(&i1[0]);
      vin1 = _mm_loadu_ps(&i1[4]);
      vin2 = _mm_loadu_ps(&i1[8]);
      vin3 = _mm_loadu_ps(&i1[12]);
      vin4 = _mm_loadu_ps(&i1[16]);
      vin5 = _mm_loadu_ps(&i1[20]);
      vin6 = _mm_loadu_ps(&i1[24]);
      vin7 = _mm_loadu_ps(&i1[28]);
      vin8 = _mm_loadu_ps(&i1[32]);
      vin9 = _mm_loadu_ps(&i1[36]);
      vin10 = _mm_loadu_ps(&i1[40]);
      vin11 = _mm_loadu_ps(&i1[44]);
      vin12 = _mm_loadu_ps(&i1[48]);
      vin13 = _mm_loadu_ps(&i1[52]);
      vin14 = _mm_loadu_ps(&i1[56]);
      vin15 = _mm_loadu_ps(&i1[60]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vacc8 = _mm_add_ps(vin8, vacc8);
      vacc9 = _mm_add_ps(vin9, vacc9);
      vacc10 = _mm_add_ps(vin10, vacc10);
      vacc11 = _mm_add_ps(vin11, vacc11);
      vacc12 = _mm_add_ps(vin12, vacc12);
      vacc13 = _mm_add_ps(vin13, vacc13);
      vacc14 = _mm_add_ps(vin14, vacc14);
      vacc15 = _mm_add_ps(vin15, vacc15);
      vin0 = _mm_loadu_ps(&i2[0]);
      vin1 = _mm_loadu_ps(&i2[4]);
      vin2 = _mm_loadu_ps(&i2[8]);
      vin3 = _mm_loadu_ps(&i2[12]);
      vin4 = _mm_loadu_ps(&i2[16]);
      vin5 = _mm_loadu_ps(&i2[20]);
      vin6 = _mm_loadu_ps(&i2[24]);
      vin7 = _mm_loadu_ps(&i2[28]);
      vin8 = _mm_loadu_ps(&i2[32]);
      vin9 = _mm_loadu_ps(&i2[36]);
      vin10 = _mm_loadu_ps(&i2[40]);
      vin11 = _mm_loadu_ps(&i2[44]);
      vin12 = _mm_loadu_ps(&i2[48]);
      vin13 = _mm_loadu_ps(&i2[52]);
      vin14 = _mm_loadu_ps(&i2[56]);
      vin15 = _mm_loadu_ps(&i2[60]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vacc8 = _mm_add_ps(vin8, vacc8);
      vacc9 = _mm_add_ps(vin9, vacc9);
      vacc10 = _mm_add_ps(vin10, vacc10);
      vacc11 = _mm_add_ps(vin11, vacc11);
      vacc12 = _mm_add_ps(vin12, vacc12);
      vacc13 = _mm_add_ps(vin13, vacc13);
      vacc14 = _mm_add_ps(vin14, vacc14);
      vacc15 = _mm_add_ps(vin15, vacc15);
      vin0 = _mm_loadu_ps(&i3[0]);
      vin1 = _mm_loadu_ps(&i3[4]);
      vin2 = _mm_loadu_ps(&i3[8]);
      vin3 = _mm_loadu_ps(&i3[12]);
      vin4 = _mm_loadu_ps(&i3[16]);
      vin5 = _mm_loadu_ps(&i3[20]);
      vin6 = _mm_loadu_ps(&i3[24]);
      vin7 = _mm_loadu_ps(&i3[28]);
      vin8 = _mm_loadu_ps(&i3[32]);
      vin9 = _mm_loadu_ps(&i3[36]);
      vin10 = _mm_loadu_ps(&i3[40]);
      vin11 = _mm_loadu_ps(&i3[44]);
      vin12 = _mm_loadu_ps(&i3[48]);
      vin13 = _mm_loadu_ps(&i3[52]);
      vin14 = _mm_loadu_ps(&i3[56]);
      vin15 = _mm_loadu_ps(&i3[60]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vacc8 = _mm_add_ps(vin8, vacc8);
      vacc9 = _mm_add_ps(vin9, vacc9);
      vacc10 = _mm_add_ps(vin10, vacc10);
      vacc11 = _mm_add_ps(vin11, vacc11);
      vacc12 = _mm_add_ps(vin12, vacc12);
      vacc13 = _mm_add_ps(vin13, vacc13);
      vacc14 = _mm_add_ps(vin14, vacc14);
      vacc15 = _mm_add_ps(vin15, vacc15);
      vin0 = _mm_loadu_ps(&i4[0]);
      vin1 = _mm_loadu_ps(&i4[4]);
      vin2 = _mm_loadu_ps(&i4[8]);
      vin3 = _mm_loadu_ps(&i4[12]);
      vin4 = _mm_loadu_ps(&i4[16]);
      vin5 = _mm_loadu_ps(&i4[20]);
      vin6 = _mm_loadu_ps(&i4[24]);
      vin7 = _mm_loadu_ps(&i4[28]);
      vin8 = _mm_loadu_ps(&i4[32]);
      vin9 = _mm_loadu_ps(&i4[36]);
      vin10 = _mm_loadu_ps(&i4[40]);
      vin11 = _mm_loadu_ps(&i4[44]);
      vin12 = _mm_loadu_ps(&i4[48]);
      vin13 = _mm_loadu_ps(&i4[52]);
      vin14 = _mm_loadu_ps(&i4[56]);
      vin15 = _mm_loadu_ps(&i4[60]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vacc8 = _mm_add_ps(vin8, vacc8);
      vacc9 = _mm_add_ps(vin9, vacc9);
      vacc10 = _mm_add_ps(vin10, vacc10);
      vacc11 = _mm_add_ps(vin11, vacc11);
      vacc12 = _mm_add_ps(vin12, vacc12);
      vacc13 = _mm_add_ps(vin13, vacc13);
      vacc14 = _mm_add_ps(vin14, vacc14);
      vacc15 = _mm_add_ps(vin15, vacc15);
      vin0 = _mm_loadu_ps(&i5[0]);
      vin1 = _mm_loadu_ps(&i5[4]);
      vin2 = _mm_loadu_ps(&i5[8]);
      vin3 = _mm_loadu_ps(&i5[12]);
      vin4 = _mm_loadu_ps(&i5[16]);
      vin5 = _mm_loadu_ps(&i5[20]);
      vin6 = _mm_loadu_ps(&i5[24]);
      vin7 = _mm_loadu_ps(&i5[28]);
      vin8 = _mm_loadu_ps(&i5[32]);
      vin9 = _mm_loadu_ps(&i5[36]);
      vin10 = _mm_loadu_ps(&i5[40]);
      vin11 = _mm_loadu_ps(&i5[44]);
      vin12 = _mm_loadu_ps(&i5[48]);
      vin13 = _mm_loadu_ps(&i5[52]);
      vin14 = _mm_loadu_ps(&i5[56]);
      vin15 = _mm_loadu_ps(&i5[60]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vacc8 = _mm_add_ps(vin8, vacc8);
      vacc9 = _mm_add_ps(vin9, vacc9);
      vacc10 = _mm_add_ps(vin10, vacc10);
      vacc11 = _mm_add_ps(vin11, vacc11);
      vacc12 = _mm_add_ps(vin12, vacc12);
      vacc13 = _mm_add_ps(vin13, vacc13);
      vacc14 = _mm_add_ps(vin14, vacc14);
      vacc15 = _mm_add_ps(vin15, vacc15);
      vin0 = _mm_loadu_ps(&i6[0]);
      vin1 = _mm_loadu_ps(&i6[4]);
      vin2 = _mm_loadu_ps(&i6[8]);
      vin3 = _mm_loadu_ps(&i6[12]);
      vin4 = _mm_loadu_ps(&i6[16]);
      vin5 = _mm_loadu_ps(&i6[20]);
      vin6 = _mm_loadu_ps(&i6[24]);
      vin7 = _mm_loadu_ps(&i6[28]);
      vin8 = _mm_loadu_ps(&i6[32]);
      vin9 = _mm_loadu_ps(&i6[36]);
      vin10 = _mm_loadu_ps(&i6[40]);
      vin11 = _mm_loadu_ps(&i6[44]);
      vin12 = _mm_loadu_ps(&i6[48]);
      vin13 = _mm_loadu_ps(&i6[52]);
      vin14 = _mm_loadu_ps(&i6[56]);
      vin15 = _mm_loadu_ps(&i6[60]);
      vacc0 = _mm_add_ps(vin0, vacc0);
      vacc1 = _mm_add_ps(vin1, vacc1);
      vacc2 = _mm_add_ps(vin2, vacc2);
      vacc3 = _mm_add_ps(vin3, vacc3);
      vacc4 = _mm_add_ps(vin4, vacc4);
      vacc5 = _mm_add_ps(vin5, vacc5);
      vacc6 = _mm_add_ps(vin6, vacc6);
      vacc7 = _mm_add_ps(vin7, vacc7);
      vacc8 = _mm_add_ps(vin8, vacc8);
      vacc9 = _mm_add_ps(vin9, vacc9);
      vacc10 = _mm_add_ps(vin10, vacc10);
      vacc11 = _mm_add_ps(vin11, vacc11);
      vacc12 = _mm_add_ps(vin12, vacc12);
      vacc13 = _mm_add_ps(vin13, vacc13);
      vacc14 = _mm_add_ps(vin14, vacc14);
      vacc15 = _mm_add_ps(vin15, vacc15);
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
    vacc4 = _mm_mul_ps(vacc4, vscale);
    vacc4 = _mm_max_ps(vacc4, vmin);
    vacc4 = _mm_min_ps(vacc4, vmax);
    vacc5 = _mm_mul_ps(vacc5, vscale);
    vacc5 = _mm_max_ps(vacc5, vmin);
    vacc5 = _mm_min_ps(vacc5, vmax);
    vacc6 = _mm_mul_ps(vacc6, vscale);
    vacc6 = _mm_max_ps(vacc6, vmin);
    vacc6 = _mm_min_ps(vacc6, vmax);
    vacc7 = _mm_mul_ps(vacc7, vscale);
    vacc7 = _mm_max_ps(vacc7, vmin);
    vacc7 = _mm_min_ps(vacc7, vmax);
    vacc8 = _mm_mul_ps(vacc8, vscale);
    vacc8 = _mm_max_ps(vacc8, vmin);
    vacc8 = _mm_min_ps(vacc8, vmax);
    vacc9 = _mm_mul_ps(vacc9, vscale);
    vacc9 = _mm_max_ps(vacc9, vmin);
    vacc9 = _mm_min_ps(vacc9, vmax);
    vacc10 = _mm_mul_ps(vacc10, vscale);
    vacc10 = _mm_max_ps(vacc10, vmin);
    vacc10 = _mm_min_ps(vacc10, vmax);
    vacc11 = _mm_mul_ps(vacc11, vscale);
    vacc11 = _mm_max_ps(vacc11, vmin);
    vacc11 = _mm_min_ps(vacc11, vmax);
    vacc12 = _mm_mul_ps(vacc12, vscale);
    vacc12 = _mm_max_ps(vacc12, vmin);
    vacc12 = _mm_min_ps(vacc12, vmax);
    vacc13 = _mm_mul_ps(vacc13, vscale);
    vacc13 = _mm_max_ps(vacc13, vmin);
    vacc13 = _mm_min_ps(vacc13, vmax);
    vacc14 = _mm_mul_ps(vacc14, vscale);
    vacc14 = _mm_max_ps(vacc14, vmin);
    vacc14 = _mm_min_ps(vacc14, vmax);
    vacc15 = _mm_mul_ps(vacc15, vscale);
    vacc15 = _mm_max_ps(vacc15, vmin);
    vacc15 = _mm_min_ps(vacc15, vmax);

    const float* o = output;
    __m128 vo0 = _mm_loadu_ps(o); o += 4;
    __m128 vo1 = _mm_loadu_ps(o); o += 4;
    __m128 vo2 = _mm_loadu_ps(o); o += 4;
    __m128 vo3 = _mm_loadu_ps(o); o += 4;
    __m128 vo4 = _mm_loadu_ps(o); o += 4;
    __m128 vo5 = _mm_loadu_ps(o); o += 4;
    __m128 vo6 = _mm_loadu_ps(o); o += 4;
    __m128 vo7 = _mm_loadu_ps(o); o += 4;
    __m128 vo8 = _mm_loadu_ps(o); o += 4;
    __m128 vo9 = _mm_loadu_ps(o); o += 4;
    __m128 vo10 = _mm_loadu_ps(o); o += 4;
    __m128 vo11 = _mm_loadu_ps(o); o += 4;
    __m128 vo12 = _mm_loadu_ps(o); o += 4;
    __m128 vo13 = _mm_loadu_ps(o); o += 4;
    __m128 vo14 = _mm_loadu_ps(o); o += 4;
    __m128 vo15 = _mm_loadu_ps(o); o += 4;
    vacc0 = _mm_add_ps(vo0, vacc0);
    vacc1 = _mm_add_ps(vo1, vacc1);
    vacc2 = _mm_add_ps(vo2, vacc2);
    vacc3 = _mm_add_ps(vo3, vacc3);
    vacc4 = _mm_add_ps(vo4, vacc4);
    vacc5 = _mm_add_ps(vo5, vacc5);
    vacc6 = _mm_add_ps(vo6, vacc6);
    vacc7 = _mm_add_ps(vo7, vacc7);
    vacc8 = _mm_add_ps(vo8, vacc8);
    vacc9 = _mm_add_ps(vo9, vacc9);
    vacc10 = _mm_add_ps(vo10, vacc10);
    vacc11 = _mm_add_ps(vo11, vacc11);
    vacc12 = _mm_add_ps(vo12, vacc12);
    vacc13 = _mm_add_ps(vo13, vacc13);
    vacc14 = _mm_add_ps(vo14, vacc14);
    vacc15 = _mm_add_ps(vo15, vacc15);
    _mm_storeu_ps(output, vacc0); output += 4;
    _mm_storeu_ps(output, vacc1); output += 4;
    _mm_storeu_ps(output, vacc2); output += 4;
    _mm_storeu_ps(output, vacc3); output += 4;
    _mm_storeu_ps(output, vacc4); output += 4;
    _mm_storeu_ps(output, vacc5); output += 4;
    _mm_storeu_ps(output, vacc6); output += 4;
    _mm_storeu_ps(output, vacc7); output += 4;
    _mm_storeu_ps(output, vacc8); output += 4;
    _mm_storeu_ps(output, vacc9); output += 4;
    _mm_storeu_ps(output, vacc10); output += 4;
    _mm_storeu_ps(output, vacc11); output += 4;
    _mm_storeu_ps(output, vacc12); output += 4;
    _mm_storeu_ps(output, vacc13); output += 4;
    _mm_storeu_ps(output, vacc14); output += 4;
    _mm_storeu_ps(output, vacc15); output += 4;

    input = (const float*) ((uintptr_t) input + 64 * sizeof(float));
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
    __m128 vacc[16];
    vacc[0] = _mm_setzero_ps();
    vacc[1] = _mm_setzero_ps();
    vacc[2] = _mm_setzero_ps();
    vacc[3] = _mm_setzero_ps();
    vacc[4] = _mm_setzero_ps();
    vacc[5] = _mm_setzero_ps();
    vacc[6] = _mm_setzero_ps();
    vacc[7] = _mm_setzero_ps();
    vacc[8] = _mm_setzero_ps();
    vacc[9] = _mm_setzero_ps();
    vacc[10] = _mm_setzero_ps();
    vacc[11] = _mm_setzero_ps();
    vacc[12] = _mm_setzero_ps();
    vacc[13] = _mm_setzero_ps();
    vacc[14] = _mm_setzero_ps();
    vacc[15] = _mm_setzero_ps();

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

    __m128 vo[16];
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
