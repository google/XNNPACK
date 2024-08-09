// Auto-generated file. Do not edit!
//   Template: src/f32-rdsum/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"


void xnn_f32_rdsum_ukernel_7p7x__neon_c64(
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

  const float32x4_t vscale = vdupq_n_f32(params->scalar.scale);
  const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
  const float32x4_t vmax = vdupq_n_f32(params->scalar.max);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 64; channels -= 64) {
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);

    float32x4_t vacc0 = vdupq_n_f32(0.f);
    float32x4_t vacc1 = vdupq_n_f32(0.f);
    float32x4_t vacc2 = vdupq_n_f32(0.f);
    float32x4_t vacc3 = vdupq_n_f32(0.f);
    float32x4_t vacc4 = vdupq_n_f32(0.f);
    float32x4_t vacc5 = vdupq_n_f32(0.f);
    float32x4_t vacc6 = vdupq_n_f32(0.f);
    float32x4_t vacc7 = vdupq_n_f32(0.f);
    float32x4_t vacc8 = vdupq_n_f32(0.f);
    float32x4_t vacc9 = vdupq_n_f32(0.f);
    float32x4_t vacc10 = vdupq_n_f32(0.f);
    float32x4_t vacc11 = vdupq_n_f32(0.f);
    float32x4_t vacc12 = vdupq_n_f32(0.f);
    float32x4_t vacc13 = vdupq_n_f32(0.f);
    float32x4_t vacc14 = vdupq_n_f32(0.f);
    float32x4_t vacc15 = vdupq_n_f32(0.f);

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
      float32x4_t vin0;
      float32x4_t vin1;
      float32x4_t vin2;
      float32x4_t vin3;
      float32x4_t vin4;
      float32x4_t vin5;
      float32x4_t vin6;
      float32x4_t vin7;
      float32x4_t vin8;
      float32x4_t vin9;
      float32x4_t vin10;
      float32x4_t vin11;
      float32x4_t vin12;
      float32x4_t vin13;
      float32x4_t vin14;
      float32x4_t vin15;
      vin0 = vld1q_f32(&i0[0]);
      vin1 = vld1q_f32(&i0[4]);
      vin2 = vld1q_f32(&i0[8]);
      vin3 = vld1q_f32(&i0[12]);
      vin4 = vld1q_f32(&i0[16]);
      vin5 = vld1q_f32(&i0[20]);
      vin6 = vld1q_f32(&i0[24]);
      vin7 = vld1q_f32(&i0[28]);
      vin8 = vld1q_f32(&i0[32]);
      vin9 = vld1q_f32(&i0[36]);
      vin10 = vld1q_f32(&i0[40]);
      vin11 = vld1q_f32(&i0[44]);
      vin12 = vld1q_f32(&i0[48]);
      vin13 = vld1q_f32(&i0[52]);
      vin14 = vld1q_f32(&i0[56]);
      vin15 = vld1q_f32(&i0[60]);
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vacc4 = vaddq_f32(vin4, vacc4);
      vacc5 = vaddq_f32(vin5, vacc5);
      vacc6 = vaddq_f32(vin6, vacc6);
      vacc7 = vaddq_f32(vin7, vacc7);
      vacc8 = vaddq_f32(vin8, vacc8);
      vacc9 = vaddq_f32(vin9, vacc9);
      vacc10 = vaddq_f32(vin10, vacc10);
      vacc11 = vaddq_f32(vin11, vacc11);
      vacc12 = vaddq_f32(vin12, vacc12);
      vacc13 = vaddq_f32(vin13, vacc13);
      vacc14 = vaddq_f32(vin14, vacc14);
      vacc15 = vaddq_f32(vin15, vacc15);
      vin0 = vld1q_f32(&i1[0]);
      vin1 = vld1q_f32(&i1[4]);
      vin2 = vld1q_f32(&i1[8]);
      vin3 = vld1q_f32(&i1[12]);
      vin4 = vld1q_f32(&i1[16]);
      vin5 = vld1q_f32(&i1[20]);
      vin6 = vld1q_f32(&i1[24]);
      vin7 = vld1q_f32(&i1[28]);
      vin8 = vld1q_f32(&i1[32]);
      vin9 = vld1q_f32(&i1[36]);
      vin10 = vld1q_f32(&i1[40]);
      vin11 = vld1q_f32(&i1[44]);
      vin12 = vld1q_f32(&i1[48]);
      vin13 = vld1q_f32(&i1[52]);
      vin14 = vld1q_f32(&i1[56]);
      vin15 = vld1q_f32(&i1[60]);
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vacc4 = vaddq_f32(vin4, vacc4);
      vacc5 = vaddq_f32(vin5, vacc5);
      vacc6 = vaddq_f32(vin6, vacc6);
      vacc7 = vaddq_f32(vin7, vacc7);
      vacc8 = vaddq_f32(vin8, vacc8);
      vacc9 = vaddq_f32(vin9, vacc9);
      vacc10 = vaddq_f32(vin10, vacc10);
      vacc11 = vaddq_f32(vin11, vacc11);
      vacc12 = vaddq_f32(vin12, vacc12);
      vacc13 = vaddq_f32(vin13, vacc13);
      vacc14 = vaddq_f32(vin14, vacc14);
      vacc15 = vaddq_f32(vin15, vacc15);
      vin0 = vld1q_f32(&i2[0]);
      vin1 = vld1q_f32(&i2[4]);
      vin2 = vld1q_f32(&i2[8]);
      vin3 = vld1q_f32(&i2[12]);
      vin4 = vld1q_f32(&i2[16]);
      vin5 = vld1q_f32(&i2[20]);
      vin6 = vld1q_f32(&i2[24]);
      vin7 = vld1q_f32(&i2[28]);
      vin8 = vld1q_f32(&i2[32]);
      vin9 = vld1q_f32(&i2[36]);
      vin10 = vld1q_f32(&i2[40]);
      vin11 = vld1q_f32(&i2[44]);
      vin12 = vld1q_f32(&i2[48]);
      vin13 = vld1q_f32(&i2[52]);
      vin14 = vld1q_f32(&i2[56]);
      vin15 = vld1q_f32(&i2[60]);
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vacc4 = vaddq_f32(vin4, vacc4);
      vacc5 = vaddq_f32(vin5, vacc5);
      vacc6 = vaddq_f32(vin6, vacc6);
      vacc7 = vaddq_f32(vin7, vacc7);
      vacc8 = vaddq_f32(vin8, vacc8);
      vacc9 = vaddq_f32(vin9, vacc9);
      vacc10 = vaddq_f32(vin10, vacc10);
      vacc11 = vaddq_f32(vin11, vacc11);
      vacc12 = vaddq_f32(vin12, vacc12);
      vacc13 = vaddq_f32(vin13, vacc13);
      vacc14 = vaddq_f32(vin14, vacc14);
      vacc15 = vaddq_f32(vin15, vacc15);
      vin0 = vld1q_f32(&i3[0]);
      vin1 = vld1q_f32(&i3[4]);
      vin2 = vld1q_f32(&i3[8]);
      vin3 = vld1q_f32(&i3[12]);
      vin4 = vld1q_f32(&i3[16]);
      vin5 = vld1q_f32(&i3[20]);
      vin6 = vld1q_f32(&i3[24]);
      vin7 = vld1q_f32(&i3[28]);
      vin8 = vld1q_f32(&i3[32]);
      vin9 = vld1q_f32(&i3[36]);
      vin10 = vld1q_f32(&i3[40]);
      vin11 = vld1q_f32(&i3[44]);
      vin12 = vld1q_f32(&i3[48]);
      vin13 = vld1q_f32(&i3[52]);
      vin14 = vld1q_f32(&i3[56]);
      vin15 = vld1q_f32(&i3[60]);
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vacc4 = vaddq_f32(vin4, vacc4);
      vacc5 = vaddq_f32(vin5, vacc5);
      vacc6 = vaddq_f32(vin6, vacc6);
      vacc7 = vaddq_f32(vin7, vacc7);
      vacc8 = vaddq_f32(vin8, vacc8);
      vacc9 = vaddq_f32(vin9, vacc9);
      vacc10 = vaddq_f32(vin10, vacc10);
      vacc11 = vaddq_f32(vin11, vacc11);
      vacc12 = vaddq_f32(vin12, vacc12);
      vacc13 = vaddq_f32(vin13, vacc13);
      vacc14 = vaddq_f32(vin14, vacc14);
      vacc15 = vaddq_f32(vin15, vacc15);
      vin0 = vld1q_f32(&i4[0]);
      vin1 = vld1q_f32(&i4[4]);
      vin2 = vld1q_f32(&i4[8]);
      vin3 = vld1q_f32(&i4[12]);
      vin4 = vld1q_f32(&i4[16]);
      vin5 = vld1q_f32(&i4[20]);
      vin6 = vld1q_f32(&i4[24]);
      vin7 = vld1q_f32(&i4[28]);
      vin8 = vld1q_f32(&i4[32]);
      vin9 = vld1q_f32(&i4[36]);
      vin10 = vld1q_f32(&i4[40]);
      vin11 = vld1q_f32(&i4[44]);
      vin12 = vld1q_f32(&i4[48]);
      vin13 = vld1q_f32(&i4[52]);
      vin14 = vld1q_f32(&i4[56]);
      vin15 = vld1q_f32(&i4[60]);
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vacc4 = vaddq_f32(vin4, vacc4);
      vacc5 = vaddq_f32(vin5, vacc5);
      vacc6 = vaddq_f32(vin6, vacc6);
      vacc7 = vaddq_f32(vin7, vacc7);
      vacc8 = vaddq_f32(vin8, vacc8);
      vacc9 = vaddq_f32(vin9, vacc9);
      vacc10 = vaddq_f32(vin10, vacc10);
      vacc11 = vaddq_f32(vin11, vacc11);
      vacc12 = vaddq_f32(vin12, vacc12);
      vacc13 = vaddq_f32(vin13, vacc13);
      vacc14 = vaddq_f32(vin14, vacc14);
      vacc15 = vaddq_f32(vin15, vacc15);
      vin0 = vld1q_f32(&i5[0]);
      vin1 = vld1q_f32(&i5[4]);
      vin2 = vld1q_f32(&i5[8]);
      vin3 = vld1q_f32(&i5[12]);
      vin4 = vld1q_f32(&i5[16]);
      vin5 = vld1q_f32(&i5[20]);
      vin6 = vld1q_f32(&i5[24]);
      vin7 = vld1q_f32(&i5[28]);
      vin8 = vld1q_f32(&i5[32]);
      vin9 = vld1q_f32(&i5[36]);
      vin10 = vld1q_f32(&i5[40]);
      vin11 = vld1q_f32(&i5[44]);
      vin12 = vld1q_f32(&i5[48]);
      vin13 = vld1q_f32(&i5[52]);
      vin14 = vld1q_f32(&i5[56]);
      vin15 = vld1q_f32(&i5[60]);
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vacc4 = vaddq_f32(vin4, vacc4);
      vacc5 = vaddq_f32(vin5, vacc5);
      vacc6 = vaddq_f32(vin6, vacc6);
      vacc7 = vaddq_f32(vin7, vacc7);
      vacc8 = vaddq_f32(vin8, vacc8);
      vacc9 = vaddq_f32(vin9, vacc9);
      vacc10 = vaddq_f32(vin10, vacc10);
      vacc11 = vaddq_f32(vin11, vacc11);
      vacc12 = vaddq_f32(vin12, vacc12);
      vacc13 = vaddq_f32(vin13, vacc13);
      vacc14 = vaddq_f32(vin14, vacc14);
      vacc15 = vaddq_f32(vin15, vacc15);
      vin0 = vld1q_f32(&i6[0]);
      vin1 = vld1q_f32(&i6[4]);
      vin2 = vld1q_f32(&i6[8]);
      vin3 = vld1q_f32(&i6[12]);
      vin4 = vld1q_f32(&i6[16]);
      vin5 = vld1q_f32(&i6[20]);
      vin6 = vld1q_f32(&i6[24]);
      vin7 = vld1q_f32(&i6[28]);
      vin8 = vld1q_f32(&i6[32]);
      vin9 = vld1q_f32(&i6[36]);
      vin10 = vld1q_f32(&i6[40]);
      vin11 = vld1q_f32(&i6[44]);
      vin12 = vld1q_f32(&i6[48]);
      vin13 = vld1q_f32(&i6[52]);
      vin14 = vld1q_f32(&i6[56]);
      vin15 = vld1q_f32(&i6[60]);
      vacc0 = vaddq_f32(vin0, vacc0);
      vacc1 = vaddq_f32(vin1, vacc1);
      vacc2 = vaddq_f32(vin2, vacc2);
      vacc3 = vaddq_f32(vin3, vacc3);
      vacc4 = vaddq_f32(vin4, vacc4);
      vacc5 = vaddq_f32(vin5, vacc5);
      vacc6 = vaddq_f32(vin6, vacc6);
      vacc7 = vaddq_f32(vin7, vacc7);
      vacc8 = vaddq_f32(vin8, vacc8);
      vacc9 = vaddq_f32(vin9, vacc9);
      vacc10 = vaddq_f32(vin10, vacc10);
      vacc11 = vaddq_f32(vin11, vacc11);
      vacc12 = vaddq_f32(vin12, vacc12);
      vacc13 = vaddq_f32(vin13, vacc13);
      vacc14 = vaddq_f32(vin14, vacc14);
      vacc15 = vaddq_f32(vin15, vacc15);
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = vmulq_f32(vacc0, vscale);
    vacc0 = vmaxq_f32(vacc0, vmin);
    vacc0 = vminq_f32(vacc0, vmax);
    vacc1 = vmulq_f32(vacc1, vscale);
    vacc1 = vmaxq_f32(vacc1, vmin);
    vacc1 = vminq_f32(vacc1, vmax);
    vacc2 = vmulq_f32(vacc2, vscale);
    vacc2 = vmaxq_f32(vacc2, vmin);
    vacc2 = vminq_f32(vacc2, vmax);
    vacc3 = vmulq_f32(vacc3, vscale);
    vacc3 = vmaxq_f32(vacc3, vmin);
    vacc3 = vminq_f32(vacc3, vmax);
    vacc4 = vmulq_f32(vacc4, vscale);
    vacc4 = vmaxq_f32(vacc4, vmin);
    vacc4 = vminq_f32(vacc4, vmax);
    vacc5 = vmulq_f32(vacc5, vscale);
    vacc5 = vmaxq_f32(vacc5, vmin);
    vacc5 = vminq_f32(vacc5, vmax);
    vacc6 = vmulq_f32(vacc6, vscale);
    vacc6 = vmaxq_f32(vacc6, vmin);
    vacc6 = vminq_f32(vacc6, vmax);
    vacc7 = vmulq_f32(vacc7, vscale);
    vacc7 = vmaxq_f32(vacc7, vmin);
    vacc7 = vminq_f32(vacc7, vmax);
    vacc8 = vmulq_f32(vacc8, vscale);
    vacc8 = vmaxq_f32(vacc8, vmin);
    vacc8 = vminq_f32(vacc8, vmax);
    vacc9 = vmulq_f32(vacc9, vscale);
    vacc9 = vmaxq_f32(vacc9, vmin);
    vacc9 = vminq_f32(vacc9, vmax);
    vacc10 = vmulq_f32(vacc10, vscale);
    vacc10 = vmaxq_f32(vacc10, vmin);
    vacc10 = vminq_f32(vacc10, vmax);
    vacc11 = vmulq_f32(vacc11, vscale);
    vacc11 = vmaxq_f32(vacc11, vmin);
    vacc11 = vminq_f32(vacc11, vmax);
    vacc12 = vmulq_f32(vacc12, vscale);
    vacc12 = vmaxq_f32(vacc12, vmin);
    vacc12 = vminq_f32(vacc12, vmax);
    vacc13 = vmulq_f32(vacc13, vscale);
    vacc13 = vmaxq_f32(vacc13, vmin);
    vacc13 = vminq_f32(vacc13, vmax);
    vacc14 = vmulq_f32(vacc14, vscale);
    vacc14 = vmaxq_f32(vacc14, vmin);
    vacc14 = vminq_f32(vacc14, vmax);
    vacc15 = vmulq_f32(vacc15, vscale);
    vacc15 = vmaxq_f32(vacc15, vmin);
    vacc15 = vminq_f32(vacc15, vmax);

    const float* o = output;
    float32x4_t vo0 = vld1q_f32(o); o += 4;
    float32x4_t vo1 = vld1q_f32(o); o += 4;
    float32x4_t vo2 = vld1q_f32(o); o += 4;
    float32x4_t vo3 = vld1q_f32(o); o += 4;
    float32x4_t vo4 = vld1q_f32(o); o += 4;
    float32x4_t vo5 = vld1q_f32(o); o += 4;
    float32x4_t vo6 = vld1q_f32(o); o += 4;
    float32x4_t vo7 = vld1q_f32(o); o += 4;
    float32x4_t vo8 = vld1q_f32(o); o += 4;
    float32x4_t vo9 = vld1q_f32(o); o += 4;
    float32x4_t vo10 = vld1q_f32(o); o += 4;
    float32x4_t vo11 = vld1q_f32(o); o += 4;
    float32x4_t vo12 = vld1q_f32(o); o += 4;
    float32x4_t vo13 = vld1q_f32(o); o += 4;
    float32x4_t vo14 = vld1q_f32(o); o += 4;
    float32x4_t vo15 = vld1q_f32(o); o += 4;
    vacc0 = vaddq_f32(vo0, vacc0);
    vacc1 = vaddq_f32(vo1, vacc1);
    vacc2 = vaddq_f32(vo2, vacc2);
    vacc3 = vaddq_f32(vo3, vacc3);
    vacc4 = vaddq_f32(vo4, vacc4);
    vacc5 = vaddq_f32(vo5, vacc5);
    vacc6 = vaddq_f32(vo6, vacc6);
    vacc7 = vaddq_f32(vo7, vacc7);
    vacc8 = vaddq_f32(vo8, vacc8);
    vacc9 = vaddq_f32(vo9, vacc9);
    vacc10 = vaddq_f32(vo10, vacc10);
    vacc11 = vaddq_f32(vo11, vacc11);
    vacc12 = vaddq_f32(vo12, vacc12);
    vacc13 = vaddq_f32(vo13, vacc13);
    vacc14 = vaddq_f32(vo14, vacc14);
    vacc15 = vaddq_f32(vo15, vacc15);
    vst1q_f32(output, vacc0); output += 4;
    vst1q_f32(output, vacc1); output += 4;
    vst1q_f32(output, vacc2); output += 4;
    vst1q_f32(output, vacc3); output += 4;
    vst1q_f32(output, vacc4); output += 4;
    vst1q_f32(output, vacc5); output += 4;
    vst1q_f32(output, vacc6); output += 4;
    vst1q_f32(output, vacc7); output += 4;
    vst1q_f32(output, vacc8); output += 4;
    vst1q_f32(output, vacc9); output += 4;
    vst1q_f32(output, vacc10); output += 4;
    vst1q_f32(output, vacc11); output += 4;
    vst1q_f32(output, vacc12); output += 4;
    vst1q_f32(output, vacc13); output += 4;
    vst1q_f32(output, vacc14); output += 4;
    vst1q_f32(output, vacc15); output += 4;

    input = (const float*) ((uintptr_t) input + 64 * sizeof(float));
  }
  if (channels != 0) {
    size_t input_increment = 7 * input_stride;
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);
    float32x4_t vacc[16];
    vacc[0] = vdupq_n_f32(0.f);
    vacc[1] = vdupq_n_f32(0.f);
    vacc[2] = vdupq_n_f32(0.f);
    vacc[3] = vdupq_n_f32(0.f);
    vacc[4] = vdupq_n_f32(0.f);
    vacc[5] = vdupq_n_f32(0.f);
    vacc[6] = vdupq_n_f32(0.f);
    vacc[7] = vdupq_n_f32(0.f);
    vacc[8] = vdupq_n_f32(0.f);
    vacc[9] = vdupq_n_f32(0.f);
    vacc[10] = vdupq_n_f32(0.f);
    vacc[11] = vdupq_n_f32(0.f);
    vacc[12] = vdupq_n_f32(0.f);
    vacc[13] = vdupq_n_f32(0.f);
    vacc[14] = vdupq_n_f32(0.f);
    vacc[15] = vdupq_n_f32(0.f);

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
        vacc[i] = vaddq_f32(vld1q_f32(&i0[i*4]), vacc[i]);
        vacc[i] = vaddq_f32(vld1q_f32(&i1[i*4]), vacc[i]);
        vacc[i] = vaddq_f32(vld1q_f32(&i2[i*4]), vacc[i]);
        vacc[i] = vaddq_f32(vld1q_f32(&i3[i*4]), vacc[i]);
        vacc[i] = vaddq_f32(vld1q_f32(&i4[i*4]), vacc[i]);
        vacc[i] = vaddq_f32(vld1q_f32(&i5[i*4]), vacc[i]);
        vacc[i] = vaddq_f32(vld1q_f32(&i6[i*4]), vacc[i]);
      }
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    for (int i = 0; i < (channels + 4) >> 2; ++i) {
      vacc[i] = vmulq_f32(vacc[i], vscale);
      vacc[i] = vmaxq_f32(vacc[i], vmin);
      vacc[i] = vminq_f32(vacc[i], vmax);
    }

    float32x4_t vo[16];
    const float* o = output;
    for (int i = 0; i < channels >> 2; ++i) {
      vo[i] = vld1q_f32(o); o += 4;
    }
    for (int i = 0; i < channels >> 2; ++i) {
      vacc[i] = vaddq_f32(vo[i], vacc[i]);
    }
    for (int i = 0; i < channels >> 2; ++i) {
      vst1q_f32(output, vacc[i]); output += 4;
    }
    size_t pos = channels >> 2;
    channels &= 0x3;
    float32x2_t vacc_low = vget_low_f32(vacc[pos]);
    if (channels & 2) {
      vst1_f32(output, vadd_f32(vld1_f32(output), vacc_low)); output += 2;
      vacc_low = vget_high_f32(vacc[pos]);
    }
    if (channels & 1) {
      vst1_lane_f32(output, vadd_f32(vld1_dup_f32(output), vacc_low), 0);
    }
  }
}
