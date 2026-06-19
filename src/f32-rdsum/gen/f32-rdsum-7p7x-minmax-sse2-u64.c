// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rdsum/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/simd/f32-sse2.h"


void xnn_f32_rdsum_ukernel_7p7x__sse2_u64(
    size_t channels,
    size_t k1,
    size_t k2,
    size_t k3,
    const float* input,
    size_t input_stride1,
    size_t input_stride2,
    size_t input_stride3,
    const float* zero,
    float* output,
    const struct xnn_f32_scale_params* restrict params)
{
  assert(k1 != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const xnn_simd_f32_t vscale = xnn_set1_f32(params->scalar.scale);
  float* original_output = output;
  size_t original_channels = channels;

  for (size_t k = 0; k < k3; ++k) {
    for (size_t j = 0; j < k2; ++j) {
      const float* input_row = (const float*)((uintptr_t)input + j * input_stride2 + k * input_stride3);
      output = original_output;
      channels = original_channels;

      assert(input_row != NULL);

      size_t input_increment = 7 * input_stride1;
      for (; channels >= 64; channels -= 64) {
        const float* i0 = input_row;
        const float* i1 = (const float*) ((uintptr_t) input_row + 1 * input_stride1);
        const float* i2 = (const float*) ((uintptr_t) input_row + 2 * input_stride1);
        const float* i3 = (const float*) ((uintptr_t) input_row + 3 * input_stride1);
        const float* i4 = (const float*) ((uintptr_t) input_row + 4 * input_stride1);
        const float* i5 = (const float*) ((uintptr_t) input_row + 5 * input_stride1);
        const float* i6 = (const float*) ((uintptr_t) input_row + 6 * input_stride1);

        xnn_simd_f32_t vacc0 = xnn_zero_f32();
        xnn_simd_f32_t vacc1 = xnn_zero_f32();
        xnn_simd_f32_t vacc2 = xnn_zero_f32();
        xnn_simd_f32_t vacc3 = xnn_zero_f32();
        xnn_simd_f32_t vacc4 = xnn_zero_f32();
        xnn_simd_f32_t vacc5 = xnn_zero_f32();
        xnn_simd_f32_t vacc6 = xnn_zero_f32();
        xnn_simd_f32_t vacc7 = xnn_zero_f32();
        xnn_simd_f32_t vacc8 = xnn_zero_f32();
        xnn_simd_f32_t vacc9 = xnn_zero_f32();
        xnn_simd_f32_t vacc10 = xnn_zero_f32();
        xnn_simd_f32_t vacc11 = xnn_zero_f32();
        xnn_simd_f32_t vacc12 = xnn_zero_f32();
        xnn_simd_f32_t vacc13 = xnn_zero_f32();
        xnn_simd_f32_t vacc14 = xnn_zero_f32();
        xnn_simd_f32_t vacc15 = xnn_zero_f32();

        for (int r = k1; r > 0; r -= 7) {
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
          xnn_simd_f32_t vin0;
          xnn_simd_f32_t vin1;
          xnn_simd_f32_t vin2;
          xnn_simd_f32_t vin3;
          xnn_simd_f32_t vin4;
          xnn_simd_f32_t vin5;
          xnn_simd_f32_t vin6;
          xnn_simd_f32_t vin7;
          xnn_simd_f32_t vin8;
          xnn_simd_f32_t vin9;
          xnn_simd_f32_t vin10;
          xnn_simd_f32_t vin11;
          xnn_simd_f32_t vin12;
          xnn_simd_f32_t vin13;
          xnn_simd_f32_t vin14;
          xnn_simd_f32_t vin15;
          vin0 = xnn_loadu_f32(&i0[0]);
          vin1 = xnn_loadu_f32(&i0[4]);
          vin2 = xnn_loadu_f32(&i0[8]);
          vin3 = xnn_loadu_f32(&i0[12]);
          vin4 = xnn_loadu_f32(&i0[16]);
          vin5 = xnn_loadu_f32(&i0[20]);
          vin6 = xnn_loadu_f32(&i0[24]);
          vin7 = xnn_loadu_f32(&i0[28]);
          vin8 = xnn_loadu_f32(&i0[32]);
          vin9 = xnn_loadu_f32(&i0[36]);
          vin10 = xnn_loadu_f32(&i0[40]);
          vin11 = xnn_loadu_f32(&i0[44]);
          vin12 = xnn_loadu_f32(&i0[48]);
          vin13 = xnn_loadu_f32(&i0[52]);
          vin14 = xnn_loadu_f32(&i0[56]);
          vin15 = xnn_loadu_f32(&i0[60]);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vacc4 = xnn_add_f32(vin4, vacc4);
          vacc5 = xnn_add_f32(vin5, vacc5);
          vacc6 = xnn_add_f32(vin6, vacc6);
          vacc7 = xnn_add_f32(vin7, vacc7);
          vacc8 = xnn_add_f32(vin8, vacc8);
          vacc9 = xnn_add_f32(vin9, vacc9);
          vacc10 = xnn_add_f32(vin10, vacc10);
          vacc11 = xnn_add_f32(vin11, vacc11);
          vacc12 = xnn_add_f32(vin12, vacc12);
          vacc13 = xnn_add_f32(vin13, vacc13);
          vacc14 = xnn_add_f32(vin14, vacc14);
          vacc15 = xnn_add_f32(vin15, vacc15);
          vin0 = xnn_loadu_f32(&i1[0]);
          vin1 = xnn_loadu_f32(&i1[4]);
          vin2 = xnn_loadu_f32(&i1[8]);
          vin3 = xnn_loadu_f32(&i1[12]);
          vin4 = xnn_loadu_f32(&i1[16]);
          vin5 = xnn_loadu_f32(&i1[20]);
          vin6 = xnn_loadu_f32(&i1[24]);
          vin7 = xnn_loadu_f32(&i1[28]);
          vin8 = xnn_loadu_f32(&i1[32]);
          vin9 = xnn_loadu_f32(&i1[36]);
          vin10 = xnn_loadu_f32(&i1[40]);
          vin11 = xnn_loadu_f32(&i1[44]);
          vin12 = xnn_loadu_f32(&i1[48]);
          vin13 = xnn_loadu_f32(&i1[52]);
          vin14 = xnn_loadu_f32(&i1[56]);
          vin15 = xnn_loadu_f32(&i1[60]);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vacc4 = xnn_add_f32(vin4, vacc4);
          vacc5 = xnn_add_f32(vin5, vacc5);
          vacc6 = xnn_add_f32(vin6, vacc6);
          vacc7 = xnn_add_f32(vin7, vacc7);
          vacc8 = xnn_add_f32(vin8, vacc8);
          vacc9 = xnn_add_f32(vin9, vacc9);
          vacc10 = xnn_add_f32(vin10, vacc10);
          vacc11 = xnn_add_f32(vin11, vacc11);
          vacc12 = xnn_add_f32(vin12, vacc12);
          vacc13 = xnn_add_f32(vin13, vacc13);
          vacc14 = xnn_add_f32(vin14, vacc14);
          vacc15 = xnn_add_f32(vin15, vacc15);
          vin0 = xnn_loadu_f32(&i2[0]);
          vin1 = xnn_loadu_f32(&i2[4]);
          vin2 = xnn_loadu_f32(&i2[8]);
          vin3 = xnn_loadu_f32(&i2[12]);
          vin4 = xnn_loadu_f32(&i2[16]);
          vin5 = xnn_loadu_f32(&i2[20]);
          vin6 = xnn_loadu_f32(&i2[24]);
          vin7 = xnn_loadu_f32(&i2[28]);
          vin8 = xnn_loadu_f32(&i2[32]);
          vin9 = xnn_loadu_f32(&i2[36]);
          vin10 = xnn_loadu_f32(&i2[40]);
          vin11 = xnn_loadu_f32(&i2[44]);
          vin12 = xnn_loadu_f32(&i2[48]);
          vin13 = xnn_loadu_f32(&i2[52]);
          vin14 = xnn_loadu_f32(&i2[56]);
          vin15 = xnn_loadu_f32(&i2[60]);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vacc4 = xnn_add_f32(vin4, vacc4);
          vacc5 = xnn_add_f32(vin5, vacc5);
          vacc6 = xnn_add_f32(vin6, vacc6);
          vacc7 = xnn_add_f32(vin7, vacc7);
          vacc8 = xnn_add_f32(vin8, vacc8);
          vacc9 = xnn_add_f32(vin9, vacc9);
          vacc10 = xnn_add_f32(vin10, vacc10);
          vacc11 = xnn_add_f32(vin11, vacc11);
          vacc12 = xnn_add_f32(vin12, vacc12);
          vacc13 = xnn_add_f32(vin13, vacc13);
          vacc14 = xnn_add_f32(vin14, vacc14);
          vacc15 = xnn_add_f32(vin15, vacc15);
          vin0 = xnn_loadu_f32(&i3[0]);
          vin1 = xnn_loadu_f32(&i3[4]);
          vin2 = xnn_loadu_f32(&i3[8]);
          vin3 = xnn_loadu_f32(&i3[12]);
          vin4 = xnn_loadu_f32(&i3[16]);
          vin5 = xnn_loadu_f32(&i3[20]);
          vin6 = xnn_loadu_f32(&i3[24]);
          vin7 = xnn_loadu_f32(&i3[28]);
          vin8 = xnn_loadu_f32(&i3[32]);
          vin9 = xnn_loadu_f32(&i3[36]);
          vin10 = xnn_loadu_f32(&i3[40]);
          vin11 = xnn_loadu_f32(&i3[44]);
          vin12 = xnn_loadu_f32(&i3[48]);
          vin13 = xnn_loadu_f32(&i3[52]);
          vin14 = xnn_loadu_f32(&i3[56]);
          vin15 = xnn_loadu_f32(&i3[60]);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vacc4 = xnn_add_f32(vin4, vacc4);
          vacc5 = xnn_add_f32(vin5, vacc5);
          vacc6 = xnn_add_f32(vin6, vacc6);
          vacc7 = xnn_add_f32(vin7, vacc7);
          vacc8 = xnn_add_f32(vin8, vacc8);
          vacc9 = xnn_add_f32(vin9, vacc9);
          vacc10 = xnn_add_f32(vin10, vacc10);
          vacc11 = xnn_add_f32(vin11, vacc11);
          vacc12 = xnn_add_f32(vin12, vacc12);
          vacc13 = xnn_add_f32(vin13, vacc13);
          vacc14 = xnn_add_f32(vin14, vacc14);
          vacc15 = xnn_add_f32(vin15, vacc15);
          vin0 = xnn_loadu_f32(&i4[0]);
          vin1 = xnn_loadu_f32(&i4[4]);
          vin2 = xnn_loadu_f32(&i4[8]);
          vin3 = xnn_loadu_f32(&i4[12]);
          vin4 = xnn_loadu_f32(&i4[16]);
          vin5 = xnn_loadu_f32(&i4[20]);
          vin6 = xnn_loadu_f32(&i4[24]);
          vin7 = xnn_loadu_f32(&i4[28]);
          vin8 = xnn_loadu_f32(&i4[32]);
          vin9 = xnn_loadu_f32(&i4[36]);
          vin10 = xnn_loadu_f32(&i4[40]);
          vin11 = xnn_loadu_f32(&i4[44]);
          vin12 = xnn_loadu_f32(&i4[48]);
          vin13 = xnn_loadu_f32(&i4[52]);
          vin14 = xnn_loadu_f32(&i4[56]);
          vin15 = xnn_loadu_f32(&i4[60]);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vacc4 = xnn_add_f32(vin4, vacc4);
          vacc5 = xnn_add_f32(vin5, vacc5);
          vacc6 = xnn_add_f32(vin6, vacc6);
          vacc7 = xnn_add_f32(vin7, vacc7);
          vacc8 = xnn_add_f32(vin8, vacc8);
          vacc9 = xnn_add_f32(vin9, vacc9);
          vacc10 = xnn_add_f32(vin10, vacc10);
          vacc11 = xnn_add_f32(vin11, vacc11);
          vacc12 = xnn_add_f32(vin12, vacc12);
          vacc13 = xnn_add_f32(vin13, vacc13);
          vacc14 = xnn_add_f32(vin14, vacc14);
          vacc15 = xnn_add_f32(vin15, vacc15);
          vin0 = xnn_loadu_f32(&i5[0]);
          vin1 = xnn_loadu_f32(&i5[4]);
          vin2 = xnn_loadu_f32(&i5[8]);
          vin3 = xnn_loadu_f32(&i5[12]);
          vin4 = xnn_loadu_f32(&i5[16]);
          vin5 = xnn_loadu_f32(&i5[20]);
          vin6 = xnn_loadu_f32(&i5[24]);
          vin7 = xnn_loadu_f32(&i5[28]);
          vin8 = xnn_loadu_f32(&i5[32]);
          vin9 = xnn_loadu_f32(&i5[36]);
          vin10 = xnn_loadu_f32(&i5[40]);
          vin11 = xnn_loadu_f32(&i5[44]);
          vin12 = xnn_loadu_f32(&i5[48]);
          vin13 = xnn_loadu_f32(&i5[52]);
          vin14 = xnn_loadu_f32(&i5[56]);
          vin15 = xnn_loadu_f32(&i5[60]);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vacc4 = xnn_add_f32(vin4, vacc4);
          vacc5 = xnn_add_f32(vin5, vacc5);
          vacc6 = xnn_add_f32(vin6, vacc6);
          vacc7 = xnn_add_f32(vin7, vacc7);
          vacc8 = xnn_add_f32(vin8, vacc8);
          vacc9 = xnn_add_f32(vin9, vacc9);
          vacc10 = xnn_add_f32(vin10, vacc10);
          vacc11 = xnn_add_f32(vin11, vacc11);
          vacc12 = xnn_add_f32(vin12, vacc12);
          vacc13 = xnn_add_f32(vin13, vacc13);
          vacc14 = xnn_add_f32(vin14, vacc14);
          vacc15 = xnn_add_f32(vin15, vacc15);
          vin0 = xnn_loadu_f32(&i6[0]);
          vin1 = xnn_loadu_f32(&i6[4]);
          vin2 = xnn_loadu_f32(&i6[8]);
          vin3 = xnn_loadu_f32(&i6[12]);
          vin4 = xnn_loadu_f32(&i6[16]);
          vin5 = xnn_loadu_f32(&i6[20]);
          vin6 = xnn_loadu_f32(&i6[24]);
          vin7 = xnn_loadu_f32(&i6[28]);
          vin8 = xnn_loadu_f32(&i6[32]);
          vin9 = xnn_loadu_f32(&i6[36]);
          vin10 = xnn_loadu_f32(&i6[40]);
          vin11 = xnn_loadu_f32(&i6[44]);
          vin12 = xnn_loadu_f32(&i6[48]);
          vin13 = xnn_loadu_f32(&i6[52]);
          vin14 = xnn_loadu_f32(&i6[56]);
          vin15 = xnn_loadu_f32(&i6[60]);
          vacc0 = xnn_add_f32(vin0, vacc0);
          vacc1 = xnn_add_f32(vin1, vacc1);
          vacc2 = xnn_add_f32(vin2, vacc2);
          vacc3 = xnn_add_f32(vin3, vacc3);
          vacc4 = xnn_add_f32(vin4, vacc4);
          vacc5 = xnn_add_f32(vin5, vacc5);
          vacc6 = xnn_add_f32(vin6, vacc6);
          vacc7 = xnn_add_f32(vin7, vacc7);
          vacc8 = xnn_add_f32(vin8, vacc8);
          vacc9 = xnn_add_f32(vin9, vacc9);
          vacc10 = xnn_add_f32(vin10, vacc10);
          vacc11 = xnn_add_f32(vin11, vacc11);
          vacc12 = xnn_add_f32(vin12, vacc12);
          vacc13 = xnn_add_f32(vin13, vacc13);
          vacc14 = xnn_add_f32(vin14, vacc14);
          vacc15 = xnn_add_f32(vin15, vacc15);
          i0 = (const float*) ((uintptr_t) i0 + input_increment);
          i1 = (const float*) ((uintptr_t) i1 + input_increment);
          i2 = (const float*) ((uintptr_t) i2 + input_increment);
          i3 = (const float*) ((uintptr_t) i3 + input_increment);
          i4 = (const float*) ((uintptr_t) i4 + input_increment);
          i5 = (const float*) ((uintptr_t) i5 + input_increment);
          i6 = (const float*) ((uintptr_t) i6 + input_increment);
        }
        vacc0 = xnn_mul_f32(vacc0, vscale);
        vacc1 = xnn_mul_f32(vacc1, vscale);
        vacc2 = xnn_mul_f32(vacc2, vscale);
        vacc3 = xnn_mul_f32(vacc3, vscale);
        vacc4 = xnn_mul_f32(vacc4, vscale);
        vacc5 = xnn_mul_f32(vacc5, vscale);
        vacc6 = xnn_mul_f32(vacc6, vscale);
        vacc7 = xnn_mul_f32(vacc7, vscale);
        vacc8 = xnn_mul_f32(vacc8, vscale);
        vacc9 = xnn_mul_f32(vacc9, vscale);
        vacc10 = xnn_mul_f32(vacc10, vscale);
        vacc11 = xnn_mul_f32(vacc11, vscale);
        vacc12 = xnn_mul_f32(vacc12, vscale);
        vacc13 = xnn_mul_f32(vacc13, vscale);
        vacc14 = xnn_mul_f32(vacc14, vscale);
        vacc15 = xnn_mul_f32(vacc15, vscale);

        const float* o = output;
        xnn_simd_f32_t vo0 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo1 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo2 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo3 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo4 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo5 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo6 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo7 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo8 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo9 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo10 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo11 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo12 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo13 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo14 = xnn_loadu_f32(o); o += 4;
        xnn_simd_f32_t vo15 = xnn_loadu_f32(o); o += 4;
        vacc0 = xnn_add_f32(vo0, vacc0);
        vacc1 = xnn_add_f32(vo1, vacc1);
        vacc2 = xnn_add_f32(vo2, vacc2);
        vacc3 = xnn_add_f32(vo3, vacc3);
        vacc4 = xnn_add_f32(vo4, vacc4);
        vacc5 = xnn_add_f32(vo5, vacc5);
        vacc6 = xnn_add_f32(vo6, vacc6);
        vacc7 = xnn_add_f32(vo7, vacc7);
        vacc8 = xnn_add_f32(vo8, vacc8);
        vacc9 = xnn_add_f32(vo9, vacc9);
        vacc10 = xnn_add_f32(vo10, vacc10);
        vacc11 = xnn_add_f32(vo11, vacc11);
        vacc12 = xnn_add_f32(vo12, vacc12);
        vacc13 = xnn_add_f32(vo13, vacc13);
        vacc14 = xnn_add_f32(vo14, vacc14);
        vacc15 = xnn_add_f32(vo15, vacc15);
        xnn_storeu_f32(output, vacc0); output += 4;
        xnn_storeu_f32(output, vacc1); output += 4;
        xnn_storeu_f32(output, vacc2); output += 4;
        xnn_storeu_f32(output, vacc3); output += 4;
        xnn_storeu_f32(output, vacc4); output += 4;
        xnn_storeu_f32(output, vacc5); output += 4;
        xnn_storeu_f32(output, vacc6); output += 4;
        xnn_storeu_f32(output, vacc7); output += 4;
        xnn_storeu_f32(output, vacc8); output += 4;
        xnn_storeu_f32(output, vacc9); output += 4;
        xnn_storeu_f32(output, vacc10); output += 4;
        xnn_storeu_f32(output, vacc11); output += 4;
        xnn_storeu_f32(output, vacc12); output += 4;
        xnn_storeu_f32(output, vacc13); output += 4;
        xnn_storeu_f32(output, vacc14); output += 4;
        xnn_storeu_f32(output, vacc15); output += 4;

        input_row = (const float*) ((uintptr_t) input_row + 64 * sizeof(float));
      }
      if (channels != 0) {
        input_increment = 7 * input_stride1;
        const float* i0 = input_row;
        const float* i1 = (const float*) ((uintptr_t) input_row + 1 * input_stride1);
        const float* i2 = (const float*) ((uintptr_t) input_row + 2 * input_stride1);
        const float* i3 = (const float*) ((uintptr_t) input_row + 3 * input_stride1);
        const float* i4 = (const float*) ((uintptr_t) input_row + 4 * input_stride1);
        const float* i5 = (const float*) ((uintptr_t) input_row + 5 * input_stride1);
        const float* i6 = (const float*) ((uintptr_t) input_row + 6 * input_stride1);
        xnn_simd_f32_t vacc[16];
        vacc[0] = xnn_zero_f32();
        vacc[1] = xnn_zero_f32();
        vacc[2] = xnn_zero_f32();
        vacc[3] = xnn_zero_f32();
        vacc[4] = xnn_zero_f32();
        vacc[5] = xnn_zero_f32();
        vacc[6] = xnn_zero_f32();
        vacc[7] = xnn_zero_f32();
        vacc[8] = xnn_zero_f32();
        vacc[9] = xnn_zero_f32();
        vacc[10] = xnn_zero_f32();
        vacc[11] = xnn_zero_f32();
        vacc[12] = xnn_zero_f32();
        vacc[13] = xnn_zero_f32();
        vacc[14] = xnn_zero_f32();
        vacc[15] = xnn_zero_f32();

        const size_t num_full_chunks = channels >> 2;
        const size_t num_chunks = round_up_po2(channels, 4) >> 2;
        const size_t remainder = channels & 3;
        for (int r = k1; r > 0; r -= 7) {
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
            vacc[i] = xnn_add_f32(xnn_loadu_f32(&i0[i*4]), vacc[i]);
            vacc[i] = xnn_add_f32(xnn_loadu_f32(&i1[i*4]), vacc[i]);
            vacc[i] = xnn_add_f32(xnn_loadu_f32(&i2[i*4]), vacc[i]);
            vacc[i] = xnn_add_f32(xnn_loadu_f32(&i3[i*4]), vacc[i]);
            vacc[i] = xnn_add_f32(xnn_loadu_f32(&i4[i*4]), vacc[i]);
            vacc[i] = xnn_add_f32(xnn_loadu_f32(&i5[i*4]), vacc[i]);
            vacc[i] = xnn_add_f32(xnn_loadu_f32(&i6[i*4]), vacc[i]);
          }

          if (remainder) {
            vacc[num_full_chunks] = xnn_add_f32(xnn_load_tail_f32(&i0[num_full_chunks*4], remainder), vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(xnn_load_tail_f32(&i1[num_full_chunks*4], remainder), vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(xnn_load_tail_f32(&i2[num_full_chunks*4], remainder), vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(xnn_load_tail_f32(&i3[num_full_chunks*4], remainder), vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(xnn_load_tail_f32(&i4[num_full_chunks*4], remainder), vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(xnn_load_tail_f32(&i5[num_full_chunks*4], remainder), vacc[num_full_chunks]);
            vacc[num_full_chunks] = xnn_add_f32(xnn_load_tail_f32(&i6[num_full_chunks*4], remainder), vacc[num_full_chunks]);
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
          vacc[i] = xnn_mul_f32(vacc[i], vscale);
        }

        xnn_simd_f32_t vo[16];
        const float* o = output;
        for (int i = 0; i < channels >> 2; ++i) {
          vo[i] = xnn_loadu_f32(o); o += 4;
        }
        for (int i = 0; i < channels >> 2; ++i) {
          vacc[i] = xnn_add_f32(vo[i], vacc[i]);
        }
        for (int i = 0; i < channels >> 2; ++i) {
          xnn_storeu_f32(output, vacc[i]); output += 4;
        }
        if (remainder) {
          const size_t pos = num_full_chunks;
          xnn_simd_f32_t vout = vacc[pos];
          const xnn_simd_f32_t vdata = xnn_load_tail_safe_f32(output, remainder);
          vout = xnn_add_f32(vout, vdata);
          xnn_store_tail_f32(output, vout, remainder);
        }
      }
    }
  }
}
