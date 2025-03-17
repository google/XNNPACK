// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rdminmax/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"

#include "src/xnnpack/simd/f32-neon.h"


void xnn_f32_rdmin_ukernel_2p2x__neon_c32(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const void* params)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 2 * input_stride;
  for (; channels >= 32; channels -= 32) {
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);

    xnn_simd_f32_t vmin0 = xnn_loadu_f32(output);
    xnn_simd_f32_t vmin4 = xnn_loadu_f32((float*)((uintptr_t) output + 4 * sizeof(float)));
    xnn_simd_f32_t vmin8 = xnn_loadu_f32((float*)((uintptr_t) output + 8 * sizeof(float)));
    xnn_simd_f32_t vmin12 = xnn_loadu_f32((float*)((uintptr_t) output + 12 * sizeof(float)));
    xnn_simd_f32_t vmin16 = xnn_loadu_f32((float*)((uintptr_t) output + 16 * sizeof(float)));
    xnn_simd_f32_t vmin20 = xnn_loadu_f32((float*)((uintptr_t) output + 20 * sizeof(float)));
    xnn_simd_f32_t vmin24 = xnn_loadu_f32((float*)((uintptr_t) output + 24 * sizeof(float)));
    xnn_simd_f32_t vmin28 = xnn_loadu_f32((float*)((uintptr_t) output + 28 * sizeof(float)));

    for (int r = rows; r > 0; r -= 2) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = i0;
      }
      xnn_simd_f32_t vin_0_0 = xnn_loadu_f32(&i0[0]);
      xnn_simd_f32_t vin_0_4 = xnn_loadu_f32(&i0[4]);
      xnn_simd_f32_t vin_0_8 = xnn_loadu_f32(&i0[8]);
      xnn_simd_f32_t vin_0_12 = xnn_loadu_f32(&i0[12]);
      xnn_simd_f32_t vin_0_16 = xnn_loadu_f32(&i0[16]);
      xnn_simd_f32_t vin_0_20 = xnn_loadu_f32(&i0[20]);
      xnn_simd_f32_t vin_0_24 = xnn_loadu_f32(&i0[24]);
      xnn_simd_f32_t vin_0_28 = xnn_loadu_f32(&i0[28]);
      xnn_simd_f32_t vin_1_0 = xnn_loadu_f32(&i1[0]);
      xnn_simd_f32_t vin_1_4 = xnn_loadu_f32(&i1[4]);
      xnn_simd_f32_t vin_1_8 = xnn_loadu_f32(&i1[8]);
      xnn_simd_f32_t vin_1_12 = xnn_loadu_f32(&i1[12]);
      xnn_simd_f32_t vin_1_16 = xnn_loadu_f32(&i1[16]);
      xnn_simd_f32_t vin_1_20 = xnn_loadu_f32(&i1[20]);
      xnn_simd_f32_t vin_1_24 = xnn_loadu_f32(&i1[24]);
      xnn_simd_f32_t vin_1_28 = xnn_loadu_f32(&i1[28]);
      vmin0 = xnn_min_f32(vmin0, vin_0_0);
      vmin4 = xnn_min_f32(vmin4, vin_0_4);
      vmin8 = xnn_min_f32(vmin8, vin_0_8);
      vmin12 = xnn_min_f32(vmin12, vin_0_12);
      vmin16 = xnn_min_f32(vmin16, vin_0_16);
      vmin20 = xnn_min_f32(vmin20, vin_0_20);
      vmin24 = xnn_min_f32(vmin24, vin_0_24);
      vmin28 = xnn_min_f32(vmin28, vin_0_28);
      vmin0 = xnn_min_f32(vmin0, vin_1_0);
      vmin4 = xnn_min_f32(vmin4, vin_1_4);
      vmin8 = xnn_min_f32(vmin8, vin_1_8);
      vmin12 = xnn_min_f32(vmin12, vin_1_12);
      vmin16 = xnn_min_f32(vmin16, vin_1_16);
      vmin20 = xnn_min_f32(vmin20, vin_1_20);
      vmin24 = xnn_min_f32(vmin24, vin_1_24);
      vmin28 = xnn_min_f32(vmin28, vin_1_28);

      i0 = (float*) ((uintptr_t) i0 + input_increment);
      i1 = (float*) ((uintptr_t) i1 + input_increment);
    }

    xnn_storeu_f32(output, vmin0);
    output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);
    xnn_storeu_f32(output, vmin4);
    output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);
    xnn_storeu_f32(output, vmin8);
    output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);
    xnn_storeu_f32(output, vmin12);
    output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);
    xnn_storeu_f32(output, vmin16);
    output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);
    xnn_storeu_f32(output, vmin20);
    output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);
    xnn_storeu_f32(output, vmin24);
    output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);
    xnn_storeu_f32(output, vmin28);
    output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);

    input = (float*) ((uintptr_t) input + 32 * sizeof(float));
  }
  if (channels != 0) {
    input_increment = 2 * input_stride;
    do {
      const float* i0 = input;
      const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);

      xnn_simd_f32_t vmin;

      if (channels >= xnn_simd_size_f32) {
        vmin = xnn_loadu_f32(output);
      } else {
        vmin = xnn_load_tail_safe_f32(output, channels);
      }

      for (int r = rows; r > 0; r -= 2) {
        if XNN_UNPREDICTABLE(r < 2) {
          i1 = i0;
        }
        xnn_simd_f32_t vin0;
        xnn_simd_f32_t vin1;
        if (channels >= xnn_simd_size_f32) {
          vin0 = xnn_loadu_f32(&i0[0]);
        } else {
          vin0 = xnn_load_tail_safe_f32(&i0[0], channels);
        }
        if (channels >= xnn_simd_size_f32) {
          vin1 = xnn_loadu_f32(&i1[0]);
        } else {
          vin1 = xnn_load_tail_safe_f32(&i1[0], channels);
        }
        vmin = xnn_min_f32(vmin, vin0);
        vmin = xnn_min_f32(vmin, vin1);
        i0 = (float*) ((uintptr_t) i0 + input_increment);
        i1 = (float*) ((uintptr_t) i1 + input_increment);
      }

      if (channels >= xnn_simd_size_f32) {
        xnn_storeu_f32(output, vmin);
        output = (float*) ((uintptr_t) output + xnn_simd_bytes_f32);
        input = (float*) ((uintptr_t) input + xnn_simd_bytes_f32);
        channels -= xnn_simd_size_f32;
      } else {
        xnn_store_tail_f32(output, vmin, channels);

        channels = 0;
      }
    } while (channels != 0);
  }
}
