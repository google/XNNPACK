// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-rdminmax/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"

#include "src/xnnpack/simd/f16-scalar.h"


void xnn_f16_rdmin_ukernel_2p2x__scalar_c2(
    size_t rows,
    size_t channels,
    const xnn_float16* input,
    size_t input_stride,
    const xnn_float16* zero,
    xnn_float16* output,
    const void* params)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 2 * input_stride;
  for (; channels >= 2; channels -= 2) {
    const xnn_float16* i0 = input;
    const xnn_float16* i1 = (const xnn_float16*) ((uintptr_t) input + 1 * input_stride);

    xnn_simd_f16_t vmin0 = xnn_loadu_f16(output);
    xnn_simd_f16_t vmin1 = xnn_loadu_f16((xnn_float16*)((uintptr_t) output + 1 * sizeof(xnn_float16)));

    for (int r = rows; r > 0; r -= 2) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = i0;
      }
      xnn_simd_f16_t vin_0_0 = xnn_loadu_f16(&i0[0]);
      xnn_simd_f16_t vin_0_1 = xnn_loadu_f16(&i0[1]);
      xnn_simd_f16_t vin_1_0 = xnn_loadu_f16(&i1[0]);
      xnn_simd_f16_t vin_1_1 = xnn_loadu_f16(&i1[1]);
      vmin0 = xnn_min_f16(vmin0, vin_0_0);
      vmin1 = xnn_min_f16(vmin1, vin_0_1);
      vmin0 = xnn_min_f16(vmin0, vin_1_0);
      vmin1 = xnn_min_f16(vmin1, vin_1_1);

      i0 = (xnn_float16*) ((uintptr_t) i0 + input_increment);
      i1 = (xnn_float16*) ((uintptr_t) i1 + input_increment);
    }

    xnn_storeu_f16(output, vmin0);
    output = (xnn_float16*) ((uintptr_t) output + xnn_simd_bytes_f16);
    xnn_storeu_f16(output, vmin1);
    output = (xnn_float16*) ((uintptr_t) output + xnn_simd_bytes_f16);

    input = (xnn_float16*) ((uintptr_t) input + 2 * sizeof(xnn_float16));
  }
  if (channels != 0) {
    input_increment = 2 * input_stride;
    do {
      const xnn_float16* i0 = input;
      const xnn_float16* i1 = (const xnn_float16*) ((uintptr_t) input + 1 * input_stride);

      xnn_simd_f16_t vmin;

      if (channels >= xnn_simd_size_f16) {
        vmin = xnn_loadu_f16(output);
      } else {
        vmin = xnn_load_tail_safe_f16(output, channels);
      }

      for (int r = rows; r > 0; r -= 2) {
        if XNN_UNPREDICTABLE(r < 2) {
          i1 = i0;
        }
        xnn_simd_f16_t vin0 = xnn_loadu_f16(&i0[0]);
        xnn_simd_f16_t vin1 = xnn_loadu_f16(&i1[0]);
        vmin = xnn_min_f16(vmin, vin0);
        vmin = xnn_min_f16(vmin, vin1);
        i0 = (xnn_float16*) ((uintptr_t) i0 + input_increment);
        i1 = (xnn_float16*) ((uintptr_t) i1 + input_increment);
      }

      if (channels >= xnn_simd_size_f16) {
        xnn_storeu_f16(output, vmin);
        output = (xnn_float16*) ((uintptr_t) output + xnn_simd_bytes_f16);
        input = (xnn_float16*) ((uintptr_t) input + xnn_simd_bytes_f16);
        channels -= xnn_simd_size_f16;
      } else {
        xnn_store_tail_f16(output, vmin, channels);

        channels = 0;
      }
    } while (channels != 0);
  }
}
