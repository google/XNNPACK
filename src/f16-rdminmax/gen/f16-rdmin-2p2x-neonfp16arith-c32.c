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

#include "src/xnnpack/simd/f16-neonfp16arith.h"


void xnn_f16_rdmin_ukernel_2p2x__neonfp16arith_c32(
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
  for (; channels >= 32; channels -= 32) {
    const xnn_float16* i0 = input;
    const xnn_float16* i1 = (const xnn_float16*) ((uintptr_t) input + 1 * input_stride);

    xnn_simd_f16_t vmin0 = xnn_loadu_f16(output);
    xnn_simd_f16_t vmin8 = xnn_loadu_f16((xnn_float16*)((uintptr_t) output + 8 * sizeof(xnn_float16)));
    xnn_simd_f16_t vmin16 = xnn_loadu_f16((xnn_float16*)((uintptr_t) output + 16 * sizeof(xnn_float16)));
    xnn_simd_f16_t vmin24 = xnn_loadu_f16((xnn_float16*)((uintptr_t) output + 24 * sizeof(xnn_float16)));

    for (int r = rows; r > 0; r -= 2) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = i0;
      }
      xnn_simd_f16_t vin_0_0 = xnn_loadu_f16(&i0[0]);
      xnn_simd_f16_t vin_0_8 = xnn_loadu_f16(&i0[8]);
      xnn_simd_f16_t vin_0_16 = xnn_loadu_f16(&i0[16]);
      xnn_simd_f16_t vin_0_24 = xnn_loadu_f16(&i0[24]);
      xnn_simd_f16_t vin_1_0 = xnn_loadu_f16(&i1[0]);
      xnn_simd_f16_t vin_1_8 = xnn_loadu_f16(&i1[8]);
      xnn_simd_f16_t vin_1_16 = xnn_loadu_f16(&i1[16]);
      xnn_simd_f16_t vin_1_24 = xnn_loadu_f16(&i1[24]);
      vmin0 = xnn_min_f16(vmin0, vin_0_0);
      vmin8 = xnn_min_f16(vmin8, vin_0_8);
      vmin16 = xnn_min_f16(vmin16, vin_0_16);
      vmin24 = xnn_min_f16(vmin24, vin_0_24);
      vmin0 = xnn_min_f16(vmin0, vin_1_0);
      vmin8 = xnn_min_f16(vmin8, vin_1_8);
      vmin16 = xnn_min_f16(vmin16, vin_1_16);
      vmin24 = xnn_min_f16(vmin24, vin_1_24);

      i0 = (xnn_float16*) ((uintptr_t) i0 + input_increment);
      i1 = (xnn_float16*) ((uintptr_t) i1 + input_increment);
    }

    xnn_storeu_f16(output, vmin0);
    output = (xnn_float16*) ((uintptr_t) output + xnn_simd_bytes_f16);
    xnn_storeu_f16(output, vmin8);
    output = (xnn_float16*) ((uintptr_t) output + xnn_simd_bytes_f16);
    xnn_storeu_f16(output, vmin16);
    output = (xnn_float16*) ((uintptr_t) output + xnn_simd_bytes_f16);
    xnn_storeu_f16(output, vmin24);
    output = (xnn_float16*) ((uintptr_t) output + xnn_simd_bytes_f16);

    input = (xnn_float16*) ((uintptr_t) input + 32 * sizeof(xnn_float16));
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
