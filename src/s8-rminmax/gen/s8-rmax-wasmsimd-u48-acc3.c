// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/s8-rminmax/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"

#include "src/xnnpack/simd/s8-wasmsimd.h"

void xnn_s8_rmax_ukernel__wasmsimd_u48_acc3(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_s8 == 16);

  xnn_simd_s8_t vmax0 = xnn_set1_s8(output[0]);
  xnn_simd_s8_t vmax1 = vmax0;
  xnn_simd_s8_t vmax2 = vmax0;

  for (; batch >= 48 * sizeof(int8_t); batch -= 48 * sizeof(int8_t)) {
    xnn_simd_s8_t vt0 = xnn_loadu_s8(input + 0 * xnn_simd_size_s8);
    xnn_simd_s8_t vt1 = xnn_loadu_s8(input + 1 * xnn_simd_size_s8);
    xnn_simd_s8_t vt2 = xnn_loadu_s8(input + 2 * xnn_simd_size_s8);
    input += 3 * xnn_simd_size_s8;

    vmax0 = xnn_max_s8(vmax0, vt0);
    vmax1 = xnn_max_s8(vmax1, vt1);
    vmax2 = xnn_max_s8(vmax2, vt2);
  }
  vmax0 = xnn_max_s8(vmax0, vmax1);
  vmax0 = xnn_max_s8(vmax0, vmax2);

  for (; batch >= xnn_simd_bytes_s8; batch -= xnn_simd_bytes_s8) {
    xnn_simd_s8_t vt = xnn_loadu_s8(input);
    input += xnn_simd_size_s8;

    vmax0 = xnn_max_s8(vmax0, vt);
  }

  int8_t max0 = xnn_reduce_max_s8(vmax0);

  if XNN_UNLIKELY(batch != 0) {
    do {
      const int8_t vt = *input++;

      max0 = vt > max0 ? vt : max0;
    } while (--batch != 0);
  }

  *output = max0;
}
