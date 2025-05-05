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

#include "src/xnnpack/simd/u8-neon.h"

void xnn_u8_rminmax_ukernel__neon_u16(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_u8 == 16);

  xnn_simd_u8_t vmin0 = xnn_set1_u8(output[0]);
  xnn_simd_u8_t vmax0 = xnn_set1_u8(output[1]);


  for (; batch >= xnn_simd_bytes_u8; batch -= xnn_simd_bytes_u8) {
    xnn_simd_u8_t vt = xnn_loadu_u8(input);
    input += xnn_simd_size_u8;

    vmin0 = xnn_min_u8(vmin0, vt);
    vmax0 = xnn_max_u8(vmax0, vt);
  }

  uint8_t min0 = xnn_reduce_min_u8(vmin0);
  uint8_t max0 = xnn_reduce_max_u8(vmax0);

  if XNN_UNLIKELY(batch != 0) {
    do {
      const uint8_t vt = *input++;

      min0 = vt > min0 ? min0 : vt;
      max0 = vt > max0 ? vt : max0;
    } while (--batch != 0);
  }

  *output = min0;
  output += sizeof(uint8_t);
  *output = max0;
}
