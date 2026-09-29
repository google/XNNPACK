// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-rminmax/wasmrelaxedsimdfp16.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"


void xnn_f16_rmax_ukernel__wasmrelaxedsimdfp16_u16_acc2(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f16_t vmax0 = xnn_set1_f16(output[0]);
  xnn_simd_f16_t vmax1 = vmax0;

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const xnn_simd_f16_t vt0 = xnn_loadu_f16(input + 0);
    const xnn_simd_f16_t vt1 = xnn_loadu_f16(input + 8);
    input += 16;

    vmax0 = xnn_max_f16(vmax0, vt0);
    vmax1 = xnn_max_f16(vmax1, vt1);
  }
  vmax0 = xnn_max_f16(vmax0, vmax1);

  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const xnn_simd_f16_t vt = xnn_loadu_f16(input);
    input += 8;
    vmax0 = xnn_max_f16(vmax0, vt);
  }

  vmax0 = xnn_max_f16(vmax0, wasm_i64x2_shuffle(vmax0, vmax0, 1, 1));
  vmax0 = xnn_max_f16(vmax0, wasm_i32x4_shuffle(vmax0, vmax0, 1, 0, 3, 2));
  vmax0 = xnn_max_f16(vmax0, wasm_i16x8_shuffle(vmax0, vmax0, 1, 0, 3, 2, 5, 4, 7, 6));
  int16_t vmax = math_signcomplement_f16((uint16_t) wasm_i16x8_extract_lane(vmax0, 0));

  if XNN_UNLIKELY(batch != 0) {
    const uint16_t* tail = (const uint16_t*) input;
    do {
      const int16_t vt = math_signcomplement_f16(*tail++);
      vmax = math_max_s16(vmax, vt);
      batch -= sizeof(uint16_t);
    } while (batch != 0);
  }

  *((uint16_t*) output) = (uint16_t) math_signcomplement_f16((uint16_t) vmax);
}
