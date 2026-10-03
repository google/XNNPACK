// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vunary/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <wasm_simd128.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vabs_ukernel__wasmsimd_u16(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  const v128_t vnonsign_mask = wasm_u16x8_const_splat(UINT16_C(0x7FFF));
  XNN_FORCE_REALIZATION(vnonsign_mask);
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    v128_t vacc0 = wasm_v128_load(i);
    v128_t vacc1 = wasm_v128_load(i + 8);
    i += 16;

    vacc0 = wasm_v128_and(vacc0, vnonsign_mask);
    vacc1 = wasm_v128_and(vacc1, vnonsign_mask);

    wasm_v128_store(o, vacc0);
    wasm_v128_store(o + 8, vacc1);
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    v128_t vacc = wasm_v128_load(i);
    i += 8;
    vacc = wasm_v128_and(vacc, vnonsign_mask);
    wasm_v128_store(o, vacc);
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vacc = wasm_v128_load(i);
    vacc = wasm_v128_and(vacc, vnonsign_mask);
    if (batch & (4 * sizeof(uint16_t))) {
      wasm_v128_store64_lane(o, vacc, 0);
      vacc = wasm_i64x2_shuffle(vacc, vacc, 1, 1);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      wasm_v128_store32_lane(o, vacc, 0);
      vacc = wasm_i64x2_shr(vacc, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      wasm_v128_store16_lane(o, vacc, 0);
    }
  }
}
