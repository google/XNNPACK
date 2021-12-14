// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/vunary.h>


void xnn_s8_vclamp_ukernel__wasmsimd_x64(
    size_t n,
    const int8_t* x,
    int8_t* y,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);

  const v128_t voutput_max = wasm_v128_load64_splat(params->wasmsimd.max);
  const v128_t voutput_min = wasm_v128_load64_splat(params->wasmsimd.min);
  for (; n >= 64; n -= 64) {
    v128_t vacc0 = wasm_v128_load(x);
    v128_t vacc1 = wasm_v128_load(x + 16);
    v128_t vacc2 = wasm_v128_load(x + 32);
    v128_t vacc3 = wasm_v128_load(x + 48);
    x += 64;

    vacc0 = wasm_i8x16_max(vacc0, voutput_min);
    vacc1 = wasm_i8x16_max(vacc1, voutput_min);
    vacc2 = wasm_i8x16_max(vacc2, voutput_min);
    vacc3 = wasm_i8x16_max(vacc3, voutput_min);

    vacc0 = wasm_i8x16_min(vacc0, voutput_max);
    vacc1 = wasm_i8x16_min(vacc1, voutput_max);
    vacc2 = wasm_i8x16_min(vacc2, voutput_max);
    vacc3 = wasm_i8x16_min(vacc3, voutput_max);

    wasm_v128_store(y, vacc0);
    wasm_v128_store(y + 16, vacc1);
    wasm_v128_store(y + 32, vacc2);
    wasm_v128_store(y + 48, vacc3);
    y += 64;
  }
  for (; n >= 16; n -= 16) {
    v128_t vacc = wasm_v128_load(x);
    x += 16;

    vacc = wasm_i8x16_min(vacc, voutput_max);
    vacc = wasm_i8x16_max(vacc, voutput_min);

    wasm_v128_store(y, vacc);
    y += 16;
  }
  if XNN_UNLIKELY(n != 0) {
    v128_t vacc = wasm_v128_load(x);

    vacc = wasm_i8x16_min(vacc, voutput_max);
    vacc = wasm_i8x16_max(vacc, voutput_min);

    if (n & 8) {
      *((double*) y) = wasm_f64x2_extract_lane(vacc, 0);
      y += 8;
      vacc = wasm_v64x2_shuffle(vacc, vacc, 1, 1);
    }
    if (n & 4) {
      *((float*) y) = wasm_f32x4_extract_lane(vacc, 0);
      y += 4;
      vacc = wasm_u64x2_shr(vacc, 32);
    }
    uint32_t vacc_lo = wasm_i32x4_extract_lane(vacc, 0);
    if (n & 2) {
      *((uint16_t*) y) = (uint16_t) vacc_lo;
      vacc_lo >>= 16;
      y += 2;
    }
    if (n & 1) {
      *y = (uint8_t) vacc_lo;
    }
  }
}
