// Auto-generated file. Do not edit!
//   Template: src/s8-ibilinear/wasmsimd-dot16x2.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"


void xnn_s8_ibilinear_ukernel__wasmsimd_dot16x2_c16(
    size_t output_pixels,
    size_t channels,
    const int8_t** restrict input,
    size_t input_offset,
    const int16_t* restrict weights,
    int8_t* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const int8_t* i0 = (const int8_t*) ((uintptr_t) input[0] + input_offset);
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input[1] + input_offset);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input[2] + input_offset);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const v128_t valphah =
      wasm_i16x8_add(
        wasm_v128_xor(
          wasm_v128_load16_splat(weights),
          wasm_i32x4_const_splat(0xFFFF0000)),
        wasm_i32x4_const_splat(0x08010000));
    const v128_t valphav = wasm_i32x4_extend_low_i16x8(wasm_v128_load16_splat(weights + 1));
    weights += 2;

    const v128_t vrounding = wasm_i32x4_const_splat(0x00200000);

    size_t c = channels;
    for (; c >= 16 * sizeof(int8_t); c -= 16 * sizeof(int8_t)) {
      const v128_t vtl01234567 = wasm_i16x8_load8x8(i0);
      const v128_t vtr01234567 = wasm_i16x8_load8x8(i1);
      const v128_t vbl01234567 = wasm_i16x8_load8x8(i2);
      const v128_t vbr01234567 = wasm_i16x8_load8x8(i3);
      const v128_t vtl89ABCDEF = wasm_i16x8_load8x8(i0 + 8);
      const v128_t vtr89ABCDEF = wasm_i16x8_load8x8(i1 + 8);
      const v128_t vbl89ABCDEF = wasm_i16x8_load8x8(i2 + 8);
      const v128_t vbr89ABCDEF = wasm_i16x8_load8x8(i3 + 8);
      i0 += 16;
      i1 += 16;
      i2 += 16;
      i3 += 16;

      const v128_t vdr01234567 = wasm_i16x8_sub(vbr01234567, vtr01234567);
      const v128_t vt0123 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vtr01234567, vtl01234567, 0, 8, 1, 9, 2, 10, 3, 11), valphah);
      const v128_t vdl01234567 = wasm_i16x8_sub(vbl01234567, vtl01234567);
      const v128_t vt4567 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vtr01234567, vtl01234567, 4, 12, 5, 13, 6, 14, 7, 15), valphah);
      const v128_t vdr89ABCDEF = wasm_i16x8_sub(vbr89ABCDEF, vtr89ABCDEF);
      const v128_t vt89AB = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vtr89ABCDEF, vtl89ABCDEF, 0, 8, 1, 9, 2, 10, 3, 11), valphah);
      const v128_t vdl89ABCDEF = wasm_i16x8_sub(vbl89ABCDEF, vtl89ABCDEF);
      const v128_t vtCDEF = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vtr89ABCDEF, vtl89ABCDEF, 4, 12, 5, 13, 6, 14, 7, 15), valphah);

      const v128_t vd0123 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vdr01234567, vdl01234567, 0, 8, 1, 9, 2, 10, 3, 11), valphah);
      const v128_t vd4567 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vdr01234567, vdl01234567, 4, 12, 5, 13, 6, 14, 7, 15), valphah);
      const v128_t vd89AB = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vdr89ABCDEF, vdl89ABCDEF, 0, 8, 1, 9, 2, 10, 3, 11), valphah);
      const v128_t vdCDEF = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vdr89ABCDEF, vdl89ABCDEF, 4, 12, 5, 13, 6, 14, 7, 15), valphah);

      v128_t vacc0123 = wasm_i32x4_mul(vd0123, valphav);
      v128_t vacc4567 = wasm_i32x4_mul(vd4567, valphav);
      v128_t vacc89AB = wasm_i32x4_mul(vd89AB, valphav);
      v128_t vaccCDEF = wasm_i32x4_mul(vdCDEF, valphav);

      vacc0123 = wasm_i32x4_add(wasm_i32x4_shl(vt0123, 11), vacc0123);
      vacc4567 = wasm_i32x4_add(wasm_i32x4_shl(vt4567, 11), vacc4567);
      vacc89AB = wasm_i32x4_add(wasm_i32x4_shl(vt89AB, 11), vacc89AB);
      vaccCDEF = wasm_i32x4_add(wasm_i32x4_shl(vtCDEF, 11), vaccCDEF);

      vacc0123 = wasm_i32x4_shr(wasm_i16x8_add(vacc0123, vrounding), 22);
      vacc4567 = wasm_i32x4_shr(wasm_i16x8_add(vacc4567, vrounding), 22);
      vacc89AB = wasm_i32x4_shr(wasm_i16x8_add(vacc89AB, vrounding), 22);
      vaccCDEF = wasm_i32x4_shr(wasm_i16x8_add(vaccCDEF, vrounding), 22);

      const v128_t vacc01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
      const v128_t vacc89ABCDEF = wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF);

      const v128_t vo0123456789ABCDEF = wasm_i8x16_narrow_i16x8(vacc01234567, vacc89ABCDEF);

      wasm_v128_store(output, vo0123456789ABCDEF);
      output += 16;
    }
    for (; c >= 8 * sizeof(int8_t); c -= 8 * sizeof(int8_t)) {
      const v128_t vtl01234567 = wasm_i16x8_load8x8(i0);
      i0 += 8;
      const v128_t vtr01234567 = wasm_i16x8_load8x8(i1);
      i1 += 8;
      const v128_t vbl01234567 = wasm_i16x8_load8x8(i2);
      i2 += 8;
      const v128_t vbr01234567 = wasm_i16x8_load8x8(i3);
      i3 += 8;

      const v128_t vdr01234567 = wasm_i16x8_sub(vbr01234567, vtr01234567);
      const v128_t vt0123 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vtr01234567, vtl01234567, 0, 8, 1, 9, 2, 10, 3, 11), valphah);
      const v128_t vdl01234567 = wasm_i16x8_sub(vbl01234567, vtl01234567);
      const v128_t vt4567 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vtr01234567, vtl01234567, 4, 12, 5, 13, 6, 14, 7, 15), valphah);

      const v128_t vd0123 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vdr01234567, vdl01234567, 0, 8, 1, 9, 2, 10, 3, 11), valphah);
      const v128_t vd4567 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vdr01234567, vdl01234567, 4, 12, 5, 13, 6, 14, 7, 15), valphah);

      v128_t vacc0123 = wasm_i32x4_mul(vd0123, valphav);
      v128_t vacc4567 = wasm_i32x4_mul(vd4567, valphav);

      vacc0123 = wasm_i32x4_add(wasm_i32x4_shl(vt0123, 11), vacc0123);
      vacc4567 = wasm_i32x4_add(wasm_i32x4_shl(vt4567, 11), vacc4567);

      vacc0123 = wasm_i32x4_shr(wasm_i16x8_add(vacc0123, vrounding), 22);
      vacc4567 = wasm_i32x4_shr(wasm_i16x8_add(vacc4567, vrounding), 22);

      const v128_t vacc01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);

      const v128_t vo01234567 = wasm_i8x16_narrow_i16x8(vacc01234567, vacc01234567);

      wasm_v128_store64_lane(output, vo01234567, 0);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const v128_t vtl01234567 = wasm_i16x8_load8x8(i0);
      const v128_t vtr01234567 = wasm_i16x8_load8x8(i1);
      const v128_t vbl01234567 = wasm_i16x8_load8x8(i2);
      const v128_t vbr01234567 = wasm_i16x8_load8x8(i3);

      const v128_t vdr01234567 = wasm_i16x8_sub(vbr01234567, vtr01234567);
      const v128_t vt0123 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vtr01234567, vtl01234567, 0, 8, 1, 9, 2, 10, 3, 11), valphah);
      const v128_t vdl01234567 = wasm_i16x8_sub(vbl01234567, vtl01234567);
      const v128_t vt4567 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vtr01234567, vtl01234567, 4, 12, 5, 13, 6, 14, 7, 15), valphah);

      const v128_t vd0123 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vdr01234567, vdl01234567, 0, 8, 1, 9, 2, 10, 3, 11), valphah);
      const v128_t vd4567 = wasm_i32x4_dot_i16x8(wasm_v16x8_shuffle(vdr01234567, vdl01234567, 4, 12, 5, 13, 6, 14, 7, 15), valphah);

      v128_t vacc0123 = wasm_i32x4_mul(vd0123, valphav);
      v128_t vacc4567 = wasm_i32x4_mul(vd4567, valphav);

      vacc0123 = wasm_i32x4_add(wasm_i32x4_shl(vt0123, 11), vacc0123);
      vacc4567 = wasm_i32x4_add(wasm_i32x4_shl(vt4567, 11), vacc4567);

      vacc0123 = wasm_i32x4_shr(wasm_i16x8_add(vacc0123, vrounding), 22);
      vacc4567 = wasm_i32x4_shr(wasm_i16x8_add(vacc4567, vrounding), 22);

      const v128_t vacc01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);

      v128_t vo01234567 = wasm_i8x16_narrow_i16x8(vacc01234567, vacc01234567);

      if (c & (4 * sizeof(int8_t))) {
        wasm_v128_store32_lane(output, vo01234567, 0);
        vo01234567 = wasm_u64x2_shr(vo01234567, 32);
        output += 4;
      }
      if (c & (2 * sizeof(int8_t))) {
        wasm_v128_store16_lane(output, vo01234567, 0);
        vo01234567 = wasm_u32x4_shr(vo01234567, 16);
        output += 2;
      }
      if (c & (1 * sizeof(int8_t))) {
        wasm_v128_store8_lane(output, vo01234567, 0);
        output += 1;
      }
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
