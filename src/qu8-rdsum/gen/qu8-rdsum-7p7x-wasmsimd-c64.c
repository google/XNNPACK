// Auto-generated file. Do not edit!
//   Template: src/qs8-rdsum/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <math.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"

void xnn_qu8_rdsum_ukernel_7p7x__wasmsimd_c64(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint32_t* output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 64; channels -= 64) {
    const uint8_t* i0 = input;
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input + 1 * input_stride);
    const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input + 2 * input_stride);
    const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input + 3 * input_stride);
    const uint8_t* i4 = (const uint8_t*) ((uintptr_t) input + 4 * input_stride);
    const uint8_t* i5 = (const uint8_t*) ((uintptr_t) input + 5 * input_stride);
    const uint8_t* i6 = (const uint8_t*) ((uintptr_t) input + 6 * input_stride);

    v128_t vacc0 = wasm_i32x4_const_splat(0);
    v128_t vacc4 = wasm_i32x4_const_splat(0);
    v128_t vacc8 = wasm_i32x4_const_splat(0);
    v128_t vacc12 = wasm_i32x4_const_splat(0);
    v128_t vacc16 = wasm_i32x4_const_splat(0);
    v128_t vacc20 = wasm_i32x4_const_splat(0);
    v128_t vacc24 = wasm_i32x4_const_splat(0);
    v128_t vacc28 = wasm_i32x4_const_splat(0);
    v128_t vacc32 = wasm_i32x4_const_splat(0);
    v128_t vacc36 = wasm_i32x4_const_splat(0);
    v128_t vacc40 = wasm_i32x4_const_splat(0);
    v128_t vacc44 = wasm_i32x4_const_splat(0);
    v128_t vacc48 = wasm_i32x4_const_splat(0);
    v128_t vacc52 = wasm_i32x4_const_splat(0);
    v128_t vacc56 = wasm_i32x4_const_splat(0);
    v128_t vacc60 = wasm_i32x4_const_splat(0);

    // 256 int8s may be summed into an int16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int r = rows;
    while (r > 0) {
      v128_t vacc16_0 = wasm_i16x8_const_splat(0);
      v128_t vacc16_8 = wasm_i16x8_const_splat(0);
      v128_t vacc16_16 = wasm_i16x8_const_splat(0);
      v128_t vacc16_24 = wasm_i16x8_const_splat(0);
      v128_t vacc16_32 = wasm_i16x8_const_splat(0);
      v128_t vacc16_40 = wasm_i16x8_const_splat(0);
      v128_t vacc16_48 = wasm_i16x8_const_splat(0);
      v128_t vacc16_56 = wasm_i16x8_const_splat(0);
      for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
        if XNN_UNPREDICTABLE(current_batch < 2) {
          i1 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 2) {
          i2 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 4) {
          i3 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 4) {
          i4 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 6) {
          i5 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 6) {
          i6 = zero;
        }
        v128_t vin0;
        v128_t vin8;
        v128_t vin16;
        v128_t vin24;
        v128_t vin32;
        v128_t vin40;
        v128_t vin48;
        v128_t vin56;
        vin0 = wasm_u16x8_load8x8(&i0[0]);
        vin8 = wasm_u16x8_load8x8(&i0[8]);
        vin16 = wasm_u16x8_load8x8(&i0[16]);
        vin24 = wasm_u16x8_load8x8(&i0[24]);
        vin32 = wasm_u16x8_load8x8(&i0[32]);
        vin40 = wasm_u16x8_load8x8(&i0[40]);
        vin48 = wasm_u16x8_load8x8(&i0[48]);
        vin56 = wasm_u16x8_load8x8(&i0[56]);
        vacc16_0 = wasm_i16x8_add(vacc16_0, vin0);
        vacc16_8 = wasm_i16x8_add(vacc16_8, vin8);
        vacc16_16 = wasm_i16x8_add(vacc16_16, vin16);
        vacc16_24 = wasm_i16x8_add(vacc16_24, vin24);
        vacc16_32 = wasm_i16x8_add(vacc16_32, vin32);
        vacc16_40 = wasm_i16x8_add(vacc16_40, vin40);
        vacc16_48 = wasm_i16x8_add(vacc16_48, vin48);
        vacc16_56 = wasm_i16x8_add(vacc16_56, vin56);
        vin0 = wasm_u16x8_load8x8(&i1[0]);
        vin8 = wasm_u16x8_load8x8(&i1[8]);
        vin16 = wasm_u16x8_load8x8(&i1[16]);
        vin24 = wasm_u16x8_load8x8(&i1[24]);
        vin32 = wasm_u16x8_load8x8(&i1[32]);
        vin40 = wasm_u16x8_load8x8(&i1[40]);
        vin48 = wasm_u16x8_load8x8(&i1[48]);
        vin56 = wasm_u16x8_load8x8(&i1[56]);
        vacc16_0 = wasm_i16x8_add(vacc16_0, vin0);
        vacc16_8 = wasm_i16x8_add(vacc16_8, vin8);
        vacc16_16 = wasm_i16x8_add(vacc16_16, vin16);
        vacc16_24 = wasm_i16x8_add(vacc16_24, vin24);
        vacc16_32 = wasm_i16x8_add(vacc16_32, vin32);
        vacc16_40 = wasm_i16x8_add(vacc16_40, vin40);
        vacc16_48 = wasm_i16x8_add(vacc16_48, vin48);
        vacc16_56 = wasm_i16x8_add(vacc16_56, vin56);
        vin0 = wasm_u16x8_load8x8(&i2[0]);
        vin8 = wasm_u16x8_load8x8(&i2[8]);
        vin16 = wasm_u16x8_load8x8(&i2[16]);
        vin24 = wasm_u16x8_load8x8(&i2[24]);
        vin32 = wasm_u16x8_load8x8(&i2[32]);
        vin40 = wasm_u16x8_load8x8(&i2[40]);
        vin48 = wasm_u16x8_load8x8(&i2[48]);
        vin56 = wasm_u16x8_load8x8(&i2[56]);
        vacc16_0 = wasm_i16x8_add(vacc16_0, vin0);
        vacc16_8 = wasm_i16x8_add(vacc16_8, vin8);
        vacc16_16 = wasm_i16x8_add(vacc16_16, vin16);
        vacc16_24 = wasm_i16x8_add(vacc16_24, vin24);
        vacc16_32 = wasm_i16x8_add(vacc16_32, vin32);
        vacc16_40 = wasm_i16x8_add(vacc16_40, vin40);
        vacc16_48 = wasm_i16x8_add(vacc16_48, vin48);
        vacc16_56 = wasm_i16x8_add(vacc16_56, vin56);
        vin0 = wasm_u16x8_load8x8(&i3[0]);
        vin8 = wasm_u16x8_load8x8(&i3[8]);
        vin16 = wasm_u16x8_load8x8(&i3[16]);
        vin24 = wasm_u16x8_load8x8(&i3[24]);
        vin32 = wasm_u16x8_load8x8(&i3[32]);
        vin40 = wasm_u16x8_load8x8(&i3[40]);
        vin48 = wasm_u16x8_load8x8(&i3[48]);
        vin56 = wasm_u16x8_load8x8(&i3[56]);
        vacc16_0 = wasm_i16x8_add(vacc16_0, vin0);
        vacc16_8 = wasm_i16x8_add(vacc16_8, vin8);
        vacc16_16 = wasm_i16x8_add(vacc16_16, vin16);
        vacc16_24 = wasm_i16x8_add(vacc16_24, vin24);
        vacc16_32 = wasm_i16x8_add(vacc16_32, vin32);
        vacc16_40 = wasm_i16x8_add(vacc16_40, vin40);
        vacc16_48 = wasm_i16x8_add(vacc16_48, vin48);
        vacc16_56 = wasm_i16x8_add(vacc16_56, vin56);
        vin0 = wasm_u16x8_load8x8(&i4[0]);
        vin8 = wasm_u16x8_load8x8(&i4[8]);
        vin16 = wasm_u16x8_load8x8(&i4[16]);
        vin24 = wasm_u16x8_load8x8(&i4[24]);
        vin32 = wasm_u16x8_load8x8(&i4[32]);
        vin40 = wasm_u16x8_load8x8(&i4[40]);
        vin48 = wasm_u16x8_load8x8(&i4[48]);
        vin56 = wasm_u16x8_load8x8(&i4[56]);
        vacc16_0 = wasm_i16x8_add(vacc16_0, vin0);
        vacc16_8 = wasm_i16x8_add(vacc16_8, vin8);
        vacc16_16 = wasm_i16x8_add(vacc16_16, vin16);
        vacc16_24 = wasm_i16x8_add(vacc16_24, vin24);
        vacc16_32 = wasm_i16x8_add(vacc16_32, vin32);
        vacc16_40 = wasm_i16x8_add(vacc16_40, vin40);
        vacc16_48 = wasm_i16x8_add(vacc16_48, vin48);
        vacc16_56 = wasm_i16x8_add(vacc16_56, vin56);
        vin0 = wasm_u16x8_load8x8(&i5[0]);
        vin8 = wasm_u16x8_load8x8(&i5[8]);
        vin16 = wasm_u16x8_load8x8(&i5[16]);
        vin24 = wasm_u16x8_load8x8(&i5[24]);
        vin32 = wasm_u16x8_load8x8(&i5[32]);
        vin40 = wasm_u16x8_load8x8(&i5[40]);
        vin48 = wasm_u16x8_load8x8(&i5[48]);
        vin56 = wasm_u16x8_load8x8(&i5[56]);
        vacc16_0 = wasm_i16x8_add(vacc16_0, vin0);
        vacc16_8 = wasm_i16x8_add(vacc16_8, vin8);
        vacc16_16 = wasm_i16x8_add(vacc16_16, vin16);
        vacc16_24 = wasm_i16x8_add(vacc16_24, vin24);
        vacc16_32 = wasm_i16x8_add(vacc16_32, vin32);
        vacc16_40 = wasm_i16x8_add(vacc16_40, vin40);
        vacc16_48 = wasm_i16x8_add(vacc16_48, vin48);
        vacc16_56 = wasm_i16x8_add(vacc16_56, vin56);
        vin0 = wasm_u16x8_load8x8(&i6[0]);
        vin8 = wasm_u16x8_load8x8(&i6[8]);
        vin16 = wasm_u16x8_load8x8(&i6[16]);
        vin24 = wasm_u16x8_load8x8(&i6[24]);
        vin32 = wasm_u16x8_load8x8(&i6[32]);
        vin40 = wasm_u16x8_load8x8(&i6[40]);
        vin48 = wasm_u16x8_load8x8(&i6[48]);
        vin56 = wasm_u16x8_load8x8(&i6[56]);
        vacc16_0 = wasm_i16x8_add(vacc16_0, vin0);
        vacc16_8 = wasm_i16x8_add(vacc16_8, vin8);
        vacc16_16 = wasm_i16x8_add(vacc16_16, vin16);
        vacc16_24 = wasm_i16x8_add(vacc16_24, vin24);
        vacc16_32 = wasm_i16x8_add(vacc16_32, vin32);
        vacc16_40 = wasm_i16x8_add(vacc16_40, vin40);
        vacc16_48 = wasm_i16x8_add(vacc16_48, vin48);
        vacc16_56 = wasm_i16x8_add(vacc16_56, vin56);
        i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
      }
      vacc0 = wasm_i32x4_add(vacc0, wasm_u32x4_extend_low_u16x8(vacc16_0));
      vacc4 = wasm_i32x4_add(vacc4, wasm_u32x4_extend_high_u16x8(vacc16_0));
      vacc8 = wasm_i32x4_add(vacc8, wasm_u32x4_extend_low_u16x8(vacc16_8));
      vacc12 = wasm_i32x4_add(vacc12, wasm_u32x4_extend_high_u16x8(vacc16_8));
      vacc16 = wasm_i32x4_add(vacc16, wasm_u32x4_extend_low_u16x8(vacc16_16));
      vacc20 = wasm_i32x4_add(vacc20, wasm_u32x4_extend_high_u16x8(vacc16_16));
      vacc24 = wasm_i32x4_add(vacc24, wasm_u32x4_extend_low_u16x8(vacc16_24));
      vacc28 = wasm_i32x4_add(vacc28, wasm_u32x4_extend_high_u16x8(vacc16_24));
      vacc32 = wasm_i32x4_add(vacc32, wasm_u32x4_extend_low_u16x8(vacc16_32));
      vacc36 = wasm_i32x4_add(vacc36, wasm_u32x4_extend_high_u16x8(vacc16_32));
      vacc40 = wasm_i32x4_add(vacc40, wasm_u32x4_extend_low_u16x8(vacc16_40));
      vacc44 = wasm_i32x4_add(vacc44, wasm_u32x4_extend_high_u16x8(vacc16_40));
      vacc48 = wasm_i32x4_add(vacc48, wasm_u32x4_extend_low_u16x8(vacc16_48));
      vacc52 = wasm_i32x4_add(vacc52, wasm_u32x4_extend_high_u16x8(vacc16_48));
      vacc56 = wasm_i32x4_add(vacc56, wasm_u32x4_extend_low_u16x8(vacc16_56));
      vacc60 = wasm_i32x4_add(vacc60, wasm_u32x4_extend_high_u16x8(vacc16_56));
      r = doz(r, 252);
    }

    const uint32_t* o = output;
    v128_t vo0 = wasm_v128_load(o); o += 4;
    v128_t vo4 = wasm_v128_load(o); o += 4;
    v128_t vo8 = wasm_v128_load(o); o += 4;
    v128_t vo12 = wasm_v128_load(o); o += 4;
    v128_t vo16 = wasm_v128_load(o); o += 4;
    v128_t vo20 = wasm_v128_load(o); o += 4;
    v128_t vo24 = wasm_v128_load(o); o += 4;
    v128_t vo28 = wasm_v128_load(o); o += 4;
    v128_t vo32 = wasm_v128_load(o); o += 4;
    v128_t vo36 = wasm_v128_load(o); o += 4;
    v128_t vo40 = wasm_v128_load(o); o += 4;
    v128_t vo44 = wasm_v128_load(o); o += 4;
    v128_t vo48 = wasm_v128_load(o); o += 4;
    v128_t vo52 = wasm_v128_load(o); o += 4;
    v128_t vo56 = wasm_v128_load(o); o += 4;
    v128_t vo60 = wasm_v128_load(o); o += 4;
    vo0 = wasm_i32x4_add(vo0, vacc0);
    vo4 = wasm_i32x4_add(vo4, vacc4);
    vo8 = wasm_i32x4_add(vo8, vacc8);
    vo12 = wasm_i32x4_add(vo12, vacc12);
    vo16 = wasm_i32x4_add(vo16, vacc16);
    vo20 = wasm_i32x4_add(vo20, vacc20);
    vo24 = wasm_i32x4_add(vo24, vacc24);
    vo28 = wasm_i32x4_add(vo28, vacc28);
    vo32 = wasm_i32x4_add(vo32, vacc32);
    vo36 = wasm_i32x4_add(vo36, vacc36);
    vo40 = wasm_i32x4_add(vo40, vacc40);
    vo44 = wasm_i32x4_add(vo44, vacc44);
    vo48 = wasm_i32x4_add(vo48, vacc48);
    vo52 = wasm_i32x4_add(vo52, vacc52);
    vo56 = wasm_i32x4_add(vo56, vacc56);
    vo60 = wasm_i32x4_add(vo60, vacc60);
    wasm_v128_store(output, vo0); output += 4;
    wasm_v128_store(output, vo4); output += 4;
    wasm_v128_store(output, vo8); output += 4;
    wasm_v128_store(output, vo12); output += 4;
    wasm_v128_store(output, vo16); output += 4;
    wasm_v128_store(output, vo20); output += 4;
    wasm_v128_store(output, vo24); output += 4;
    wasm_v128_store(output, vo28); output += 4;
    wasm_v128_store(output, vo32); output += 4;
    wasm_v128_store(output, vo36); output += 4;
    wasm_v128_store(output, vo40); output += 4;
    wasm_v128_store(output, vo44); output += 4;
    wasm_v128_store(output, vo48); output += 4;
    wasm_v128_store(output, vo52); output += 4;
    wasm_v128_store(output, vo56); output += 4;
    wasm_v128_store(output, vo60); output += 4;

    input = (const int8_t*) ((uintptr_t) input + 64 * sizeof(int8_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    // 256 int8s may be summed into an int16 before overflowing.
    do {
      int num_batches = floor((rows + 251) / 252);
      int r = rows;
      const int8_t* i0 = input;
      const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
      const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
      const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
      const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
      const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
      const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

      v128_t vacc0 = wasm_i32x4_const_splat(0);
      v128_t vacc1 = wasm_i32x4_const_splat(0);

      for (; num_batches > 0; --num_batches) {
        v128_t vacc16 = wasm_i16x8_const_splat(0);
        for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
          if XNN_UNPREDICTABLE(current_batch < 2) {
            i1 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 2) {
            i2 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 4) {
            i3 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 4) {
            i4 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 6) {
            i5 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 6) {
            i6 = zero;
          }

          v128_t vin0 = wasm_u16x8_load8x8(&i0[0]);
          v128_t vin1 = wasm_u16x8_load8x8(&i1[0]);
          v128_t vin2 = wasm_u16x8_load8x8(&i2[0]);
          v128_t vin3 = wasm_u16x8_load8x8(&i3[0]);
          v128_t vin4 = wasm_u16x8_load8x8(&i4[0]);
          v128_t vin5 = wasm_u16x8_load8x8(&i5[0]);
          v128_t vin6 = wasm_u16x8_load8x8(&i6[0]);
          vacc16 = wasm_i16x8_add(vacc16, vin0);
          vacc16 = wasm_i16x8_add(vacc16, vin1);
          vacc16 = wasm_i16x8_add(vacc16, vin2);
          vacc16 = wasm_i16x8_add(vacc16, vin3);
          vacc16 = wasm_i16x8_add(vacc16, vin4);
          vacc16 = wasm_i16x8_add(vacc16, vin5);
          vacc16 = wasm_i16x8_add(vacc16, vin6);
          i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
        }
        vacc0 = wasm_i32x4_add(vacc0, wasm_u32x4_extend_low_u16x8(vacc16));
        vacc1 = wasm_i32x4_add(vacc1, wasm_u32x4_extend_high_u16x8(vacc16));
        r = doz(r, 252);
      }

      if XNN_LIKELY(channels >= 8) {
        v128_t vo0 = wasm_v128_load(output);
        v128_t vo1 = wasm_v128_load(output + 4);
        vo0 = wasm_i32x4_add(vo0, vacc0);
        vo1 = wasm_i32x4_add(vo1, vacc1);
        wasm_v128_store(output, vo0); output += 4;
        wasm_v128_store(output, vo1); output += 4;
        channels -= 8;
        input = (const uint8_t*) ((uintptr_t) input + 8 * sizeof(uint8_t));
      } else {
        if (channels & 4) {
          v128_t vo = wasm_v128_load(output);
          vo = wasm_i32x4_add(vo, vacc0);
          wasm_v128_store(output, vo); output += 4;
          vacc0 = vacc1;
        }
        if (channels & 2) {
          v128_t vo = wasm_u32x4_make(output[0], output[1], 0, 0);
          vo = wasm_i32x4_add(vo, vacc0);
          wasm_v128_store64_lane(output, vo, 0); output += 2;
          vacc0 = wasm_v64x2_shuffle(vacc0, vacc0, 1, 1);
        }
        if (channels & 1) {
          *output += wasm_u32x4_extract_lane(vacc0, 0);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
