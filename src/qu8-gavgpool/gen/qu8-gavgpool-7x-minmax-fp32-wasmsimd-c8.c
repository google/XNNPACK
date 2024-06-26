// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/unipass-wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gavgpool.h"


void xnn_qu8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c8(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const v128_t vinit_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.init_bias);
  const v128_t vscale = wasm_v128_load64_splat(params->fp32_wasmsimd.scale);
  const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
  const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
  const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
  const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
  for (; channels >= 8; channels -= 8) {
    const v128_t vxi0x01234567 = wasm_u16x8_load8x8(i0);
    i0 += 8;
    const v128_t vxi1x01234567 = wasm_u16x8_load8x8(i1);
    i1 += 8;

    v128_t vacc01234567 = wasm_i16x8_add(vxi0x01234567, vxi1x01234567);
    const v128_t vxi2x01234567 = wasm_u16x8_load8x8(i2);
    i2 += 8;

    vacc01234567 = wasm_i16x8_add(vacc01234567, vxi2x01234567);
    const v128_t vxi3x01234567 = wasm_u16x8_load8x8(i3);
    i3 += 8;
    vacc01234567 = wasm_i16x8_add(vacc01234567, vxi3x01234567);
    const v128_t vxi4x01234567 = wasm_u16x8_load8x8(i4);
    i4 += 8;
    vacc01234567 = wasm_i16x8_add(vacc01234567, vxi4x01234567);
    const v128_t vxi5x01234567 = wasm_u16x8_load8x8(i5);
    i5 += 8;
    vacc01234567 = wasm_i16x8_add(vacc01234567, vxi5x01234567);
    const v128_t vxi6x01234567 = wasm_u16x8_load8x8(i6);
    i6 += 8;

    vacc01234567 = wasm_i16x8_add(vacc01234567, vxi6x01234567);

    v128_t vacc0123 = wasm_i32x4_add(vinit_bias, wasm_u32x4_extend_low_u16x8(vacc01234567));
    v128_t vacc4567 = wasm_i32x4_add(vinit_bias, wasm_u32x4_extend_high_u16x8(vacc01234567));

    vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
    vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);

    vacc0123 = wasm_f32x4_mul(vacc0123, vscale);
    vacc4567 = wasm_f32x4_mul(vacc4567, vscale);

    vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
    vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);

    vacc0123 = wasm_i32x4_max(vacc0123, vmagic_min);
    vacc4567 = wasm_i32x4_max(vacc4567, vmagic_min);

    vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
    vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);

    v128_t vout01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);

    v128_t vout0123456701234567 = wasm_u8x16_narrow_i16x8(vout01234567, vout01234567);

    vout0123456701234567 = wasm_u8x16_min(vout0123456701234567, voutput_max);

    wasm_v128_store64_lane(output, vout0123456701234567, 0);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const v128_t vxi0x01234567 = wasm_u16x8_load8x8(i0);
      i0 += 8;
      const v128_t vxi1x01234567 = wasm_u16x8_load8x8(i1);
      i1 += 8;

      v128_t vacc01234567 = wasm_i16x8_add(vxi0x01234567, vxi1x01234567);
      const v128_t vxi2x01234567 = wasm_u16x8_load8x8(i2);
      i2 += 8;

      vacc01234567 = wasm_i16x8_add(vacc01234567, vxi2x01234567);
      const v128_t vxi3x01234567 = wasm_u16x8_load8x8(i3);
      i3 += 8;
      vacc01234567 = wasm_i16x8_add(vacc01234567, vxi3x01234567);
      const v128_t vxi4x01234567 = wasm_u16x8_load8x8(i4);
      i4 += 8;
      vacc01234567 = wasm_i16x8_add(vacc01234567, vxi4x01234567);
      const v128_t vxi5x01234567 = wasm_u16x8_load8x8(i5);
      i5 += 8;
      vacc01234567 = wasm_i16x8_add(vacc01234567, vxi5x01234567);
      const v128_t vxi6x01234567 = wasm_u16x8_load8x8(i6);
      i6 += 8;

      vacc01234567 = wasm_i16x8_add(vacc01234567, vxi6x01234567);

      v128_t vacc0123 = wasm_i32x4_add(vinit_bias, wasm_u32x4_extend_low_u16x8(vacc01234567));
      v128_t vacc4567 = wasm_i32x4_add(vinit_bias, wasm_u32x4_extend_high_u16x8(vacc01234567));

      vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
      vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);

      vacc0123 = wasm_f32x4_mul(vacc0123, vscale);
      vacc4567 = wasm_f32x4_mul(vacc4567, vscale);

      vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
      vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);

      vacc0123 = wasm_i32x4_max(vacc0123, vmagic_min);
      vacc4567 = wasm_i32x4_max(vacc4567, vmagic_min);

      vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
      vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);

      const v128_t vout01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
      v128_t vout0123456701234567 = wasm_u8x16_narrow_i16x8(vout01234567, vout01234567);
      vout0123456701234567 = wasm_u8x16_min(vout0123456701234567, voutput_max);

      if (channels & 4) {
        wasm_v128_store32_lane(output, vout0123456701234567, 0);
        vout0123456701234567 = wasm_u64x2_shr(vout0123456701234567, 32);
        output += 4;
      }
      if (channels & 2) {
        wasm_v128_store16_lane(output, vout0123456701234567, 0);
        vout0123456701234567 = wasm_u32x4_shr(vout0123456701234567, 16);
        output += 2;
      }
      if (channels & 1) {
        wasm_v128_store8_lane(output, vout0123456701234567, 0);
      }
    }
  }
}
