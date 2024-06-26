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


void xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__wasmsimd_c32(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const v128_t vinit_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.init_bias);
  const v128_t vscale = wasm_v128_load64_splat(params->fp32_wasmsimd.scale);
  const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
  const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
  const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
  const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
  for (; channels >= 32; channels -= 32) {
    const v128_t vxi0x01234567 = wasm_i16x8_load8x8(i0);
    const v128_t vxi0x89ABCDEF = wasm_i16x8_load8x8(i0 + 8);
    const v128_t vxi0xGHIJKLMN = wasm_i16x8_load8x8(i0 + 16);
    const v128_t vxi0xOPQRSTUV = wasm_i16x8_load8x8(i0 + 24);
    i0 += 32;
    const v128_t vxi1x01234567 = wasm_i16x8_load8x8(i1);
    const v128_t vxi1x89ABCDEF = wasm_i16x8_load8x8(i1 + 8);
    const v128_t vxi1xGHIJKLMN = wasm_i16x8_load8x8(i1 + 16);
    const v128_t vxi1xOPQRSTUV = wasm_i16x8_load8x8(i1 + 24);
    i1 += 32;

    v128_t vacc01234567 = wasm_i16x8_add(vxi0x01234567, vxi1x01234567);
    const v128_t vxi2x01234567 = wasm_i16x8_load8x8(i2);
    v128_t vacc89ABCDEF = wasm_i16x8_add(vxi0x89ABCDEF, vxi1x89ABCDEF);
    const v128_t vxi2x89ABCDEF = wasm_i16x8_load8x8(i2 + 8);
    v128_t vaccGHIJKLMN = wasm_i16x8_add(vxi0xGHIJKLMN, vxi1xGHIJKLMN);
    const v128_t vxi2xGHIJKLMN = wasm_i16x8_load8x8(i2 + 16);
    v128_t vaccOPQRSTUV = wasm_i16x8_add(vxi0xOPQRSTUV, vxi1xOPQRSTUV);
    const v128_t vxi2xOPQRSTUV = wasm_i16x8_load8x8(i2 + 24);
    i2 += 32;

    vacc01234567 = wasm_i16x8_add(vacc01234567, vxi2x01234567);
    const v128_t vxi3x01234567 = wasm_i16x8_load8x8(i3);
    vacc89ABCDEF = wasm_i16x8_add(vacc89ABCDEF, vxi2x89ABCDEF);
    const v128_t vxi3x89ABCDEF = wasm_i16x8_load8x8(i3 + 8);
    vaccGHIJKLMN = wasm_i16x8_add(vaccGHIJKLMN, vxi2xGHIJKLMN);
    const v128_t vxi3xGHIJKLMN = wasm_i16x8_load8x8(i3 + 16);
    vaccOPQRSTUV = wasm_i16x8_add(vaccOPQRSTUV, vxi2xOPQRSTUV);
    const v128_t vxi3xOPQRSTUV = wasm_i16x8_load8x8(i3 + 24);
    i3 += 32;
    vacc01234567 = wasm_i16x8_add(vacc01234567, vxi3x01234567);
    const v128_t vxi4x01234567 = wasm_i16x8_load8x8(i4);
    vacc89ABCDEF = wasm_i16x8_add(vacc89ABCDEF, vxi3x89ABCDEF);
    const v128_t vxi4x89ABCDEF = wasm_i16x8_load8x8(i4 + 8);
    vaccGHIJKLMN = wasm_i16x8_add(vaccGHIJKLMN, vxi3xGHIJKLMN);
    const v128_t vxi4xGHIJKLMN = wasm_i16x8_load8x8(i4 + 16);
    vaccOPQRSTUV = wasm_i16x8_add(vaccOPQRSTUV, vxi3xOPQRSTUV);
    const v128_t vxi4xOPQRSTUV = wasm_i16x8_load8x8(i4 + 24);
    i4 += 32;
    vacc01234567 = wasm_i16x8_add(vacc01234567, vxi4x01234567);
    const v128_t vxi5x01234567 = wasm_i16x8_load8x8(i5);
    vacc89ABCDEF = wasm_i16x8_add(vacc89ABCDEF, vxi4x89ABCDEF);
    const v128_t vxi5x89ABCDEF = wasm_i16x8_load8x8(i5 + 8);
    vaccGHIJKLMN = wasm_i16x8_add(vaccGHIJKLMN, vxi4xGHIJKLMN);
    const v128_t vxi5xGHIJKLMN = wasm_i16x8_load8x8(i5 + 16);
    vaccOPQRSTUV = wasm_i16x8_add(vaccOPQRSTUV, vxi4xOPQRSTUV);
    const v128_t vxi5xOPQRSTUV = wasm_i16x8_load8x8(i5 + 24);
    i5 += 32;
    vacc01234567 = wasm_i16x8_add(vacc01234567, vxi5x01234567);
    const v128_t vxi6x01234567 = wasm_i16x8_load8x8(i6);
    vacc89ABCDEF = wasm_i16x8_add(vacc89ABCDEF, vxi5x89ABCDEF);
    const v128_t vxi6x89ABCDEF = wasm_i16x8_load8x8(i6 + 8);
    vaccGHIJKLMN = wasm_i16x8_add(vaccGHIJKLMN, vxi5xGHIJKLMN);
    const v128_t vxi6xGHIJKLMN = wasm_i16x8_load8x8(i6 + 16);
    vaccOPQRSTUV = wasm_i16x8_add(vaccOPQRSTUV, vxi5xOPQRSTUV);
    const v128_t vxi6xOPQRSTUV = wasm_i16x8_load8x8(i6 + 24);
    i6 += 32;

    vacc01234567 = wasm_i16x8_add(vacc01234567, vxi6x01234567);
    vacc89ABCDEF = wasm_i16x8_add(vacc89ABCDEF, vxi6x89ABCDEF);
    vaccGHIJKLMN = wasm_i16x8_add(vaccGHIJKLMN, vxi6xGHIJKLMN);
    vaccOPQRSTUV = wasm_i16x8_add(vaccOPQRSTUV, vxi6xOPQRSTUV);

    v128_t vacc0123 = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_low_i16x8(vacc01234567));
    v128_t vacc4567 = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_high_i16x8(vacc01234567));
    v128_t vacc89AB = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_low_i16x8(vacc89ABCDEF));
    v128_t vaccCDEF = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_high_i16x8(vacc89ABCDEF));
    v128_t vaccGHIJ = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_low_i16x8(vaccGHIJKLMN));
    v128_t vaccKLMN = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_high_i16x8(vaccGHIJKLMN));
    v128_t vaccOPQR = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_low_i16x8(vaccOPQRSTUV));
    v128_t vaccSTUV = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_high_i16x8(vaccOPQRSTUV));

    vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
    vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);
    vacc89AB = wasm_f32x4_convert_i32x4(vacc89AB);
    vaccCDEF = wasm_f32x4_convert_i32x4(vaccCDEF);
    vaccGHIJ = wasm_f32x4_convert_i32x4(vaccGHIJ);
    vaccKLMN = wasm_f32x4_convert_i32x4(vaccKLMN);
    vaccOPQR = wasm_f32x4_convert_i32x4(vaccOPQR);
    vaccSTUV = wasm_f32x4_convert_i32x4(vaccSTUV);

    vacc0123 = wasm_f32x4_mul(vacc0123, vscale);
    vacc4567 = wasm_f32x4_mul(vacc4567, vscale);
    vacc89AB = wasm_f32x4_mul(vacc89AB, vscale);
    vaccCDEF = wasm_f32x4_mul(vaccCDEF, vscale);
    vaccGHIJ = wasm_f32x4_mul(vaccGHIJ, vscale);
    vaccKLMN = wasm_f32x4_mul(vaccKLMN, vscale);
    vaccOPQR = wasm_f32x4_mul(vaccOPQR, vscale);
    vaccSTUV = wasm_f32x4_mul(vaccSTUV, vscale);

    vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
    vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);
    vacc89AB = wasm_f32x4_add(vacc89AB, vmagic_bias);
    vaccCDEF = wasm_f32x4_add(vaccCDEF, vmagic_bias);
    vaccGHIJ = wasm_f32x4_add(vaccGHIJ, vmagic_bias);
    vaccKLMN = wasm_f32x4_add(vaccKLMN, vmagic_bias);
    vaccOPQR = wasm_f32x4_add(vaccOPQR, vmagic_bias);
    vaccSTUV = wasm_f32x4_add(vaccSTUV, vmagic_bias);

    vacc0123 = wasm_i32x4_max(vacc0123, vmagic_min);
    vacc4567 = wasm_i32x4_max(vacc4567, vmagic_min);
    vacc89AB = wasm_i32x4_max(vacc89AB, vmagic_min);
    vaccCDEF = wasm_i32x4_max(vaccCDEF, vmagic_min);
    vaccGHIJ = wasm_i32x4_max(vaccGHIJ, vmagic_min);
    vaccKLMN = wasm_i32x4_max(vaccKLMN, vmagic_min);
    vaccOPQR = wasm_i32x4_max(vaccOPQR, vmagic_min);
    vaccSTUV = wasm_i32x4_max(vaccSTUV, vmagic_min);

    vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
    vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);
    vacc89AB = wasm_i32x4_sub(vacc89AB, vmagic_bias_less_output_zero_point);
    vaccCDEF = wasm_i32x4_sub(vaccCDEF, vmagic_bias_less_output_zero_point);
    vaccGHIJ = wasm_i32x4_sub(vaccGHIJ, vmagic_bias_less_output_zero_point);
    vaccKLMN = wasm_i32x4_sub(vaccKLMN, vmagic_bias_less_output_zero_point);
    vaccOPQR = wasm_i32x4_sub(vaccOPQR, vmagic_bias_less_output_zero_point);
    vaccSTUV = wasm_i32x4_sub(vaccSTUV, vmagic_bias_less_output_zero_point);

    v128_t vout01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
    v128_t vout89ABCDEF = wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF);
    v128_t voutGHIJKLMN = wasm_i16x8_narrow_i32x4(vaccGHIJ, vaccKLMN);
    v128_t voutOPQRSTUV = wasm_i16x8_narrow_i32x4(vaccOPQR, vaccSTUV);

    v128_t vout0123456789ABCDEF = wasm_i8x16_narrow_i16x8(vout01234567, vout89ABCDEF);
    v128_t voutGHIJKLMNOPQRSTUV = wasm_i8x16_narrow_i16x8(voutGHIJKLMN, voutOPQRSTUV);

    vout0123456789ABCDEF = wasm_i8x16_min(vout0123456789ABCDEF, voutput_max);
    voutGHIJKLMNOPQRSTUV = wasm_i8x16_min(voutGHIJKLMNOPQRSTUV, voutput_max);

    wasm_v128_store(output, vout0123456789ABCDEF);
    wasm_v128_store(output + 16, voutGHIJKLMNOPQRSTUV);
    output += 32;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {
      const v128_t vxi0x01234567 = wasm_i16x8_load8x8(i0);
      i0 += 8;
      const v128_t vxi1x01234567 = wasm_i16x8_load8x8(i1);
      i1 += 8;

      v128_t vacc01234567 = wasm_i16x8_add(vxi0x01234567, vxi1x01234567);
      const v128_t vxi2x01234567 = wasm_i16x8_load8x8(i2);
      i2 += 8;

      vacc01234567 = wasm_i16x8_add(vacc01234567, vxi2x01234567);
      const v128_t vxi3x01234567 = wasm_i16x8_load8x8(i3);
      i3 += 8;
      vacc01234567 = wasm_i16x8_add(vacc01234567, vxi3x01234567);
      const v128_t vxi4x01234567 = wasm_i16x8_load8x8(i4);
      i4 += 8;
      vacc01234567 = wasm_i16x8_add(vacc01234567, vxi4x01234567);
      const v128_t vxi5x01234567 = wasm_i16x8_load8x8(i5);
      i5 += 8;
      vacc01234567 = wasm_i16x8_add(vacc01234567, vxi5x01234567);
      const v128_t vxi6x01234567 = wasm_i16x8_load8x8(i6);
      i6 += 8;

      vacc01234567 = wasm_i16x8_add(vacc01234567, vxi6x01234567);

      v128_t vacc0123 = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_low_i16x8(vacc01234567));
      v128_t vacc4567 = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_high_i16x8(vacc01234567));

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
      v128_t vout0123456701234567 = wasm_i8x16_narrow_i16x8(vout01234567, vout01234567);
      vout0123456701234567 = wasm_i8x16_min(vout0123456701234567, voutput_max);

      if XNN_LIKELY(channels >= 8) {
        wasm_v128_store64_lane(output, vout0123456701234567, 0);
        output += 8;
        channels -= 8;
      } else {
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
          output += 1;
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
