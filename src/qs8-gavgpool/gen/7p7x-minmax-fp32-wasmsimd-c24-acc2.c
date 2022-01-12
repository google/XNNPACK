// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/multipass-wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__wasmsimd_c24_acc2(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* buffer,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8);

  const v128_t vinit_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.init_bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c >= 24; c -= 24) {
    const v128_t vxi0x01234567 = wasm_i16x8_load8x8(i0);
    const v128_t vxi0x89ABCDEF = wasm_i16x8_load8x8(i0 + 8);
    const v128_t vxi0xGHIJKLMN = wasm_i16x8_load8x8(i0 + 16);
    i0 += 24;
    const v128_t vxi1x01234567 = wasm_i16x8_load8x8(i1);
    const v128_t vxi1x89ABCDEF = wasm_i16x8_load8x8(i1 + 8);
    const v128_t vxi1xGHIJKLMN = wasm_i16x8_load8x8(i1 + 16);
    i1 += 24;
    const v128_t vxi2x01234567 = wasm_i16x8_load8x8(i2);
    const v128_t vxi2x89ABCDEF = wasm_i16x8_load8x8(i2 + 8);
    const v128_t vxi2xGHIJKLMN = wasm_i16x8_load8x8(i2 + 16);
    i2 += 24;
    const v128_t vxi3x01234567 = wasm_i16x8_load8x8(i3);
    const v128_t vxi3x89ABCDEF = wasm_i16x8_load8x8(i3 + 8);
    const v128_t vxi3xGHIJKLMN = wasm_i16x8_load8x8(i3 + 16);
    i3 += 24;
    const v128_t vxi4x01234567 = wasm_i16x8_load8x8(i4);
    const v128_t vxi4x89ABCDEF = wasm_i16x8_load8x8(i4 + 8);
    const v128_t vxi4xGHIJKLMN = wasm_i16x8_load8x8(i4 + 16);
    i4 += 24;
    const v128_t vxi5x01234567 = wasm_i16x8_load8x8(i5);
    const v128_t vxi5x89ABCDEF = wasm_i16x8_load8x8(i5 + 8);
    const v128_t vxi5xGHIJKLMN = wasm_i16x8_load8x8(i5 + 16);
    i5 += 24;
    const v128_t vxi6x01234567 = wasm_i16x8_load8x8(i6);
    const v128_t vxi6x89ABCDEF = wasm_i16x8_load8x8(i6 + 8);
    const v128_t vxi6xGHIJKLMN = wasm_i16x8_load8x8(i6 + 16);
    i6 += 24;

    v128_t vacc0x01234567 = wasm_i16x8_add(vxi0x01234567, vxi1x01234567);
    v128_t vacc0x89ABCDEF = wasm_i16x8_add(vxi0x89ABCDEF, vxi1x89ABCDEF);
    v128_t vacc0xGHIJKLMN = wasm_i16x8_add(vxi0xGHIJKLMN, vxi1xGHIJKLMN);
    v128_t vacc1x01234567 = wasm_i16x8_add(vxi2x01234567, vxi3x01234567);
    v128_t vacc1x89ABCDEF = wasm_i16x8_add(vxi2x89ABCDEF, vxi3x89ABCDEF);
    v128_t vacc1xGHIJKLMN = wasm_i16x8_add(vxi2xGHIJKLMN, vxi3xGHIJKLMN);

    vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi4x01234567);
    vacc0x89ABCDEF = wasm_i16x8_add(vacc0x89ABCDEF, vxi4x89ABCDEF);
    vacc0xGHIJKLMN = wasm_i16x8_add(vacc0xGHIJKLMN, vxi4xGHIJKLMN);
    vacc1x01234567 = wasm_i16x8_add(vacc1x01234567, vxi5x01234567);
    vacc1x89ABCDEF = wasm_i16x8_add(vacc1x89ABCDEF, vxi5x89ABCDEF);
    vacc1xGHIJKLMN = wasm_i16x8_add(vacc1xGHIJKLMN, vxi5xGHIJKLMN);
    vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi6x01234567);
    vacc0x89ABCDEF = wasm_i16x8_add(vacc0x89ABCDEF, vxi6x89ABCDEF);
    vacc0xGHIJKLMN = wasm_i16x8_add(vacc0xGHIJKLMN, vxi6xGHIJKLMN);

    // Add up all accumulators to vacc0x0123456789ABCDEFGHIJKLMN
    vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vacc1x01234567);
    vacc0x89ABCDEF = wasm_i16x8_add(vacc0x89ABCDEF, vacc1x89ABCDEF);
    vacc0xGHIJKLMN = wasm_i16x8_add(vacc0xGHIJKLMN, vacc1xGHIJKLMN);

    const v128_t vacc0123 = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_low_i16x8(vacc0x01234567));
    const v128_t vacc4567 = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_high_i16x8(vacc0x01234567));
    const v128_t vacc89AB = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_low_i16x8(vacc0x89ABCDEF));
    const v128_t vaccCDEF = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_high_i16x8(vacc0x89ABCDEF));
    const v128_t vaccGHIJ = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_low_i16x8(vacc0xGHIJKLMN));
    const v128_t vaccKLMN = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_high_i16x8(vacc0xGHIJKLMN));

    wasm_v128_store(b, vacc0123);
    wasm_v128_store(b + 4, vacc4567);
    wasm_v128_store(b + 8, vacc89AB);
    wasm_v128_store(b + 12, vaccCDEF);
    wasm_v128_store(b + 16, vaccGHIJ);
    wasm_v128_store(b + 20, vaccKLMN);
    b += 24;
  }
  if XNN_UNLIKELY(c != 0) {
    do {
      const v128_t vxi0x01234567 = wasm_i16x8_load8x8(i0);
      i0 += 8;
      const v128_t vxi1x01234567 = wasm_i16x8_load8x8(i1);
      i1 += 8;
      const v128_t vxi2x01234567 = wasm_i16x8_load8x8(i2);
      i2 += 8;
      const v128_t vxi3x01234567 = wasm_i16x8_load8x8(i3);
      i3 += 8;
      const v128_t vxi4x01234567 = wasm_i16x8_load8x8(i4);
      i4 += 8;
      const v128_t vxi5x01234567 = wasm_i16x8_load8x8(i5);
      i5 += 8;
      const v128_t vxi6x01234567 = wasm_i16x8_load8x8(i6);
      i6 += 8;

      v128_t vacc0x01234567 = wasm_i16x8_add(vxi0x01234567, vxi1x01234567);
      v128_t vacc1x01234567 = wasm_i16x8_add(vxi2x01234567, vxi3x01234567);

      vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi4x01234567);
      vacc1x01234567 = wasm_i16x8_add(vacc1x01234567, vxi5x01234567);
      vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi6x01234567);

      // Add up all accumulators to vacc0x01234567
      vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vacc1x01234567);

      const v128_t vacc0123 = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_low_i16x8(vacc0x01234567));
      const v128_t vacc4567 = wasm_i32x4_add(vinit_bias, wasm_i32x4_extend_high_i16x8(vacc0x01234567));

      wasm_v128_store(b, vacc0123);
      wasm_v128_store(b + 4, vacc4567);
      b += 8;

      c = doz(c, 8);
    } while (c != 0);
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    for (; c >= 24; c -= 24) {
      const v128_t vxi0x01234567 = wasm_i16x8_load8x8(i0);
      const v128_t vxi0x89ABCDEF = wasm_i16x8_load8x8(i0 + 8);
      const v128_t vxi0xGHIJKLMN = wasm_i16x8_load8x8(i0 + 16);
      i0 += 24;
      const v128_t vxi1x01234567 = wasm_i16x8_load8x8(i1);
      const v128_t vxi1x89ABCDEF = wasm_i16x8_load8x8(i1 + 8);
      const v128_t vxi1xGHIJKLMN = wasm_i16x8_load8x8(i1 + 16);
      i1 += 24;
      const v128_t vxi2x01234567 = wasm_i16x8_load8x8(i2);
      const v128_t vxi2x89ABCDEF = wasm_i16x8_load8x8(i2 + 8);
      const v128_t vxi2xGHIJKLMN = wasm_i16x8_load8x8(i2 + 16);
      i2 += 24;
      const v128_t vxi3x01234567 = wasm_i16x8_load8x8(i3);
      const v128_t vxi3x89ABCDEF = wasm_i16x8_load8x8(i3 + 8);
      const v128_t vxi3xGHIJKLMN = wasm_i16x8_load8x8(i3 + 16);
      i3 += 24;
      const v128_t vxi4x01234567 = wasm_i16x8_load8x8(i4);
      const v128_t vxi4x89ABCDEF = wasm_i16x8_load8x8(i4 + 8);
      const v128_t vxi4xGHIJKLMN = wasm_i16x8_load8x8(i4 + 16);
      i4 += 24;
      const v128_t vxi5x01234567 = wasm_i16x8_load8x8(i5);
      const v128_t vxi5x89ABCDEF = wasm_i16x8_load8x8(i5 + 8);
      const v128_t vxi5xGHIJKLMN = wasm_i16x8_load8x8(i5 + 16);
      i5 += 24;
      const v128_t vxi6x01234567 = wasm_i16x8_load8x8(i6);
      const v128_t vxi6x89ABCDEF = wasm_i16x8_load8x8(i6 + 8);
      const v128_t vxi6xGHIJKLMN = wasm_i16x8_load8x8(i6 + 16);
      i6 += 24;

      v128_t vacc0x01234567 = wasm_i16x8_add(vxi0x01234567, vxi1x01234567);
      v128_t vacc0x89ABCDEF = wasm_i16x8_add(vxi0x89ABCDEF, vxi1x89ABCDEF);
      v128_t vacc0xGHIJKLMN = wasm_i16x8_add(vxi0xGHIJKLMN, vxi1xGHIJKLMN);
      v128_t vacc1x01234567 = wasm_i16x8_add(vxi2x01234567, vxi3x01234567);
      v128_t vacc1x89ABCDEF = wasm_i16x8_add(vxi2x89ABCDEF, vxi3x89ABCDEF);
      v128_t vacc1xGHIJKLMN = wasm_i16x8_add(vxi2xGHIJKLMN, vxi3xGHIJKLMN);

      vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi4x01234567);
      vacc0x89ABCDEF = wasm_i16x8_add(vacc0x89ABCDEF, vxi4x89ABCDEF);
      vacc0xGHIJKLMN = wasm_i16x8_add(vacc0xGHIJKLMN, vxi4xGHIJKLMN);
      vacc1x01234567 = wasm_i16x8_add(vacc1x01234567, vxi5x01234567);
      vacc1x89ABCDEF = wasm_i16x8_add(vacc1x89ABCDEF, vxi5x89ABCDEF);
      vacc1xGHIJKLMN = wasm_i16x8_add(vacc1xGHIJKLMN, vxi5xGHIJKLMN);
      vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi6x01234567);
      vacc0x89ABCDEF = wasm_i16x8_add(vacc0x89ABCDEF, vxi6x89ABCDEF);
      vacc0xGHIJKLMN = wasm_i16x8_add(vacc0xGHIJKLMN, vxi6xGHIJKLMN);

      // Add up all accumulators to vacc0x0123456789ABCDEFGHIJKLMN
      vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vacc1x01234567);
      vacc0x89ABCDEF = wasm_i16x8_add(vacc0x89ABCDEF, vacc1x89ABCDEF);
      vacc0xGHIJKLMN = wasm_i16x8_add(vacc0xGHIJKLMN, vacc1xGHIJKLMN);

      const v128_t vacc0123 = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(vacc0x01234567), wasm_v128_load(b + 0));
      const v128_t vacc4567 = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(vacc0x01234567), wasm_v128_load(b + 4));
      const v128_t vacc89AB = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(vacc0x89ABCDEF), wasm_v128_load(b + 8));
      const v128_t vaccCDEF = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(vacc0x89ABCDEF), wasm_v128_load(b + 12));
      const v128_t vaccGHIJ = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(vacc0xGHIJKLMN), wasm_v128_load(b + 16));
      const v128_t vaccKLMN = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(vacc0xGHIJKLMN), wasm_v128_load(b + 20));

      wasm_v128_store(b, vacc0123);
      wasm_v128_store(b + 4, vacc4567);
      wasm_v128_store(b + 8, vacc89AB);
      wasm_v128_store(b + 12, vaccCDEF);
      wasm_v128_store(b + 16, vaccGHIJ);
      wasm_v128_store(b + 20, vaccKLMN);
      b += 24;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        const v128_t vxi0x01234567 = wasm_i16x8_load8x8(i0);
        i0 += 8;
        const v128_t vxi1x01234567 = wasm_i16x8_load8x8(i1);
        i1 += 8;
        const v128_t vxi2x01234567 = wasm_i16x8_load8x8(i2);
        i2 += 8;
        const v128_t vxi3x01234567 = wasm_i16x8_load8x8(i3);
        i3 += 8;
        const v128_t vxi4x01234567 = wasm_i16x8_load8x8(i4);
        i4 += 8;
        const v128_t vxi5x01234567 = wasm_i16x8_load8x8(i5);
        i5 += 8;
        const v128_t vxi6x01234567 = wasm_i16x8_load8x8(i6);
        i6 += 8;

        v128_t vacc0x01234567 = wasm_i16x8_add(vxi0x01234567, vxi1x01234567);
        v128_t vacc1x01234567 = wasm_i16x8_add(vxi2x01234567, vxi3x01234567);

        vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi4x01234567);
        vacc1x01234567 = wasm_i16x8_add(vacc1x01234567, vxi5x01234567);
        vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi6x01234567);

        // Add up all accumulators to vacc0x01234567
        vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vacc1x01234567);

        const v128_t vacc0123 = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(vacc0x01234567), wasm_v128_load(b));
        const v128_t vacc4567 = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(vacc0x01234567), wasm_v128_load(b + 4));

        wasm_v128_store(b, vacc0123);
        wasm_v128_store(b + 4, vacc4567);
        b += 8;

        c = doz(c, 8);
      } while (c != 0);
    }
  }

  i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const v128_t vscale = wasm_v128_load64_splat(params->fp32_wasmsimd.scale);
  const v128_t vmagic_bias = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias);
  const v128_t vmagic_min = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_min);
  const v128_t vmagic_bias_less_output_zero_point = wasm_v128_load64_splat(params->fp32_wasmsimd.magic_bias_less_output_zero_point);
  const v128_t voutput_max = wasm_v128_load64_splat(params->fp32_wasmsimd.output_max);
  while (channels >= 24) {
    const v128_t vxi0x01234567 = wasm_i16x8_load8x8(i0);
    const v128_t vxi0x89ABCDEF = wasm_i16x8_load8x8(i0 + 8);
    const v128_t vxi0xGHIJKLMN = wasm_i16x8_load8x8(i0 + 16);
    i0 += 24;
    const v128_t vxi1x01234567 = wasm_i16x8_load8x8(i1);
    const v128_t vxi1x89ABCDEF = wasm_i16x8_load8x8(i1 + 8);
    const v128_t vxi1xGHIJKLMN = wasm_i16x8_load8x8(i1 + 16);
    i1 += 24;
    const v128_t vxi2x01234567 = wasm_i16x8_load8x8(i2);
    const v128_t vxi2x89ABCDEF = wasm_i16x8_load8x8(i2 + 8);
    const v128_t vxi2xGHIJKLMN = wasm_i16x8_load8x8(i2 + 16);
    i2 += 24;
    const v128_t vxi3x01234567 = wasm_i16x8_load8x8(i3);
    const v128_t vxi3x89ABCDEF = wasm_i16x8_load8x8(i3 + 8);
    const v128_t vxi3xGHIJKLMN = wasm_i16x8_load8x8(i3 + 16);
    i3 += 24;
    const v128_t vxi4x01234567 = wasm_i16x8_load8x8(i4);
    const v128_t vxi4x89ABCDEF = wasm_i16x8_load8x8(i4 + 8);
    const v128_t vxi4xGHIJKLMN = wasm_i16x8_load8x8(i4 + 16);
    i4 += 24;
    const v128_t vxi5x01234567 = wasm_i16x8_load8x8(i5);
    const v128_t vxi5x89ABCDEF = wasm_i16x8_load8x8(i5 + 8);
    const v128_t vxi5xGHIJKLMN = wasm_i16x8_load8x8(i5 + 16);
    i5 += 24;
    const v128_t vxi6x01234567 = wasm_i16x8_load8x8(i6);
    const v128_t vxi6x89ABCDEF = wasm_i16x8_load8x8(i6 + 8);
    const v128_t vxi6xGHIJKLMN = wasm_i16x8_load8x8(i6 + 16);
    i6 += 24;

    v128_t vacc0x01234567 = wasm_i16x8_add(vxi0x01234567, vxi1x01234567);
    v128_t vacc0x89ABCDEF = wasm_i16x8_add(vxi0x89ABCDEF, vxi1x89ABCDEF);
    v128_t vacc0xGHIJKLMN = wasm_i16x8_add(vxi0xGHIJKLMN, vxi1xGHIJKLMN);
    v128_t vacc1x01234567 = wasm_i16x8_add(vxi2x01234567, vxi3x01234567);
    v128_t vacc1x89ABCDEF = wasm_i16x8_add(vxi2x89ABCDEF, vxi3x89ABCDEF);
    v128_t vacc1xGHIJKLMN = wasm_i16x8_add(vxi2xGHIJKLMN, vxi3xGHIJKLMN);

    vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi4x01234567);
    vacc0x89ABCDEF = wasm_i16x8_add(vacc0x89ABCDEF, vxi4x89ABCDEF);
    vacc0xGHIJKLMN = wasm_i16x8_add(vacc0xGHIJKLMN, vxi4xGHIJKLMN);
    vacc1x01234567 = wasm_i16x8_add(vacc1x01234567, vxi5x01234567);
    vacc1x89ABCDEF = wasm_i16x8_add(vacc1x89ABCDEF, vxi5x89ABCDEF);
    vacc1xGHIJKLMN = wasm_i16x8_add(vacc1xGHIJKLMN, vxi5xGHIJKLMN);
    vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi6x01234567);
    vacc0x89ABCDEF = wasm_i16x8_add(vacc0x89ABCDEF, vxi6x89ABCDEF);
    vacc0xGHIJKLMN = wasm_i16x8_add(vacc0xGHIJKLMN, vxi6xGHIJKLMN);

    // Add up all accumulators to vacc0x0123456789ABCDEFGHIJKLMN
    vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vacc1x01234567);
    vacc0x89ABCDEF = wasm_i16x8_add(vacc0x89ABCDEF, vacc1x89ABCDEF);
    vacc0xGHIJKLMN = wasm_i16x8_add(vacc0xGHIJKLMN, vacc1xGHIJKLMN);

    v128_t vacc0123 = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(vacc0x01234567), wasm_v128_load(buffer + 0));
    v128_t vacc4567 = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(vacc0x01234567), wasm_v128_load(buffer + 4));
    v128_t vacc89AB = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(vacc0x89ABCDEF), wasm_v128_load(buffer + 8));
    v128_t vaccCDEF = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(vacc0x89ABCDEF), wasm_v128_load(buffer + 12));
    v128_t vaccGHIJ = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(vacc0xGHIJKLMN), wasm_v128_load(buffer + 16));
    v128_t vaccKLMN = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(vacc0xGHIJKLMN), wasm_v128_load(buffer + 20));
    buffer += 24;

    vacc0123 = wasm_f32x4_convert_i32x4(vacc0123);
    vacc4567 = wasm_f32x4_convert_i32x4(vacc4567);
    vacc89AB = wasm_f32x4_convert_i32x4(vacc89AB);
    vaccCDEF = wasm_f32x4_convert_i32x4(vaccCDEF);
    vaccGHIJ = wasm_f32x4_convert_i32x4(vaccGHIJ);
    vaccKLMN = wasm_f32x4_convert_i32x4(vaccKLMN);

    vacc0123 = wasm_f32x4_mul(vacc0123, vscale);
    vacc4567 = wasm_f32x4_mul(vacc4567, vscale);
    vacc89AB = wasm_f32x4_mul(vacc89AB, vscale);
    vaccCDEF = wasm_f32x4_mul(vaccCDEF, vscale);
    vaccGHIJ = wasm_f32x4_mul(vaccGHIJ, vscale);
    vaccKLMN = wasm_f32x4_mul(vaccKLMN, vscale);

    vacc0123 = wasm_f32x4_add(vacc0123, vmagic_bias);
    vacc4567 = wasm_f32x4_add(vacc4567, vmagic_bias);
    vacc89AB = wasm_f32x4_add(vacc89AB, vmagic_bias);
    vaccCDEF = wasm_f32x4_add(vaccCDEF, vmagic_bias);
    vaccGHIJ = wasm_f32x4_add(vaccGHIJ, vmagic_bias);
    vaccKLMN = wasm_f32x4_add(vaccKLMN, vmagic_bias);

    vacc0123 = wasm_i32x4_max(vacc0123, vmagic_min);
    vacc4567 = wasm_i32x4_max(vacc4567, vmagic_min);
    vacc89AB = wasm_i32x4_max(vacc89AB, vmagic_min);
    vaccCDEF = wasm_i32x4_max(vaccCDEF, vmagic_min);
    vaccGHIJ = wasm_i32x4_max(vaccGHIJ, vmagic_min);
    vaccKLMN = wasm_i32x4_max(vaccKLMN, vmagic_min);

    vacc0123 = wasm_i32x4_sub(vacc0123, vmagic_bias_less_output_zero_point);
    vacc4567 = wasm_i32x4_sub(vacc4567, vmagic_bias_less_output_zero_point);
    vacc89AB = wasm_i32x4_sub(vacc89AB, vmagic_bias_less_output_zero_point);
    vaccCDEF = wasm_i32x4_sub(vaccCDEF, vmagic_bias_less_output_zero_point);
    vaccGHIJ = wasm_i32x4_sub(vaccGHIJ, vmagic_bias_less_output_zero_point);
    vaccKLMN = wasm_i32x4_sub(vaccKLMN, vmagic_bias_less_output_zero_point);

    v128_t vout01234567 = wasm_i16x8_narrow_i32x4(vacc0123, vacc4567);
    v128_t vout89ABCDEF = wasm_i16x8_narrow_i32x4(vacc89AB, vaccCDEF);
    v128_t voutGHIJKLMN = wasm_i16x8_narrow_i32x4(vaccGHIJ, vaccKLMN);

    v128_t vout0123456789ABCDEF = wasm_i8x16_narrow_i16x8(vout01234567, vout89ABCDEF);
    v128_t voutGHIJKLMNGHIJKLMN = wasm_i8x16_narrow_i16x8(voutGHIJKLMN, voutGHIJKLMN);

    vout0123456789ABCDEF = wasm_i8x16_min(vout0123456789ABCDEF, voutput_max);
    voutGHIJKLMNGHIJKLMN = wasm_i8x16_min(voutGHIJKLMNGHIJKLMN, voutput_max);

    wasm_v128_store(output, vout0123456789ABCDEF);
    *((double*) (output + 16)) = wasm_f64x2_extract_lane(voutGHIJKLMNGHIJKLMN, 0);
    output += 24;

    channels -= 24;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {
      const v128_t vxi0x01234567 = wasm_i16x8_load8x8(i0);
      i0 += 8;
      const v128_t vxi1x01234567 = wasm_i16x8_load8x8(i1);
      i1 += 8;
      const v128_t vxi2x01234567 = wasm_i16x8_load8x8(i2);
      i2 += 8;
      const v128_t vxi3x01234567 = wasm_i16x8_load8x8(i3);
      i3 += 8;
      const v128_t vxi4x01234567 = wasm_i16x8_load8x8(i4);
      i4 += 8;
      const v128_t vxi5x01234567 = wasm_i16x8_load8x8(i5);
      i5 += 8;
      const v128_t vxi6x01234567 = wasm_i16x8_load8x8(i6);
      i6 += 8;

      v128_t vacc0x01234567 = wasm_i16x8_add(vxi0x01234567, vxi1x01234567);
      v128_t vacc1x01234567 = wasm_i16x8_add(vxi2x01234567, vxi3x01234567);

      vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi4x01234567);
      vacc1x01234567 = wasm_i16x8_add(vacc1x01234567, vxi5x01234567);
      vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vxi6x01234567);

      // Add up all accumulators to vacc0x01234567
      vacc0x01234567 = wasm_i16x8_add(vacc0x01234567, vacc1x01234567);

      v128_t vacc0123 = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(vacc0x01234567), wasm_v128_load(buffer));
      v128_t vacc4567 = wasm_i32x4_add(wasm_i32x4_extend_high_i16x8(vacc0x01234567), wasm_v128_load(buffer + 4));
      buffer += 8;

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
        *((double*) output) = wasm_f64x2_extract_lane(vout0123456701234567, 0);
        output += 8;
        channels -= 8;
      } else {
        if (channels & 4) {
          *((float*) output) = wasm_f32x4_extract_lane(vout0123456701234567, 0);
          vout0123456701234567 = wasm_u64x2_shr(vout0123456701234567, 32);
          output += 4;
        }
        uint32_t vout0123 = wasm_i32x4_extract_lane(vout0123456701234567, 0);
        if (channels & 2) {
          *((uint16_t*) output) = (uint16_t) vout0123;
          vout0123 >>= 16;
          output += 2;
        }
        if (channels & 1) {
          *output = (int8_t) vout0123;
          output += 1;
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
