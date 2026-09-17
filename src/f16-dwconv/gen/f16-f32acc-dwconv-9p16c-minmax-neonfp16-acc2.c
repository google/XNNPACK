// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/unipass-f16-f32acc.c.in
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
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f32-neon.h"


static XNN_INLINE xnn_simd_f32_t xnn_loadu_f16_f32(const xnn_float16* input) {
  return xnn_cvt_f32_f16(xnn_loadu_f16(input));
}

static XNN_INLINE xnn_simd_f32_t xnn_load_tail_f16_f32(
    const xnn_float16* input, size_t elements)
{
  return xnn_cvt_f32_f16(xnn_load_tail_f16(input, elements));
}

static XNN_INLINE void xnn_store_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value)
{
  xnn_store_tail_f16(output, xnn_cvt_f16_f32(value), xnn_simd_size_f32);
}

static XNN_INLINE void xnn_store_tail_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value, size_t elements)
{
  xnn_store_tail_f16(output, xnn_cvt_f16_f32(value), elements);
}


void xnn_f16_f32acc_dwconv_minmax_ukernel_9p16c__neonfp16_acc2(
    size_t channels,
    size_t output_width,
    const xnn_float16** input,
    const xnn_float16* weights,
    xnn_float16* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const xnn_float16* zero,
    const struct xnn_f16_minmax_params* restrict params) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(xnn_simd_size_f32 == 4);

  const xnn_simd_f32_t vmin = xnn_set1_f32(xnn_float16_to_float(params->scalar.min));
  const xnn_simd_f32_t vmax = xnn_set1_f32(xnn_float16_to_float(params->scalar.max));
  do {
    const xnn_float16* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const xnn_float16*) ((uintptr_t) i0 + input_offset);
    }
    const xnn_float16* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const xnn_float16*) ((uintptr_t) i1 + input_offset);
    }
    const xnn_float16* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const xnn_float16*) ((uintptr_t) i2 + input_offset);
    }
    const xnn_float16* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const xnn_float16*) ((uintptr_t) i3 + input_offset);
    }
    const xnn_float16* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const xnn_float16*) ((uintptr_t) i4 + input_offset);
    }
    const xnn_float16* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const xnn_float16*) ((uintptr_t) i5 + input_offset);
    }
    const xnn_float16* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const xnn_float16*) ((uintptr_t) i6 + input_offset);
    }
    const xnn_float16* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const xnn_float16*) ((uintptr_t) i7 + input_offset);
    }
    const xnn_float16* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const xnn_float16*) ((uintptr_t) i8 + input_offset);
    }
    input = (const xnn_float16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const xnn_float16* w = weights;
    for (; c >= 16; c -= 16) {
      xnn_simd_f32_t vacc0123p0 = xnn_loadu_f16_f32(w + 0);
      xnn_simd_f32_t vacc4567p0 = xnn_loadu_f16_f32(w + 4);
      xnn_simd_f32_t vacc89ABp0 = xnn_loadu_f16_f32(w + 8);
      xnn_simd_f32_t vaccCDEFp0 = xnn_loadu_f16_f32(w + 12);

      const xnn_simd_f32_t vi0x0123 = xnn_loadu_f16_f32(i0 + 0);
      const xnn_simd_f32_t vk0x0123 = xnn_loadu_f16_f32(w + 16);
      vacc0123p0 = xnn_fmadd_f32(vi0x0123, vk0x0123, vacc0123p0);
      const xnn_simd_f32_t vi0x4567 = xnn_loadu_f16_f32(i0 + 4);
      const xnn_simd_f32_t vk0x4567 = xnn_loadu_f16_f32(w + 20);
      vacc4567p0 = xnn_fmadd_f32(vi0x4567, vk0x4567, vacc4567p0);
      const xnn_simd_f32_t vi0x89AB = xnn_loadu_f16_f32(i0 + 8);
      const xnn_simd_f32_t vk0x89AB = xnn_loadu_f16_f32(w + 24);
      vacc89ABp0 = xnn_fmadd_f32(vi0x89AB, vk0x89AB, vacc89ABp0);
      const xnn_simd_f32_t vi0xCDEF = xnn_loadu_f16_f32(i0 + 12);
      const xnn_simd_f32_t vk0xCDEF = xnn_loadu_f16_f32(w + 28);
      vaccCDEFp0 = xnn_fmadd_f32(vi0xCDEF, vk0xCDEF, vaccCDEFp0);
      i0 += 16;
      const xnn_simd_f32_t vi1x0123 = xnn_loadu_f16_f32(i1 + 0);
      const xnn_simd_f32_t vk1x0123 = xnn_loadu_f16_f32(w + 32);
      xnn_simd_f32_t vacc0123p1 = xnn_mul_f32(vi1x0123, vk1x0123);
      const xnn_simd_f32_t vi1x4567 = xnn_loadu_f16_f32(i1 + 4);
      const xnn_simd_f32_t vk1x4567 = xnn_loadu_f16_f32(w + 36);
      xnn_simd_f32_t vacc4567p1 = xnn_mul_f32(vi1x4567, vk1x4567);
      const xnn_simd_f32_t vi1x89AB = xnn_loadu_f16_f32(i1 + 8);
      const xnn_simd_f32_t vk1x89AB = xnn_loadu_f16_f32(w + 40);
      xnn_simd_f32_t vacc89ABp1 = xnn_mul_f32(vi1x89AB, vk1x89AB);
      const xnn_simd_f32_t vi1xCDEF = xnn_loadu_f16_f32(i1 + 12);
      const xnn_simd_f32_t vk1xCDEF = xnn_loadu_f16_f32(w + 44);
      xnn_simd_f32_t vaccCDEFp1 = xnn_mul_f32(vi1xCDEF, vk1xCDEF);
      i1 += 16;
      const xnn_simd_f32_t vi2x0123 = xnn_loadu_f16_f32(i2 + 0);
      const xnn_simd_f32_t vk2x0123 = xnn_loadu_f16_f32(w + 48);
      vacc0123p0 = xnn_fmadd_f32(vi2x0123, vk2x0123, vacc0123p0);
      const xnn_simd_f32_t vi2x4567 = xnn_loadu_f16_f32(i2 + 4);
      const xnn_simd_f32_t vk2x4567 = xnn_loadu_f16_f32(w + 52);
      vacc4567p0 = xnn_fmadd_f32(vi2x4567, vk2x4567, vacc4567p0);
      const xnn_simd_f32_t vi2x89AB = xnn_loadu_f16_f32(i2 + 8);
      const xnn_simd_f32_t vk2x89AB = xnn_loadu_f16_f32(w + 56);
      vacc89ABp0 = xnn_fmadd_f32(vi2x89AB, vk2x89AB, vacc89ABp0);
      const xnn_simd_f32_t vi2xCDEF = xnn_loadu_f16_f32(i2 + 12);
      const xnn_simd_f32_t vk2xCDEF = xnn_loadu_f16_f32(w + 60);
      vaccCDEFp0 = xnn_fmadd_f32(vi2xCDEF, vk2xCDEF, vaccCDEFp0);
      i2 += 16;
      const xnn_simd_f32_t vi3x0123 = xnn_loadu_f16_f32(i3 + 0);
      const xnn_simd_f32_t vk3x0123 = xnn_loadu_f16_f32(w + 64);
      vacc0123p1 = xnn_fmadd_f32(vi3x0123, vk3x0123, vacc0123p1);
      const xnn_simd_f32_t vi3x4567 = xnn_loadu_f16_f32(i3 + 4);
      const xnn_simd_f32_t vk3x4567 = xnn_loadu_f16_f32(w + 68);
      vacc4567p1 = xnn_fmadd_f32(vi3x4567, vk3x4567, vacc4567p1);
      const xnn_simd_f32_t vi3x89AB = xnn_loadu_f16_f32(i3 + 8);
      const xnn_simd_f32_t vk3x89AB = xnn_loadu_f16_f32(w + 72);
      vacc89ABp1 = xnn_fmadd_f32(vi3x89AB, vk3x89AB, vacc89ABp1);
      const xnn_simd_f32_t vi3xCDEF = xnn_loadu_f16_f32(i3 + 12);
      const xnn_simd_f32_t vk3xCDEF = xnn_loadu_f16_f32(w + 76);
      vaccCDEFp1 = xnn_fmadd_f32(vi3xCDEF, vk3xCDEF, vaccCDEFp1);
      i3 += 16;
      const xnn_simd_f32_t vi4x0123 = xnn_loadu_f16_f32(i4 + 0);
      const xnn_simd_f32_t vk4x0123 = xnn_loadu_f16_f32(w + 80);
      vacc0123p0 = xnn_fmadd_f32(vi4x0123, vk4x0123, vacc0123p0);
      const xnn_simd_f32_t vi4x4567 = xnn_loadu_f16_f32(i4 + 4);
      const xnn_simd_f32_t vk4x4567 = xnn_loadu_f16_f32(w + 84);
      vacc4567p0 = xnn_fmadd_f32(vi4x4567, vk4x4567, vacc4567p0);
      const xnn_simd_f32_t vi4x89AB = xnn_loadu_f16_f32(i4 + 8);
      const xnn_simd_f32_t vk4x89AB = xnn_loadu_f16_f32(w + 88);
      vacc89ABp0 = xnn_fmadd_f32(vi4x89AB, vk4x89AB, vacc89ABp0);
      const xnn_simd_f32_t vi4xCDEF = xnn_loadu_f16_f32(i4 + 12);
      const xnn_simd_f32_t vk4xCDEF = xnn_loadu_f16_f32(w + 92);
      vaccCDEFp0 = xnn_fmadd_f32(vi4xCDEF, vk4xCDEF, vaccCDEFp0);
      i4 += 16;
      const xnn_simd_f32_t vi5x0123 = xnn_loadu_f16_f32(i5 + 0);
      const xnn_simd_f32_t vk5x0123 = xnn_loadu_f16_f32(w + 96);
      vacc0123p1 = xnn_fmadd_f32(vi5x0123, vk5x0123, vacc0123p1);
      const xnn_simd_f32_t vi5x4567 = xnn_loadu_f16_f32(i5 + 4);
      const xnn_simd_f32_t vk5x4567 = xnn_loadu_f16_f32(w + 100);
      vacc4567p1 = xnn_fmadd_f32(vi5x4567, vk5x4567, vacc4567p1);
      const xnn_simd_f32_t vi5x89AB = xnn_loadu_f16_f32(i5 + 8);
      const xnn_simd_f32_t vk5x89AB = xnn_loadu_f16_f32(w + 104);
      vacc89ABp1 = xnn_fmadd_f32(vi5x89AB, vk5x89AB, vacc89ABp1);
      const xnn_simd_f32_t vi5xCDEF = xnn_loadu_f16_f32(i5 + 12);
      const xnn_simd_f32_t vk5xCDEF = xnn_loadu_f16_f32(w + 108);
      vaccCDEFp1 = xnn_fmadd_f32(vi5xCDEF, vk5xCDEF, vaccCDEFp1);
      i5 += 16;
      const xnn_simd_f32_t vi6x0123 = xnn_loadu_f16_f32(i6 + 0);
      const xnn_simd_f32_t vk6x0123 = xnn_loadu_f16_f32(w + 112);
      vacc0123p0 = xnn_fmadd_f32(vi6x0123, vk6x0123, vacc0123p0);
      const xnn_simd_f32_t vi6x4567 = xnn_loadu_f16_f32(i6 + 4);
      const xnn_simd_f32_t vk6x4567 = xnn_loadu_f16_f32(w + 116);
      vacc4567p0 = xnn_fmadd_f32(vi6x4567, vk6x4567, vacc4567p0);
      const xnn_simd_f32_t vi6x89AB = xnn_loadu_f16_f32(i6 + 8);
      const xnn_simd_f32_t vk6x89AB = xnn_loadu_f16_f32(w + 120);
      vacc89ABp0 = xnn_fmadd_f32(vi6x89AB, vk6x89AB, vacc89ABp0);
      const xnn_simd_f32_t vi6xCDEF = xnn_loadu_f16_f32(i6 + 12);
      const xnn_simd_f32_t vk6xCDEF = xnn_loadu_f16_f32(w + 124);
      vaccCDEFp0 = xnn_fmadd_f32(vi6xCDEF, vk6xCDEF, vaccCDEFp0);
      i6 += 16;
      const xnn_simd_f32_t vi7x0123 = xnn_loadu_f16_f32(i7 + 0);
      const xnn_simd_f32_t vk7x0123 = xnn_loadu_f16_f32(w + 128);
      vacc0123p1 = xnn_fmadd_f32(vi7x0123, vk7x0123, vacc0123p1);
      const xnn_simd_f32_t vi7x4567 = xnn_loadu_f16_f32(i7 + 4);
      const xnn_simd_f32_t vk7x4567 = xnn_loadu_f16_f32(w + 132);
      vacc4567p1 = xnn_fmadd_f32(vi7x4567, vk7x4567, vacc4567p1);
      const xnn_simd_f32_t vi7x89AB = xnn_loadu_f16_f32(i7 + 8);
      const xnn_simd_f32_t vk7x89AB = xnn_loadu_f16_f32(w + 136);
      vacc89ABp1 = xnn_fmadd_f32(vi7x89AB, vk7x89AB, vacc89ABp1);
      const xnn_simd_f32_t vi7xCDEF = xnn_loadu_f16_f32(i7 + 12);
      const xnn_simd_f32_t vk7xCDEF = xnn_loadu_f16_f32(w + 140);
      vaccCDEFp1 = xnn_fmadd_f32(vi7xCDEF, vk7xCDEF, vaccCDEFp1);
      i7 += 16;
      const xnn_simd_f32_t vi8x0123 = xnn_loadu_f16_f32(i8 + 0);
      const xnn_simd_f32_t vk8x0123 = xnn_loadu_f16_f32(w + 144);
      vacc0123p0 = xnn_fmadd_f32(vi8x0123, vk8x0123, vacc0123p0);
      const xnn_simd_f32_t vi8x4567 = xnn_loadu_f16_f32(i8 + 4);
      const xnn_simd_f32_t vk8x4567 = xnn_loadu_f16_f32(w + 148);
      vacc4567p0 = xnn_fmadd_f32(vi8x4567, vk8x4567, vacc4567p0);
      const xnn_simd_f32_t vi8x89AB = xnn_loadu_f16_f32(i8 + 8);
      const xnn_simd_f32_t vk8x89AB = xnn_loadu_f16_f32(w + 152);
      vacc89ABp0 = xnn_fmadd_f32(vi8x89AB, vk8x89AB, vacc89ABp0);
      const xnn_simd_f32_t vi8xCDEF = xnn_loadu_f16_f32(i8 + 12);
      const xnn_simd_f32_t vk8xCDEF = xnn_loadu_f16_f32(w + 156);
      vaccCDEFp0 = xnn_fmadd_f32(vi8xCDEF, vk8xCDEF, vaccCDEFp0);
      i8 += 16;

      w += 160;

      vacc0123p0 = xnn_add_f32(vacc0123p0, vacc0123p1);
      vacc4567p0 = xnn_add_f32(vacc4567p0, vacc4567p1);
      vacc89ABp0 = xnn_add_f32(vacc89ABp0, vacc89ABp1);
      vaccCDEFp0 = xnn_add_f32(vaccCDEFp0, vaccCDEFp1);

      xnn_simd_f32_t vacc0123 = xnn_max_f32(vacc0123p0, vmin);
      vacc0123 = xnn_min_f32(vacc0123, vmax);
      xnn_store_f32_f16(output + 0, vacc0123);
      xnn_simd_f32_t vacc4567 = xnn_max_f32(vacc4567p0, vmin);
      vacc4567 = xnn_min_f32(vacc4567, vmax);
      xnn_store_f32_f16(output + 4, vacc4567);
      xnn_simd_f32_t vacc89AB = xnn_max_f32(vacc89ABp0, vmin);
      vacc89AB = xnn_min_f32(vacc89AB, vmax);
      xnn_store_f32_f16(output + 8, vacc89AB);
      xnn_simd_f32_t vaccCDEF = xnn_max_f32(vaccCDEFp0, vmin);
      vaccCDEF = xnn_min_f32(vaccCDEF, vmax);
      xnn_store_f32_f16(output + 12, vaccCDEF);
      output += 16;
    }
    for (; c >= 4; c -= 4) {
      xnn_simd_f32_t vacc0123p0 = xnn_loadu_f16_f32(w);

      const xnn_simd_f32_t vi0x0123 = xnn_loadu_f16_f32(i0);
      const xnn_simd_f32_t vk0x0123 = xnn_loadu_f16_f32(w + 16);
      vacc0123p0 = xnn_fmadd_f32(vi0x0123, vk0x0123, vacc0123p0);
      i0 += 4;
      const xnn_simd_f32_t vi1x0123 = xnn_loadu_f16_f32(i1);
      const xnn_simd_f32_t vk1x0123 = xnn_loadu_f16_f32(w + 32);
      xnn_simd_f32_t vacc0123p1 = xnn_mul_f32(vi1x0123, vk1x0123);
      i1 += 4;
      const xnn_simd_f32_t vi2x0123 = xnn_loadu_f16_f32(i2);
      const xnn_simd_f32_t vk2x0123 = xnn_loadu_f16_f32(w + 48);
      vacc0123p0 = xnn_fmadd_f32(vi2x0123, vk2x0123, vacc0123p0);
      i2 += 4;
      const xnn_simd_f32_t vi3x0123 = xnn_loadu_f16_f32(i3);
      const xnn_simd_f32_t vk3x0123 = xnn_loadu_f16_f32(w + 64);
      vacc0123p1 = xnn_fmadd_f32(vi3x0123, vk3x0123, vacc0123p1);
      i3 += 4;
      const xnn_simd_f32_t vi4x0123 = xnn_loadu_f16_f32(i4);
      const xnn_simd_f32_t vk4x0123 = xnn_loadu_f16_f32(w + 80);
      vacc0123p0 = xnn_fmadd_f32(vi4x0123, vk4x0123, vacc0123p0);
      i4 += 4;
      const xnn_simd_f32_t vi5x0123 = xnn_loadu_f16_f32(i5);
      const xnn_simd_f32_t vk5x0123 = xnn_loadu_f16_f32(w + 96);
      vacc0123p1 = xnn_fmadd_f32(vi5x0123, vk5x0123, vacc0123p1);
      i5 += 4;
      const xnn_simd_f32_t vi6x0123 = xnn_loadu_f16_f32(i6);
      const xnn_simd_f32_t vk6x0123 = xnn_loadu_f16_f32(w + 112);
      vacc0123p0 = xnn_fmadd_f32(vi6x0123, vk6x0123, vacc0123p0);
      i6 += 4;
      const xnn_simd_f32_t vi7x0123 = xnn_loadu_f16_f32(i7);
      const xnn_simd_f32_t vk7x0123 = xnn_loadu_f16_f32(w + 128);
      vacc0123p1 = xnn_fmadd_f32(vi7x0123, vk7x0123, vacc0123p1);
      i7 += 4;
      const xnn_simd_f32_t vi8x0123 = xnn_loadu_f16_f32(i8);
      const xnn_simd_f32_t vk8x0123 = xnn_loadu_f16_f32(w + 144);
      vacc0123p0 = xnn_fmadd_f32(vi8x0123, vk8x0123, vacc0123p0);
      i8 += 4;

      w += 4;

      vacc0123p0 = xnn_add_f32(vacc0123p0, vacc0123p1);

      xnn_simd_f32_t vacc0123 = xnn_max_f32(vacc0123p0, vmin);
      vacc0123 = xnn_min_f32(vacc0123, vmax);
      xnn_store_f32_f16(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      xnn_simd_f32_t vacc0123p0 = xnn_loadu_f16_f32(w);

      const xnn_simd_f32_t vi0x0123 = xnn_load_tail_f16_f32(i0, c);
      const xnn_simd_f32_t vk0x0123 = xnn_loadu_f16_f32(w + 16);
      vacc0123p0 = xnn_fmadd_f32(vi0x0123, vk0x0123, vacc0123p0);
      const xnn_simd_f32_t vi1x0123 = xnn_load_tail_f16_f32(i1, c);
      const xnn_simd_f32_t vk1x0123 = xnn_loadu_f16_f32(w + 32);
      xnn_simd_f32_t vacc0123p1 = xnn_mul_f32(vi1x0123, vk1x0123);
      const xnn_simd_f32_t vi2x0123 = xnn_load_tail_f16_f32(i2, c);
      const xnn_simd_f32_t vk2x0123 = xnn_loadu_f16_f32(w + 48);
      vacc0123p0 = xnn_fmadd_f32(vi2x0123, vk2x0123, vacc0123p0);
      const xnn_simd_f32_t vi3x0123 = xnn_load_tail_f16_f32(i3, c);
      const xnn_simd_f32_t vk3x0123 = xnn_loadu_f16_f32(w + 64);
      vacc0123p1 = xnn_fmadd_f32(vi3x0123, vk3x0123, vacc0123p1);
      const xnn_simd_f32_t vi4x0123 = xnn_load_tail_f16_f32(i4, c);
      const xnn_simd_f32_t vk4x0123 = xnn_loadu_f16_f32(w + 80);
      vacc0123p0 = xnn_fmadd_f32(vi4x0123, vk4x0123, vacc0123p0);
      const xnn_simd_f32_t vi5x0123 = xnn_load_tail_f16_f32(i5, c);
      const xnn_simd_f32_t vk5x0123 = xnn_loadu_f16_f32(w + 96);
      vacc0123p1 = xnn_fmadd_f32(vi5x0123, vk5x0123, vacc0123p1);
      const xnn_simd_f32_t vi6x0123 = xnn_load_tail_f16_f32(i6, c);
      const xnn_simd_f32_t vk6x0123 = xnn_loadu_f16_f32(w + 112);
      vacc0123p0 = xnn_fmadd_f32(vi6x0123, vk6x0123, vacc0123p0);
      const xnn_simd_f32_t vi7x0123 = xnn_load_tail_f16_f32(i7, c);
      const xnn_simd_f32_t vk7x0123 = xnn_loadu_f16_f32(w + 128);
      vacc0123p1 = xnn_fmadd_f32(vi7x0123, vk7x0123, vacc0123p1);
      const xnn_simd_f32_t vi8x0123 = xnn_load_tail_f16_f32(i8, c);
      const xnn_simd_f32_t vk8x0123 = xnn_loadu_f16_f32(w + 144);
      vacc0123p0 = xnn_fmadd_f32(vi8x0123, vk8x0123, vacc0123p0);

      vacc0123p0 = xnn_add_f32(vacc0123p0, vacc0123p1);

      xnn_simd_f32_t vacc0123 = xnn_max_f32(vacc0123p0, vmin);
      vacc0123 = xnn_min_f32(vacc0123, vmax);
      xnn_store_tail_f32_f16(output, vacc0123, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
