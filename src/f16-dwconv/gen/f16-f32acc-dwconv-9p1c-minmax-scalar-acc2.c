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
#include "src/xnnpack/simd/f32-scalar.h"
#include "src/xnnpack/simd/f16-scalar.h"


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


void xnn_f16_f32acc_dwconv_minmax_ukernel_9p1c__scalar_acc2(
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
  assert(xnn_simd_size_f32 == 1);

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
    for (; c >= 1; c -= 1) {
      xnn_simd_f32_t vacc0p0 = xnn_loadu_f16_f32(w + 0);

      const xnn_simd_f32_t vi0x0 = xnn_loadu_f16_f32(i0 + 0);
      const xnn_simd_f32_t vk0x0 = xnn_loadu_f16_f32(w + 1);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);
      i0 += 1;
      const xnn_simd_f32_t vi1x0 = xnn_loadu_f16_f32(i1 + 0);
      const xnn_simd_f32_t vk1x0 = xnn_loadu_f16_f32(w + 2);
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);
      i1 += 1;
      const xnn_simd_f32_t vi2x0 = xnn_loadu_f16_f32(i2 + 0);
      const xnn_simd_f32_t vk2x0 = xnn_loadu_f16_f32(w + 3);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);
      i2 += 1;
      const xnn_simd_f32_t vi3x0 = xnn_loadu_f16_f32(i3 + 0);
      const xnn_simd_f32_t vk3x0 = xnn_loadu_f16_f32(w + 4);
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);
      i3 += 1;
      const xnn_simd_f32_t vi4x0 = xnn_loadu_f16_f32(i4 + 0);
      const xnn_simd_f32_t vk4x0 = xnn_loadu_f16_f32(w + 5);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);
      i4 += 1;
      const xnn_simd_f32_t vi5x0 = xnn_loadu_f16_f32(i5 + 0);
      const xnn_simd_f32_t vk5x0 = xnn_loadu_f16_f32(w + 6);
      vacc0p1 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p1);
      i5 += 1;
      const xnn_simd_f32_t vi6x0 = xnn_loadu_f16_f32(i6 + 0);
      const xnn_simd_f32_t vk6x0 = xnn_loadu_f16_f32(w + 7);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);
      i6 += 1;
      const xnn_simd_f32_t vi7x0 = xnn_loadu_f16_f32(i7 + 0);
      const xnn_simd_f32_t vk7x0 = xnn_loadu_f16_f32(w + 8);
      vacc0p1 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p1);
      i7 += 1;
      const xnn_simd_f32_t vi8x0 = xnn_loadu_f16_f32(i8 + 0);
      const xnn_simd_f32_t vk8x0 = xnn_loadu_f16_f32(w + 9);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);
      i8 += 1;

      w += 10;

      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);

      xnn_simd_f32_t vacc0 = xnn_max_f32(vacc0p0, vmin);
      vacc0 = xnn_min_f32(vacc0, vmax);
      xnn_store_f32_f16(output + 0, vacc0);
      output += 1;
    }
    if XNN_UNLIKELY(c != 0) {
      xnn_simd_f32_t vacc0p0 = xnn_loadu_f16_f32(w);

      const xnn_simd_f32_t vi0x0 = xnn_load_tail_f16_f32(i0, c);
      const xnn_simd_f32_t vk0x0 = xnn_loadu_f16_f32(w + 1);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);
      const xnn_simd_f32_t vi1x0 = xnn_load_tail_f16_f32(i1, c);
      const xnn_simd_f32_t vk1x0 = xnn_loadu_f16_f32(w + 2);
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);
      const xnn_simd_f32_t vi2x0 = xnn_load_tail_f16_f32(i2, c);
      const xnn_simd_f32_t vk2x0 = xnn_loadu_f16_f32(w + 3);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);
      const xnn_simd_f32_t vi3x0 = xnn_load_tail_f16_f32(i3, c);
      const xnn_simd_f32_t vk3x0 = xnn_loadu_f16_f32(w + 4);
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);
      const xnn_simd_f32_t vi4x0 = xnn_load_tail_f16_f32(i4, c);
      const xnn_simd_f32_t vk4x0 = xnn_loadu_f16_f32(w + 5);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);
      const xnn_simd_f32_t vi5x0 = xnn_load_tail_f16_f32(i5, c);
      const xnn_simd_f32_t vk5x0 = xnn_loadu_f16_f32(w + 6);
      vacc0p1 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p1);
      const xnn_simd_f32_t vi6x0 = xnn_load_tail_f16_f32(i6, c);
      const xnn_simd_f32_t vk6x0 = xnn_loadu_f16_f32(w + 7);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);
      const xnn_simd_f32_t vi7x0 = xnn_load_tail_f16_f32(i7, c);
      const xnn_simd_f32_t vk7x0 = xnn_loadu_f16_f32(w + 8);
      vacc0p1 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p1);
      const xnn_simd_f32_t vi8x0 = xnn_load_tail_f16_f32(i8, c);
      const xnn_simd_f32_t vk8x0 = xnn_loadu_f16_f32(w + 9);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);

      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);

      xnn_simd_f32_t vacc0 = xnn_max_f32(vacc0p0, vmin);
      vacc0 = xnn_min_f32(vacc0, vmax);
      xnn_store_tail_f32_f16(output, vacc0, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
