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


void xnn_f16_f32acc_dwconv_minmax_ukernel_3p8c__neonfp16_acc2(
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
    input = (const xnn_float16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const xnn_float16* w = weights;
    for (; c >= 8; c -= 8) {
      xnn_simd_f32_t vacc0123p0 = xnn_loadu_f16_f32(w + 0);
      xnn_simd_f32_t vacc4567p0 = xnn_loadu_f16_f32(w + 4);

      const xnn_simd_f32_t vi0x0123 = xnn_loadu_f16_f32(i0 + 0);
      const xnn_simd_f32_t vk0x0123 = xnn_loadu_f16_f32(w + 8);
      vacc0123p0 = xnn_fmadd_f32(vi0x0123, vk0x0123, vacc0123p0);
      const xnn_simd_f32_t vi0x4567 = xnn_loadu_f16_f32(i0 + 4);
      const xnn_simd_f32_t vk0x4567 = xnn_loadu_f16_f32(w + 12);
      vacc4567p0 = xnn_fmadd_f32(vi0x4567, vk0x4567, vacc4567p0);
      i0 += 8;
      const xnn_simd_f32_t vi1x0123 = xnn_loadu_f16_f32(i1 + 0);
      const xnn_simd_f32_t vk1x0123 = xnn_loadu_f16_f32(w + 16);
      xnn_simd_f32_t vacc0123p1 = xnn_mul_f32(vi1x0123, vk1x0123);
      const xnn_simd_f32_t vi1x4567 = xnn_loadu_f16_f32(i1 + 4);
      const xnn_simd_f32_t vk1x4567 = xnn_loadu_f16_f32(w + 20);
      xnn_simd_f32_t vacc4567p1 = xnn_mul_f32(vi1x4567, vk1x4567);
      i1 += 8;
      const xnn_simd_f32_t vi2x0123 = xnn_loadu_f16_f32(i2 + 0);
      const xnn_simd_f32_t vk2x0123 = xnn_loadu_f16_f32(w + 24);
      vacc0123p0 = xnn_fmadd_f32(vi2x0123, vk2x0123, vacc0123p0);
      const xnn_simd_f32_t vi2x4567 = xnn_loadu_f16_f32(i2 + 4);
      const xnn_simd_f32_t vk2x4567 = xnn_loadu_f16_f32(w + 28);
      vacc4567p0 = xnn_fmadd_f32(vi2x4567, vk2x4567, vacc4567p0);
      i2 += 8;

      w += 32;

      vacc0123p0 = xnn_add_f32(vacc0123p0, vacc0123p1);
      vacc4567p0 = xnn_add_f32(vacc4567p0, vacc4567p1);

      xnn_simd_f32_t vacc0123 = xnn_max_f32(vacc0123p0, vmin);
      vacc0123 = xnn_min_f32(vacc0123, vmax);
      xnn_store_f32_f16(output + 0, vacc0123);
      xnn_simd_f32_t vacc4567 = xnn_max_f32(vacc4567p0, vmin);
      vacc4567 = xnn_min_f32(vacc4567, vmax);
      xnn_store_f32_f16(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      xnn_simd_f32_t vacc0123p0 = xnn_loadu_f16_f32(w);

      const xnn_simd_f32_t vi0x0123 = xnn_loadu_f16_f32(i0);
      const xnn_simd_f32_t vk0x0123 = xnn_loadu_f16_f32(w + 8);
      vacc0123p0 = xnn_fmadd_f32(vi0x0123, vk0x0123, vacc0123p0);
      i0 += 4;
      const xnn_simd_f32_t vi1x0123 = xnn_loadu_f16_f32(i1);
      const xnn_simd_f32_t vk1x0123 = xnn_loadu_f16_f32(w + 16);
      xnn_simd_f32_t vacc0123p1 = xnn_mul_f32(vi1x0123, vk1x0123);
      i1 += 4;
      const xnn_simd_f32_t vi2x0123 = xnn_loadu_f16_f32(i2);
      const xnn_simd_f32_t vk2x0123 = xnn_loadu_f16_f32(w + 24);
      vacc0123p0 = xnn_fmadd_f32(vi2x0123, vk2x0123, vacc0123p0);
      i2 += 4;

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
      const xnn_simd_f32_t vk0x0123 = xnn_loadu_f16_f32(w + 8);
      vacc0123p0 = xnn_fmadd_f32(vi0x0123, vk0x0123, vacc0123p0);
      const xnn_simd_f32_t vi1x0123 = xnn_load_tail_f16_f32(i1, c);
      const xnn_simd_f32_t vk1x0123 = xnn_loadu_f16_f32(w + 16);
      xnn_simd_f32_t vacc0123p1 = xnn_mul_f32(vi1x0123, vk1x0123);
      const xnn_simd_f32_t vi2x0123 = xnn_load_tail_f16_f32(i2, c);
      const xnn_simd_f32_t vk2x0123 = xnn_loadu_f16_f32(w + 24);
      vacc0123p0 = xnn_fmadd_f32(vi2x0123, vk2x0123, vacc0123p0);

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
