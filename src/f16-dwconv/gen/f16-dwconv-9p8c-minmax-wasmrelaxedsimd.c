// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/unipass.c.in
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
#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"


void xnn_f16_dwconv_minmax_ukernel_9p8c__wasmrelaxedsimd(
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
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(xnn_simd_size_f16 == 8);

  const xnn_simd_f16_t vmin = xnn_set1_f16(params->scalar.min);
  const xnn_simd_f16_t vmax = xnn_set1_f16(params->scalar.max);
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
    for (; c >= 8; c -= 8) {
      xnn_simd_f16_t vacc01234567p0 = xnn_loadu_f16(w + 0);

      const xnn_simd_f16_t vi0x01234567 = xnn_loadu_f16(i0 + 0);
      const xnn_simd_f16_t vk0x01234567 = xnn_loadu_f16(w + 8);
      vacc01234567p0 = xnn_fmadd_f16(vi0x01234567, vk0x01234567, vacc01234567p0);
      i0 += 8;
      const xnn_simd_f16_t vi1x01234567 = xnn_loadu_f16(i1 + 0);
      const xnn_simd_f16_t vk1x01234567 = xnn_loadu_f16(w + 16);
      vacc01234567p0 = xnn_fmadd_f16(vi1x01234567, vk1x01234567, vacc01234567p0);
      i1 += 8;
      const xnn_simd_f16_t vi2x01234567 = xnn_loadu_f16(i2 + 0);
      const xnn_simd_f16_t vk2x01234567 = xnn_loadu_f16(w + 24);
      vacc01234567p0 = xnn_fmadd_f16(vi2x01234567, vk2x01234567, vacc01234567p0);
      i2 += 8;
      const xnn_simd_f16_t vi3x01234567 = xnn_loadu_f16(i3 + 0);
      const xnn_simd_f16_t vk3x01234567 = xnn_loadu_f16(w + 32);
      vacc01234567p0 = xnn_fmadd_f16(vi3x01234567, vk3x01234567, vacc01234567p0);
      i3 += 8;
      const xnn_simd_f16_t vi4x01234567 = xnn_loadu_f16(i4 + 0);
      const xnn_simd_f16_t vk4x01234567 = xnn_loadu_f16(w + 40);
      vacc01234567p0 = xnn_fmadd_f16(vi4x01234567, vk4x01234567, vacc01234567p0);
      i4 += 8;
      const xnn_simd_f16_t vi5x01234567 = xnn_loadu_f16(i5 + 0);
      const xnn_simd_f16_t vk5x01234567 = xnn_loadu_f16(w + 48);
      vacc01234567p0 = xnn_fmadd_f16(vi5x01234567, vk5x01234567, vacc01234567p0);
      i5 += 8;
      const xnn_simd_f16_t vi6x01234567 = xnn_loadu_f16(i6 + 0);
      const xnn_simd_f16_t vk6x01234567 = xnn_loadu_f16(w + 56);
      vacc01234567p0 = xnn_fmadd_f16(vi6x01234567, vk6x01234567, vacc01234567p0);
      i6 += 8;
      const xnn_simd_f16_t vi7x01234567 = xnn_loadu_f16(i7 + 0);
      const xnn_simd_f16_t vk7x01234567 = xnn_loadu_f16(w + 64);
      vacc01234567p0 = xnn_fmadd_f16(vi7x01234567, vk7x01234567, vacc01234567p0);
      i7 += 8;
      const xnn_simd_f16_t vi8x01234567 = xnn_loadu_f16(i8 + 0);
      const xnn_simd_f16_t vk8x01234567 = xnn_loadu_f16(w + 72);
      vacc01234567p0 = xnn_fmadd_f16(vi8x01234567, vk8x01234567, vacc01234567p0);
      i8 += 8;

      w += 80;


      xnn_simd_f16_t vacc01234567 = xnn_max_f16(vacc01234567p0, vmin);
      vacc01234567 = xnn_min_f16(vacc01234567, vmax);
      xnn_storeu_f16(output + 0, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      xnn_simd_f16_t vacc01234567p0 = xnn_loadu_f16(w);

      const xnn_simd_f16_t vi0x01234567 = xnn_load_tail_f16(i0, c);
      const xnn_simd_f16_t vk0x01234567 = xnn_loadu_f16(w + 8);
      vacc01234567p0 = xnn_fmadd_f16(vi0x01234567, vk0x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi1x01234567 = xnn_load_tail_f16(i1, c);
      const xnn_simd_f16_t vk1x01234567 = xnn_loadu_f16(w + 16);
      vacc01234567p0 = xnn_fmadd_f16(vi1x01234567, vk1x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi2x01234567 = xnn_load_tail_f16(i2, c);
      const xnn_simd_f16_t vk2x01234567 = xnn_loadu_f16(w + 24);
      vacc01234567p0 = xnn_fmadd_f16(vi2x01234567, vk2x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi3x01234567 = xnn_load_tail_f16(i3, c);
      const xnn_simd_f16_t vk3x01234567 = xnn_loadu_f16(w + 32);
      vacc01234567p0 = xnn_fmadd_f16(vi3x01234567, vk3x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi4x01234567 = xnn_load_tail_f16(i4, c);
      const xnn_simd_f16_t vk4x01234567 = xnn_loadu_f16(w + 40);
      vacc01234567p0 = xnn_fmadd_f16(vi4x01234567, vk4x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi5x01234567 = xnn_load_tail_f16(i5, c);
      const xnn_simd_f16_t vk5x01234567 = xnn_loadu_f16(w + 48);
      vacc01234567p0 = xnn_fmadd_f16(vi5x01234567, vk5x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi6x01234567 = xnn_load_tail_f16(i6, c);
      const xnn_simd_f16_t vk6x01234567 = xnn_loadu_f16(w + 56);
      vacc01234567p0 = xnn_fmadd_f16(vi6x01234567, vk6x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi7x01234567 = xnn_load_tail_f16(i7, c);
      const xnn_simd_f16_t vk7x01234567 = xnn_loadu_f16(w + 64);
      vacc01234567p0 = xnn_fmadd_f16(vi7x01234567, vk7x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi8x01234567 = xnn_load_tail_f16(i8, c);
      const xnn_simd_f16_t vk8x01234567 = xnn_loadu_f16(w + 72);
      vacc01234567p0 = xnn_fmadd_f16(vi8x01234567, vk8x01234567, vacc01234567p0);


      xnn_simd_f16_t vacc01234567 = xnn_max_f16(vacc01234567p0, vmin);
      vacc01234567 = xnn_min_f16(vacc01234567, vmax);
      xnn_store_tail_f16(output, vacc01234567, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
