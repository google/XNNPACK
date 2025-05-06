// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/simd/f32-hvx.h"

#include "src/xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_9p64c__hvx(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const xnn_simd_f32_t vmin = xnn_set1_f32(params->scalar.min);
  const xnn_simd_f32_t vmax = xnn_set1_f32(params->scalar.max);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 64; c -= 64) {
      xnn_simd_f32_t vacc0p0 = xnn_load_f32(w + 0);
      xnn_simd_f32_t vacc32p0 = xnn_load_f32(w + 32);


      const xnn_simd_f32_t vi0x0 = xnn_loadu_f32(i0 + 0);
      const xnn_simd_f32_t vi0x32 = xnn_loadu_f32(i0 + 32);
      i0 += 64;

      const xnn_simd_f32_t vk0x0 = xnn_load_f32(w + 64);
      const xnn_simd_f32_t vk0x32 = xnn_load_f32(w + 96);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);
      vacc32p0 = xnn_fmadd_f32(vi0x32, vk0x32, vacc32p0);

      const xnn_simd_f32_t vi1x0 = xnn_loadu_f32(i1 + 0);
      const xnn_simd_f32_t vi1x32 = xnn_loadu_f32(i1 + 32);
      i1 += 64;

      const xnn_simd_f32_t vk1x0 = xnn_load_f32(w + 128);
      const xnn_simd_f32_t vk1x32 = xnn_load_f32(w + 160);
      vacc0p0 = xnn_fmadd_f32(vi1x0, vk1x0, vacc0p0);
      vacc32p0 = xnn_fmadd_f32(vi1x32, vk1x32, vacc32p0);

      const xnn_simd_f32_t vi2x0 = xnn_loadu_f32(i2 + 0);
      const xnn_simd_f32_t vi2x32 = xnn_loadu_f32(i2 + 32);
      i2 += 64;

      const xnn_simd_f32_t vk2x0 = xnn_load_f32(w + 192);
      const xnn_simd_f32_t vk2x32 = xnn_load_f32(w + 224);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);
      vacc32p0 = xnn_fmadd_f32(vi2x32, vk2x32, vacc32p0);

      const xnn_simd_f32_t vi3x0 = xnn_loadu_f32(i3 + 0);
      const xnn_simd_f32_t vi3x32 = xnn_loadu_f32(i3 + 32);
      i3 += 64;

      const xnn_simd_f32_t vk3x0 = xnn_load_f32(w + 256);
      const xnn_simd_f32_t vk3x32 = xnn_load_f32(w + 288);
      vacc0p0 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p0);
      vacc32p0 = xnn_fmadd_f32(vi3x32, vk3x32, vacc32p0);

      const xnn_simd_f32_t vi4x0 = xnn_loadu_f32(i4 + 0);
      const xnn_simd_f32_t vi4x32 = xnn_loadu_f32(i4 + 32);
      i4 += 64;

      const xnn_simd_f32_t vk4x0 = xnn_load_f32(w + 320);
      const xnn_simd_f32_t vk4x32 = xnn_load_f32(w + 352);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);
      vacc32p0 = xnn_fmadd_f32(vi4x32, vk4x32, vacc32p0);

      const xnn_simd_f32_t vi5x0 = xnn_loadu_f32(i5 + 0);
      const xnn_simd_f32_t vi5x32 = xnn_loadu_f32(i5 + 32);
      i5 += 64;

      const xnn_simd_f32_t vk5x0 = xnn_load_f32(w + 384);
      const xnn_simd_f32_t vk5x32 = xnn_load_f32(w + 416);
      vacc0p0 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p0);
      vacc32p0 = xnn_fmadd_f32(vi5x32, vk5x32, vacc32p0);

      const xnn_simd_f32_t vi6x0 = xnn_loadu_f32(i6 + 0);
      const xnn_simd_f32_t vi6x32 = xnn_loadu_f32(i6 + 32);
      i6 += 64;

      const xnn_simd_f32_t vk6x0 = xnn_load_f32(w + 448);
      const xnn_simd_f32_t vk6x32 = xnn_load_f32(w + 480);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);
      vacc32p0 = xnn_fmadd_f32(vi6x32, vk6x32, vacc32p0);

      const xnn_simd_f32_t vi7x0 = xnn_loadu_f32(i7 + 0);
      const xnn_simd_f32_t vi7x32 = xnn_loadu_f32(i7 + 32);
      i7 += 64;

      const xnn_simd_f32_t vk7x0 = xnn_load_f32(w + 512);
      const xnn_simd_f32_t vk7x32 = xnn_load_f32(w + 544);
      vacc0p0 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p0);
      vacc32p0 = xnn_fmadd_f32(vi7x32, vk7x32, vacc32p0);

      const xnn_simd_f32_t vi8x0 = xnn_loadu_f32(i8 + 0);
      const xnn_simd_f32_t vi8x32 = xnn_loadu_f32(i8 + 32);
      i8 += 64;

      const xnn_simd_f32_t vk8x0 = xnn_load_f32(w + 576);
      const xnn_simd_f32_t vk8x32 = xnn_load_f32(w + 608);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);
      vacc32p0 = xnn_fmadd_f32(vi8x32, vk8x32, vacc32p0);

      w += 640;


      xnn_simd_f32_t vacc0 = xnn_max_f32(vmin, vacc0p0);
      xnn_simd_f32_t vacc32 = xnn_max_f32(vmin, vacc32p0);
      vacc0 = xnn_min_f32(vmax, vacc0);
      vacc32 = xnn_min_f32(vmax, vacc32);

      xnn_storeu_f32(output + 0, vacc0);
      xnn_storeu_f32(output + 32, vacc32);
      output += 64;
    }
    for (; c >= 32; c -= 32) {
      xnn_simd_f32_t vacc0p0 = xnn_load_f32(w);

      const xnn_simd_f32_t vi0x0 = xnn_loadu_f32(i0);
      i0 += 32;

      const xnn_simd_f32_t vk0x0 = xnn_load_f32(w + 64);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);

      const xnn_simd_f32_t vi1x0 = xnn_loadu_f32(i1);
      i1 += 32;

      const xnn_simd_f32_t vk1x0 = xnn_load_f32(w + 128);
      vacc0p0 = xnn_fmadd_f32(vi1x0, vk1x0, vacc0p0);

      const xnn_simd_f32_t vi2x0 = xnn_loadu_f32(i2);
      i2 += 32;

      const xnn_simd_f32_t vk2x0 = xnn_load_f32(w + 192);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);

      const xnn_simd_f32_t vi3x0 = xnn_loadu_f32(i3);
      i3 += 32;

      const xnn_simd_f32_t vk3x0 = xnn_load_f32(w + 256);
      vacc0p0 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p0);

      const xnn_simd_f32_t vi4x0 = xnn_loadu_f32(i4);
      i4 += 32;

      const xnn_simd_f32_t vk4x0 = xnn_load_f32(w + 320);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);

      const xnn_simd_f32_t vi5x0 = xnn_loadu_f32(i5);
      i5 += 32;

      const xnn_simd_f32_t vk5x0 = xnn_load_f32(w + 384);
      vacc0p0 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p0);

      const xnn_simd_f32_t vi6x0 = xnn_loadu_f32(i6);
      i6 += 32;

      const xnn_simd_f32_t vk6x0 = xnn_load_f32(w + 448);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);

      const xnn_simd_f32_t vi7x0 = xnn_loadu_f32(i7);
      i7 += 32;

      const xnn_simd_f32_t vk7x0 = xnn_load_f32(w + 512);
      vacc0p0 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p0);

      const xnn_simd_f32_t vi8x0 = xnn_loadu_f32(i8);
      i8 += 32;

      const xnn_simd_f32_t vk8x0 = xnn_load_f32(w + 576);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);

      w += 32;


      xnn_simd_f32_t vacc0 = xnn_max_f32(vmin, vacc0p0);
      vacc0 = xnn_min_f32(vmax, vacc0);

      xnn_storeu_f32(output, vacc0);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      xnn_simd_f32_t vacc0p0 = xnn_load_tail_f32(w, c);

      const xnn_simd_f32_t vi0x0 = xnn_load_tail_f32(i0, c);
      const xnn_simd_f32_t vk0x0 = xnn_load_tail_f32(w + 64, c);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);

      const xnn_simd_f32_t vi1x0 = xnn_load_tail_f32(i1, c);
      const xnn_simd_f32_t vk1x0 = xnn_load_tail_f32(w + 128, c);
      vacc0p0 = xnn_fmadd_f32(vi1x0, vk1x0, vacc0p0);

      const xnn_simd_f32_t vi2x0 = xnn_load_tail_f32(i2, c);
      const xnn_simd_f32_t vk2x0 = xnn_load_tail_f32(w + 192, c);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);

      const xnn_simd_f32_t vi3x0 = xnn_load_tail_f32(i3, c);
      const xnn_simd_f32_t vk3x0 = xnn_load_tail_f32(w + 256, c);
      vacc0p0 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p0);

      const xnn_simd_f32_t vi4x0 = xnn_load_tail_f32(i4, c);
      const xnn_simd_f32_t vk4x0 = xnn_load_tail_f32(w + 320, c);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);

      const xnn_simd_f32_t vi5x0 = xnn_load_tail_f32(i5, c);
      const xnn_simd_f32_t vk5x0 = xnn_load_tail_f32(w + 384, c);
      vacc0p0 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p0);

      const xnn_simd_f32_t vi6x0 = xnn_load_tail_f32(i6, c);
      const xnn_simd_f32_t vk6x0 = xnn_load_tail_f32(w + 448, c);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);

      const xnn_simd_f32_t vi7x0 = xnn_load_tail_f32(i7, c);
      const xnn_simd_f32_t vk7x0 = xnn_load_tail_f32(w + 512, c);
      vacc0p0 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p0);

      const xnn_simd_f32_t vi8x0 = xnn_load_tail_f32(i8, c);
      const xnn_simd_f32_t vk8x0 = xnn_load_tail_f32(w + 576, c);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);


      xnn_simd_f32_t vacc0 = xnn_max_f32(vmin, vacc0p0);
      vacc0 = xnn_min_f32(vmax, vacc0);

      xnn_store_tail_f32(output, vacc0, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
