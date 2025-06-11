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


void xnn_f32_dwconv_minmax_ukernel_4p64c__hvx_acc2(
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
    const struct xnn_f32_minmax_params* restrict params)
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
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);
      xnn_simd_f32_t vacc32p1 = xnn_mul_f32(vi1x32, vk1x32);

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
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);
      vacc32p1 = xnn_fmadd_f32(vi3x32, vk3x32, vacc32p1);

      w += 320;

      // Add up all accumulators to vacc0p0
      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);
      vacc32p0 = xnn_add_f32(vacc32p0, vacc32p1);

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
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);

      const xnn_simd_f32_t vi2x0 = xnn_loadu_f32(i2);
      i2 += 32;

      const xnn_simd_f32_t vk2x0 = xnn_load_f32(w + 192);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);

      const xnn_simd_f32_t vi3x0 = xnn_loadu_f32(i3);
      i3 += 32;

      const xnn_simd_f32_t vk3x0 = xnn_load_f32(w + 256);
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);

      w += 32;

      // Add up all accumulators to vacc0p0
      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);

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
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);

      const xnn_simd_f32_t vi2x0 = xnn_load_tail_f32(i2, c);
      const xnn_simd_f32_t vk2x0 = xnn_load_tail_f32(w + 192, c);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);

      const xnn_simd_f32_t vi3x0 = xnn_load_tail_f32(i3, c);
      const xnn_simd_f32_t vk3x0 = xnn_load_tail_f32(w + 256, c);
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);

      // Add up all accumulators to vacc0p0
      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);

      xnn_simd_f32_t vacc0 = xnn_max_f32(vmin, vacc0p0);
      vacc0 = xnn_min_f32(vmax, vacc0);

      xnn_store_tail_f32(output, vacc0, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
