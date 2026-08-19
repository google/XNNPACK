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
#include "src/xnnpack/simd/f16-scalar.h"


void xnn_f16_dwconv_minmax_ukernel_3p1c__scalar_acc2(
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
  assert(xnn_simd_size_f16 == 1);

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
    input = (const xnn_float16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const xnn_float16* w = weights;
    for (; c >= 1; c -= 1) {
      xnn_simd_f16_t vacc0p0 = xnn_loadu_f16(w + 0);

      const xnn_simd_f16_t vi0x0 = xnn_loadu_f16(i0 + 0);
      const xnn_simd_f16_t vk0x0 = xnn_loadu_f16(w + 1);
      vacc0p0 = xnn_fmadd_f16(vi0x0, vk0x0, vacc0p0);
      i0 += 1;
      const xnn_simd_f16_t vi1x0 = xnn_loadu_f16(i1 + 0);
      const xnn_simd_f16_t vk1x0 = xnn_loadu_f16(w + 2);
      xnn_simd_f16_t vacc0p1 = xnn_mul_f16(vi1x0, vk1x0);
      i1 += 1;
      const xnn_simd_f16_t vi2x0 = xnn_loadu_f16(i2 + 0);
      const xnn_simd_f16_t vk2x0 = xnn_loadu_f16(w + 3);
      vacc0p0 = xnn_fmadd_f16(vi2x0, vk2x0, vacc0p0);
      i2 += 1;

      w += 4;

      vacc0p0 = xnn_add_f16(vacc0p0, vacc0p1);

      xnn_simd_f16_t vacc0 = xnn_max_f16(vacc0p0, vmin);
      vacc0 = xnn_min_f16(vacc0, vmax);
      xnn_storeu_f16(output + 0, vacc0);
      output += 1;
    }
    if XNN_UNLIKELY(c != 0) {
      xnn_simd_f16_t vacc0p0 = xnn_loadu_f16(w);

      const xnn_simd_f16_t vi0x0 = xnn_load_tail_f16(i0, c);
      const xnn_simd_f16_t vk0x0 = xnn_loadu_f16(w + 1);
      vacc0p0 = xnn_fmadd_f16(vi0x0, vk0x0, vacc0p0);
      const xnn_simd_f16_t vi1x0 = xnn_load_tail_f16(i1, c);
      const xnn_simd_f16_t vk1x0 = xnn_loadu_f16(w + 2);
      xnn_simd_f16_t vacc0p1 = xnn_mul_f16(vi1x0, vk1x0);
      const xnn_simd_f16_t vi2x0 = xnn_load_tail_f16(i2, c);
      const xnn_simd_f16_t vk2x0 = xnn_loadu_f16(w + 3);
      vacc0p0 = xnn_fmadd_f16(vi2x0, vk2x0, vacc0p0);

      vacc0p0 = xnn_add_f16(vacc0p0, vacc0p1);

      xnn_simd_f16_t vacc0 = xnn_max_f16(vacc0p0, vmin);
      vacc0 = xnn_min_f16(vacc0, vmax);
      xnn_store_tail_f16(output, vacc0, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
