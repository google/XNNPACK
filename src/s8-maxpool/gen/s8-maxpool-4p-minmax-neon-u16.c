// Auto-generated file. Do not edit!
//   Template: src/f32-maxpool/maxpool.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>


// On some architectures, we have max(u8, u8) but not max(s8, s8). We can emulate max(s8, s8) on these architectures by
// xoring with the sign bit mask.
#define xnn_load_impl(x) xnn_loadu_s8(x)
#define xnn_load_tail_impl(x, c) xnn_load_tail_s8(x, c)
#define xnn_load_tail_safe_impl(x, c) xnn_load_tail_safe_s8(x, c)
#define xnn_pre_store_impl(x) x

#include "src/xnnpack/simd/s8-neon.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

void xnn_s8_maxpool_minmax_ukernel_4p__neon_u16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(channels != 0);

  const xnn_simd_s8_t vmin = xnn_set1_s8(params->scalar.min);
  const xnn_simd_s8_t vmax = xnn_set1_s8(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    const int8_t** i = (const int8_t**) input;

    // First pass: load the inputs, store the max pool in the output.
    const int8_t* i0 = *i++;
    const int8_t* i1 = 1 < kernel_elements ? *i++ : i0;
    const int8_t* i2 = 2 < kernel_elements ? *i++ : i0;
    const int8_t* i3 = 3 < kernel_elements ? *i++ : i0;
    i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);

    int8_t* o = (int8_t*) output;
    size_t c = channels;
    for (; c >= 16; c -= 16) {
      const xnn_simd_s8_t vi0 = xnn_load_impl(i0); i0 += 16;
      const xnn_simd_s8_t vi1 = xnn_load_impl(i1); i1 += 16;
      const xnn_simd_s8_t vi2 = xnn_load_impl(i2); i2 += 16;
      const xnn_simd_s8_t vi3 = xnn_load_impl(i3); i3 += 16;

      const xnn_simd_s8_t vmax01 = xnn_max_s8(vi0, vi1);
      const xnn_simd_s8_t vmax23 = xnn_max_s8(vi2, vi3);
      xnn_simd_s8_t vacc = xnn_max_s8(vmax01, vmax23);

      vacc = xnn_max_s8(vacc, vmin);
      vacc = xnn_min_s8(vacc, vmax);

      vacc = xnn_pre_store_impl(vacc);

      xnn_storeu_s8(o, vacc); o += 16;
    }
    if (c > 0) {
      const xnn_simd_s8_t vi0 = xnn_load_tail_impl(i0, c);
      const xnn_simd_s8_t vi1 = xnn_load_tail_impl(i1, c);
      const xnn_simd_s8_t vi2 = xnn_load_tail_impl(i2, c);
      const xnn_simd_s8_t vi3 = xnn_load_tail_impl(i3, c);

      const xnn_simd_s8_t vmax01 = xnn_max_s8(vi0, vi1);
      const xnn_simd_s8_t vmax23 = xnn_max_s8(vi2, vi3);
      xnn_simd_s8_t vacc = xnn_max_s8(vmax01, vmax23);

      vacc = xnn_max_s8(vacc, vmin);
      vacc = xnn_min_s8(vacc, vmax);

      vacc = xnn_pre_store_impl(vacc);

      xnn_store_tail_s8(o, vacc, c); o += c;
    }

    // Passes 1 - n: Max more inputs to the output.
    o = (int8_t*) output;
    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 4; k > 0; k -= 4) {
      const int8_t* i0 = *i++;
      const int8_t* i1 = 1 < k ? *i++ : i0;
      const int8_t* i2 = 2 < k ? *i++ : i0;
      const int8_t* i3 = 3 < k ? *i++ : i0;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);

      int8_t* o = (int8_t*) output;
      size_t c = channels;
      for (; c >= 16; c -= 16) {
        const xnn_simd_s8_t vi0 = xnn_load_impl(i0); i0 += 16;
        const xnn_simd_s8_t vi1 = xnn_load_impl(i1); i1 += 16;
        const xnn_simd_s8_t vi2 = xnn_load_impl(i2); i2 += 16;
        const xnn_simd_s8_t vi3 = xnn_load_impl(i3); i3 += 16;
        const xnn_simd_s8_t vprev = xnn_load_impl(o);

        const xnn_simd_s8_t vmax01 = xnn_max_s8(vi0, vi1);
        const xnn_simd_s8_t vmax23 = xnn_max_s8(vi2, vi3);
        const xnn_simd_s8_t vmax0123 = xnn_max_s8(vmax01, vmax23);

        xnn_simd_s8_t vacc = xnn_max_s8(vprev, vmax0123);

        vacc = xnn_min_s8(vacc, vmax);

        vacc = xnn_pre_store_impl(vacc);

        xnn_storeu_s8(o, vacc); o += 16;
      }
      if (c > 0) {
        const xnn_simd_s8_t vi0 = xnn_load_tail_impl(i0, c);
        const xnn_simd_s8_t vi1 = xnn_load_tail_impl(i1, c);
        const xnn_simd_s8_t vi2 = xnn_load_tail_impl(i2, c);
        const xnn_simd_s8_t vi3 = xnn_load_tail_impl(i3, c);
        const xnn_simd_s8_t vprev = xnn_load_tail_safe_impl(o, c);

        const xnn_simd_s8_t vmax01 = xnn_max_s8(vi0, vi1);
        const xnn_simd_s8_t vmax23 = xnn_max_s8(vi2, vi3);
        const xnn_simd_s8_t vmax0123 = xnn_max_s8(vmax01, vmax23);

        xnn_simd_s8_t vacc = xnn_max_s8(vprev, vmax0123);

        vacc = xnn_min_s8(vacc, vmax);

        vacc = xnn_pre_store_impl(vacc);

        xnn_store_tail_s8(o, vacc, c);
      }
    }

    input = (const int8_t**) ((uintptr_t) input + input_increment);
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
