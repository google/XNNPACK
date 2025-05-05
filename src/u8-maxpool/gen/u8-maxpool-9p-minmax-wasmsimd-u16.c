// clang-format off
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
#define xnn_load_impl(x) xnn_loadu_u8(x)
#define xnn_load_tail_impl(x, c) xnn_load_tail_u8(x, c)
#define xnn_load_tail_safe_impl(x, c) xnn_load_tail_safe_u8(x, c)
#define xnn_pre_store_impl(x) x

#include "src/xnnpack/simd/u8-wasmsimd.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

void xnn_u8_maxpool_minmax_ukernel_9p__wasmsimd_u16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    size_t input_pixel_stride,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(channels != 0);

  const xnn_simd_u8_t vmin = xnn_set1_u8(params->scalar.min);
  const xnn_simd_u8_t vmax = xnn_set1_u8(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    const uint8_t** i = (const uint8_t**) input;

    // First pass: load the inputs, store the max pool in the output.
    const uint8_t* i0 = *i++;
    const uint8_t* i1 = 1 < kernel_elements ? *i++ : i0;
    const uint8_t* i2 = 2 < kernel_elements ? *i++ : i0;
    const uint8_t* i3 = 3 < kernel_elements ? *i++ : i0;
    const uint8_t* i4 = 4 < kernel_elements ? *i++ : i0;
    const uint8_t* i5 = 5 < kernel_elements ? *i++ : i0;
    const uint8_t* i6 = 6 < kernel_elements ? *i++ : i0;
    const uint8_t* i7 = 7 < kernel_elements ? *i++ : i0;
    const uint8_t* i8 = 8 < kernel_elements ? *i++ : i0;
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);

    uint8_t* o = (uint8_t*) output;
    size_t c = channels;
    for (; c >= 16; c -= 16) {
      const xnn_simd_u8_t vi0 = xnn_load_impl(i0); i0 += 16;
      const xnn_simd_u8_t vi1 = xnn_load_impl(i1); i1 += 16;
      const xnn_simd_u8_t vi2 = xnn_load_impl(i2); i2 += 16;
      const xnn_simd_u8_t vi3 = xnn_load_impl(i3); i3 += 16;
      const xnn_simd_u8_t vi4 = xnn_load_impl(i4); i4 += 16;
      const xnn_simd_u8_t vi5 = xnn_load_impl(i5); i5 += 16;
      const xnn_simd_u8_t vi6 = xnn_load_impl(i6); i6 += 16;
      const xnn_simd_u8_t vi7 = xnn_load_impl(i7); i7 += 16;
      const xnn_simd_u8_t vi8 = xnn_load_impl(i8); i8 += 16;

      const xnn_simd_u8_t vmax018 = xnn_max_u8(xnn_max_u8(vi0, vi1), vi8);
      const xnn_simd_u8_t vmax23 = xnn_max_u8(vi2, vi3);
      const xnn_simd_u8_t vmax45 = xnn_max_u8(vi4, vi5);
      const xnn_simd_u8_t vmax67 = xnn_max_u8(vi6, vi7);

      const xnn_simd_u8_t vmax2345 = xnn_max_u8(vmax23, vmax45);
      const xnn_simd_u8_t vmax01678 = xnn_max_u8(vmax018, vmax67);
      xnn_simd_u8_t vacc = xnn_max_u8(vmax2345, vmax01678);

      vacc = xnn_max_u8(vacc, vmin);
      vacc = xnn_min_u8(vacc, vmax);

      vacc = xnn_pre_store_impl(vacc);

      xnn_storeu_u8(o, vacc); o += 16;
    }
    if (c > 0) {
      const xnn_simd_u8_t vi0 = xnn_load_tail_impl(i0, c);
      const xnn_simd_u8_t vi1 = xnn_load_tail_impl(i1, c);
      const xnn_simd_u8_t vi2 = xnn_load_tail_impl(i2, c);
      const xnn_simd_u8_t vi3 = xnn_load_tail_impl(i3, c);
      const xnn_simd_u8_t vi4 = xnn_load_tail_impl(i4, c);
      const xnn_simd_u8_t vi5 = xnn_load_tail_impl(i5, c);
      const xnn_simd_u8_t vi6 = xnn_load_tail_impl(i6, c);
      const xnn_simd_u8_t vi7 = xnn_load_tail_impl(i7, c);
      const xnn_simd_u8_t vi8 = xnn_load_tail_impl(i8, c);

      const xnn_simd_u8_t vmax018 = xnn_max_u8(xnn_max_u8(vi0, vi1), vi8);
      const xnn_simd_u8_t vmax23 = xnn_max_u8(vi2, vi3);
      const xnn_simd_u8_t vmax45 = xnn_max_u8(vi4, vi5);
      const xnn_simd_u8_t vmax67 = xnn_max_u8(vi6, vi7);

      const xnn_simd_u8_t vmax2345 = xnn_max_u8(vmax23, vmax45);
      const xnn_simd_u8_t vmax01678 = xnn_max_u8(vmax018, vmax67);
      xnn_simd_u8_t vacc = xnn_max_u8(vmax2345, vmax01678);

      vacc = xnn_max_u8(vacc, vmin);
      vacc = xnn_min_u8(vacc, vmax);

      vacc = xnn_pre_store_impl(vacc);

      xnn_store_tail_u8(o, vacc, c); o += c;
    }

    // Passes 1 - n: Max more inputs to the output.
    o = (uint8_t*) output;
    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 9) {
      const uint8_t* i0 = *i++;
      const uint8_t* i1 = 1 < k ? *i++ : i0;
      const uint8_t* i2 = 2 < k ? *i++ : i0;
      const uint8_t* i3 = 3 < k ? *i++ : i0;
      const uint8_t* i4 = 4 < k ? *i++ : i0;
      const uint8_t* i5 = 5 < k ? *i++ : i0;
      const uint8_t* i6 = 6 < k ? *i++ : i0;
      const uint8_t* i7 = 7 < k ? *i++ : i0;
      const uint8_t* i8 = 8 < k ? *i++ : i0;
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);

      uint8_t* o = (uint8_t*) output;
      size_t c = channels;
      for (; c >= 16; c -= 16) {
        const xnn_simd_u8_t vi0 = xnn_load_impl(i0); i0 += 16;
        const xnn_simd_u8_t vi1 = xnn_load_impl(i1); i1 += 16;
        const xnn_simd_u8_t vi2 = xnn_load_impl(i2); i2 += 16;
        const xnn_simd_u8_t vi3 = xnn_load_impl(i3); i3 += 16;
        const xnn_simd_u8_t vi4 = xnn_load_impl(i4); i4 += 16;
        const xnn_simd_u8_t vi5 = xnn_load_impl(i5); i5 += 16;
        const xnn_simd_u8_t vi6 = xnn_load_impl(i6); i6 += 16;
        const xnn_simd_u8_t vi7 = xnn_load_impl(i7); i7 += 16;
        const xnn_simd_u8_t vi8 = xnn_load_impl(i8); i8 += 16;
        const xnn_simd_u8_t vprev = xnn_load_impl(o);

        const xnn_simd_u8_t vmax018 = xnn_max_u8(xnn_max_u8(vi0, vi1), vi8);
        const xnn_simd_u8_t vmax23 = xnn_max_u8(vi2, vi3);
        const xnn_simd_u8_t vmax45 = xnn_max_u8(vi4, vi5);
        const xnn_simd_u8_t vmax67 = xnn_max_u8(vi6, vi7);

        const xnn_simd_u8_t vmax2345 = xnn_max_u8(vmax23, vmax45);
        const xnn_simd_u8_t vmax01678 = xnn_max_u8(vmax018, vmax67);
        const xnn_simd_u8_t vmax012345678 = xnn_max_u8(vmax2345, vmax01678);

        xnn_simd_u8_t vacc = xnn_max_u8(vprev, vmax012345678);

        vacc = xnn_min_u8(vacc, vmax);

        vacc = xnn_pre_store_impl(vacc);

        xnn_storeu_u8(o, vacc); o += 16;
      }
      if (c > 0) {
        const xnn_simd_u8_t vi0 = xnn_load_tail_impl(i0, c);
        const xnn_simd_u8_t vi1 = xnn_load_tail_impl(i1, c);
        const xnn_simd_u8_t vi2 = xnn_load_tail_impl(i2, c);
        const xnn_simd_u8_t vi3 = xnn_load_tail_impl(i3, c);
        const xnn_simd_u8_t vi4 = xnn_load_tail_impl(i4, c);
        const xnn_simd_u8_t vi5 = xnn_load_tail_impl(i5, c);
        const xnn_simd_u8_t vi6 = xnn_load_tail_impl(i6, c);
        const xnn_simd_u8_t vi7 = xnn_load_tail_impl(i7, c);
        const xnn_simd_u8_t vi8 = xnn_load_tail_impl(i8, c);
        const xnn_simd_u8_t vprev = xnn_load_tail_safe_impl(o, c);

        const xnn_simd_u8_t vmax018 = xnn_max_u8(xnn_max_u8(vi0, vi1), vi8);
        const xnn_simd_u8_t vmax23 = xnn_max_u8(vi2, vi3);
        const xnn_simd_u8_t vmax45 = xnn_max_u8(vi4, vi5);
        const xnn_simd_u8_t vmax67 = xnn_max_u8(vi6, vi7);

        const xnn_simd_u8_t vmax2345 = xnn_max_u8(vmax23, vmax45);
        const xnn_simd_u8_t vmax01678 = xnn_max_u8(vmax018, vmax67);
        const xnn_simd_u8_t vmax012345678 = xnn_max_u8(vmax2345, vmax01678);

        xnn_simd_u8_t vacc = xnn_max_u8(vprev, vmax012345678);

        vacc = xnn_min_u8(vacc, vmax);

        vacc = xnn_pre_store_impl(vacc);

        xnn_store_tail_u8(o, vacc, c);
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
