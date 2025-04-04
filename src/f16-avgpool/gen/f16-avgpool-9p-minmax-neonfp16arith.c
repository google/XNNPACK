// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-avgpool/avgpool.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/simd/f16-neonfp16arith.h"

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

void xnn_f16_avgpool_minmax_ukernel_9p__neonfp16arith_u8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const xnn_float16** input,
    size_t input_offset,
    size_t input_pixel_stride,
    const xnn_float16* zero,
    const xnn_float16* multiplier,
    xnn_float16* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(channels != 0);

  const xnn_simd_f16_t vmin = xnn_set1_f16(params->scalar.min);
  const xnn_simd_f16_t vmax = xnn_set1_f16(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  xnn_simd_f16_t vscale = xnn_set1_f16(params->scalar.scale);

  do {
    // Start with the previous output as the zero buffer.
    const xnn_float16* prev_output = zero;

    const xnn_float16** i = input;

    // Passes 0 - n-1: load the output, add 9 inputs.
    size_t k = kernel_elements;
    for (; k > 9; k -= 9) {
      const xnn_float16* i0 = *i++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const xnn_float16*) ((uintptr_t) i0 + input_offset);
      }
      const xnn_float16* i1 = *i++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const xnn_float16*) ((uintptr_t) i1 + input_offset);
      }
      const xnn_float16* i2 = *i++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const xnn_float16*) ((uintptr_t) i2 + input_offset);
      }
      const xnn_float16* i3 = *i++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const xnn_float16*) ((uintptr_t) i3 + input_offset);
      }
      const xnn_float16* i4 = *i++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const xnn_float16*) ((uintptr_t) i4 + input_offset);
      }
      const xnn_float16* i5 = *i++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const xnn_float16*) ((uintptr_t) i5 + input_offset);
      }
      const xnn_float16* i6 = *i++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const xnn_float16*) ((uintptr_t) i6 + input_offset);
      }
      const xnn_float16* i7 = *i++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const xnn_float16*) ((uintptr_t) i7 + input_offset);
      }
      const xnn_float16* i8 = *i++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const xnn_float16*) ((uintptr_t) i8 + input_offset);
      }

      xnn_float16* o = output;
      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const xnn_simd_f16_t vi0 = xnn_loadu_f16(i0); i0 += 8;
        const xnn_simd_f16_t vi1 = xnn_loadu_f16(i1); i1 += 8;
        const xnn_simd_f16_t vi2 = xnn_loadu_f16(i2); i2 += 8;
        const xnn_simd_f16_t vi3 = xnn_loadu_f16(i3); i3 += 8;
        const xnn_simd_f16_t vi4 = xnn_loadu_f16(i4); i4 += 8;
        const xnn_simd_f16_t vi5 = xnn_loadu_f16(i5); i5 += 8;
        const xnn_simd_f16_t vi6 = xnn_loadu_f16(i6); i6 += 8;
        const xnn_simd_f16_t vi7 = xnn_loadu_f16(i7); i7 += 8;
        const xnn_simd_f16_t vi8 = xnn_loadu_f16(i8); i8 += 8;
        const xnn_simd_f16_t vprev = xnn_loadu_f16(prev_output); prev_output += 8;

        const xnn_simd_f16_t vsum018 = xnn_add_f16(xnn_add_f16(vi0, vi1), vi8);
        const xnn_simd_f16_t vsum23 = xnn_add_f16(vi2, vi3);
        const xnn_simd_f16_t vsum45 = xnn_add_f16(vi4, vi5);
        const xnn_simd_f16_t vsum67 = xnn_add_f16(vi6, vi7);

        const xnn_simd_f16_t vsum2345 = xnn_add_f16(vsum23, vsum45);
        const xnn_simd_f16_t vsum01678 = xnn_add_f16(vsum018, vsum67);
        const xnn_simd_f16_t vsum012345678 = xnn_add_f16(vsum2345, vsum01678);
        const xnn_simd_f16_t vacc = xnn_add_f16(vprev, vsum012345678);
        xnn_storeu_f16(o, vacc); o += 8;
      }
      if (c > 0) {
        const xnn_simd_f16_t vi0 = xnn_load_tail_f16(i0, c);
        const xnn_simd_f16_t vi1 = xnn_load_tail_f16(i1, c);
        const xnn_simd_f16_t vi2 = xnn_load_tail_f16(i2, c);
        const xnn_simd_f16_t vi3 = xnn_load_tail_f16(i3, c);
        const xnn_simd_f16_t vi4 = xnn_load_tail_f16(i4, c);
        const xnn_simd_f16_t vi5 = xnn_load_tail_f16(i5, c);
        const xnn_simd_f16_t vi6 = xnn_load_tail_f16(i6, c);
        const xnn_simd_f16_t vi7 = xnn_load_tail_f16(i7, c);
        const xnn_simd_f16_t vi8 = xnn_load_tail_f16(i8, c);
        const xnn_simd_f16_t vprev = xnn_load_tail_safe_f16(prev_output, c);

        const xnn_simd_f16_t vsum018 = xnn_add_f16(xnn_add_f16(vi0, vi1), vi8);
        const xnn_simd_f16_t vsum23 = xnn_add_f16(vi2, vi3);
        const xnn_simd_f16_t vsum45 = xnn_add_f16(vi4, vi5);
        const xnn_simd_f16_t vsum67 = xnn_add_f16(vi6, vi7);

        const xnn_simd_f16_t vsum2345 = xnn_add_f16(vsum23, vsum45);
        const xnn_simd_f16_t vsum01678 = xnn_add_f16(vsum018, vsum67);
        const xnn_simd_f16_t vsum012345678 = xnn_add_f16(vsum2345, vsum01678);
        const xnn_simd_f16_t vacc = xnn_add_f16(vprev, vsum012345678);
        xnn_store_tail_f16(o, vacc, c);
      }

      // Subsequent passes read from the previous output.
      prev_output = output;
    }

    // Final pass: load the output, add remaining kernel elements, apply scaling/min/max
    const xnn_float16* i0 = 0 < k ? *i++ : zero;
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const xnn_float16*) ((uintptr_t) i0 + input_offset);
    }
    const xnn_float16* i1 = 1 < k ? *i++ : zero;
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const xnn_float16*) ((uintptr_t) i1 + input_offset);
    }
    const xnn_float16* i2 = 2 < k ? *i++ : zero;
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const xnn_float16*) ((uintptr_t) i2 + input_offset);
    }
    const xnn_float16* i3 = 3 < k ? *i++ : zero;
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const xnn_float16*) ((uintptr_t) i3 + input_offset);
    }
    const xnn_float16* i4 = 4 < k ? *i++ : zero;
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const xnn_float16*) ((uintptr_t) i4 + input_offset);
    }
    const xnn_float16* i5 = 5 < k ? *i++ : zero;
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const xnn_float16*) ((uintptr_t) i5 + input_offset);
    }
    const xnn_float16* i6 = 6 < k ? *i++ : zero;
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const xnn_float16*) ((uintptr_t) i6 + input_offset);
    }
    const xnn_float16* i7 = 7 < k ? *i++ : zero;
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const xnn_float16*) ((uintptr_t) i7 + input_offset);
    }
    const xnn_float16* i8 = 8 < k ? *i++ : zero;
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const xnn_float16*) ((uintptr_t) i8 + input_offset);
    }

    if (multiplier != NULL) {
      vscale = xnn_set1_f16(*multiplier++);
    }
    xnn_float16* o = output;
    size_t c = channels;
    for (; c >= 8; c -= 8) {
      const xnn_simd_f16_t vi0 = xnn_loadu_f16(i0); i0 += 8;
      const xnn_simd_f16_t vi1 = xnn_loadu_f16(i1); i1 += 8;
      const xnn_simd_f16_t vi2 = xnn_loadu_f16(i2); i2 += 8;
      const xnn_simd_f16_t vi3 = xnn_loadu_f16(i3); i3 += 8;
      const xnn_simd_f16_t vi4 = xnn_loadu_f16(i4); i4 += 8;
      const xnn_simd_f16_t vi5 = xnn_loadu_f16(i5); i5 += 8;
      const xnn_simd_f16_t vi6 = xnn_loadu_f16(i6); i6 += 8;
      const xnn_simd_f16_t vi7 = xnn_loadu_f16(i7); i7 += 8;
      const xnn_simd_f16_t vi8 = xnn_loadu_f16(i8); i8 += 8;
      const xnn_simd_f16_t vprev = xnn_loadu_f16(prev_output); prev_output += 8;

      const xnn_simd_f16_t vsum018 = xnn_add_f16(xnn_add_f16(vi0, vi1), vi8);
      const xnn_simd_f16_t vsum23 = xnn_add_f16(vi2, vi3);
      const xnn_simd_f16_t vsum45 = xnn_add_f16(vi4, vi5);
      const xnn_simd_f16_t vsum67 = xnn_add_f16(vi6, vi7);

      const xnn_simd_f16_t vsum2345 = xnn_add_f16(vsum23, vsum45);
      const xnn_simd_f16_t vsum01678 = xnn_add_f16(vsum018, vsum67);
      const xnn_simd_f16_t vsum012345678 = xnn_add_f16(vsum2345, vsum01678);

      xnn_simd_f16_t vacc = xnn_add_f16(vprev, vsum012345678);

      vacc = xnn_mul_f16(vacc, vscale);
      vacc = xnn_max_f16(vacc, vmin);
      vacc = xnn_min_f16(vacc, vmax);

      xnn_storeu_f16(o, vacc); o += 8;
    }
    if (c > 0) {
      const xnn_simd_f16_t vi0 = xnn_load_tail_f16(i0, c);
      const xnn_simd_f16_t vi1 = xnn_load_tail_f16(i1, c);
      const xnn_simd_f16_t vi2 = xnn_load_tail_f16(i2, c);
      const xnn_simd_f16_t vi3 = xnn_load_tail_f16(i3, c);
      const xnn_simd_f16_t vi4 = xnn_load_tail_f16(i4, c);
      const xnn_simd_f16_t vi5 = xnn_load_tail_f16(i5, c);
      const xnn_simd_f16_t vi6 = xnn_load_tail_f16(i6, c);
      const xnn_simd_f16_t vi7 = xnn_load_tail_f16(i7, c);
      const xnn_simd_f16_t vi8 = xnn_load_tail_f16(i8, c);
      const xnn_simd_f16_t vprev = xnn_load_tail_safe_f16(prev_output, c);

      const xnn_simd_f16_t vsum018 = xnn_add_f16(xnn_add_f16(vi0, vi1), vi8);
      const xnn_simd_f16_t vsum23 = xnn_add_f16(vi2, vi3);
      const xnn_simd_f16_t vsum45 = xnn_add_f16(vi4, vi5);
      const xnn_simd_f16_t vsum67 = xnn_add_f16(vi6, vi7);

      const xnn_simd_f16_t vsum2345 = xnn_add_f16(vsum23, vsum45);
      const xnn_simd_f16_t vsum01678 = xnn_add_f16(vsum018, vsum67);
      const xnn_simd_f16_t vsum012345678 = xnn_add_f16(vsum2345, vsum01678);

      xnn_simd_f16_t vacc = xnn_add_f16(vprev, vsum012345678);

      vacc = xnn_mul_f16(vacc, vscale);
      vacc = xnn_max_f16(vacc, vmin);
      vacc = xnn_min_f16(vacc, vmax);

      xnn_store_tail_f16(o, vacc, c); o += c;
    }

    input = (const xnn_float16**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
