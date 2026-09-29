// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-avgpool/avgpool-f16-f32acc.c.in
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
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f32-scalar.h"
#include "src/xnnpack/simd/f16-scalar.h"


static XNN_INLINE xnn_simd_f32_t xnn_loadu_f16_f32(
    const xnn_float16* input)
{
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


void xnn_f16_f32acc_avgpool_minmax_ukernel_9p__scalar_u1(
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
    const struct xnn_f16_scaleminmax_params* restrict params)
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(xnn_simd_size_f32 == 1);

  const xnn_simd_f32_t vmin =
      xnn_set1_f32(xnn_float16_to_float(params->scalar.min));
  const xnn_simd_f32_t vmax =
      xnn_set1_f32(xnn_float16_to_float(params->scalar.max));
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  xnn_simd_f32_t vscale =
      xnn_set1_f32(xnn_float16_to_float(params->scalar.scale));

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
      for (; c >= 1; c -= 1) {
        const xnn_simd_f32_t vi0 = xnn_loadu_f16_f32(i0); i0 += 1;
        const xnn_simd_f32_t vi1 = xnn_loadu_f16_f32(i1); i1 += 1;
        const xnn_simd_f32_t vi2 = xnn_loadu_f16_f32(i2); i2 += 1;
        const xnn_simd_f32_t vi3 = xnn_loadu_f16_f32(i3); i3 += 1;
        const xnn_simd_f32_t vi4 = xnn_loadu_f16_f32(i4); i4 += 1;
        const xnn_simd_f32_t vi5 = xnn_loadu_f16_f32(i5); i5 += 1;
        const xnn_simd_f32_t vi6 = xnn_loadu_f16_f32(i6); i6 += 1;
        const xnn_simd_f32_t vi7 = xnn_loadu_f16_f32(i7); i7 += 1;
        const xnn_simd_f32_t vi8 = xnn_loadu_f16_f32(i8); i8 += 1;
        const xnn_simd_f32_t vprev = xnn_loadu_f16_f32(prev_output); prev_output += 1;

        const xnn_simd_f32_t vsum018 = xnn_add_f32(xnn_add_f32(vi0, vi1), vi8);
        const xnn_simd_f32_t vsum23 = xnn_add_f32(vi2, vi3);
        const xnn_simd_f32_t vsum45 = xnn_add_f32(vi4, vi5);
        const xnn_simd_f32_t vsum67 = xnn_add_f32(vi6, vi7);

        const xnn_simd_f32_t vsum2345 = xnn_add_f32(vsum23, vsum45);
        const xnn_simd_f32_t vsum01678 = xnn_add_f32(vsum018, vsum67);
        const xnn_simd_f32_t vsum012345678 = xnn_add_f32(vsum2345, vsum01678);
        const xnn_simd_f32_t vacc = xnn_add_f32(vprev, vsum012345678);
        xnn_store_f32_f16(o, vacc); o += 1;
      }

      // Subsequent passes read from the previous output.
      prev_output = output;
    }

    // Final pass: add remaining kernel elements, then scale and clamp.
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
      vscale = xnn_set1_f32(xnn_float16_to_float(*multiplier++));
    }
    xnn_float16* o = output;
    size_t c = channels;
    for (; c >= 1; c -= 1) {
      const xnn_simd_f32_t vi0 = xnn_loadu_f16_f32(i0); i0 += 1;
      const xnn_simd_f32_t vi1 = xnn_loadu_f16_f32(i1); i1 += 1;
      const xnn_simd_f32_t vi2 = xnn_loadu_f16_f32(i2); i2 += 1;
      const xnn_simd_f32_t vi3 = xnn_loadu_f16_f32(i3); i3 += 1;
      const xnn_simd_f32_t vi4 = xnn_loadu_f16_f32(i4); i4 += 1;
      const xnn_simd_f32_t vi5 = xnn_loadu_f16_f32(i5); i5 += 1;
      const xnn_simd_f32_t vi6 = xnn_loadu_f16_f32(i6); i6 += 1;
      const xnn_simd_f32_t vi7 = xnn_loadu_f16_f32(i7); i7 += 1;
      const xnn_simd_f32_t vi8 = xnn_loadu_f16_f32(i8); i8 += 1;
      const xnn_simd_f32_t vprev = xnn_loadu_f16_f32(prev_output); prev_output += 1;

      const xnn_simd_f32_t vsum018 = xnn_add_f32(xnn_add_f32(vi0, vi1), vi8);
      const xnn_simd_f32_t vsum23 = xnn_add_f32(vi2, vi3);
      const xnn_simd_f32_t vsum45 = xnn_add_f32(vi4, vi5);
      const xnn_simd_f32_t vsum67 = xnn_add_f32(vi6, vi7);

      const xnn_simd_f32_t vsum2345 = xnn_add_f32(vsum23, vsum45);
      const xnn_simd_f32_t vsum01678 = xnn_add_f32(vsum018, vsum67);
      const xnn_simd_f32_t vsum012345678 = xnn_add_f32(vsum2345, vsum01678);

      xnn_simd_f32_t vacc = xnn_add_f32(vprev, vsum012345678);
      vacc = xnn_mul_f32(vacc, vscale);
      vacc = xnn_max_f32(vacc, vmin);
      vacc = xnn_min_f32(vacc, vmax);

      xnn_store_f32_f16(o, vacc); o += 1;
    }

    input = (const xnn_float16**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
