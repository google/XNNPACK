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
#include "src/xnnpack/simd/f32-hvx.h"


static XNN_INLINE xnn_simd_f32_t xnn_loadu_f16_f32(const xnn_float16* input) {
  return xnn_cvt_f32_f16(xnn_loadu_f16(input));
}

static XNN_INLINE xnn_simd_f32_t xnn_load_sum9_f16_f32(
    const xnn_float16* i0, const xnn_float16* i1, const xnn_float16* i2,
    const xnn_float16* i3, const xnn_float16* i4, const xnn_float16* i5,
    const xnn_float16* i6, const xnn_float16* i7, const xnn_float16* i8,
    uintptr_t idx) {
  const xnn_simd_f32_t vi0 = xnn_loadu_f16_f32(
      (const xnn_float16*) ((uintptr_t) i0 + idx));
  const xnn_simd_f32_t vi1 = xnn_loadu_f16_f32(
      (const xnn_float16*) ((uintptr_t) i1 + idx));
  const xnn_simd_f32_t vi2 = xnn_loadu_f16_f32(
      (const xnn_float16*) ((uintptr_t) i2 + idx));
  const xnn_simd_f32_t vi3 = xnn_loadu_f16_f32(
      (const xnn_float16*) ((uintptr_t) i3 + idx));
  const xnn_simd_f32_t vi4 = xnn_loadu_f16_f32(
      (const xnn_float16*) ((uintptr_t) i4 + idx));
  const xnn_simd_f32_t vi5 = xnn_loadu_f16_f32(
      (const xnn_float16*) ((uintptr_t) i5 + idx));
  const xnn_simd_f32_t vi6 = xnn_loadu_f16_f32(
      (const xnn_float16*) ((uintptr_t) i6 + idx));
  const xnn_simd_f32_t vi7 = xnn_loadu_f16_f32(
      (const xnn_float16*) ((uintptr_t) i7 + idx));
  const xnn_simd_f32_t vi8 = xnn_loadu_f16_f32(
      (const xnn_float16*) ((uintptr_t) i8 + idx));
  const xnn_simd_f32_t vsum018 = xnn_add_f32(xnn_add_f32(vi0, vi1), vi8);
  const xnn_simd_f32_t vsum23 = xnn_add_f32(vi2, vi3);
  const xnn_simd_f32_t vsum45 = xnn_add_f32(vi4, vi5);
  const xnn_simd_f32_t vsum67 = xnn_add_f32(vi6, vi7);
  return xnn_add_f32(
      xnn_add_f32(vsum23, vsum45), xnn_add_f32(vsum018, vsum67));
}

static XNN_INLINE void xnn_store_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value) {
  xnn_store_tail_f16(output, xnn_cvt_f16_f32(value), xnn_simd_size_f32);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_tail_sum9_f16_f32(
    const xnn_float16* i0, const xnn_float16* i1, const xnn_float16* i2,
    const xnn_float16* i3, const xnn_float16* i4, const xnn_float16* i5,
    const xnn_float16* i6, const xnn_float16* i7, const xnn_float16* i8,
    uintptr_t idx, size_t elements) {
  const xnn_simd_f32_t vi0 = xnn_cvt_f32_f16(xnn_load_tail_f16(
      (const xnn_float16*) ((uintptr_t) i0 + idx), elements));
  const xnn_simd_f32_t vi1 = xnn_cvt_f32_f16(xnn_load_tail_f16(
      (const xnn_float16*) ((uintptr_t) i1 + idx), elements));
  const xnn_simd_f32_t vi2 = xnn_cvt_f32_f16(xnn_load_tail_f16(
      (const xnn_float16*) ((uintptr_t) i2 + idx), elements));
  const xnn_simd_f32_t vi3 = xnn_cvt_f32_f16(xnn_load_tail_f16(
      (const xnn_float16*) ((uintptr_t) i3 + idx), elements));
  const xnn_simd_f32_t vi4 = xnn_cvt_f32_f16(xnn_load_tail_f16(
      (const xnn_float16*) ((uintptr_t) i4 + idx), elements));
  const xnn_simd_f32_t vi5 = xnn_cvt_f32_f16(xnn_load_tail_f16(
      (const xnn_float16*) ((uintptr_t) i5 + idx), elements));
  const xnn_simd_f32_t vi6 = xnn_cvt_f32_f16(xnn_load_tail_f16(
      (const xnn_float16*) ((uintptr_t) i6 + idx), elements));
  const xnn_simd_f32_t vi7 = xnn_cvt_f32_f16(xnn_load_tail_f16(
      (const xnn_float16*) ((uintptr_t) i7 + idx), elements));
  const xnn_simd_f32_t vi8 = xnn_cvt_f32_f16(xnn_load_tail_f16(
      (const xnn_float16*) ((uintptr_t) i8 + idx), elements));
  const xnn_simd_f32_t vsum018 = xnn_add_f32(xnn_add_f32(vi0, vi1), vi8);
  const xnn_simd_f32_t vsum23 = xnn_add_f32(vi2, vi3);
  const xnn_simd_f32_t vsum45 = xnn_add_f32(vi4, vi5);
  const xnn_simd_f32_t vsum67 = xnn_add_f32(vi6, vi7);
  return xnn_add_f32(
      xnn_add_f32(vsum23, vsum45), xnn_add_f32(vsum018, vsum67));
}

static XNN_INLINE void xnn_store_tail_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value, size_t elements) {
  xnn_store_tail_f16(output, xnn_cvt_f16_f32(value), elements);
}

static XNN_INLINE void xnn_avgpool_finalize_channels(
    size_t channels,
    const xnn_float16* i0, const xnn_float16* i1, const xnn_float16* i2,
    const xnn_float16* i3, const xnn_float16* i4, const xnn_float16* i5,
    const xnn_float16* i6, const xnn_float16* i7, const xnn_float16* i8,
    const float* acc,
    xnn_simd_f32_t vscale, xnn_simd_f32_t vmin, xnn_simd_f32_t vmax,
    xnn_float16* output) {
  uintptr_t idx = 0;
  for (; channels >= 32; channels -= 32) {
    xnn_simd_f32_t vacc = xnn_load_sum9_f16_f32(
        i0, i1, i2, i3, i4, i5, i6, i7, i8, idx);
    if (acc != NULL) {
      vacc = xnn_add_f32(
          vacc, xnn_loadu_f32((const float*) ((uintptr_t) acc + idx * 2)));
    }
    vacc = xnn_mul_f32(vacc, vscale);
    vacc = xnn_max_f32(vacc, vmin);
    vacc = xnn_min_f32(vacc, vmax);
    xnn_store_f32_f16((xnn_float16*) ((uintptr_t) output + idx), vacc);
    idx += 32 * sizeof(xnn_float16);
  }
  if (channels != 0) {
    xnn_simd_f32_t vacc = xnn_load_tail_sum9_f16_f32(
        i0, i1, i2, i3, i4, i5, i6, i7, i8, idx, channels);
    if (acc != NULL) {
      vacc = xnn_add_f32(
          vacc, xnn_loadu_f32((const float*) ((uintptr_t) acc + idx * 2)));
    }
    vacc = xnn_mul_f32(vacc, vscale);
    vacc = xnn_max_f32(vacc, vmin);
    vacc = xnn_min_f32(vacc, vmax);
    xnn_store_tail_f32_f16(
        (xnn_float16*) ((uintptr_t) output + idx), vacc, channels);
  }
}


void xnn_f16_f32acc_avgpool_minmax_ukernel_9p__hvx_u32(
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
  assert(xnn_simd_size_f32 == 32);

  const xnn_simd_f32_t vmin =
      xnn_set1_f32(xnn_float16_to_float(params->scalar.min));
  const xnn_simd_f32_t vmax =
      xnn_set1_f32(xnn_float16_to_float(params->scalar.max));
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  xnn_simd_f32_t vscale =
      xnn_set1_f32(xnn_float16_to_float(params->scalar.scale));

  if XNN_LIKELY(kernel_elements <= 9) {
    do {
      if (multiplier != NULL) {
        vscale = xnn_set1_f32(xnn_float16_to_float(*multiplier++));
      }

      const xnn_float16** i = input;
      const xnn_float16* i0 = 0 < kernel_elements ? *i++ : zero;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const xnn_float16*) ((uintptr_t) i0 + input_offset);
      }
      const xnn_float16* i1 = 1 < kernel_elements ? *i++ : zero;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const xnn_float16*) ((uintptr_t) i1 + input_offset);
      }
      const xnn_float16* i2 = 2 < kernel_elements ? *i++ : zero;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const xnn_float16*) ((uintptr_t) i2 + input_offset);
      }
      const xnn_float16* i3 = 3 < kernel_elements ? *i++ : zero;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const xnn_float16*) ((uintptr_t) i3 + input_offset);
      }
      const xnn_float16* i4 = 4 < kernel_elements ? *i++ : zero;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const xnn_float16*) ((uintptr_t) i4 + input_offset);
      }
      const xnn_float16* i5 = 5 < kernel_elements ? *i++ : zero;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const xnn_float16*) ((uintptr_t) i5 + input_offset);
      }
      const xnn_float16* i6 = 6 < kernel_elements ? *i++ : zero;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const xnn_float16*) ((uintptr_t) i6 + input_offset);
      }
      const xnn_float16* i7 = 7 < kernel_elements ? *i++ : zero;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const xnn_float16*) ((uintptr_t) i7 + input_offset);
      }
      const xnn_float16* i8 = 8 < kernel_elements ? *i++ : zero;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const xnn_float16*) ((uintptr_t) i8 + input_offset);
      }

      xnn_avgpool_finalize_channels(
          channels, i0, i1, i2, i3, i4, i5, i6, i7, i8, /*acc=*/NULL,
          vscale, vmin, vmax, output);

      input = (const xnn_float16**) ((uintptr_t) input + input_increment);
      input_offset += input_pixel_stride;
      output = (xnn_float16*) ((uintptr_t) output + output_increment);
    } while (--output_pixels != 0);
  } else {
    // Accumulate in L1-resident FP32 strips so kernel_elements > 9 preserves
    // full FP32 precision without repeating pointer setup per SIMD vector.
    XNN_ALIGN(128) float acc[256];

    do {
      if (multiplier != NULL) {
        vscale = xnn_set1_f32(xnn_float16_to_float(*multiplier++));
      }

      size_t c_rem = channels;
      uintptr_t base_idx = 0;
      do {
        const size_t strip_c = min(c_rem, 256);
        const xnn_float16** i = input;
        size_t k = kernel_elements;

        for (; k > 9; k -= 9) {
          const xnn_float16* i0 = *i++;
          assert(i0 != NULL);
          if XNN_UNPREDICTABLE(i0 != zero) {
            i0 = (const xnn_float16*) ((uintptr_t) i0 + input_offset);
          }
          i0 = (const xnn_float16*) ((uintptr_t) i0 + base_idx);
          const xnn_float16* i1 = *i++;
          assert(i1 != NULL);
          if XNN_UNPREDICTABLE(i1 != zero) {
            i1 = (const xnn_float16*) ((uintptr_t) i1 + input_offset);
          }
          i1 = (const xnn_float16*) ((uintptr_t) i1 + base_idx);
          const xnn_float16* i2 = *i++;
          assert(i2 != NULL);
          if XNN_UNPREDICTABLE(i2 != zero) {
            i2 = (const xnn_float16*) ((uintptr_t) i2 + input_offset);
          }
          i2 = (const xnn_float16*) ((uintptr_t) i2 + base_idx);
          const xnn_float16* i3 = *i++;
          assert(i3 != NULL);
          if XNN_UNPREDICTABLE(i3 != zero) {
            i3 = (const xnn_float16*) ((uintptr_t) i3 + input_offset);
          }
          i3 = (const xnn_float16*) ((uintptr_t) i3 + base_idx);
          const xnn_float16* i4 = *i++;
          assert(i4 != NULL);
          if XNN_UNPREDICTABLE(i4 != zero) {
            i4 = (const xnn_float16*) ((uintptr_t) i4 + input_offset);
          }
          i4 = (const xnn_float16*) ((uintptr_t) i4 + base_idx);
          const xnn_float16* i5 = *i++;
          assert(i5 != NULL);
          if XNN_UNPREDICTABLE(i5 != zero) {
            i5 = (const xnn_float16*) ((uintptr_t) i5 + input_offset);
          }
          i5 = (const xnn_float16*) ((uintptr_t) i5 + base_idx);
          const xnn_float16* i6 = *i++;
          assert(i6 != NULL);
          if XNN_UNPREDICTABLE(i6 != zero) {
            i6 = (const xnn_float16*) ((uintptr_t) i6 + input_offset);
          }
          i6 = (const xnn_float16*) ((uintptr_t) i6 + base_idx);
          const xnn_float16* i7 = *i++;
          assert(i7 != NULL);
          if XNN_UNPREDICTABLE(i7 != zero) {
            i7 = (const xnn_float16*) ((uintptr_t) i7 + input_offset);
          }
          i7 = (const xnn_float16*) ((uintptr_t) i7 + base_idx);
          const xnn_float16* i8 = *i++;
          assert(i8 != NULL);
          if XNN_UNPREDICTABLE(i8 != zero) {
            i8 = (const xnn_float16*) ((uintptr_t) i8 + input_offset);
          }
          i8 = (const xnn_float16*) ((uintptr_t) i8 + base_idx);

          uintptr_t idx = 0;
          size_t sc = strip_c;
          for (; sc >= 32; sc -= 32) {
            xnn_simd_f32_t vacc = xnn_load_sum9_f16_f32(
                i0, i1, i2, i3, i4, i5, i6, i7, i8, idx);
            if (k != kernel_elements) {
              vacc = xnn_add_f32(
                  vacc,
                  xnn_loadu_f32((const float*) ((uintptr_t) acc + idx * 2)));
            }
            xnn_storeu_f32((float*) ((uintptr_t) acc + idx * 2), vacc);
            idx += 32 * sizeof(xnn_float16);
          }
          if (sc != 0) {
            xnn_simd_f32_t vacc = xnn_load_tail_sum9_f16_f32(
                i0, i1, i2, i3, i4, i5, i6, i7, i8, idx, sc);
            if (k != kernel_elements) {
              vacc = xnn_add_f32(
                  vacc,
                  xnn_loadu_f32((const float*) ((uintptr_t) acc + idx * 2)));
            }
            xnn_storeu_f32((float*) ((uintptr_t) acc + idx * 2), vacc);
          }
        }

        const xnn_float16* i0 = 0 < k ? *i++ : zero;
        assert(i0 != NULL);
        if XNN_UNPREDICTABLE(i0 != zero) {
          i0 = (const xnn_float16*) ((uintptr_t) i0 + input_offset);
        }
        i0 = (const xnn_float16*) ((uintptr_t) i0 + base_idx);
        const xnn_float16* i1 = 1 < k ? *i++ : zero;
        assert(i1 != NULL);
        if XNN_UNPREDICTABLE(i1 != zero) {
          i1 = (const xnn_float16*) ((uintptr_t) i1 + input_offset);
        }
        i1 = (const xnn_float16*) ((uintptr_t) i1 + base_idx);
        const xnn_float16* i2 = 2 < k ? *i++ : zero;
        assert(i2 != NULL);
        if XNN_UNPREDICTABLE(i2 != zero) {
          i2 = (const xnn_float16*) ((uintptr_t) i2 + input_offset);
        }
        i2 = (const xnn_float16*) ((uintptr_t) i2 + base_idx);
        const xnn_float16* i3 = 3 < k ? *i++ : zero;
        assert(i3 != NULL);
        if XNN_UNPREDICTABLE(i3 != zero) {
          i3 = (const xnn_float16*) ((uintptr_t) i3 + input_offset);
        }
        i3 = (const xnn_float16*) ((uintptr_t) i3 + base_idx);
        const xnn_float16* i4 = 4 < k ? *i++ : zero;
        assert(i4 != NULL);
        if XNN_UNPREDICTABLE(i4 != zero) {
          i4 = (const xnn_float16*) ((uintptr_t) i4 + input_offset);
        }
        i4 = (const xnn_float16*) ((uintptr_t) i4 + base_idx);
        const xnn_float16* i5 = 5 < k ? *i++ : zero;
        assert(i5 != NULL);
        if XNN_UNPREDICTABLE(i5 != zero) {
          i5 = (const xnn_float16*) ((uintptr_t) i5 + input_offset);
        }
        i5 = (const xnn_float16*) ((uintptr_t) i5 + base_idx);
        const xnn_float16* i6 = 6 < k ? *i++ : zero;
        assert(i6 != NULL);
        if XNN_UNPREDICTABLE(i6 != zero) {
          i6 = (const xnn_float16*) ((uintptr_t) i6 + input_offset);
        }
        i6 = (const xnn_float16*) ((uintptr_t) i6 + base_idx);
        const xnn_float16* i7 = 7 < k ? *i++ : zero;
        assert(i7 != NULL);
        if XNN_UNPREDICTABLE(i7 != zero) {
          i7 = (const xnn_float16*) ((uintptr_t) i7 + input_offset);
        }
        i7 = (const xnn_float16*) ((uintptr_t) i7 + base_idx);
        const xnn_float16* i8 = 8 < k ? *i++ : zero;
        assert(i8 != NULL);
        if XNN_UNPREDICTABLE(i8 != zero) {
          i8 = (const xnn_float16*) ((uintptr_t) i8 + input_offset);
        }
        i8 = (const xnn_float16*) ((uintptr_t) i8 + base_idx);

        xnn_avgpool_finalize_channels(
            strip_c, i0, i1, i2, i3, i4, i5, i6, i7, i8, acc,
            vscale, vmin, vmax,
            (xnn_float16*) ((uintptr_t) output + base_idx));

        base_idx += strip_c * sizeof(xnn_float16);
        c_rem -= strip_c;
      } while (c_rem != 0);

      input = (const xnn_float16**) ((uintptr_t) input + input_increment);
      input_offset += input_pixel_stride;
      output = (xnn_float16*) ((uintptr_t) output + output_increment);
    } while (--output_pixels != 0);
  }
}
