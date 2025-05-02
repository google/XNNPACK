// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-avgpool/f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/microparams.h"

static XNN_INLINE void xnn_store_tail_f16(uint16_t* o, __m128i vh, size_t c) {
  assert(c > 0);
  assert(c < 8);
  if (c & 4) {
    _mm_storel_epi64((__m128i*) o, vh);
    vh = _mm_unpackhi_epi64(vh, vh);
    o += 4;
  }
  if (c & 2) {
    _mm_storeu_si32(o, vh);
    vh = _mm_srli_epi64(vh, 32);
    o += 2;
  }
  if (c & 1) {
    *o = (uint16_t) _mm_extract_epi16(vh, 0);
  }
}

static XNN_INLINE __m128i xnn_load_tail_safe_f16(const uint16_t* i, size_t c) {
  assert(c > 0);
  assert(c < 8);

  XNN_ALIGN(16) uint16_t padded[8];
  uint16_t* dst = padded;
  switch (c) {
  case 7: *dst++ = *i++;
  case 6: *dst++ = *i++;
  case 5: *dst++ = *i++;
  case 4: *dst++ = *i++;
  case 3: *dst++ = *i++;
  case 2: *dst++ = *i++;
  default: *dst++ = *i++;
  }
  return _mm_load_si128((const __m128i*) padded);
}

void xnn_f16_avgpool_minmax_ukernel_9p__f16c_u8(
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

  const __m256 vmin = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m256 vmax = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.max));
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  __m256 vscale = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.scale));

  do {
    // Start with the previous output as the zero buffer.
    const uint16_t* prev_output = (const uint16_t*) zero;

    const xnn_float16** i = input;

    // Passes 0 - n-1: load the output, add 9 inputs.
    size_t k = kernel_elements;
    for (; k > 9; k -= 9) {
      const uint16_t* i0 = (const uint16_t*) *i++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *i++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *i++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *i++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *i++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *i++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *i++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *i++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint16_t* i8 = (const uint16_t*) *i++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
        i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
      }

      uint16_t* o = (uint16_t*) output;
      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7)); i7 += 8;
        const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8)); i8 += 8;
        const __m256 vprev = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) prev_output)); prev_output += 8;

        const __m256 vsum018 = _mm256_add_ps(_mm256_add_ps(vi0, vi1), vi8);
        const __m256 vsum23 = _mm256_add_ps(vi2, vi3);
        const __m256 vsum45 = _mm256_add_ps(vi4, vi5);
        const __m256 vsum67 = _mm256_add_ps(vi6, vi7);

        const __m256 vsum2345 = _mm256_add_ps(vsum23, vsum45);
        const __m256 vsum01678 = _mm256_add_ps(vsum018, vsum67);
        const __m256 vsum012345678 = _mm256_add_ps(vsum2345, vsum01678);

        const __m256 vacc = _mm256_add_ps(vprev, vsum012345678);

        _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT)); o += 8;
      }
      if (c > 0) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
        const __m256 vprev = _mm256_cvtph_ps(xnn_load_tail_safe_f16((const uint16_t*) prev_output, c));

        const __m256 vsum018 = _mm256_add_ps(_mm256_add_ps(vi0, vi1), vi8);
        const __m256 vsum23 = _mm256_add_ps(vi2, vi3);
        const __m256 vsum45 = _mm256_add_ps(vi4, vi5);
        const __m256 vsum67 = _mm256_add_ps(vi6, vi7);

        const __m256 vsum2345 = _mm256_add_ps(vsum23, vsum45);
        const __m256 vsum01678 = _mm256_add_ps(vsum018, vsum67);
        const __m256 vsum012345678 = _mm256_add_ps(vsum2345, vsum01678);

        const __m256 vacc = _mm256_add_ps(vprev, vsum012345678);

        xnn_store_tail_f16(o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT), c); o += c;
      }

      // Subsequent passes read from the previous output.
      prev_output = (const uint16_t*) output;
    }

    // Final pass: load the output, add remaining kernel elements, apply scaling/min/max
    const uint16_t* i0 = (const uint16_t*) (0 < k ? *i++ : zero);
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) (1 < k ? *i++ : zero);
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) (2 < k ? *i++ : zero);
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = (const uint16_t*) (3 < k ? *i++ : zero);
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = (const uint16_t*) (4 < k ? *i++ : zero);
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = (const uint16_t*) (5 < k ? *i++ : zero);
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = (const uint16_t*) (6 < k ? *i++ : zero);
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = (const uint16_t*) (7 < k ? *i++ : zero);
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = (const uint16_t*) (8 < k ? *i++ : zero);
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }

    if (multiplier) {
      vscale = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) (multiplier++)));
    }
    uint16_t* o = (uint16_t*) output;
    size_t c = channels;
    for (; c >= 8; c -= 8) {
      const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
      const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
      const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
      const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
      const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
      const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
      const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
      const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7)); i7 += 8;
      const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8)); i8 += 8;
      const __m256 vprev = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) prev_output)); prev_output += 8;

      const __m256 vsum018 = _mm256_add_ps(_mm256_add_ps(vi0, vi1), vi8);
      const __m256 vsum23 = _mm256_add_ps(vi2, vi3);
      const __m256 vsum45 = _mm256_add_ps(vi4, vi5);
      const __m256 vsum67 = _mm256_add_ps(vi6, vi7);

      const __m256 vsum2345 = _mm256_add_ps(vsum23, vsum45);
      const __m256 vsum01678 = _mm256_add_ps(vsum018, vsum67);
      const __m256 vsum012345678 = _mm256_add_ps(vsum2345, vsum01678);

      __m256 vacc = _mm256_add_ps(vprev, vsum012345678);

      vacc = _mm256_mul_ps(vacc, vscale);
      vacc = _mm256_max_ps(vacc, vmin);
      vacc = _mm256_min_ps(vacc, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT)); o += 8;
    }
    if (c > 0) {
      const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      const __m256 vprev = _mm256_cvtph_ps(xnn_load_tail_safe_f16((const uint16_t*) prev_output, c));

      const __m256 vsum018 = _mm256_add_ps(_mm256_add_ps(vi0, vi1), vi8);
      const __m256 vsum23 = _mm256_add_ps(vi2, vi3);
      const __m256 vsum45 = _mm256_add_ps(vi4, vi5);
      const __m256 vsum67 = _mm256_add_ps(vi6, vi7);

      const __m256 vsum2345 = _mm256_add_ps(vsum23, vsum45);
      const __m256 vsum01678 = _mm256_add_ps(vsum018, vsum67);
      const __m256 vsum012345678 = _mm256_add_ps(vsum2345, vsum01678);

      __m256 vacc = _mm256_add_ps(vprev, vsum012345678);

      vacc = _mm256_mul_ps(vacc, vscale);
      vacc = _mm256_max_ps(vacc, vmin);
      vacc = _mm256_min_ps(vacc, vmax);

      xnn_store_tail_f16((uint16_t*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT), c); o += c;
    }

    input = (const xnn_float16**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
