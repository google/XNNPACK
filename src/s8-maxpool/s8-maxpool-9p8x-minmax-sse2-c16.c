// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/unaligned.h"


void xnn_s8_maxpool_minmax_ukernel_9p8x__sse2_c16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const __m128i vbias = _mm_set1_epi8(UINT8_C(0x80));
  const __m128i voutput_max_with_bias = _mm_set1_epi8(UINT8_C(0x80) ^ params->scalar.max);
  const __m128i voutput_min_with_bias = _mm_set1_epi8(UINT8_C(0x80) ^ params->scalar.min);
  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(voutput_max_with_bias);
  XNN_FORCE_REALIZATION(voutput_min_with_bias);

  do {
    int8_t* o = output;
    {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      const int8_t* i8 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      for (; c >= 16; c -= 16) {
        const __m128i vi0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i0), vbias);
        i0 += 16;
        const __m128i vi1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i1), vbias);
        i1 += 16;
        const __m128i vi2 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i2), vbias);
        i2 += 16;
        const __m128i vi3 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i3), vbias);
        i3 += 16;
        const __m128i vi4 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i4), vbias);
        i4 += 16;
        const __m128i vi5 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i5), vbias);
        i5 += 16;
        const __m128i vi6 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i6), vbias);
        i6 += 16;
        const __m128i vi7 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i7), vbias);
        i7 += 16;
        const __m128i vi8 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i8), vbias);
        i8 += 16;

        const __m128i vmax018 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vi8);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax01678 = _mm_max_epu8(vmax018, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax01678);
        vout = _mm_max_epu8(vout, voutput_min_with_bias);
        vout = _mm_min_epu8(vout, voutput_max_with_bias);
        vout = _mm_xor_si128(vout, vbias);

        _mm_storeu_si128((__m128i*) o, vout); o += 16;
      }
      if (c != 0) {
        const __m128i vi0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i0), vbias);
        const __m128i vi1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i1), vbias);
        const __m128i vi2 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i2), vbias);
        const __m128i vi3 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i3), vbias);
        const __m128i vi4 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i4), vbias);
        const __m128i vi5 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i5), vbias);
        const __m128i vi6 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i6), vbias);
        const __m128i vi7 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i7), vbias);
        const __m128i vi8 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i8), vbias);

        const __m128i vmax018 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vi8);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax01678 = _mm_max_epu8(vmax018, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax01678);
        vout = _mm_max_epu8(vout, voutput_min_with_bias);
        vout = _mm_min_epu8(vout, voutput_max_with_bias);
        vout = _mm_xor_si128(vout, vbias);

        if (c & 8) {
          _mm_storel_epi64((__m128i*) o, vout);
          vout = _mm_unpackhi_epi64(vout, vout);
          o += 8;
        }
        if (c & 4) {
          unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vout));
          vout = _mm_srli_epi64(vout, 32);
          o += 4;
        }
        if (c & 2) {
          unaligned_store_u16(o, (uint16_t) _mm_extract_epi16(vout, 0));
          vout = _mm_srli_epi32(vout, 16);
          o += 2;
        }
        if (c & 1) {
          *((int8_t*) o) = (int8_t) _mm_cvtsi128_si32(vout);
          o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const int8_t* i0 = *input++;
      const int8_t* i1 = *input++;
      const int8_t* i2 = *input++;
      const int8_t* i3 = *input++;
      const int8_t* i4 = *input++;
      const int8_t* i5 = *input++;
      const int8_t* i6 = *input++;
      const int8_t* i7 = *input++;
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      for (; c >= 16; c -= 16) {
        const __m128i vi0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i0), vbias);
        i0 += 16;
        const __m128i vi1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i1), vbias);
        i1 += 16;
        const __m128i vi2 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i2), vbias);
        i2 += 16;
        const __m128i vi3 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i3), vbias);
        i3 += 16;
        const __m128i vi4 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i4), vbias);
        i4 += 16;
        const __m128i vi5 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i5), vbias);
        i5 += 16;
        const __m128i vi6 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i6), vbias);
        i6 += 16;
        const __m128i vi7 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i7), vbias);
        i7 += 16;
        const __m128i vo = _mm_xor_si128(_mm_loadu_si128((const __m128i*) o), vbias);

        const __m128i vmax01 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vo);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax0167 = _mm_max_epu8(vmax01, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax0167);
        vout = _mm_max_epu8(vout, voutput_min_with_bias);
        vout = _mm_min_epu8(vout, voutput_max_with_bias);
        vout = _mm_xor_si128(vout, vbias);

        _mm_storeu_si128((__m128i*) o, vout);
        o += 16;
      }
      if (c != 0) {
        const __m128i vi0 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i0), vbias);
        const __m128i vi1 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i1), vbias);
        const __m128i vi2 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i2), vbias);
        const __m128i vi3 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i3), vbias);
        const __m128i vi4 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i4), vbias);
        const __m128i vi5 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i5), vbias);
        const __m128i vi6 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i6), vbias);
        const __m128i vi7 = _mm_xor_si128(_mm_loadu_si128((const __m128i*) i7), vbias);
        const __m128i vo = _mm_xor_si128(_mm_loadu_si128((const __m128i*) o), vbias);

        const __m128i vmax01 = _mm_max_epu8(_mm_max_epu8(vi0, vi1), vo);
        const __m128i vmax23 = _mm_max_epu8(vi2, vi3);
        const __m128i vmax45 = _mm_max_epu8(vi4, vi5);
        const __m128i vmax67 = _mm_max_epu8(vi6, vi7);

        const __m128i vmax2345 = _mm_max_epu8(vmax23, vmax45);
        const __m128i vmax0167 = _mm_max_epu8(vmax01, vmax67);
        __m128i vout = _mm_max_epu8(vmax2345, vmax0167);
        vout = _mm_max_epu8(vout, voutput_min_with_bias);
        vout = _mm_min_epu8(vout, voutput_max_with_bias);
        vout = _mm_xor_si128(vout, vbias);

        if (c & 8) {
          _mm_storel_epi64((__m128i*) o, vout);
          vout = _mm_unpackhi_epi64(vout, vout);
          o += 8;
        }
        if (c & 4) {
          unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vout));
          vout = _mm_srli_epi64(vout, 32);
          o += 4;
        }
        if (c & 2) {
          unaligned_store_u16(o, (uint16_t) _mm_extract_epi16(vout, 0));
          vout = _mm_srli_epi32(vout, 16);
          o += 2;
        }
        if (c & 1) {
          *o = (int8_t) _mm_cvtsi128_si32(vout);
          o += 1;
        }
      }
    }
    input = (const int8_t**) ((uintptr_t) input + input_increment);
    output = (int8_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
