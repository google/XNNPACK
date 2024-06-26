// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <emmintrin.h>

#include "xnnpack/zip.h"
#include "xnnpack/unaligned.h"


void xnn_x8_zip_xm_ukernel__sse2(
    size_t n,
    size_t m,
    const uint8_t* input,
    uint8_t* output)
{
  const uint8_t* w = input;
  const size_t input_increment = n * 3;
  const size_t output_increment = 4 - m * n;
  const uint8_t* last_input = w + n * (m - 1);
  uint8_t* last_output = (uint8_t*) ((uintptr_t) output + (m - 4));

  if (n >= 8) {
    for (size_t i = 0; i < m; i += 4) {
      size_t k = n;
      w = (const uint8_t*) ((uintptr_t) w + input_increment);
      if (w >= last_input) {
        w = last_input;
      }
      const uint8_t* z = (const uint8_t*) ((uintptr_t) w - n);
      const uint8_t* y = (const uint8_t*) ((uintptr_t) z - n);
      const uint8_t* x = (const uint8_t*) ((uintptr_t) y - n);
      while (k >= 16) {
        const __m128i vx = _mm_loadu_si128((const __m128i*) x);
        x += 16;
        const __m128i vy = _mm_loadu_si128((const __m128i*) y);
        y += 16;
        const __m128i vz = _mm_loadu_si128((const __m128i*) z);
        z += 16;
        const __m128i vw = _mm_loadu_si128((const __m128i*) w);
        w += 16;
        const __m128i vxy_lo = _mm_unpacklo_epi8(vx, vy);
        const __m128i vxy_hi = _mm_unpackhi_epi8(vx, vy);
        const __m128i vzw_lo = _mm_unpacklo_epi8(vz, vw);
        const __m128i vzw_hi = _mm_unpackhi_epi8(vz, vw);
        __m128i vxyzw0 = _mm_unpacklo_epi16(vxy_lo, vzw_lo);
        __m128i vxyzw1 = _mm_unpackhi_epi16(vxy_lo, vzw_lo);
        __m128i vxyzw2 = _mm_unpacklo_epi16(vxy_hi, vzw_hi);
        __m128i vxyzw3 = _mm_unpackhi_epi16(vxy_hi, vzw_hi);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_unpackhi_epi64(vxyzw1, vxyzw1);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw2));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw2 = _mm_shufflelo_epi16(vxyzw2, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw2));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw2 = _mm_unpackhi_epi64(vxyzw2, vxyzw2);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw2));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw2 = _mm_shufflelo_epi16(vxyzw2, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw2));
        output = (uint8_t*) ((uintptr_t) output + m);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw3));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw3 = _mm_shufflelo_epi16(vxyzw3, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw3));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw3 = _mm_unpackhi_epi64(vxyzw3, vxyzw3);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw3));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw3 = _mm_shufflelo_epi16(vxyzw3, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw3));
        output = (uint8_t*) ((uintptr_t) output + m);
        k -= 16;
      };
      if (k >= 8) {
        const __m128i vx = _mm_loadl_epi64((const __m128i*) x);
        x += 8;
        const __m128i vy = _mm_loadl_epi64((const __m128i*) y);
        y += 8;
        const __m128i vz = _mm_loadl_epi64((const __m128i*) z);
        z += 8;
        const __m128i vw = _mm_loadl_epi64((const __m128i*) w);
        w += 8;
        const __m128i vxy = _mm_unpacklo_epi8(vx, vy);
        const __m128i vzw = _mm_unpacklo_epi8(vz, vw);
        __m128i vxyzw0 = _mm_unpacklo_epi16(vxy, vzw);
        __m128i vxyzw1 = _mm_unpackhi_epi16(vxy, vzw);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
        output = (uint8_t*) ((uintptr_t) output + m);

        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_unpackhi_epi64(vxyzw1, vxyzw1);
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        vxyzw1 = _mm_shufflelo_epi16(vxyzw1, _MM_SHUFFLE(3, 2, 3, 2));
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw1));
        output = (uint8_t*) ((uintptr_t) output + m);
        k -= 8;
      }
      if (k != 0) {
        const size_t address_decrement = 8 - k;
        x -= address_decrement;
        y -= address_decrement;
        z -= address_decrement;
        w -= address_decrement;
        const __m128i vshift = _mm_cvtsi32_si128((int) address_decrement * 8);

        const __m128i vx = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) x), vshift);
        const __m128i vy = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) y), vshift);
        const __m128i vz = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) z), vshift);
        const __m128i vw = _mm_srl_epi64(_mm_loadl_epi64((const __m128i*) w), vshift);
        w += 8;
        const __m128i vxy = _mm_unpacklo_epi8(vx, vy);
        const __m128i vzw = _mm_unpacklo_epi8(vz, vw);
        __m128i vxyzw0 = _mm_unpacklo_epi16(vxy, vzw);
        __m128i vxyzw1 = _mm_unpackhi_epi16(vxy, vzw);

        if (k & 4) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = vxyzw1;
        }

        if (k & 2) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = _mm_shufflelo_epi16(vxyzw0, _MM_SHUFFLE(3, 2, 3, 2));
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
          vxyzw0 = _mm_unpackhi_epi64(vxyzw0, vxyzw0);
        }
        if (k & 1) {
          unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vxyzw0));
          output = (uint8_t*) ((uintptr_t) output + m);
        }
      }
      output = (uint8_t*) ((uintptr_t) output + output_increment);
      if (output > last_output) {
        output = last_output;
      }
    }
  } else {
    const uint8_t* i = input;
    uint8_t* o = output;
    size_t k = n;
    do {
      size_t l = m;
      const uint8_t* ii = i++;
      do {
        *o++ = *ii;
        ii += n;
      } while (--l != 0);
    } while (--k != 0);
  }
}
