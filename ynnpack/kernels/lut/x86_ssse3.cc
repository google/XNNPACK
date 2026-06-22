// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <tmmintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/lut/lut.h"

namespace ynn {

bool lut_u2_u8_ssse3(size_t n, const void* idx, size_t lut_size,
                     const void* lut, void* out) {
  if (lut_size <= 3) {
    // Bounds check needed. Fallback to scalar.
    return lut_u2_u8(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint8_t* lut_ptr = static_cast<const uint8_t*>(lut);
  uint8_t* out_ptr = static_cast<uint8_t*>(out);

  int val = 0;
  memcpy(&val, lut_ptr, 4);
  __m128i table = _mm_cvtsi32_si128(val);

  __m128i three = _mm_set1_epi8(3);

  while (n >= 64) {
    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(idx_ptr));
    idx_ptr += 16;

    __m128i i0 = _mm_and_si128(packed, three);
    __m128i i1 = _mm_and_si128(_mm_srli_epi16(packed, 2), three);
    __m128i i2 = _mm_and_si128(_mm_srli_epi16(packed, 4), three);
    __m128i i3 = _mm_and_si128(_mm_srli_epi16(packed, 6), three);

    __m128i r0 = _mm_shuffle_epi8(table, i0);
    __m128i r1 = _mm_shuffle_epi8(table, i1);
    __m128i r2 = _mm_shuffle_epi8(table, i2);
    __m128i r3 = _mm_shuffle_epi8(table, i3);

    __m128i ab_lo = _mm_unpacklo_epi8(r0, r1);
    __m128i ab_hi = _mm_unpackhi_epi8(r0, r1);
    __m128i cd_lo = _mm_unpacklo_epi8(r2, r3);
    __m128i cd_hi = _mm_unpackhi_epi8(r2, r3);

    __m128i out0 = _mm_unpacklo_epi16(ab_lo, cd_lo);
    __m128i out1 = _mm_unpackhi_epi16(ab_lo, cd_lo);
    __m128i out2 = _mm_unpacklo_epi16(ab_hi, cd_hi);
    __m128i out3 = _mm_unpackhi_epi16(ab_hi, cd_hi);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr), out0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr + 16), out1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr + 32), out2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr + 48), out3);
    out_ptr += 64;
    n -= 64;
  }

  if (n > 0) {
    lut_u2_u8(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

bool lut_u2_u16_ssse3(size_t n, const void* idx, size_t lut_size,
                      const void* lut, void* out) {
  if (lut_size <= 3) {
    // Bounds check needed. Fallback to scalar.
    return lut_u2_u16(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint16_t* lut_ptr = static_cast<const uint16_t*>(lut);
  uint16_t* out_ptr = static_cast<uint16_t*>(out);

  uint64_t val = 0;
  memcpy(&val, lut_ptr, 8);
  __m128i table = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&val));

  __m128i mask_ff = _mm_set1_epi16(0x00FF);
  __m128i zero = _mm_setzero_si128();
  __m128i table_lo = _mm_packus_epi16(_mm_and_si128(table, mask_ff), zero);
  __m128i table_hi = _mm_packus_epi16(_mm_srli_epi16(table, 8), zero);

  __m128i three = _mm_set1_epi8(3);

  while (n >= 32) {
    __m128i packed = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(idx_ptr));
    idx_ptr += 8;

    __m128i i0 = _mm_and_si128(packed, three);
    __m128i i1 = _mm_and_si128(_mm_srli_epi16(packed, 2), three);
    __m128i i2 = _mm_and_si128(_mm_srli_epi16(packed, 4), three);
    __m128i i3 = _mm_and_si128(_mm_srli_epi16(packed, 6), three);

    __m128i r0_lo = _mm_shuffle_epi8(table_lo, i0);
    __m128i r0_hi = _mm_shuffle_epi8(table_hi, i0);
    __m128i r1_lo = _mm_shuffle_epi8(table_lo, i1);
    __m128i r1_hi = _mm_shuffle_epi8(table_hi, i1);
    __m128i r2_lo = _mm_shuffle_epi8(table_lo, i2);
    __m128i r2_hi = _mm_shuffle_epi8(table_hi, i2);
    __m128i r3_lo = _mm_shuffle_epi8(table_lo, i3);
    __m128i r3_hi = _mm_shuffle_epi8(table_hi, i3);

    __m128i r0_ans = _mm_unpacklo_epi8(r0_lo, r0_hi);
    __m128i r1_ans = _mm_unpacklo_epi8(r1_lo, r1_hi);
    __m128i r2_ans = _mm_unpacklo_epi8(r2_lo, r2_hi);
    __m128i r3_ans = _mm_unpacklo_epi8(r3_lo, r3_hi);

    __m128i ab_lo = _mm_unpacklo_epi16(r0_ans, r1_ans);
    __m128i ab_hi = _mm_unpackhi_epi16(r0_ans, r1_ans);
    __m128i cd_lo = _mm_unpacklo_epi16(r2_ans, r3_ans);
    __m128i cd_hi = _mm_unpackhi_epi16(r2_ans, r3_ans);

    __m128i out0 = _mm_unpacklo_epi32(ab_lo, cd_lo);
    __m128i out1 = _mm_unpackhi_epi32(ab_lo, cd_lo);
    __m128i out2 = _mm_unpacklo_epi32(ab_hi, cd_hi);
    __m128i out3 = _mm_unpackhi_epi32(ab_hi, cd_hi);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr), out0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr + 8), out1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr + 16), out2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr + 24), out3);
    out_ptr += 32;
    n -= 32;
  }

  if (n > 0) {
    lut_u2_u16(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

bool lut_u4_u8_ssse3(size_t n, const void* idx, size_t lut_size,
                     const void* lut, void* out) {
  if (lut_size <= 15) {
    // Bounds check needed. Fallback to scalar.
    return lut_u4_u8(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint8_t* lut_ptr = static_cast<const uint8_t*>(lut);
  uint8_t* out_ptr = static_cast<uint8_t*>(out);

  __m128i table = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_ptr));
  __m128i vdup_15 = _mm_set1_epi8(15);

  while (n >= 32) {
    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(idx_ptr));
    idx_ptr += 16;

    __m128i i0 = _mm_and_si128(packed, vdup_15);
    __m128i i1 = _mm_and_si128(_mm_srli_epi16(packed, 4), vdup_15);

    __m128i r0 = _mm_shuffle_epi8(table, i0);
    __m128i r1 = _mm_shuffle_epi8(table, i1);

    __m128i out0 = _mm_unpacklo_epi8(r0, r1);
    __m128i out1 = _mm_unpackhi_epi8(r0, r1);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr), out0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr + 16), out1);
    out_ptr += 32;
    n -= 32;
  }

  if (n > 0) {
    lut_u4_u8(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

bool lut_u4_u16_ssse3(size_t n, const void* idx, size_t lut_size,
                      const void* lut, void* out) {
  if (lut_size <= 15) {
    // Bounds check needed. Fallback to scalar.
    return lut_u4_u16(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint16_t* lut_ptr = static_cast<const uint16_t*>(lut);
  uint16_t* out_ptr = static_cast<uint16_t*>(out);

  __m128i table0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_ptr));
  __m128i table1 =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(lut_ptr + 8));

  __m128i mask_ff = _mm_set1_epi16(0x00FF);
  __m128i table_lo = _mm_packus_epi16(_mm_and_si128(table0, mask_ff),
                                      _mm_and_si128(table1, mask_ff));
  __m128i table_hi =
      _mm_packus_epi16(_mm_srli_epi16(table0, 8), _mm_srli_epi16(table1, 8));

  __m128i vdup_15 = _mm_set1_epi8(15);

  while (n >= 32) {
    __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(idx_ptr));
    idx_ptr += 16;

    __m128i i0 = _mm_and_si128(packed, vdup_15);
    __m128i i1 = _mm_and_si128(_mm_srli_epi16(packed, 4), vdup_15);

    __m128i r0_lo = _mm_shuffle_epi8(table_lo, i0);
    __m128i r0_hi = _mm_shuffle_epi8(table_hi, i0);
    __m128i r1_lo = _mm_shuffle_epi8(table_lo, i1);
    __m128i r1_hi = _mm_shuffle_epi8(table_hi, i1);

    __m128i r0_ans_lo = _mm_unpacklo_epi8(r0_lo, r0_hi);
    __m128i r0_ans_hi = _mm_unpackhi_epi8(r0_lo, r0_hi);
    __m128i r1_ans_lo = _mm_unpacklo_epi8(r1_lo, r1_hi);
    __m128i r1_ans_hi = _mm_unpackhi_epi8(r1_lo, r1_hi);

    __m128i out0 = _mm_unpacklo_epi16(r0_ans_lo, r1_ans_lo);
    __m128i out1 = _mm_unpackhi_epi16(r0_ans_lo, r1_ans_lo);
    __m128i out2 = _mm_unpacklo_epi16(r0_ans_hi, r1_ans_hi);
    __m128i out3 = _mm_unpackhi_epi16(r0_ans_hi, r1_ans_hi);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr), out0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr + 8), out1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr + 16), out2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out_ptr + 24), out3);

    out_ptr += 32;
    n -= 32;
  }

  if (n > 0) {
    lut_u4_u16(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

}  // namespace ynn
