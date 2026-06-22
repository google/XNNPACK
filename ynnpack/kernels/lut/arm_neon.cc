// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/type.h"
#include "ynnpack/kernels/lut/lut.h"

namespace ynn {

bool lut_u2_u8_arm_neon(size_t n, const void* idx, size_t lut_size,
                        const void* lut, void* out) {
  if (lut_size <= 3) {
    // Bounds check needed. Fallback to scalar.
    return lut_u2_u8(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint8_t* lut_ptr = static_cast<const uint8_t*>(lut);
  uint8_t* out_ptr = static_cast<uint8_t*>(out);

  uint8x8_t table;
  memcpy(&table, lut_ptr, 4);

  while (n >= 32) {
    uint8x8_t packed = vld1_u8(idx_ptr);
    idx_ptr += 8;

    uint8x8_t i0 = vand_u8(packed, vdup_n_u8(3));
    uint8x8_t i1 = vand_u8(vshr_n_u8(packed, 2), vdup_n_u8(3));
    uint8x8_t i2 = vand_u8(vshr_n_u8(packed, 4), vdup_n_u8(3));
    uint8x8_t i3 = vshr_n_u8(packed, 6);

    uint8x8x4_t r = {
        vtbl1_u8(table, i0),
        vtbl1_u8(table, i1),
        vtbl1_u8(table, i2),
        vtbl1_u8(table, i3),
    };
    vst4_u8(out_ptr, r);
    out_ptr += 32;
    n -= 32;
  }

  if (n > 0) {
    lut_u2_u8(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

bool lut_u2_u16_arm_neon(size_t n, const void* idx, size_t lut_size,
                         const void* lut, void* out) {
  if (lut_size <= 3) {
    // Bounds check needed. Fallback to scalar.
    return lut_u2_u16(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint16_t* lut_ptr = static_cast<const uint16_t*>(lut);
  uint16_t* out_ptr = static_cast<uint16_t*>(out);

  // There is no vtbl1_u16, so we split it into two tables of the low and high
  // u8 values.
  uint8x8_t table_u8x2 = vld1_u8(reinterpret_cast<const uint8_t*>(lut_ptr));
  uint8x8x2_t table = vuzp_u8(table_u8x2, table_u8x2);

  while (n >= 32) {
    uint8x8_t packed = vld1_u8(idx_ptr);
    idx_ptr += 8;

    uint8x8_t i0 = vand_u8(packed, vdup_n_u8(3));
    uint8x8_t i1 = vand_u8(vshr_n_u8(packed, 2), vdup_n_u8(3));
    uint8x8_t i2 = vand_u8(vshr_n_u8(packed, 4), vdup_n_u8(3));
    uint8x8_t i3 = vshr_n_u8(packed, 6);

    uint8x8x2_t zipped0 =
        vzip_u8(vtbl1_u8(table.val[0], i0), vtbl1_u8(table.val[1], i0));
    uint8x8x2_t zipped1 =
        vzip_u8(vtbl1_u8(table.val[0], i1), vtbl1_u8(table.val[1], i1));
    uint8x8x2_t zipped2 =
        vzip_u8(vtbl1_u8(table.val[0], i2), vtbl1_u8(table.val[1], i2));
    uint8x8x2_t zipped3 =
        vzip_u8(vtbl1_u8(table.val[0], i3), vtbl1_u8(table.val[1], i3));

    uint16x8x4_t r = {
        vreinterpretq_u16_u8(vcombine_u8(zipped0.val[0], zipped0.val[1])),
        vreinterpretq_u16_u8(vcombine_u8(zipped1.val[0], zipped1.val[1])),
        vreinterpretq_u16_u8(vcombine_u8(zipped2.val[0], zipped2.val[1])),
        vreinterpretq_u16_u8(vcombine_u8(zipped3.val[0], zipped3.val[1])),
    };
    vst4q_u16(out_ptr, r);
    out_ptr += 32;
    n -= 32;
  }

  if (n > 0) {
    lut_u2_u16(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

bool lut_u4_u8_arm_neon(size_t n, const void* idx, size_t lut_size,
                        const void* lut, void* out) {
  if (lut_size <= 15) {
    // Bounds check needed. Fallback to scalar.
    return lut_u4_u8(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint8_t* lut_ptr = static_cast<const uint8_t*>(lut);
  uint8_t* out_ptr = static_cast<uint8_t*>(out);

  // Since lut_size >= 16, we can safely load exactly 16 bytes (indices 0..15).
  uint8x8x2_t table = {vld1_u8(lut_ptr), vld1_u8(lut_ptr + 8)};

  while (n >= 16) {
    uint8x8_t packed = vld1_u8(idx_ptr);
    idx_ptr += 8;

    uint8x8_t i0 = vand_u8(packed, vdup_n_u8(15));
    uint8x8_t i1 = vshr_n_u8(packed, 4);

    uint8x8x2_t r = {
        vtbl2_u8(table, i0),
        vtbl2_u8(table, i1),
    };
    vst2_u8(out_ptr, r);
    out_ptr += 16;
    n -= 16;
  }

  if (n > 0) {
    lut_u4_u8(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

bool lut_u4_u16_arm_neon(size_t n, const void* idx, size_t lut_size,
                         const void* lut, void* out) {
  if (lut_size <= 15) {
    // Bounds check needed. Fallback to scalar.
    return lut_u4_u16(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint16_t* lut_ptr = static_cast<const uint16_t*>(lut);
  uint16_t* out_ptr = static_cast<uint16_t*>(out);

  uint16x8x2_t table;
  table.val[0] = vld1q_u16(lut_ptr);
  table.val[1] = vld1q_u16(lut_ptr + 8);
  uint8x16x2_t table_lohi = vuzpq_u8(vreinterpretq_u8_u16(table.val[0]),
                                     vreinterpretq_u8_u16(table.val[1]));
  uint8x8x2_t table_lo = {vget_low_u8(table_lohi.val[0]),
                          vget_high_u8(table_lohi.val[0])};
  uint8x8x2_t table_hi = {vget_low_u8(table_lohi.val[1]),
                          vget_high_u8(table_lohi.val[1])};

  while (n >= 16) {
    uint8x8_t packed = vld1_u8(idx_ptr);
    idx_ptr += 8;

    uint8x8_t i0 = vand_u8(packed, vdup_n_u8(15));
    uint8x8_t i1 = vshr_n_u8(packed, 4);

    uint8x8_t r0_lo = vtbl2_u8(table_lo, i0);
    uint8x8_t r0_hi = vtbl2_u8(table_hi, i0);
    uint8x8_t r1_lo = vtbl2_u8(table_lo, i1);
    uint8x8_t r1_hi = vtbl2_u8(table_hi, i1);

    uint8x8x2_t zipped0 = vzip_u8(r0_lo, r0_hi);
    uint8x8x2_t zipped1 = vzip_u8(r1_lo, r1_hi);

    uint16x4x2_t out0 = {
        vreinterpret_u16_u8(zipped0.val[0]),
        vreinterpret_u16_u8(zipped1.val[0]),
    };
    vst2_u16(out_ptr, out0);

    uint16x4x2_t out1 = {
        vreinterpret_u16_u8(zipped0.val[1]),
        vreinterpret_u16_u8(zipped1.val[1]),
    };
    vst2_u16(out_ptr + 8, out1);

    out_ptr += 16;
    n -= 16;
  }

  if (n > 0) {
    lut_u4_u16(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

}  // namespace ynn
