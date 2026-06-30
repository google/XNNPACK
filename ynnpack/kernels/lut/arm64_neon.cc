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

bool lut_u2_u8_arm64_neon(size_t n, const void* idx, size_t lut_size,
                          const void* lut, void* out) {
  if (lut_size <= 3) {
    // Bounds check needed. Fallback to scalar.
    return lut_u2_u8(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint8_t* lut_ptr = static_cast<const uint8_t*>(lut);
  uint8_t* out_ptr = static_cast<uint8_t*>(out);

  uint8x16_t table;
  memcpy(&table, lut_ptr, 4);

  while (n >= 64) {
    uint8x16_t packed = vld1q_u8(idx_ptr);
    idx_ptr += 16;

    uint8x16_t i0 = vandq_u8(packed, vdupq_n_u8(3));
    uint8x16_t i1 = vandq_u8(vshrq_n_u8(packed, 2), vdupq_n_u8(3));
    uint8x16_t i2 = vandq_u8(vshrq_n_u8(packed, 4), vdupq_n_u8(3));
    uint8x16_t i3 = vshrq_n_u8(packed, 6);

    uint8x16x4_t r = {
        vqtbl1q_u8(table, i0),
        vqtbl1q_u8(table, i1),
        vqtbl1q_u8(table, i2),
        vqtbl1q_u8(table, i3),
    };
    vst4q_u8(out_ptr, r);
    out_ptr += 64;
    n -= 64;
  }

  if (n > 0) {
    lut_u2_u8(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

bool lut_u4_u8_arm64_neon(size_t n, const void* idx, size_t lut_size,
                          const void* lut, void* out) {
  if (lut_size <= 15) {
    // Bounds check needed. Fallback to scalar.
    return lut_u4_u8(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint8_t* lut_ptr = static_cast<const uint8_t*>(lut);
  uint8_t* out_ptr = static_cast<uint8_t*>(out);

  // Since lut_size >= 16, we can safely load exactly 16 bytes (indices 0..15).
  uint8x16_t table = vld1q_u8(lut_ptr);

  while (n >= 32) {
    uint8x16_t packed = vld1q_u8(idx_ptr);
    idx_ptr += 16;

    uint8x16_t i0 = vandq_u8(packed, vdupq_n_u8(15));
    uint8x16_t i1 = vshrq_n_u8(packed, 4);

    uint8x16x2_t r = {
        vqtbl1q_u8(table, i0),
        vqtbl1q_u8(table, i1),
    };
    vst2q_u8(out_ptr, r);
    out_ptr += 32;
    n -= 32;
  }

  if (n > 0) {
    lut_u4_u8(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

bool lut_u4_u16_arm64_neon(size_t n, const void* idx, size_t lut_size,
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

  while (n >= 32) {
    uint8x16_t packed = vld1q_u8(idx_ptr);
    idx_ptr += 16;

    uint8x16_t i0 = vandq_u8(packed, vdupq_n_u8(15));
    uint8x16_t i1 = vshrq_n_u8(packed, 4);

    uint8x16_t r0_lo = vqtbl1q_u8(table_lohi.val[0], i0);
    uint8x16_t r0_hi = vqtbl1q_u8(table_lohi.val[1], i0);
    uint8x16_t r1_lo = vqtbl1q_u8(table_lohi.val[0], i1);
    uint8x16_t r1_hi = vqtbl1q_u8(table_lohi.val[1], i1);

    uint8x16x2_t zipped0 = vzipq_u8(r0_lo, r0_hi);
    uint8x16x2_t zipped1 = vzipq_u8(r1_lo, r1_hi);

    uint16x8x2_t out0 = {
        vreinterpretq_u16_u8(zipped0.val[0]),
        vreinterpretq_u16_u8(zipped1.val[0]),
    };
    vst2q_u16(out_ptr, out0);

    uint16x8x2_t out1 = {
        vreinterpretq_u16_u8(zipped0.val[1]),
        vreinterpretq_u16_u8(zipped1.val[1]),
    };
    vst2q_u16(out_ptr + 16, out1);

    out_ptr += 32;
    n -= 32;
  }

  if (n > 0) {
    lut_u4_u16(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

bool lut_u8_u8_arm64_neon(size_t n, const void* idx, size_t lut_size,
                          const void* lut, void* out) {
  if (lut_size <= 255) {
    // Bounds check needed. Fallback to scalar.
    return lut_u8_u8(n, idx, lut_size, lut, out);
  }

  const uint8_t* idx_ptr = static_cast<const uint8_t*>(idx);
  const uint8_t* lut_ptr = static_cast<const uint8_t*>(lut);
  uint8_t* out_ptr = static_cast<uint8_t*>(out);

  // Since lut_size >= 256, we can safely load all 256 bytes of the table.
  // We load them as 4 blocks of 64 bytes.
  uint8x16x4_t table0 = vld1q_u8_x4(lut_ptr);
  uint8x16x4_t table1 = vld1q_u8_x4(lut_ptr + 64);
  uint8x16x4_t table2 = vld1q_u8_x4(lut_ptr + 128);
  uint8x16x4_t table3 = vld1q_u8_x4(lut_ptr + 192);

  while (n >= 16) {
    uint8x16_t idx0 = vld1q_u8(idx_ptr);
    idx_ptr += 16;

    uint8x16_t idx1 = vsubq_u8(idx0, vdupq_n_u8(64));
    uint8x16_t idx2 = vsubq_u8(idx0, vdupq_n_u8(128));
    uint8x16_t idx3 = vsubq_u8(idx0, vdupq_n_u8(192));

    uint8x16_t r = vqtbl4q_u8(table0, idx0);
    r = vqtbx4q_u8(r, table1, idx1);
    r = vqtbx4q_u8(r, table2, idx2);
    r = vqtbx4q_u8(r, table3, idx3);

    vst1q_u8(out_ptr, r);
    out_ptr += 16;
    n -= 16;
  }

  if (n > 0) {
    lut_u8_u8(n, idx_ptr, lut_size, lut_ptr, out_ptr);
  }

  return true;
}

}  // namespace ynn
