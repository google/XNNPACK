// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/packx.h>


void xnn_x32_packx_ukernel_4x__psimd(
    size_t m,
    size_t k,
    const uint32_t* restrict x,
    size_t x_stride,
    uint32_t* restrict y)
{
  assert(m != 0);
  assert(k != 0);

  const uint32_t* x0 = x;
  const uint32_t* x1 = (const uint32_t*) ((uintptr_t) x0 + x_stride);
  if (m < 2) {
    x1 = x0;
  }
  const uint32_t* x2 = (const uint32_t*) ((uintptr_t) x1 + x_stride);
  if (m <= 2) {
    x2 = x1;
  }
  const uint32_t* x3 = (const uint32_t*) ((uintptr_t) x2 + x_stride);
  if (m != 4) {
    x3 = x2;
  }

  for (; k >= 4; k -= 4) {
    const psimd_u32 vx0 = psimd_load_u32(x0);
    x0 += 4;
    const psimd_u32 vx1 = psimd_load_u32(x1);
    x1 += 4;
    const psimd_u32 vx2 = psimd_load_u32(x2);
    x2 += 4;
    const psimd_u32 vx3 = psimd_load_u32(x3);
    x3 += 4;

    const psimd_u32 vt0 = psimd_interleave_lo_u32(vx0, vx1);
    const psimd_u32 vt1 = psimd_interleave_hi_u32(vx0, vx1);
    const psimd_u32 vt2 = psimd_interleave_lo_u32(vx2, vx3);
    const psimd_u32 vt3 = psimd_interleave_hi_u32(vx2, vx3);

    const psimd_u32 vy0 = psimd_concat_lo_u32(vt0, vt2);
    psimd_store_u32(y, vy0);

    const psimd_u32 vy1 = psimd_concat_hi_u32(vt0, vt2);
    psimd_store_u32(y + 4, vy1);

    const psimd_u32 vy2 = psimd_concat_lo_u32(vt1, vt3);
    psimd_store_u32(y + 8, vy2);

    const psimd_u32 vy3 = psimd_concat_hi_u32(vt1, vt3);
    psimd_store_u32(y + 12, vy3);

    y += 16;
  }
  if XNN_UNLIKELY(k != 0) {
    do {
      const psimd_u32 vx0 = psimd_load1_u32(x0);
      x0 += 1;
      const psimd_u32 vx1 = psimd_load1_u32(x1);
      x1 += 1;
      const psimd_u32 vx2 = psimd_load1_u32(x2);
      x2 += 1;
      const psimd_u32 vx3 = psimd_load1_u32(x3);
      x3 += 1;
      const psimd_u32 vx01 = psimd_interleave_lo_u32(vx0, vx1);
      const psimd_u32 vx23 = psimd_interleave_lo_u32(vx2, vx3);
      const psimd_u32 vy = psimd_concat_lo_u32(vx01, vx23);
      psimd_store_u32(y, vy);
      y += 4;
    } while (--k != 0);
  }
}
