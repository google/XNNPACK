// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/zip.h>


void xnn_x32_zip_x2_ukernel__psimd(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  uint32_t* o = output;

  while (n >= 16) {
    const psimd_u32 vx = psimd_load_u32(x);
    x += 4;
    const psimd_u32 vy = psimd_load_u32(y);
    y += 4;
    const psimd_u32 vxy_lo = psimd_interleave_lo_u32(vx, vy);
    const psimd_u32 vxy_hi = psimd_interleave_hi_u32(vx, vy);
    psimd_store_u32(o, vxy_lo);
    psimd_store_u32(o + 4, vxy_hi);
    o += 8;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      const psimd_u32 vx = psimd_load2_u32(x);
      x += 2;
      const psimd_u32 vy = psimd_load2_u32(y);
      y += 2;
      const psimd_u32 vxy = psimd_interleave_lo_u32(vx, vy);
      psimd_store_u32((psimd_u32*) o, vxy);
      o += 4;
    }
    if (n & 4) {
      const uint32_t vx = *x;
      const uint32_t vy = *y;
      o[0] = vx;
      o[1] = vy;
    }
  }
}
