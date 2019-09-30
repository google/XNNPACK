// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/zip.h>


void xnn_x32_zip_x4_ukernel__psimd(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  const uint32_t* z = (const uint32_t*) ((uintptr_t) y + n);
  const uint32_t* w = (const uint32_t*) ((uintptr_t) z + n);
  uint32_t* o = output;

  while (n >= 16) {
    const psimd_u32 vx = psimd_load_u32(x);
    x += 4;
    const psimd_u32 vy = psimd_load_u32(y);
    y += 4;
    const psimd_u32 vz = psimd_load_u32(z);
    z += 4;
    const psimd_u32 vw = psimd_load_u32(w);
    w += 4;

    const psimd_u32 vxy_lo = psimd_interleave_lo_u32(vx, vy);
    const psimd_u32 vxy_hi = psimd_interleave_hi_u32(vx, vy);
    const psimd_u32 vzw_lo = psimd_interleave_lo_u32(vz, vw);
    const psimd_u32 vzw_hi = psimd_interleave_hi_u32(vz, vw);

    const psimd_u32 vxyzw0 = psimd_concat_lo_u32(vxy_lo, vzw_lo);
    const psimd_u32 vxyzw1 = psimd_concat_hi_u32(vxy_lo, vzw_lo);
    const psimd_u32 vxyzw2 = psimd_concat_lo_u32(vxy_hi, vzw_hi);
    const psimd_u32 vxyzw3 = psimd_concat_hi_u32(vxy_hi, vzw_hi);

    psimd_store_u32(o, vxyzw0);
    psimd_store_u32(o + 4, vxyzw1);
    psimd_store_u32(o + 8, vxyzw2);
    psimd_store_u32(o + 12, vxyzw3);
    o += 16;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      const psimd_u32 vx = psimd_load2_u32(x);
      x += 2;
      const psimd_u32 vy = psimd_load2_u32(y);
      y += 2;
      const psimd_u32 vz = psimd_load2_u32(z);
      z += 2;
      const psimd_u32 vw = psimd_load2_u32(w);
      w += 2;

      const psimd_u32 vxy = psimd_interleave_lo_u32(vx, vy);
      const psimd_u32 vzw = psimd_interleave_lo_u32(vz, vw);

      const psimd_u32 vxyzw_lo = psimd_concat_lo_u32(vxy, vzw);
      const psimd_u32 vxyzw_hi = psimd_concat_hi_u32(vxy, vzw);

      psimd_store_u32(o, vxyzw_lo);
      psimd_store_u32(o + 4, vxyzw_hi);
      o += 8;
    }
    if (n & 4) {
      const uint32_t vx = *x;
      const uint32_t vy = *y;
      const uint32_t vz = *z;
      const uint32_t vw = *w;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o[3] = vw;
    }
  }
}
