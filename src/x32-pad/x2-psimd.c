// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/pad.h>


void xnn_x32_pad_x2__psimd(
    size_t m,
    size_t n,
    size_t l,
    size_t r,
    uint32_t c,
    const void* x,
    size_t x_stride,
    void* y,
    size_t y_stride)
{
  assert(m <= 2);
  assert(l % 4 == 0);
  assert(n % 4 == 0);
  assert(r % 4 == 0);

  const uint32_t* x0 = x;
  uint32_t* y0 = y;

  const uint32_t* x1 = (const uint32_t*) ((uintptr_t) x0 + x_stride);
  uint32_t* y1 = (uint32_t*) ((uintptr_t) y0 + y_stride);
  if (m != 2) {
    x1 = x0;
    y1 = y0;
  }
  const psimd_u32 vc = psimd_splat_u32(c);

  // Pre-pad input channels.
  for (; l >= 16; l -= 16) {
    psimd_store_u32(y0, vc); y0 += 4;
    psimd_store_u32(y1, vc); y1 += 4;
  }
  if (l & 8) {
    psimd_store2_u32(y0, vc); y0 += 2;
    psimd_store2_u32(y1, vc); y1 += 2;
  }
  if (l & 4) {
    psimd_store1_u32(y0, vc); y0 += 1;
    psimd_store1_u32(y1, vc); y1 += 1;
  }

  // Copy input channels.
  for (; n >= 16; n -= 16) {
    const psimd_u32 vt0 = psimd_load_u32(x0); x0 += 4;
    const psimd_u32 vt1 = psimd_load_u32(x1); x1 += 4;
    psimd_store_u32(y0, vt0); y0 += 4;
    psimd_store_u32(y1, vt1); y1 += 4;
  }
  if (n != 0) {
    psimd_u32 vt0 = psimd_load_u32(x0);
    psimd_u32 vt1 = psimd_load_u32(x1);
    if (n & 8) {
      psimd_store2_u32(y0, vt0); y0 += 2;
      psimd_store2_u32(y1, vt1); y1 += 2;
      vt0 = psimd_concat_hi_u32(vt0, vt0);
      vt1 = psimd_concat_hi_u32(vt1, vt1);
    }
    if (n & 4) {
      psimd_store1_u32(y0, vt0); y0 += 1;
      psimd_store1_u32(y1, vt1); y1 += 1;
    }
  }

  // Post-pad input channels.
  for (; r >= 16; r -= 16) {
    psimd_store_u32(y0, vc); y0 += 4;
    psimd_store_u32(y1, vc); y1 += 4;
  }
  if (r & 8) {
    psimd_store2_u32(y0, vc); y0 += 2;
    psimd_store2_u32(y1, vc); y1 += 2;
  }
  if (r & 4) {
    psimd_store1_u32(y0, vc);
    psimd_store1_u32(y1, vc);
  }
}
