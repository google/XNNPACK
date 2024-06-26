// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_x4_ukernel__neon(
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
    uint32x4x4_t vxyzw;
    vxyzw.val[0] = vld1q_u32(x); x += 4;
    vxyzw.val[1] = vld1q_u32(y); y += 4;
    vxyzw.val[2] = vld1q_u32(z); z += 4;
    vxyzw.val[3] = vld1q_u32(w); w += 4;
    vst4q_u32(o, vxyzw); o += 16;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      uint32x2x4_t vxyzw;
      vxyzw.val[0] = vld1_u32(x); x += 2;
      vxyzw.val[1] = vld1_u32(y); y += 2;
      vxyzw.val[2] = vld1_u32(z); z += 2;
      vxyzw.val[3] = vld1_u32(w); w += 2;
      vst4_u32(o, vxyzw); o += 8;
    }
    if (n & 4) {
      uint32x4_t vxyzw = vld1q_dup_u32(x);
      vxyzw = vld1q_lane_u32(y, vxyzw, 1);
      vxyzw = vld1q_lane_u32(z, vxyzw, 2);
      vxyzw = vld1q_lane_u32(w, vxyzw, 3);
      vst1q_u32(o, vxyzw);
    }
  }
}
