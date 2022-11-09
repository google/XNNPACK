// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/zip.h>


void xnn_x32_zip_x3_ukernel__neon(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % 4 == 0);

  const uint32_t* x = input;
  const uint32_t* y = (const uint32_t*) ((uintptr_t) x + n);
  const uint32_t* z = (const uint32_t*) ((uintptr_t) y + n);
  uint32_t* o = output;

  while (n >= 16) {
    uint32x4x3_t vxyz;
    vxyz.val[0] = vld1q_u32(x); x += 4;
    vxyz.val[1] = vld1q_u32(y); y += 4;
    vxyz.val[2] = vld1q_u32(z); z += 4;
    vst3q_u32(o, vxyz); o += 12;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      uint32x2x3_t vxyz;
      vxyz.val[0] = vld1_u32(x); x += 2;
      vxyz.val[1] = vld1_u32(y); y += 2;
      vxyz.val[2] = vld1_u32(z); z += 2;
      vst3_u32(o, vxyz); o += 6;
    }
    if (n & 4) {
      uint32x2_t vxy = vld1_dup_u32(x);
      const uint32x2_t vz = vld1_dup_u32(z);
      vxy = vld1_lane_u32(y, vxy, 1);
      vst1_u32(o, vxy); o += 2;
      vst1_lane_u32(o, vz, 0);
    }
  }
}
