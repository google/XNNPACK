// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_x2_ukernel__neon(
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
    uint32x4x2_t vxy;
    vxy.val[0] = vld1q_u32(x); x += 4;
    vxy.val[1] = vld1q_u32(y); y += 4;
    vst2q_u32(o, vxy); o += 8;
    n -= 16;
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & 8) {
      uint32x2x2_t vxy;
      vxy.val[0] = vld1_u32(x); x += 2;
      vxy.val[1] = vld1_u32(y); y += 2;
      vst2_u32(o, vxy); o += 4;
    }
    if (n & 4) {
      uint32x2_t vxy = vld1_dup_u32(x);
      vxy = vld1_lane_u32(y, vxy, 1);
      vst1_u32(o, vxy);
    }
  }
}
