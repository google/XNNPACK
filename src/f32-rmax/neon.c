// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__neon(
    size_t n,
    const float* x,
    float* y)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  float32x4_t vmax0 = vld1q_dup_f32(x);
  float32x4_t vmax1 = vmax0;
  float32x4_t vmax2 = vmax0;
  float32x4_t vmax3 = vmax0;
  for (; n >= 64; n -= 64) {
    const float32x4_t vx0 = vld1q_f32(x); x += 4;
    const float32x4_t vx1 = vld1q_f32(x); x += 4;
    const float32x4_t vx2 = vld1q_f32(x); x += 4;
    const float32x4_t vx3 = vld1q_f32(x); x += 4;

    vmax0 = vmaxq_f32(vmax0, vx0);
    vmax1 = vmaxq_f32(vmax1, vx1);
    vmax2 = vmaxq_f32(vmax2, vx2);
    vmax3 = vmaxq_f32(vmax3, vx3);
  }
  float32x4_t vmax = vmaxq_f32(vmaxq_f32(vmax0, vmax1), vmaxq_f32(vmax2, vmax3));
  for (; n >= 16; n -= 16) {
    const float32x4_t vx = vld1q_f32(x); x += 4;
    vmax = vmaxq_f32(vmax, vx);
  }
#if XNN_ARCH_ARM64
  float32x2_t vmax_lo = vget_low_f32(vpmaxq_f32(vmax, vmax));
#else
  float32x2_t vmax_lo = vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
#endif
  if XNN_UNLIKELY(n != 0) {
    do {
      const float32x2_t vx = vld1_dup_f32(x); x += 1;
      vmax_lo = vmax_f32(vmax_lo, vx);
      n -= 4;
    } while (n != 0);
  }
#if XNN_ARCH_ARM64
  *y = vmaxv_f32(vmax_lo);
#else
  vst1_lane_f32(y, vpmax_f32(vmax_lo, vmax_lo), 0);
#endif
}
