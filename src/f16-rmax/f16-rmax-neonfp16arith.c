// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/rmax.h>


void xnn_f16_rmax_ukernel__neonfp16arith(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != 0);
  assert(output != 0);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  float16x8_t vmax0 = vreinterpretq_f16_u16(vld1q_dup_u16(i));
  float16x8_t vmax1 = vmax0;
  float16x8_t vmax2 = vmax0;
  float16x8_t vmax3 = vmax0;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vmax0 = vmaxq_f16(vmax0, vx0);
    vmax1 = vmaxq_f16(vmax1, vx1);
    vmax2 = vmaxq_f16(vmax2, vx2);
    vmax3 = vmaxq_f16(vmax3, vx3);
  }
  float16x8_t vmax = vmaxq_f16(vmaxq_f16(vmax0, vmax1), vmaxq_f16(vmax2, vmax3));
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vmax = vmaxq_f16(vmax, vx);
  }
  float16x4_t vmax_lo = vmax_f16(vget_low_f16(vmax), vget_high_f16(vmax));
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));
    float16x4_t vx_lo = vget_low_f16(vx);
    if (batch & (4 * sizeof(uint16_t))) {
      vmax_lo = vmax_f16(vmax_lo, vx_lo);
      vx_lo = vget_high_f16(vx);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vmax_lo = vmax_f16(vmax_lo, vext_f16(vmax_lo, vx_lo, 2));
      vx_lo = vext_f16(vx_lo, vx_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vmax_lo = vmax_f16(vmax_lo, vext_f16(vmax_lo, vx_lo, 1));
    }
  }
  #if XNN_ARCH_ARM64 && defined(__GNUC__)
    *((__fp16*) o) = vmaxv_f16(vmax_lo);
  #else
    vmax_lo = vpmax_f16(vmax_lo, vmax_lo);
    vmax_lo = vpmax_f16(vmax_lo, vmax_lo);
    vst1_lane_u16(o, vreinterpret_u16_f16(vmax_lo), 0);
  #endif
}
