// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/neonfp16.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vcvt.h>


void xnn_f32_f16_vcvt_ukernel__neonfp16_x8(
    size_t n,
    const float* input,
    void* output,
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  for (; n >= 8 * sizeof(uint16_t); n -= 8 * sizeof(uint16_t)) {
    const float32x4_t vf_lo = vld1q_f32(input); input += 4;
    const float32x4_t vf_hi = vld1q_f32(input); input += 4;

    const uint16x8_t vh = vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(vf_lo), vcvt_f16_f32(vf_hi)));

    vst1q_u16(o, vh); o += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint16_t));
    assert(n <= 7 * sizeof(uint16_t));
    float32x4_t vf = vld1q_f32(input); input += 4;

    uint16x4_t vh = vreinterpret_u16_f16(vcvt_f16_f32(vf));

    if (n & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vh); o += 4;
      vf = vld1q_f32(input);
      vh = vreinterpret_u16_f16(vcvt_f16_f32(vf));
    }
    if (n & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_u16(vh), 0); o += 2;
      vh = vext_u16(vh, vh, 2);
    }
    if (n & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vh, 0);
    }
  }
}
