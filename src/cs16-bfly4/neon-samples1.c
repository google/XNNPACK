// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/fft.h>

#include <arm_neon.h>


void xnn_cs16_bfly4_samples1_ukernel__neon(
    size_t samples,
    int16_t* data,
    const size_t stride,
    const int16_t* twiddle)
{
  assert(samples == 1);
  assert(data != NULL);
  assert(stride != 0);
  assert(twiddle != NULL);

  const int16x8_t vi = vld1q_s16(data);
  const int16x8_t vdiv4 = vdupq_n_s16(8191);
  const int16x8_t vout = vqrdmulhq_s16(vi, vdiv4);

  const int16x4_t vtmp5 = vsub_s16(vget_low_s16(vout), vget_high_s16(vout));
  int16x4_t vout0 = vadd_s16(vget_low_s16(vout), vget_high_s16(vout));

  const int16x4_t vtmp3 = vadd_s16(vget_low_s16(vout), vget_high_s16(vout));
  const int16x4_t vtmp4 = vsub_s16(vget_low_s16(vout), vget_high_s16(vout));

  const int16x4_t vtmp3hi = vext_s16(vtmp3, vtmp3, 2);
  const int16x4_t vout2 = vsub_s16(vout0, vtmp3hi);
  vout0 = vadd_s16(vout0, vtmp3hi);
  const int16x4_t vtmp4rev = vrev64_s16(vtmp4);
  const int16x4_t vout1r3i = vadd_s16(vtmp5, vtmp4rev);
  const int16x4_t vout3r1i = vsub_s16(vtmp5, vtmp4rev);

  vst1_lane_u32((void*) data, vreinterpret_u32_s16(vout0), 0); data += 2;
  vst1_lane_s16(data, vout1r3i, 0); data += 1;
  vst1_lane_s16(data, vout3r1i, 1); data += 1;
  vst1_lane_u32((void*) data, vreinterpret_u32_s16(vout2), 0); data += 2;
  vst1_lane_s16(data, vout3r1i, 0); data += 1;
  vst1_lane_s16(data, vout1r3i, 1);
}
