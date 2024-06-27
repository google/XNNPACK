// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/math.h"
#include "xnnpack/fft.h"

#include <arm_neon.h>


void xnn_cs16_bfly4_samples1_ukernel__neon(
    size_t batch,
    size_t samples,
    int16_t* data,
    const int16_t* twiddle,
    size_t stride)
{
  assert(batch != 0);
  assert(samples == sizeof(int16_t) * 2);
  assert(data != NULL);
  assert(stride != 0);
  assert(twiddle != NULL);

  const int16x4_t vdiv4 = vdup_n_s16(8191);
  const int16x4_t vnegr = vreinterpret_s16_u32(vdup_n_u32(0x0001ffff));
  uint32x2x4_t vout;

  do {
    const uint32x2x4_t vi = (vld4_dup_u32((void*)data));

    int16x4_t vout1 = vqrdmulh_s16(vreinterpret_s16_u32(vi.val[1]), vdiv4);
    int16x4_t vout3 = vqrdmulh_s16(vreinterpret_s16_u32(vi.val[3]), vdiv4);
    int16x4_t vout0 = vqrdmulh_s16(vreinterpret_s16_u32(vi.val[0]), vdiv4);
    int16x4_t vout2 = vqrdmulh_s16(vreinterpret_s16_u32(vi.val[2]), vdiv4);

    const int16x4_t vtmp4 = vsub_s16(vout1, vout3);
    const int16x4_t vtmp3 = vadd_s16(vout1, vout3);

    int16x4_t vrev4 = vmul_s16(vtmp4, vnegr);   // vrev4 = vtmp4 -r, i
    const int16x4_t vtmp5 = vsub_s16(vout0, vout2);
    vout0 = vadd_s16(vout0, vout2);
    vrev4 = vrev32_s16(vrev4);  // vrev4 = vtmp4 i, -r

    vout.val[2] = vreinterpret_u32_s16(vsub_s16(vout0, vtmp3));
    vout.val[0] = vreinterpret_u32_s16(vadd_s16(vout0, vtmp3));
    vout.val[1] = vreinterpret_u32_s16(vadd_s16(vtmp5, vrev4));
    vout.val[3] = vreinterpret_u32_s16(vsub_s16(vtmp5, vrev4));

    vst4_lane_u32((void*)data, vout, 0);
    data += 8;
  } while(--batch != 0);
}
