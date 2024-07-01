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


void xnn_cs16_bfly4_ukernel__neon_x1(
    size_t batch,
    size_t samples,
    int16_t* data,
    const int16_t* twiddle,
    size_t stride)
{
  assert(batch != 0);
  assert(samples != 0);
  assert(samples % (sizeof(int16_t) * 2) == 0);
  assert(data != NULL);
  assert(stride != 0);
  assert(twiddle != NULL);

  const int16x4_t vdiv4 = vdup_n_s16(8191);
  const int16x4_t vnegr = vreinterpret_s16_u32(vdup_n_u32(0x0001ffff));

  int16_t* data3 = data;

  do {
    int16_t* data0 = data3;
    int16_t* data1 = (int16_t*) ((uintptr_t) data0 + samples);
    int16_t* data2 = (int16_t*) ((uintptr_t) data1 + samples);
    data3 = (int16_t*) ((uintptr_t) data2 + samples);

    // First sample skips twiddle.
    {
      int16x4_t vout0 = vreinterpret_s16_u32(vld1_dup_u32((void*) data0));
      int16x4_t vout1 = vreinterpret_s16_u32(vld1_dup_u32((void*) data1));
      int16x4_t vout2 = vreinterpret_s16_u32(vld1_dup_u32((void*) data2));
      int16x4_t vout3 = vreinterpret_s16_u32(vld1_dup_u32((void*) data3));

      vout1 = vqrdmulh_s16(vout1, vdiv4);
      vout3 = vqrdmulh_s16(vout3, vdiv4);
      vout0 = vqrdmulh_s16(vout0, vdiv4);
      vout2 = vqrdmulh_s16(vout2, vdiv4);

      const int16x4_t vtmp4 = vsub_s16(vout1, vout3);
      const int16x4_t vtmp3 = vadd_s16(vout1, vout3);

      int16x4_t vrev4 = vmul_s16(vtmp4, vnegr);   // vrev4 = vtmp4 -r, i
      const int16x4_t vtmp5 = vsub_s16(vout0, vout2);
      vout0 = vadd_s16(vout0, vout2);
      vrev4 = vrev32_s16(vrev4);  // vrev4 = vtmp4 i, -r

      vout2 = vsub_s16(vout0, vtmp3);
      vout0 = vadd_s16(vout0, vtmp3);
      vout1 = vadd_s16(vtmp5, vrev4);
      vout3 = vsub_s16(vtmp5, vrev4);

      vst1_lane_u32((void*) data0, vreinterpret_u32_s16(vout0), 0);  data0 += 2;
      vst1_lane_u32((void*) data1, vreinterpret_u32_s16(vout1), 0);  data1 += 2;
      vst1_lane_u32((void*) data2, vreinterpret_u32_s16(vout2), 0);  data2 += 2;
      vst1_lane_u32((void*) data3, vreinterpret_u32_s16(vout3), 0);  data3 += 2;
    }

    size_t s = samples - sizeof(int16_t) * 2;

    if XNN_LIKELY(s != 0) {

      const int16_t* tw1 = (const int16_t*) ((uintptr_t) twiddle + stride);
      const int16_t* tw2 = (const int16_t*) ((uintptr_t) twiddle + stride * 2);
      const int16_t* tw3 = (const int16_t*) ((uintptr_t) twiddle + stride * 3);

      do {
        int16x4_t vout0 = vreinterpret_s16_u32(vld1_dup_u32((void*) data0));
        int16x4_t vout1 = vreinterpret_s16_u32(vld1_dup_u32((void*) data1));
        int16x4_t vout2 = vreinterpret_s16_u32(vld1_dup_u32((void*) data2));
        int16x4_t vout3 = vreinterpret_s16_u32(vld1_dup_u32((void*) data3));

        const int16x4_t vtw1 = vreinterpret_s16_u32(vld1_dup_u32((const void*) tw1));
        const int16x4_t vtw2 = vreinterpret_s16_u32(vld1_dup_u32((const void*) tw2));
        const int16x4_t vtw3 = vreinterpret_s16_u32(vld1_dup_u32((const void*) tw3));
        tw1 = (const int16_t*) ((uintptr_t) tw1 + stride);
        tw2 = (const int16_t*) ((uintptr_t) tw2 + stride * 2);
        tw3 = (const int16_t*) ((uintptr_t) tw3 + stride * 3);

        // Note 32767 / 4 = 8191.  Should be 8192.
        vout0 = vqrdmulh_s16(vout0, vdiv4);
        vout1 = vqrdmulh_s16(vout1, vdiv4);
        vout2 = vqrdmulh_s16(vout2, vdiv4);
        vout3 = vqrdmulh_s16(vout3, vdiv4);

        int16x4_t vnegtw1 = vmul_s16(vtw1, vnegr);  // vrevtw1 = vtw1 -r, i
        int16x4_t vnegtw2 = vmul_s16(vtw2, vnegr);  // vrevtw2 = vtw2 -r, i
        int16x4_t vnegtw3 = vmul_s16(vtw3, vnegr);  // vrevtw3 = vtw3 -r, i
        int32x4_t vaccr1 = vmull_lane_s16(vtw1, vout1, 0);
        int32x4_t vaccr2 = vmull_lane_s16(vtw2, vout2, 0);
        int32x4_t vaccr3 = vmull_lane_s16(vtw3, vout3, 0);
        int16x4_t vrevtw1 = vrev32_s16(vnegtw1);    // vrevtw1 = vtw1 i, -r
        int16x4_t vrevtw2 = vrev32_s16(vnegtw2);    // vrevtw2 = vtw2 i, -r
        int16x4_t vrevtw3 = vrev32_s16(vnegtw3);    // vrevtw3 = vtw3 i, -r
        vaccr1 = vmlsl_lane_s16(vaccr1, vrevtw1, vout1, 1);
        vaccr2 = vmlsl_lane_s16(vaccr2, vrevtw2, vout2, 1);
        vaccr3 = vmlsl_lane_s16(vaccr3, vrevtw3, vout3, 1);
        const int16x4_t vtmp0 = vrshrn_n_s32(vaccr1, 15);
        const int16x4_t vtmp1 = vrshrn_n_s32(vaccr2, 15);
        const int16x4_t vtmp2 = vrshrn_n_s32(vaccr3, 15);

        const int16x4_t vtmp4 = vsub_s16(vtmp0, vtmp2);
        const int16x4_t vtmp3 = vadd_s16(vtmp0, vtmp2);

        int16x4_t vrev4 = vmul_s16(vtmp4, vnegr);   // vrev4 = vtmp4 -r, i
        const int16x4_t vtmp5 = vsub_s16(vout0, vtmp1);
        vout0 = vadd_s16(vout0, vtmp1);
        vrev4 = vrev32_s16(vrev4);  // vrev4 = vtmp4 i, -r

        vout2 = vsub_s16(vout0, vtmp3);
        vout0 = vadd_s16(vout0, vtmp3);
        vout1 = vadd_s16(vtmp5, vrev4);
        vout3 = vsub_s16(vtmp5, vrev4);

        vst1_lane_u32((void*) data0, vreinterpret_u32_s16(vout0), 0); data0 += 2;
        vst1_lane_u32((void*) data1, vreinterpret_u32_s16(vout1), 0); data1 += 2;
        vst1_lane_u32((void*) data2, vreinterpret_u32_s16(vout2), 0); data2 += 2;
        vst1_lane_u32((void*) data3, vreinterpret_u32_s16(vout3), 0); data3 += 2;

        s -= sizeof(int16_t) * 2;
      } while (s != 0);
    }
  } while (--batch != 0);
}

