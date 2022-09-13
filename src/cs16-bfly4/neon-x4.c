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

void xnn_cs16_bfly4_ukernel__neon_x4(
    size_t batch,
    size_t samples,
    int16_t* data,
    const int16_t* twiddle,
    size_t stride)
{
  assert(batch != 0);
  assert(samples != 0);
  assert(data != NULL);
  assert(stride != 0);
  assert(twiddle != NULL);

  const int16x8_t vdiv4 = vdupq_n_s16(8191);
  const int16x8_t vnegr = vreinterpretq_s16_u32(vdupq_n_u32(0x0001ffff));

  int16_t* data3 = data;
  do {
    int16_t* data0 = data3;
    int16_t* data1 = (int16_t*) ((uintptr_t) data0 + samples);
    int16_t* data2 = (int16_t*) ((uintptr_t) data1 + samples);
    data3 = (int16_t*) ((uintptr_t) data2 + samples);

    const int16_t* tw1 = twiddle;
    const int16_t* tw2 = twiddle;
    const int16_t* tw3 = twiddle;

    size_t s = samples;
    for (; s >= sizeof(int16_t) * 8; s -= sizeof(int16_t) * 8) {
      int16x8_t vout0 = vld1q_s16(data0);
      int16x8_t vout1 = vld1q_s16(data1);
      int16x8_t vout2 = vld1q_s16(data2);
      int16x8_t vout3 = vld1q_s16(data3);

      int16x8_t vtw1 = vreinterpretq_s16_u32(vld1q_dup_u32((const void*) tw1));
      int16x8_t vtw2 = vreinterpretq_s16_u32(vld1q_dup_u32((const void*) tw2));
      int16x8_t vtw3 = vreinterpretq_s16_u32(vld1q_dup_u32((const void*) tw3));
      tw1 = (const int16_t*) ((uintptr_t) tw1 + stride);
      tw2 = (const int16_t*) ((uintptr_t) tw2 + stride * 2);
      tw3 = (const int16_t*) ((uintptr_t) tw3 + stride * 3);
      vtw1 = vreinterpretq_s16_u32(vld1q_lane_u32((const void*) tw1, vreinterpretq_u32_s16(vtw1), 1));
      vtw2 = vreinterpretq_s16_u32(vld1q_lane_u32((const void*) tw2, vreinterpretq_u32_s16(vtw2), 1));
      vtw3 = vreinterpretq_s16_u32(vld1q_lane_u32((const void*) tw3, vreinterpretq_u32_s16(vtw3), 1));
      tw1 = (const int16_t*) ((uintptr_t) tw1 + stride);
      tw2 = (const int16_t*) ((uintptr_t) tw2 + stride * 2);
      tw3 = (const int16_t*) ((uintptr_t) tw3 + stride * 3);
      vtw1 = vreinterpretq_s16_u32(vld1q_lane_u32((const void*) tw1, vreinterpretq_u32_s16(vtw1), 2));
      vtw2 = vreinterpretq_s16_u32(vld1q_lane_u32((const void*) tw2, vreinterpretq_u32_s16(vtw2), 2));
      vtw3 = vreinterpretq_s16_u32(vld1q_lane_u32((const void*) tw3, vreinterpretq_u32_s16(vtw3), 2));
      tw1 = (const int16_t*) ((uintptr_t) tw1 + stride);
      tw2 = (const int16_t*) ((uintptr_t) tw2 + stride * 2);
      tw3 = (const int16_t*) ((uintptr_t) tw3 + stride * 3);
      vtw1 = vreinterpretq_s16_u32(vld1q_lane_u32((const void*) tw1, vreinterpretq_u32_s16(vtw1), 3));
      vtw2 = vreinterpretq_s16_u32(vld1q_lane_u32((const void*) tw2, vreinterpretq_u32_s16(vtw2), 3));
      vtw3 = vreinterpretq_s16_u32(vld1q_lane_u32((const void*) tw3, vreinterpretq_u32_s16(vtw3), 3));
      tw1 = (const int16_t*) ((uintptr_t) tw1 + stride);
      tw2 = (const int16_t*) ((uintptr_t) tw2 + stride * 2);
      tw3 = (const int16_t*) ((uintptr_t) tw3 + stride * 3);

      // Note 32767 / 4 = 8191.  Should be 8192.
      vout1 = vqrdmulhq_s16(vout1, vdiv4);
      vout3 = vqrdmulhq_s16(vout3, vdiv4);
      vout0 = vqrdmulhq_s16(vout0, vdiv4);
      vout2 = vqrdmulhq_s16(vout2, vdiv4);

      int16x4x2_t vout1ri = vuzp_s16(vget_low_s16(vout1), vget_high_s16(vout1));
      int16x4x2_t vout2ri = vuzp_s16(vget_low_s16(vout2), vget_high_s16(vout2));
      int16x4x2_t vout3ri = vuzp_s16(vget_low_s16(vout3), vget_high_s16(vout3));
      const int16x4x2_t vtw1ri = vuzp_s16(vget_low_s16(vtw1), vget_high_s16(vtw1));
      const int16x4x2_t vtw2ri = vuzp_s16(vget_low_s16(vtw2), vget_high_s16(vtw2));
      const int16x4x2_t vtw3ri = vuzp_s16(vget_low_s16(vtw3), vget_high_s16(vtw3));
      int32x4_t vacc1r = vmull_s16(vout1ri.val[0], vtw1ri.val[0]);
      int32x4_t vacc2r = vmull_s16(vout2ri.val[0], vtw2ri.val[0]);
      int32x4_t vacc3r = vmull_s16(vout3ri.val[0], vtw3ri.val[0]);
      int32x4_t vacc1i = vmull_s16(vout1ri.val[0], vtw1ri.val[1]);
      int32x4_t vacc2i = vmull_s16(vout2ri.val[0], vtw2ri.val[1]);
      int32x4_t vacc3i = vmull_s16(vout3ri.val[0], vtw3ri.val[1]);
      vacc1r = vmlsl_s16(vacc1r, vout1ri.val[1], vtw1ri.val[1]);
      vacc2r = vmlsl_s16(vacc2r, vout2ri.val[1], vtw2ri.val[1]);
      vacc3r = vmlsl_s16(vacc3r, vout3ri.val[1], vtw3ri.val[1]);
      vacc1i = vmlal_s16(vacc1i, vout1ri.val[1], vtw1ri.val[0]);
      vacc2i = vmlal_s16(vacc2i, vout2ri.val[1], vtw2ri.val[0]);
      vacc3i = vmlal_s16(vacc3i, vout3ri.val[1], vtw3ri.val[0]);
      int16x4_t vout1r = vrshrn_n_s32(vacc1r, 15);
      int16x4_t vout2r = vrshrn_n_s32(vacc2r, 15);
      int16x4_t vout3r = vrshrn_n_s32(vacc3r, 15);
      int16x4_t vout1i = vrshrn_n_s32(vacc1i, 15);
      int16x4_t vout2i = vrshrn_n_s32(vacc2i, 15);
      int16x4_t vout3i = vrshrn_n_s32(vacc3i, 15);
      vout1ri = vzip_s16(vout1r, vout1i);
      vout2ri = vzip_s16(vout2r, vout2i);
      vout3ri = vzip_s16(vout3r, vout3i);
      const int16x8_t vtmp0 = vcombine_s16(vout1ri.val[0], vout1ri.val[1]);
      const int16x8_t vtmp1 = vcombine_s16(vout2ri.val[0], vout2ri.val[1]);
      const int16x8_t vtmp2 = vcombine_s16(vout3ri.val[0], vout3ri.val[1]);

      const int16x8_t vtmp4 = vsubq_s16(vtmp0, vtmp2);
      const int16x8_t vtmp3 = vaddq_s16(vtmp0, vtmp2);

      int16x8_t vrev4 = vmulq_s16(vtmp4, vnegr);   // vrev4 = vtmp4 -r, i
      const int16x8_t vtmp5 = vsubq_s16(vout0, vtmp1);
      vout0 = vaddq_s16(vout0, vtmp1);
      vrev4 = vrev32q_s16(vrev4);  // vrev4 = vtmp4 i, -r

      vout2 = vsubq_s16(vout0, vtmp3);
      vout0 = vaddq_s16(vout0, vtmp3);
      vout1 = vaddq_s16(vtmp5, vrev4);
      vout3 = vsubq_s16(vtmp5, vrev4);

      vst1q_s16(data0, vout0);  data0 += 8;
      vst1q_s16(data1, vout1);  data1 += 8;
      vst1q_s16(data2, vout2);  data2 += 8;
      vst1q_s16(data3, vout3);  data3 += 8;
    }

    if XNN_UNLIKELY(s != 0) {
      do {
        int32_t vout0r = (int32_t) data0[0];
        int32_t vout0i = (int32_t) data0[1];
        int32_t vout1r = (int32_t) data1[0];
        int32_t vout1i = (int32_t) data1[1];
        int32_t vout2r = (int32_t) data2[0];
        int32_t vout2i = (int32_t) data2[1];
        int32_t vout3r = (int32_t) data3[0];
        int32_t vout3i = (int32_t) data3[1];

        const int32_t vtw1r = (const int32_t) tw1[0];
        const int32_t vtw1i = (const int32_t) tw1[1];
        const int32_t vtw2r = (const int32_t) tw2[0];
        const int32_t vtw2i = (const int32_t) tw2[1];
        const int32_t vtw3r = (const int32_t) tw3[0];
        const int32_t vtw3i = (const int32_t) tw3[1];
        tw1 = (const int16_t*) ((uintptr_t) tw1 + stride);
        tw2 = (const int16_t*) ((uintptr_t) tw2 + stride * 2);
        tw3 = (const int16_t*) ((uintptr_t) tw3 + stride * 3);

        // Note 32767 / 4 = 8191.  Should be 8192.
        vout0r = math_asr_s32(vout0r * 8191 + 16384, 15);
        vout0i = math_asr_s32(vout0i * 8191 + 16384, 15);
        vout1r = math_asr_s32(vout1r * 8191 + 16384, 15);
        vout1i = math_asr_s32(vout1i * 8191 + 16384, 15);
        vout2r = math_asr_s32(vout2r * 8191 + 16384, 15);
        vout2i = math_asr_s32(vout2i * 8191 + 16384, 15);
        vout3r = math_asr_s32(vout3r * 8191 + 16384, 15);
        vout3i = math_asr_s32(vout3i * 8191 + 16384, 15);

        const int32_t vtmp0r = math_asr_s32(vout1r * vtw1r - vout1i * vtw1i + 16384, 15);
        const int32_t vtmp0i = math_asr_s32(vout1r * vtw1i + vout1i * vtw1r + 16384, 15);
        const int32_t vtmp1r = math_asr_s32(vout2r * vtw2r - vout2i * vtw2i + 16384, 15);
        const int32_t vtmp1i = math_asr_s32(vout2r * vtw2i + vout2i * vtw2r + 16384, 15);
        const int32_t vtmp2r = math_asr_s32(vout3r * vtw3r - vout3i * vtw3i + 16384, 15);
        const int32_t vtmp2i = math_asr_s32(vout3r * vtw3i + vout3i * vtw3r + 16384, 15);

        const int32_t vtmp5r = vout0r - vtmp1r;
        const int32_t vtmp5i = vout0i - vtmp1i;
        vout0r += vtmp1r;
        vout0i += vtmp1i;
        const int32_t vtmp3r = vtmp0r + vtmp2r;
        const int32_t vtmp3i = vtmp0i + vtmp2i;
        const int32_t vtmp4r = vtmp0i - vtmp2i;
        const int32_t vtmp4i = -(vtmp0r - vtmp2r);  // swap r,i and neg i
        vout2r = vout0r - vtmp3r;
        vout2i = vout0i - vtmp3i;

        vout0r += vtmp3r;
        vout0i += vtmp3i;

        vout1r = vtmp5r + vtmp4r;
        vout1i = vtmp5i + vtmp4i;
        vout3r = vtmp5r - vtmp4r;
        vout3i = vtmp5i - vtmp4i;

        data0[0] = (int16_t) vout0r;
        data0[1] = (int16_t) vout0i;
        data1[0] = (int16_t) vout1r;
        data1[1] = (int16_t) vout1i;
        data2[0] = (int16_t) vout2r;
        data2[1] = (int16_t) vout2i;
        data3[0] = (int16_t) vout3r;
        data3[1] = (int16_t) vout3i;
        data0 += 2;
        data1 += 2;
        data2 += 2;
        data3 += 2;

        s -= sizeof(int16_t) * 2;
      } while (s != 0);
    }
  } while (--batch != 0);
}
