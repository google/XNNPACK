// Auto-generated file. Do not edit!
//   Template: src/cs16-bfly4/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/fft.h>


void xnn_cs16_bfly4_ukernel__scalar_x1(
    size_t samples,
    int16_t* data,
    const size_t stride,
    const int16_t* twiddle)
{
  const int16_t* tw1 = twiddle;
  const int16_t* tw2 = twiddle;
  const int16_t* tw3 = twiddle;
  int16_t* data0 = data;
  int16_t* data1 = data + samples * 2;
  int16_t* data2 = data + samples * 4;
  int16_t* data3 = data + samples * 6;

  assert(samples != 0);
  assert(data != NULL);
  assert(stride != 0);
  assert(twiddle != NULL);


  if XNN_UNLIKELY(samples != 0) {
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
      tw1 += stride * 2;
      tw2 += stride * 4;
      tw3 += stride * 6;

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
      const int32_t vtmp4r = vtmp0r - vtmp2r;
      const int32_t vtmp4i = vtmp0i - vtmp2i;
      vout2r = vout0r - vtmp3r;
      vout2i = vout0i - vtmp3i;

      vout0r += vtmp3r;
      vout0i += vtmp3i;

      vout1r = vtmp5r + vtmp4i;
      vout1i = vtmp5i - vtmp4r;
      vout3r = vtmp5r - vtmp4i;
      vout3i = vtmp5i + vtmp4r;

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
    } while(--samples != 0);
  }
}
