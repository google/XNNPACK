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


void xnn_cs16_bfly4m1_ukernel__scalar_x1(
    size_t samples,
    int16_t* data,
    const size_t stride,
    const int16_t* twiddle) {

  int16_t* out0 = data;

  assert(samples == 1);
  assert(data != NULL);
  assert(stride != 0);
  assert(twiddle != NULL);


  if XNN_UNLIKELY(samples != 0) {
    do {
      int32_t vout0r = (int32_t) out0[0];
      int32_t vout0i = (int32_t) out0[1];
      int32_t vout1r = (int32_t) out0[2];
      int32_t vout1i = (int32_t) out0[3];
      int32_t vout2r = (int32_t) out0[4];
      int32_t vout2i = (int32_t) out0[5];
      int32_t vout3r = (int32_t) out0[6];
      int32_t vout3i = (int32_t) out0[7];


      // Note 32767 / 4 = 8191.  Should be 8192.
      vout0r = math_asr_s32(vout0r * 8191 + 16384, 15);
      vout0i = math_asr_s32(vout0i * 8191 + 16384, 15);
      vout1r = math_asr_s32(vout1r * 8191 + 16384, 15);
      vout1i = math_asr_s32(vout1i * 8191 + 16384, 15);
      vout2r = math_asr_s32(vout2r * 8191 + 16384, 15);
      vout2i = math_asr_s32(vout2i * 8191 + 16384, 15);
      vout3r = math_asr_s32(vout3r * 8191 + 16384, 15);
      vout3i = math_asr_s32(vout3i * 8191 + 16384, 15);

      // Note 32767 should be 32768 representing a multiply by 1.
      const int32_t vtmp0r = math_asr_s32(vout1r * 32767 + 16384, 15);
      const int32_t vtmp0i = math_asr_s32(vout1i * 32767 + 16384, 15);
      const int32_t vtmp1r = math_asr_s32(vout2r * 32767 + 16384, 15);
      const int32_t vtmp1i = math_asr_s32(vout2i * 32767 + 16384, 15);
      const int32_t vtmp2r = math_asr_s32(vout3r * 32767 + 16384, 15);
      const int32_t vtmp2i = math_asr_s32(vout3i * 32767 + 16384, 15);

      const int32_t vtmp5r = vout0r - vtmp1r;
      const int32_t vtmp5i = vout0i - vtmp1i;
      vout0r  += vtmp1r;
      vout0i  += vtmp1i;
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

      out0[0] = (int16_t) vout0r;
      out0[1] = (int16_t) vout0i;
      out0[2] = (int16_t) vout1r;
      out0[3] = (int16_t) vout1i;
      out0[4] = (int16_t) vout2r;
      out0[5] = (int16_t) vout2i;
      out0[6] = (int16_t) vout3r;
      out0[7] = (int16_t) vout3i;
    } while(--samples != 0);
  }
}
