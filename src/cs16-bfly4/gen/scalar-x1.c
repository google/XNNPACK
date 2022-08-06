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
    const int16_t* twiddle) {

  const int16_t* tw1 = twiddle;
  const int16_t* tw2 = tw1;
  const int16_t* tw3 = tw1;
  int16_t* out0 = data;
  int16_t* out1 = data + samples * 2;
  int16_t* out2 = data + samples * 4;
  int16_t* out3 = data + samples * 6;

  assert(samples != 0);
  assert(stride != 0);
  assert(twiddle != NULL);
  assert(data != NULL);


  if XNN_UNLIKELY(samples != 0) {
    do {
      int32_t vout0_r = (int32_t) out0[0];
      int32_t vout0_i = (int32_t) out0[1];
      int32_t vout1_r = (int32_t) out1[0];
      int32_t vout1_i = (int32_t) out1[1];
      int32_t vout2_r = (int32_t) out2[0];
      int32_t vout2_i = (int32_t) out2[1];
      int32_t vout3_r = (int32_t) out3[0];
      int32_t vout3_i = (int32_t) out3[1];

      const int32_t tw1_r = (const int32_t) tw1[0];
      const int32_t tw1_i = (const int32_t) tw1[1];
      const int32_t tw2_r = (const int32_t) tw2[0];
      const int32_t tw2_i = (const int32_t) tw2[1];
      const int32_t tw3_r = (const int32_t) tw3[0];
      const int32_t tw3_i = (const int32_t) tw3[1];
      tw1 += stride * 2;
      tw2 += stride * 4;
      tw3 += stride * 6;

      // Note 32767 / 4 = 8191.  Should be 8192.
      vout0_r = math_asr_s32(vout0_r * 8191 + 16384, 15);
      vout0_i = math_asr_s32(vout0_i * 8191 + 16384, 15);
      vout1_r = math_asr_s32(vout1_r * 8191 + 16384, 15);
      vout1_i = math_asr_s32(vout1_i * 8191 + 16384, 15);
      vout2_r = math_asr_s32(vout2_r * 8191 + 16384, 15);
      vout2_i = math_asr_s32(vout2_i * 8191 + 16384, 15);
      vout3_r = math_asr_s32(vout3_r * 8191 + 16384, 15);
      vout3_i = math_asr_s32(vout3_i * 8191 + 16384, 15);

      const int32_t vtmp0_r = math_asr_s32(vout1_r * tw1_r - vout1_i * tw1_i + 16384, 15);
      const int32_t vtmp0_i = math_asr_s32(vout1_r * tw1_i + vout1_i * tw1_r + 16384, 15);
      const int32_t vtmp1_r = math_asr_s32(vout2_r * tw2_r - vout2_i * tw2_i + 16384, 15);
      const int32_t vtmp1_i = math_asr_s32(vout2_r * tw2_i + vout2_i * tw2_r + 16384, 15);
      const int32_t vtmp2_r = math_asr_s32(vout3_r * tw3_r - vout3_i * tw3_i + 16384, 15);
      const int32_t vtmp2_i = math_asr_s32(vout3_r * tw3_i + vout3_i * tw3_r + 16384, 15);

      const int32_t vtmp5_r = vout0_r - vtmp1_r;
      const int32_t vtmp5_i = vout0_i - vtmp1_i;
      vout0_r  += vtmp1_r;
      vout0_i  += vtmp1_i;
      const int32_t vtmp3_r = vtmp0_r + vtmp2_r;
      const int32_t vtmp3_i = vtmp0_i + vtmp2_i;
      const int32_t vtmp4_r = vtmp0_r - vtmp2_r;
      const int32_t vtmp4_i = vtmp0_i - vtmp2_i;
      vout2_r = vout0_r - vtmp3_r;
      vout2_i = vout0_i - vtmp3_i;

      vout0_r += vtmp3_r;
      vout0_i += vtmp3_i;

      vout1_r = vtmp5_r + vtmp4_i;
      vout1_i = vtmp5_i - vtmp4_r;
      vout3_r = vtmp5_r - vtmp4_i;
      vout3_i = vtmp5_i + vtmp4_r;

      out0[0] = (int16_t) vout0_r;
      out0[1] = (int16_t) vout0_i;
      out1[0] = (int16_t) vout1_r;
      out1[1] = (int16_t) vout1_i;
      out2[0] = (int16_t) vout2_r;
      out2[1] = (int16_t) vout2_i;
      out3[0] = (int16_t) vout3_r;
      out3[1] = (int16_t) vout3_i;
      out0 += 2;
      out1 += 2;
      out2 += 2;
      out3 += 2;
    } while(--samples != 0);
  }
}
