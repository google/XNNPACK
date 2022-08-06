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

void xnn_cs16_bfly4_ukernel__scalar_x3(
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

  for (; samples >= 3; samples -= 3) {
    int32_t vout0x0_r = (int32_t) out0[0];
    int32_t vout0x0_i = (int32_t) out0[1];
    int32_t vout0x1_r = (int32_t) out0[2];
    int32_t vout0x1_i = (int32_t) out0[3];
    int32_t vout0x2_r = (int32_t) out0[4];
    int32_t vout0x2_i = (int32_t) out0[5];
    int32_t vout1x0_r = (int32_t) out1[0];
    int32_t vout1x0_i = (int32_t) out1[1];
    int32_t vout1x1_r = (int32_t) out1[2];
    int32_t vout1x1_i = (int32_t) out1[3];
    int32_t vout1x2_r = (int32_t) out1[4];
    int32_t vout1x2_i = (int32_t) out1[5];
    int32_t vout2x0_r = (int32_t) out2[0];
    int32_t vout2x0_i = (int32_t) out2[1];
    int32_t vout2x1_r = (int32_t) out2[2];
    int32_t vout2x1_i = (int32_t) out2[3];
    int32_t vout2x2_r = (int32_t) out2[4];
    int32_t vout2x2_i = (int32_t) out2[5];
    int32_t vout3x0_r = (int32_t) out3[0];
    int32_t vout3x0_i = (int32_t) out3[1];
    int32_t vout3x1_r = (int32_t) out3[2];
    int32_t vout3x1_i = (int32_t) out3[3];
    int32_t vout3x2_r = (int32_t) out3[4];
    int32_t vout3x2_i = (int32_t) out3[5];

    const int32_t tw1x0_r = (const int32_t) tw1[0];
    const int32_t tw1x0_i = (const int32_t) tw1[1];
    tw1 += stride * 2;
    const int32_t tw1x1_r = (const int32_t) tw1[0];
    const int32_t tw1x1_i = (const int32_t) tw1[1];
    tw1 += stride * 2;
    const int32_t tw1x2_r = (const int32_t) tw1[0];
    const int32_t tw1x2_i = (const int32_t) tw1[1];
    tw1 += stride * 2;
    const int32_t tw2x0_r = (const int32_t) tw2[0];
    const int32_t tw2x0_i = (const int32_t) tw2[1];
    tw2 += stride * 4;
    const int32_t tw2x1_r = (const int32_t) tw2[0];
    const int32_t tw2x1_i = (const int32_t) tw2[1];
    tw2 += stride * 4;
    const int32_t tw2x2_r = (const int32_t) tw2[0];
    const int32_t tw2x2_i = (const int32_t) tw2[1];
    tw2 += stride * 4;
    const int32_t tw3x0_r = (const int32_t) tw3[0];
    const int32_t tw3x0_i = (const int32_t) tw3[1];
    tw3 += stride * 6;
    const int32_t tw3x1_r = (const int32_t) tw3[0];
    const int32_t tw3x1_i = (const int32_t) tw3[1];
    tw3 += stride * 6;
    const int32_t tw3x2_r = (const int32_t) tw3[0];
    const int32_t tw3x2_i = (const int32_t) tw3[1];
    tw3 += stride * 6;

    // Note 32767 / 4 = 8191.  Should be 8192.
    vout0x0_r = math_asr_s32(vout0x0_r * 8191 + 16384, 15);
    vout0x0_i = math_asr_s32(vout0x0_i * 8191 + 16384, 15);
    vout0x1_r = math_asr_s32(vout0x1_r * 8191 + 16384, 15);
    vout0x1_i = math_asr_s32(vout0x1_i * 8191 + 16384, 15);
    vout0x2_r = math_asr_s32(vout0x2_r * 8191 + 16384, 15);
    vout0x2_i = math_asr_s32(vout0x2_i * 8191 + 16384, 15);
    vout1x0_r = math_asr_s32(vout1x0_r * 8191 + 16384, 15);
    vout1x0_i = math_asr_s32(vout1x0_i * 8191 + 16384, 15);
    vout1x1_r = math_asr_s32(vout1x1_r * 8191 + 16384, 15);
    vout1x1_i = math_asr_s32(vout1x1_i * 8191 + 16384, 15);
    vout1x2_r = math_asr_s32(vout1x2_r * 8191 + 16384, 15);
    vout1x2_i = math_asr_s32(vout1x2_i * 8191 + 16384, 15);
    vout2x0_r = math_asr_s32(vout2x0_r * 8191 + 16384, 15);
    vout2x0_i = math_asr_s32(vout2x0_i * 8191 + 16384, 15);
    vout2x1_r = math_asr_s32(vout2x1_r * 8191 + 16384, 15);
    vout2x1_i = math_asr_s32(vout2x1_i * 8191 + 16384, 15);
    vout2x2_r = math_asr_s32(vout2x2_r * 8191 + 16384, 15);
    vout2x2_i = math_asr_s32(vout2x2_i * 8191 + 16384, 15);
    vout3x0_r = math_asr_s32(vout3x0_r * 8191 + 16384, 15);
    vout3x0_i = math_asr_s32(vout3x0_i * 8191 + 16384, 15);
    vout3x1_r = math_asr_s32(vout3x1_r * 8191 + 16384, 15);
    vout3x1_i = math_asr_s32(vout3x1_i * 8191 + 16384, 15);
    vout3x2_r = math_asr_s32(vout3x2_r * 8191 + 16384, 15);
    vout3x2_i = math_asr_s32(vout3x2_i * 8191 + 16384, 15);

    const int32_t vtmp0x0_r = math_asr_s32(vout1x0_r * tw1x0_r - vout1x0_i * tw1x0_i + 16384, 15);
    const int32_t vtmp0x0_i = math_asr_s32(vout1x0_r * tw1x0_i + vout1x0_i * tw1x0_r + 16384, 15);
    const int32_t vtmp0x1_r = math_asr_s32(vout1x1_r * tw1x1_r - vout1x1_i * tw1x1_i + 16384, 15);
    const int32_t vtmp0x1_i = math_asr_s32(vout1x1_r * tw1x1_i + vout1x1_i * tw1x1_r + 16384, 15);
    const int32_t vtmp0x2_r = math_asr_s32(vout1x2_r * tw1x2_r - vout1x2_i * tw1x2_i + 16384, 15);
    const int32_t vtmp0x2_i = math_asr_s32(vout1x2_r * tw1x2_i + vout1x2_i * tw1x2_r + 16384, 15);
    const int32_t vtmp1x0_r = math_asr_s32(vout2x0_r * tw2x0_r - vout2x0_i * tw2x0_i + 16384, 15);
    const int32_t vtmp1x0_i = math_asr_s32(vout2x0_r * tw2x0_i + vout2x0_i * tw2x0_r + 16384, 15);
    const int32_t vtmp1x1_r = math_asr_s32(vout2x1_r * tw2x1_r - vout2x1_i * tw2x1_i + 16384, 15);
    const int32_t vtmp1x1_i = math_asr_s32(vout2x1_r * tw2x1_i + vout2x1_i * tw2x1_r + 16384, 15);
    const int32_t vtmp1x2_r = math_asr_s32(vout2x2_r * tw2x2_r - vout2x2_i * tw2x2_i + 16384, 15);
    const int32_t vtmp1x2_i = math_asr_s32(vout2x2_r * tw2x2_i + vout2x2_i * tw2x2_r + 16384, 15);
    const int32_t vtmp2x0_r = math_asr_s32(vout3x0_r * tw3x0_r - vout3x0_i * tw3x0_i + 16384, 15);
    const int32_t vtmp2x0_i = math_asr_s32(vout3x0_r * tw3x0_i + vout3x0_i * tw3x0_r + 16384, 15);
    const int32_t vtmp2x1_r = math_asr_s32(vout3x1_r * tw3x1_r - vout3x1_i * tw3x1_i + 16384, 15);
    const int32_t vtmp2x1_i = math_asr_s32(vout3x1_r * tw3x1_i + vout3x1_i * tw3x1_r + 16384, 15);
    const int32_t vtmp2x2_r = math_asr_s32(vout3x2_r * tw3x2_r - vout3x2_i * tw3x2_i + 16384, 15);
    const int32_t vtmp2x2_i = math_asr_s32(vout3x2_r * tw3x2_i + vout3x2_i * tw3x2_r + 16384, 15);

    const int32_t vtmp5x0_r = vout0x0_r - vtmp1x0_r;
    const int32_t vtmp5x0_i = vout0x0_i - vtmp1x0_i;
    const int32_t vtmp5x1_r = vout0x1_r - vtmp1x1_r;
    const int32_t vtmp5x1_i = vout0x1_i - vtmp1x1_i;
    const int32_t vtmp5x2_r = vout0x2_r - vtmp1x2_r;
    const int32_t vtmp5x2_i = vout0x2_i - vtmp1x2_i;
    vout0x0_r  += vtmp1x0_r;
    vout0x0_i  += vtmp1x0_i;
    vout0x1_r  += vtmp1x1_r;
    vout0x1_i  += vtmp1x1_i;
    vout0x2_r  += vtmp1x2_r;
    vout0x2_i  += vtmp1x2_i;
    const int32_t vtmp3x0_r = vtmp0x0_r + vtmp2x0_r;
    const int32_t vtmp3x0_i = vtmp0x0_i + vtmp2x0_i;
    const int32_t vtmp3x1_r = vtmp0x1_r + vtmp2x1_r;
    const int32_t vtmp3x1_i = vtmp0x1_i + vtmp2x1_i;
    const int32_t vtmp3x2_r = vtmp0x2_r + vtmp2x2_r;
    const int32_t vtmp3x2_i = vtmp0x2_i + vtmp2x2_i;
    const int32_t vtmp4x0_r = vtmp0x0_r - vtmp2x0_r;
    const int32_t vtmp4x0_i = vtmp0x0_i - vtmp2x0_i;
    const int32_t vtmp4x1_r = vtmp0x1_r - vtmp2x1_r;
    const int32_t vtmp4x1_i = vtmp0x1_i - vtmp2x1_i;
    const int32_t vtmp4x2_r = vtmp0x2_r - vtmp2x2_r;
    const int32_t vtmp4x2_i = vtmp0x2_i - vtmp2x2_i;
    vout2x0_r = vout0x0_r - vtmp3x0_r;
    vout2x0_i = vout0x0_i - vtmp3x0_i;
    vout2x1_r = vout0x1_r - vtmp3x1_r;
    vout2x1_i = vout0x1_i - vtmp3x1_i;
    vout2x2_r = vout0x2_r - vtmp3x2_r;
    vout2x2_i = vout0x2_i - vtmp3x2_i;
    vout0x0_r += vtmp3x0_r;
    vout0x0_i += vtmp3x0_i;
    vout0x1_r += vtmp3x1_r;
    vout0x1_i += vtmp3x1_i;
    vout0x2_r += vtmp3x2_r;
    vout0x2_i += vtmp3x2_i;
    vout1x0_r = vtmp5x0_r + vtmp4x0_i;
    vout1x0_i = vtmp5x0_i - vtmp4x0_r;
    vout1x1_r = vtmp5x1_r + vtmp4x1_i;
    vout1x1_i = vtmp5x1_i - vtmp4x1_r;
    vout1x2_r = vtmp5x2_r + vtmp4x2_i;
    vout1x2_i = vtmp5x2_i - vtmp4x2_r;
    vout3x0_r = vtmp5x0_r - vtmp4x0_i;
    vout3x0_i = vtmp5x0_i + vtmp4x0_r;
    vout3x1_r = vtmp5x1_r - vtmp4x1_i;
    vout3x1_i = vtmp5x1_i + vtmp4x1_r;
    vout3x2_r = vtmp5x2_r - vtmp4x2_i;
    vout3x2_i = vtmp5x2_i + vtmp4x2_r;

    out0[0] = (int16_t) vout0x0_r;
    out0[1] = (int16_t) vout0x0_i;
    out0[2] = (int16_t) vout0x1_r;
    out0[3] = (int16_t) vout0x1_i;
    out0[4] = (int16_t) vout0x2_r;
    out0[5] = (int16_t) vout0x2_i;
    out0 += 3 * 2;
    out1[0] = (int16_t) vout1x0_r;
    out1[1] = (int16_t) vout1x0_i;
    out1[2] = (int16_t) vout1x1_r;
    out1[3] = (int16_t) vout1x1_i;
    out1[4] = (int16_t) vout1x2_r;
    out1[5] = (int16_t) vout1x2_i;
    out1 += 3 * 2;
    out2[0] = (int16_t) vout2x0_r;
    out2[1] = (int16_t) vout2x0_i;
    out2[2] = (int16_t) vout2x1_r;
    out2[3] = (int16_t) vout2x1_i;
    out2[4] = (int16_t) vout2x2_r;
    out2[5] = (int16_t) vout2x2_i;
    out2 += 3 * 2;
    out3[0] = (int16_t) vout3x0_r;
    out3[1] = (int16_t) vout3x0_i;
    out3[2] = (int16_t) vout3x1_r;
    out3[3] = (int16_t) vout3x1_i;
    out3[4] = (int16_t) vout3x2_r;
    out3[5] = (int16_t) vout3x2_i;
    out3 += 3 * 2;
  }

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
