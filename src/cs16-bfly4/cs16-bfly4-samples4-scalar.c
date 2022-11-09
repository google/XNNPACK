// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/fft.h>


void xnn_cs16_bfly4_samples4_ukernel__scalar(
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

  const int16_t* tw1 = twiddle;
  const int16_t* tw2 = twiddle;
  const int16_t* tw3 = twiddle;
  tw1 = (const int16_t*) ((uintptr_t) tw1 + stride);
  tw2 = (const int16_t*) ((uintptr_t) tw2 + stride * 2);
  tw3 = (const int16_t*) ((uintptr_t) tw3 + stride * 3);

  const int32_t vtw1r0 = (const int32_t) tw1[0];
  const int32_t vtw1i0 = (const int32_t) tw1[1];
  tw1 = (const int16_t*) ((uintptr_t) tw1 + stride);
  const int32_t vtw1r1 = (const int32_t) tw1[0];
  const int32_t vtw1i1 = (const int32_t) tw1[1];
  tw1 = (const int16_t*) ((uintptr_t) tw1 + stride);
  const int32_t vtw1r2 = (const int32_t) tw1[0];
  const int32_t vtw1i2 = (const int32_t) tw1[1];
  const int32_t vtw2r0 = (const int32_t) tw2[0];
  const int32_t vtw2i0 = (const int32_t) tw2[1];
  tw2 = (const int16_t*) ((uintptr_t) tw2 + stride * 2);
  const int32_t vtw2r1 = (const int32_t) tw2[0];
  const int32_t vtw2i1 = (const int32_t) tw2[1];
  tw2 = (const int16_t*) ((uintptr_t) tw2 + stride * 2);
  const int32_t vtw2r2 = (const int32_t) tw2[0];
  const int32_t vtw2i2 = (const int32_t) tw2[1];
  const int32_t vtw3r0 = (const int32_t) tw3[0];
  const int32_t vtw3i0 = (const int32_t) tw3[1];
  tw3 = (const int16_t*) ((uintptr_t) tw3 + stride * 3);
  const int32_t vtw3r1 = (const int32_t) tw3[0];
  const int32_t vtw3i1 = (const int32_t) tw3[1];
  tw3 = (const int16_t*) ((uintptr_t) tw3 + stride * 3);
  const int32_t vtw3r2 = (const int32_t) tw3[0];
  const int32_t vtw3i2 = (const int32_t) tw3[1];

  int16_t* data3 = data;

  do {
    int16_t* data0 = data3;
    int16_t* data1 = (int16_t*) ((uintptr_t) data0 + samples);
    int16_t* data2 = (int16_t*) ((uintptr_t) data1 + samples);
    data3 = (int16_t*) ((uintptr_t) data2 + samples);

    // First sample skips twiddle.
    // Same code as samples=1 but supports stride
    {
      int32_t vout0r = (int32_t) data0[0];
      int32_t vout0i = (int32_t) data0[1];
      int32_t vout1r = (int32_t) data1[0];
      int32_t vout1i = (int32_t) data1[1];
      int32_t vout2r = (int32_t) data2[0];
      int32_t vout2i = (int32_t) data2[1];
      int32_t vout3r = (int32_t) data3[0];
      int32_t vout3i = (int32_t) data3[1];

      // Note 32767 / 4 = 8191.  Should be 8192.
      vout0r = math_asr_s32(vout0r * 8191 + 16384, 15);
      vout0i = math_asr_s32(vout0i * 8191 + 16384, 15);
      vout1r = math_asr_s32(vout1r * 8191 + 16384, 15);
      vout1i = math_asr_s32(vout1i * 8191 + 16384, 15);
      vout2r = math_asr_s32(vout2r * 8191 + 16384, 15);
      vout2i = math_asr_s32(vout2i * 8191 + 16384, 15);
      vout3r = math_asr_s32(vout3r * 8191 + 16384, 15);
      vout3i = math_asr_s32(vout3i * 8191 + 16384, 15);

      const int32_t vtmp5r = vout0r - vout2r;
      const int32_t vtmp5i = vout0i - vout2i;
      vout0r += vout2r;
      vout0i += vout2i;
      const int32_t vtmp3r = vout1r + vout3r;
      const int32_t vtmp3i = vout1i + vout3i;
      const int32_t vtmp4r = vout1i - vout3i;
      const int32_t vtmp4i = -(vout1r - vout3r);  // swap r,i and neg i
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
    }

    int32_t vout0r0 = (int32_t) data0[0];
    int32_t vout0i0 = (int32_t) data0[1];
    int32_t vout0r1 = (int32_t) data0[2];
    int32_t vout0i1 = (int32_t) data0[3];
    int32_t vout0r2 = (int32_t) data0[4];
    int32_t vout0i2 = (int32_t) data0[5];
    int32_t vout1r0 = (int32_t) data1[0];
    int32_t vout1i0 = (int32_t) data1[1];
    int32_t vout1r1 = (int32_t) data1[2];
    int32_t vout1i1 = (int32_t) data1[3];
    int32_t vout1r2 = (int32_t) data1[4];
    int32_t vout1i2 = (int32_t) data1[5];
    int32_t vout2r0 = (int32_t) data2[0];
    int32_t vout2i0 = (int32_t) data2[1];
    int32_t vout2r1 = (int32_t) data2[2];
    int32_t vout2i1 = (int32_t) data2[3];
    int32_t vout2r2 = (int32_t) data2[4];
    int32_t vout2i2 = (int32_t) data2[5];
    int32_t vout3r0 = (int32_t) data3[0];
    int32_t vout3i0 = (int32_t) data3[1];
    int32_t vout3r1 = (int32_t) data3[2];
    int32_t vout3i1 = (int32_t) data3[3];
    int32_t vout3r2 = (int32_t) data3[4];
    int32_t vout3i2 = (int32_t) data3[5];

    // Note 32767 / 4 = 8191.  Should be 8192.
    vout0r0 = math_asr_s32(vout0r0 * 8191 + 16384, 15);
    vout0i0 = math_asr_s32(vout0i0 * 8191 + 16384, 15);
    vout0r1 = math_asr_s32(vout0r1 * 8191 + 16384, 15);
    vout0i1 = math_asr_s32(vout0i1 * 8191 + 16384, 15);
    vout0r2 = math_asr_s32(vout0r2 * 8191 + 16384, 15);
    vout0i2 = math_asr_s32(vout0i2 * 8191 + 16384, 15);
    vout1r0 = math_asr_s32(vout1r0 * 8191 + 16384, 15);
    vout1i0 = math_asr_s32(vout1i0 * 8191 + 16384, 15);
    vout1r1 = math_asr_s32(vout1r1 * 8191 + 16384, 15);
    vout1i1 = math_asr_s32(vout1i1 * 8191 + 16384, 15);
    vout1r2 = math_asr_s32(vout1r2 * 8191 + 16384, 15);
    vout1i2 = math_asr_s32(vout1i2 * 8191 + 16384, 15);
    vout2r0 = math_asr_s32(vout2r0 * 8191 + 16384, 15);
    vout2i0 = math_asr_s32(vout2i0 * 8191 + 16384, 15);
    vout2r1 = math_asr_s32(vout2r1 * 8191 + 16384, 15);
    vout2i1 = math_asr_s32(vout2i1 * 8191 + 16384, 15);
    vout2r2 = math_asr_s32(vout2r2 * 8191 + 16384, 15);
    vout2i2 = math_asr_s32(vout2i2 * 8191 + 16384, 15);
    vout3r0 = math_asr_s32(vout3r0 * 8191 + 16384, 15);
    vout3i0 = math_asr_s32(vout3i0 * 8191 + 16384, 15);
    vout3r1 = math_asr_s32(vout3r1 * 8191 + 16384, 15);
    vout3i1 = math_asr_s32(vout3i1 * 8191 + 16384, 15);
    vout3r2 = math_asr_s32(vout3r2 * 8191 + 16384, 15);
    vout3i2 = math_asr_s32(vout3i2 * 8191 + 16384, 15);

    const int32_t vtmp0r0 = math_asr_s32(vout1r0 * vtw1r0 - vout1i0 * vtw1i0 + 16384, 15);
    const int32_t vtmp0i0 = math_asr_s32(vout1r0 * vtw1i0 + vout1i0 * vtw1r0 + 16384, 15);
    const int32_t vtmp0r1 = math_asr_s32(vout1r1 * vtw1r1 - vout1i1 * vtw1i1 + 16384, 15);
    const int32_t vtmp0i1 = math_asr_s32(vout1r1 * vtw1i1 + vout1i1 * vtw1r1 + 16384, 15);
    const int32_t vtmp0r2 = math_asr_s32(vout1r2 * vtw1r2 - vout1i2 * vtw1i2 + 16384, 15);
    const int32_t vtmp0i2 = math_asr_s32(vout1r2 * vtw1i2 + vout1i2 * vtw1r2 + 16384, 15);
    const int32_t vtmp1r0 = math_asr_s32(vout2r0 * vtw2r0 - vout2i0 * vtw2i0 + 16384, 15);
    const int32_t vtmp1i0 = math_asr_s32(vout2r0 * vtw2i0 + vout2i0 * vtw2r0 + 16384, 15);
    const int32_t vtmp1r1 = math_asr_s32(vout2r1 * vtw2r1 - vout2i1 * vtw2i1 + 16384, 15);
    const int32_t vtmp1i1 = math_asr_s32(vout2r1 * vtw2i1 + vout2i1 * vtw2r1 + 16384, 15);
    const int32_t vtmp1r2 = math_asr_s32(vout2r2 * vtw2r2 - vout2i2 * vtw2i2 + 16384, 15);
    const int32_t vtmp1i2 = math_asr_s32(vout2r2 * vtw2i2 + vout2i2 * vtw2r2 + 16384, 15);
    const int32_t vtmp2r0 = math_asr_s32(vout3r0 * vtw3r0 - vout3i0 * vtw3i0 + 16384, 15);
    const int32_t vtmp2i0 = math_asr_s32(vout3r0 * vtw3i0 + vout3i0 * vtw3r0 + 16384, 15);
    const int32_t vtmp2r1 = math_asr_s32(vout3r1 * vtw3r1 - vout3i1 * vtw3i1 + 16384, 15);
    const int32_t vtmp2i1 = math_asr_s32(vout3r1 * vtw3i1 + vout3i1 * vtw3r1 + 16384, 15);
    const int32_t vtmp2r2 = math_asr_s32(vout3r2 * vtw3r2 - vout3i2 * vtw3i2 + 16384, 15);
    const int32_t vtmp2i2 = math_asr_s32(vout3r2 * vtw3i2 + vout3i2 * vtw3r2 + 16384, 15);

    const int32_t vtmp5r0 = vout0r0 - vtmp1r0;
    const int32_t vtmp5i0 = vout0i0 - vtmp1i0;
    const int32_t vtmp5r1 = vout0r1 - vtmp1r1;
    const int32_t vtmp5i1 = vout0i1 - vtmp1i1;
    const int32_t vtmp5r2 = vout0r2 - vtmp1r2;
    const int32_t vtmp5i2 = vout0i2 - vtmp1i2;
    vout0r0 += vtmp1r0;
    vout0i0 += vtmp1i0;
    vout0r1 += vtmp1r1;
    vout0i1 += vtmp1i1;
    vout0r2 += vtmp1r2;
    vout0i2 += vtmp1i2;
    const int32_t vtmp3r0 = vtmp0r0 + vtmp2r0;
    const int32_t vtmp3i0 = vtmp0i0 + vtmp2i0;
    const int32_t vtmp3r1 = vtmp0r1 + vtmp2r1;
    const int32_t vtmp3i1 = vtmp0i1 + vtmp2i1;
    const int32_t vtmp3r2 = vtmp0r2 + vtmp2r2;
    const int32_t vtmp3i2 = vtmp0i2 + vtmp2i2;
    const int32_t vtmp4r0 = vtmp0i0 - vtmp2i0;
    const int32_t vtmp4i0 = -(vtmp0r0 - vtmp2r0);  // swap r,i and neg i
    const int32_t vtmp4r1 = vtmp0i1 - vtmp2i1;
    const int32_t vtmp4i1 = -(vtmp0r1 - vtmp2r1);  // swap r,i and neg i
    const int32_t vtmp4r2 = vtmp0i2 - vtmp2i2;
    const int32_t vtmp4i2 = -(vtmp0r2 - vtmp2r2);  // swap r,i and neg i
    vout2r0 = vout0r0 - vtmp3r0;
    vout2i0 = vout0i0 - vtmp3i0;
    vout2r1 = vout0r1 - vtmp3r1;
    vout2i1 = vout0i1 - vtmp3i1;
    vout2r2 = vout0r2 - vtmp3r2;
    vout2i2 = vout0i2 - vtmp3i2;
    vout0r0 += vtmp3r0;
    vout0i0 += vtmp3i0;
    vout0r1 += vtmp3r1;
    vout0i1 += vtmp3i1;
    vout0r2 += vtmp3r2;
    vout0i2 += vtmp3i2;
    vout1r0 = vtmp5r0 + vtmp4r0;
    vout1i0 = vtmp5i0 + vtmp4i0;
    vout1r1 = vtmp5r1 + vtmp4r1;
    vout1i1 = vtmp5i1 + vtmp4i1;
    vout1r2 = vtmp5r2 + vtmp4r2;
    vout1i2 = vtmp5i2 + vtmp4i2;
    vout3r0 = vtmp5r0 - vtmp4r0;
    vout3i0 = vtmp5i0 - vtmp4i0;
    vout3r1 = vtmp5r1 - vtmp4r1;
    vout3i1 = vtmp5i1 - vtmp4i1;
    vout3r2 = vtmp5r2 - vtmp4r2;
    vout3i2 = vtmp5i2 - vtmp4i2;

    data0[0] = (int16_t) vout0r0;
    data0[1] = (int16_t) vout0i0;
    data0[2] = (int16_t) vout0r1;
    data0[3] = (int16_t) vout0i1;
    data0[4] = (int16_t) vout0r2;
    data0[5] = (int16_t) vout0i2;
    data1[0] = (int16_t) vout1r0;
    data1[1] = (int16_t) vout1i0;
    data1[2] = (int16_t) vout1r1;
    data1[3] = (int16_t) vout1i1;
    data1[4] = (int16_t) vout1r2;
    data1[5] = (int16_t) vout1i2;
    data2[0] = (int16_t) vout2r0;
    data2[1] = (int16_t) vout2i0;
    data2[2] = (int16_t) vout2r1;
    data2[3] = (int16_t) vout2i1;
    data2[4] = (int16_t) vout2r2;
    data2[5] = (int16_t) vout2i2;
    data3[0] = (int16_t) vout3r0;
    data3[1] = (int16_t) vout3i0;
    data3[2] = (int16_t) vout3r1;
    data3[3] = (int16_t) vout3i1;
    data3[4] = (int16_t) vout3r2;
    data3[5] = (int16_t) vout3i2;
    data3 += 3 * 2;

  } while (--batch != 0);
}
