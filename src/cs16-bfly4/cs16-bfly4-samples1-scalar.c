// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/math.h>
#include <xnnpack/fft.h>


void xnn_cs16_bfly4_samples1_ukernel__scalar(
    size_t batch,
    size_t samples,
    int16_t* data,
    const int16_t* twiddle,
    size_t stride)
{
  assert(samples == sizeof(int16_t) * 2);
  assert(data != NULL);
  assert(stride != 0);
  assert(twiddle != NULL);

  do {
    int32_t vout0r = (int32_t) data[0];
    int32_t vout0i = (int32_t) data[1];
    int32_t vout1r = (int32_t) data[2];
    int32_t vout1i = (int32_t) data[3];
    int32_t vout2r = (int32_t) data[4];
    int32_t vout2i = (int32_t) data[5];
    int32_t vout3r = (int32_t) data[6];
    int32_t vout3i = (int32_t) data[7];

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

    data[0] = (int16_t) vout0r;
    data[1] = (int16_t) vout0i;
    data[2] = (int16_t) vout1r;
    data[3] = (int16_t) vout1i;
    data[4] = (int16_t) vout2r;
    data[5] = (int16_t) vout2i;
    data[6] = (int16_t) vout3r;
    data[7] = (int16_t) vout3i;
    data += 8;
  } while(--batch != 0);
}
