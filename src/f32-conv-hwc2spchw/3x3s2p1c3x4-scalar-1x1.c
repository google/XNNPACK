// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/conv.h>
#include <xnnpack/math.h>


void xnn_f32_conv_hwc2spchw_ukernel_3x3s2p1c3x4__scalar_1x1(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_increment = round_down_po2(input_width, 2) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  float* output0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const float voutput_max = params->scalar.max;
  const float voutput_min = params->scalar.min;

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 1) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }

    const float* w = weights;
    size_t c = output_channels;
    float* o0c0 = output0;
    float* o0c1 = (float*) ((uintptr_t) o0c0 + output_channel_stride);
    float* o0c2 = (float*) ((uintptr_t) o0c1 + output_channel_stride);
    float* o0c3 = (float*) ((uintptr_t) o0c2 + output_channel_stride);
    do {
      if XNN_UNPREDICTABLE(c < 2) {
        o0c1 = o0c0;
      }
      if XNN_UNPREDICTABLE(c <= 2) {
        o0c2 = o0c1;
      }
      if XNN_UNPREDICTABLE(c < 4) {
        o0c3 = o0c2;
      }

      // Left edge padding
      float vr0c0 = 0.f;
      float vr0c1 = 0.f;
      float vr0c2 = 0.f;
      float vr1c0 = 0.f;
      float vr1c1 = 0.f;
      float vr1c2 = 0.f;
      float vr2c0 = 0.f;
      float vr2c1 = 0.f;
      float vr2c2 = 0.f;

      size_t iw = input_width;
      for (; iw >= 2; iw -= 2) {
        // start with biases
        float vc0_out = w[0];
        float vc1_out = w[1];
        float vc2_out = w[2];
        float vc3_out = w[3];

        const float vk00ic0oc0 = w[4];
        const float vk00ic0oc1 = w[5];
        const float vk00ic0oc2 = w[6];
        const float vk00ic0oc3 = w[7];

        vc0_out += vk00ic0oc0 * vr0c0;
        vc1_out += vk00ic0oc1 * vr0c0;
        vc2_out += vk00ic0oc2 * vr0c0;
        vc3_out += vk00ic0oc3 * vr0c0;

        const float vk10ic0oc0 = w[8];
        const float vk10ic0oc1 = w[9];
        const float vk10ic0oc2 = w[10];
        const float vk10ic0oc3 = w[11];

        vc0_out += vk10ic0oc0 * vr1c0;
        vc1_out += vk10ic0oc1 * vr1c0;
        vc2_out += vk10ic0oc2 * vr1c0;
        vc3_out += vk10ic0oc3 * vr1c0;

        const float vk20ic0oc0 = w[12];
        const float vk20ic0oc1 = w[13];
        const float vk20ic0oc2 = w[14];
        const float vk20ic0oc3 = w[15];

        vc0_out += vk20ic0oc0 * vr2c0;
        vc1_out += vk20ic0oc1 * vr2c0;
        vc2_out += vk20ic0oc2 * vr2c0;
        vc3_out += vk20ic0oc3 * vr2c0;

        const float vk00ic1oc0 = w[16];
        const float vk00ic1oc1 = w[17];
        const float vk00ic1oc2 = w[18];
        const float vk00ic1oc3 = w[19];


        vc0_out += vk00ic1oc0 * vr0c1;
        vc1_out += vk00ic1oc1 * vr0c1;
        vc2_out += vk00ic1oc2 * vr0c1;
        vc3_out += vk00ic1oc3 * vr0c1;

        const float vk10ic1oc0 = w[20];
        const float vk10ic1oc1 = w[21];
        const float vk10ic1oc2 = w[22];
        const float vk10ic1oc3 = w[23];

        vc0_out += vk10ic1oc0 * vr1c1;
        vc1_out += vk10ic1oc1 * vr1c1;
        vc2_out += vk10ic1oc2 * vr1c1;
        vc3_out += vk10ic1oc3 * vr1c1;

        const float vk20ic1oc0 = w[24];
        const float vk20ic1oc1 = w[25];
        const float vk20ic1oc2 = w[26];
        const float vk20ic1oc3 = w[27];

        vc0_out += vk20ic1oc0 * vr2c1;
        vc1_out += vk20ic1oc1 * vr2c1;
        vc2_out += vk20ic1oc2 * vr2c1;
        vc3_out += vk20ic1oc3 * vr2c1;

        const float vk00ic2oc0 = w[28];
        const float vk00ic2oc1 = w[29];
        const float vk00ic2oc2 = w[30];
        const float vk00ic2oc3 = w[31];

        vc0_out += vk00ic2oc0 * vr0c2;
        vc1_out += vk00ic2oc1 * vr0c2;
        vc2_out += vk00ic2oc2 * vr0c2;
        vc3_out += vk00ic2oc3 * vr0c2;

        const float vk10ic2oc0 = w[32];
        const float vk10ic2oc1 = w[33];
        const float vk10ic2oc2 = w[34];
        const float vk10ic2oc3 = w[35];

        vc0_out += vk10ic2oc0 * vr1c2;
        vc1_out += vk10ic2oc1 * vr1c2;
        vc2_out += vk10ic2oc2 * vr1c2;
        vc3_out += vk10ic2oc3 * vr1c2;

        const float vk20ic2oc0 = w[36];
        const float vk20ic2oc1 = w[37];
        const float vk20ic2oc2 = w[38];
        const float vk20ic2oc3 = w[39];

        vc0_out += vk20ic2oc0 * vr2c2;
        vc1_out += vk20ic2oc1 * vr2c2;
        vc2_out += vk20ic2oc2 * vr2c2;
        vc3_out += vk20ic2oc3 * vr2c2;

        const float vk01ic0oc0 = w[40];
        const float vk01ic0oc1 = w[41];
        const float vk01ic0oc2 = w[42];
        const float vk01ic0oc3 = w[43];

        const float i00 = i0[0];

        vc0_out += vk01ic0oc0 * i00;
        vc1_out += vk01ic0oc1 * i00;
        vc2_out += vk01ic0oc2 * i00;
        vc3_out += vk01ic0oc3 * i00;

        const float vk11ic0oc0 = w[44];
        const float vk11ic0oc1 = w[45];
        const float vk11ic0oc2 = w[46];
        const float vk11ic0oc3 = w[47];

        const float i10 = i1[0];

        vc0_out += vk11ic0oc0 * i10;
        vc1_out += vk11ic0oc1 * i10;
        vc2_out += vk11ic0oc2 * i10;
        vc3_out += vk11ic0oc3 * i10;

        const float vk21ic0oc0 = w[48];
        const float vk21ic0oc1 = w[49];
        const float vk21ic0oc2 = w[50];
        const float vk21ic0oc3 = w[51];

        const float i20 = i2[0];

        vc0_out += vk21ic0oc0 * i20;
        vc1_out += vk21ic0oc1 * i20;
        vc2_out += vk21ic0oc2 * i20;
        vc3_out += vk21ic0oc3 * i20;

        const float vk01ic1oc0 = w[52];
        const float vk01ic1oc1 = w[53];
        const float vk01ic1oc2 = w[54];
        const float vk01ic1oc3 = w[55];

        const float i01 = i0[1];

        vc0_out += vk01ic1oc0 * i01;
        vc1_out += vk01ic1oc1 * i01;
        vc2_out += vk01ic1oc2 * i01;
        vc3_out += vk01ic1oc3 * i01;

        const float vk11ic1oc0 = w[56];
        const float vk11ic1oc1 = w[57];
        const float vk11ic1oc2 = w[58];
        const float vk11ic1oc3 = w[59];

        const float i11 = i1[1];

        vc0_out += vk11ic1oc0 * i11;
        vc1_out += vk11ic1oc1 * i11;
        vc2_out += vk11ic1oc2 * i11;
        vc3_out += vk11ic1oc3 * i11;

        const float vk21ic1oc0 = w[60];
        const float vk21ic1oc1 = w[61];
        const float vk21ic1oc2 = w[62];
        const float vk21ic1oc3 = w[63];

        const float i21 = i2[1];

        vc0_out += vk21ic1oc0 * i21;
        vc1_out += vk21ic1oc1 * i21;
        vc2_out += vk21ic1oc2 * i21;
        vc3_out += vk21ic1oc3 * i21;

        const float vk01ic2oc0 = w[64];
        const float vk01ic2oc1 = w[65];
        const float vk01ic2oc2 = w[66];
        const float vk01ic2oc3 = w[67];

        const float i02 = i0[2];

        vc0_out += vk01ic2oc0 * i02;
        vc1_out += vk01ic2oc1 * i02;
        vc2_out += vk01ic2oc2 * i02;
        vc3_out += vk01ic2oc3 * i02;

        const float vk11ic2oc0 = w[68];
        const float vk11ic2oc1 = w[69];
        const float vk11ic2oc2 = w[70];
        const float vk11ic2oc3 = w[71];

        const float i12 = i1[2];

        vc0_out += vk11ic2oc0 * i12;
        vc1_out += vk11ic2oc1 * i12;
        vc2_out += vk11ic2oc2 * i12;
        vc3_out += vk11ic2oc3 * i12;

        const float vk21ic2oc0 = w[72];
        const float vk21ic2oc1 = w[73];
        const float vk21ic2oc2 = w[74];
        const float vk21ic2oc3 = w[75];

        const float i22 = i2[2];

        vc0_out += vk21ic2oc0 * i22;
        vc1_out += vk21ic2oc1 * i22;
        vc2_out += vk21ic2oc2 * i22;
        vc3_out += vk21ic2oc3 * i22;

        const float vk02ic0oc0 = w[76];
        const float vk02ic0oc1 = w[77];
        const float vk02ic0oc2 = w[78];
        const float vk02ic0oc3 = w[79];

        const float i03 = i0[3];

        vc0_out += vk02ic0oc0 * i03;
        vc1_out += vk02ic0oc1 * i03;
        vc2_out += vk02ic0oc2 * i03;
        vc3_out += vk02ic0oc3 * i03;

        const float vk12ic0oc0 = w[80];
        const float vk12ic0oc1 = w[81];
        const float vk12ic0oc2 = w[82];
        const float vk12ic0oc3 = w[83];

        const float i13 = i1[3];

        vc0_out += vk12ic0oc0 * i13;
        vc1_out += vk12ic0oc1 * i13;
        vc2_out += vk12ic0oc2 * i13;
        vc3_out += vk12ic0oc3 * i13;

        const float vk22ic0oc0 = w[84];
        const float vk22ic0oc1 = w[85];
        const float vk22ic0oc2 = w[86];
        const float vk22ic0oc3 = w[87];

        const float i23 = i2[3];

        vc0_out += vk22ic0oc0 * i23;
        vc1_out += vk22ic0oc1 * i23;
        vc2_out += vk22ic0oc2 * i23;
        vc3_out += vk22ic0oc3 * i23;

        vr0c0 = i03;
        vr1c0 = i13;
        vr2c0 = i23;

        const float vk02ic1oc0 = w[88];
        const float vk02ic1oc1 = w[89];
        const float vk02ic1oc2 = w[90];
        const float vk02ic1oc3 = w[91];

        const float i04 = i0[4];

        vc0_out += vk02ic1oc0 * i04;
        vc1_out += vk02ic1oc1 * i04;
        vc2_out += vk02ic1oc2 * i04;
        vc3_out += vk02ic1oc3 * i04;

        const float vk12ic1oc0 = w[92];
        const float vk12ic1oc1 = w[93];
        const float vk12ic1oc2 = w[94];
        const float vk12ic1oc3 = w[95];

        const float i14 = i1[4];

        vc0_out += vk12ic1oc0 * i14;
        vc1_out += vk12ic1oc1 * i14;
        vc2_out += vk12ic1oc2 * i14;
        vc3_out += vk12ic1oc3 * i14;

        const float vk22ic1oc0 = w[96];
        const float vk22ic1oc1 = w[97];
        const float vk22ic1oc2 = w[98];
        const float vk22ic1oc3 = w[99];

        const float i24 = i2[4];

        vc0_out += vk22ic1oc0 * i24;
        vc1_out += vk22ic1oc1 * i24;
        vc2_out += vk22ic1oc2 * i24;
        vc3_out += vk22ic1oc3 * i24;

        vr0c1 = i04;
        vr1c1 = i14;
        vr2c1 = i24;

        const float vk02ic2oc0 = w[100];
        const float vk02ic2oc1 = w[101];
        const float vk02ic2oc2 = w[102];
        const float vk02ic2oc3 = w[103];

        const float i05 = i0[5];

        vc0_out += vk02ic2oc0 * i05;
        vc1_out += vk02ic2oc1 * i05;
        vc2_out += vk02ic2oc2 * i05;
        vc3_out += vk02ic2oc3 * i05;

        const float vk12ic2oc0 = w[104];
        const float vk12ic2oc1 = w[105];
        const float vk12ic2oc2 = w[106];
        const float vk12ic2oc3 = w[107];

        const float i15 = i1[5];

        vc0_out += vk12ic2oc0 * i15;
        vc1_out += vk12ic2oc1 * i15;
        vc2_out += vk12ic2oc2 * i15;
        vc3_out += vk12ic2oc3 * i15;

        const float vk22ic2oc0 = w[108];
        const float vk22ic2oc1 = w[109];
        const float vk22ic2oc2 = w[110];
        const float vk22ic2oc3 = w[111];

        const float i25 = i2[5];

        vc0_out += vk22ic2oc0 * i25;
        vc1_out += vk22ic2oc1 * i25;
        vc2_out += vk22ic2oc2 * i25;
        vc3_out += vk22ic2oc3 * i25;

        vr0c2 = i05;
        vr1c2 = i15;
        vr2c2 = i25;

        vc0_out = math_min_f32(vc0_out, voutput_max);
        vc0_out = math_max_f32(vc0_out, voutput_min);
        vc1_out = math_min_f32(vc1_out, voutput_max);
        vc1_out = math_max_f32(vc1_out, voutput_min);
        vc2_out = math_min_f32(vc2_out, voutput_max);
        vc2_out = math_max_f32(vc2_out, voutput_min);
        vc3_out = math_min_f32(vc3_out, voutput_max);
        vc3_out = math_max_f32(vc3_out, voutput_min);

        *o0c0 = vc0_out; o0c0 += 1;
        *o0c1 = vc1_out; o0c1 += 1;
        *o0c2 = vc2_out; o0c2 += 1;
        *o0c3 = vc3_out; o0c3 += 1;

        i0 += 6;
        i1 += 6;
        i2 += 6;
      }
      assert(iw < 2);
      if XNN_UNLIKELY(iw != 0) {
        // start with biases
        float vc0_out = w[0];
        float vc1_out = w[1];
        float vc2_out = w[2];
        float vc3_out = w[3];

        const float vk00ic0oc0 = w[4];
        const float vk00ic0oc1 = w[5];
        const float vk00ic0oc2 = w[6];
        const float vk00ic0oc3 = w[7];

        vc0_out += vk00ic0oc0 * vr0c0;
        vc1_out += vk00ic0oc1 * vr0c0;
        vc2_out += vk00ic0oc2 * vr0c0;
        vc3_out += vk00ic0oc3 * vr0c0;

        const float vk10ic0oc0 = w[8];
        const float vk10ic0oc1 = w[9];
        const float vk10ic0oc2 = w[10];
        const float vk10ic0oc3 = w[11];

        vc0_out += vk10ic0oc0 * vr1c0;
        vc1_out += vk10ic0oc1 * vr1c0;
        vc2_out += vk10ic0oc2 * vr1c0;
        vc3_out += vk10ic0oc3 * vr1c0;

        const float vk20ic0oc0 = w[12];
        const float vk20ic0oc1 = w[13];
        const float vk20ic0oc2 = w[14];
        const float vk20ic0oc3 = w[15];

        vc0_out += vk20ic0oc0 * vr2c0;
        vc1_out += vk20ic0oc1 * vr2c0;
        vc2_out += vk20ic0oc2 * vr2c0;
        vc3_out += vk20ic0oc3 * vr2c0;

        const float vk00ic1oc0 = w[16];
        const float vk00ic1oc1 = w[17];
        const float vk00ic1oc2 = w[18];
        const float vk00ic1oc3 = w[19];


        vc0_out += vk00ic1oc0 * vr0c1;
        vc1_out += vk00ic1oc1 * vr0c1;
        vc2_out += vk00ic1oc2 * vr0c1;
        vc3_out += vk00ic1oc3 * vr0c1;

        const float vk10ic1oc0 = w[20];
        const float vk10ic1oc1 = w[21];
        const float vk10ic1oc2 = w[22];
        const float vk10ic1oc3 = w[23];

        vc0_out += vk10ic1oc0 * vr1c1;
        vc1_out += vk10ic1oc1 * vr1c1;
        vc2_out += vk10ic1oc2 * vr1c1;
        vc3_out += vk10ic1oc3 * vr1c1;

        const float vk20ic1oc0 = w[24];
        const float vk20ic1oc1 = w[25];
        const float vk20ic1oc2 = w[26];
        const float vk20ic1oc3 = w[27];

        vc0_out += vk20ic1oc0 * vr2c1;
        vc1_out += vk20ic1oc1 * vr2c1;
        vc2_out += vk20ic1oc2 * vr2c1;
        vc3_out += vk20ic1oc3 * vr2c1;

        const float vk00ic2oc0 = w[28];
        const float vk00ic2oc1 = w[29];
        const float vk00ic2oc2 = w[30];
        const float vk00ic2oc3 = w[31];

        vc0_out += vk00ic2oc0 * vr0c2;
        vc1_out += vk00ic2oc1 * vr0c2;
        vc2_out += vk00ic2oc2 * vr0c2;
        vc3_out += vk00ic2oc3 * vr0c2;

        const float vk10ic2oc0 = w[32];
        const float vk10ic2oc1 = w[33];
        const float vk10ic2oc2 = w[34];
        const float vk10ic2oc3 = w[35];

        vc0_out += vk10ic2oc0 * vr1c2;
        vc1_out += vk10ic2oc1 * vr1c2;
        vc2_out += vk10ic2oc2 * vr1c2;
        vc3_out += vk10ic2oc3 * vr1c2;

        const float vk20ic2oc0 = w[36];
        const float vk20ic2oc1 = w[37];
        const float vk20ic2oc2 = w[38];
        const float vk20ic2oc3 = w[39];

        vc0_out += vk20ic2oc0 * vr2c2;
        vc1_out += vk20ic2oc1 * vr2c2;
        vc2_out += vk20ic2oc2 * vr2c2;
        vc3_out += vk20ic2oc3 * vr2c2;

        const float vk01ic0oc0 = w[40];
        const float vk01ic0oc1 = w[41];
        const float vk01ic0oc2 = w[42];
        const float vk01ic0oc3 = w[43];

        const float i00 = i0[0];

        vc0_out += vk01ic0oc0 * i00;
        vc1_out += vk01ic0oc1 * i00;
        vc2_out += vk01ic0oc2 * i00;
        vc3_out += vk01ic0oc3 * i00;

        const float vk11ic0oc0 = w[44];
        const float vk11ic0oc1 = w[45];
        const float vk11ic0oc2 = w[46];
        const float vk11ic0oc3 = w[47];

        const float i10 = i1[0];

        vc0_out += vk11ic0oc0 * i10;
        vc1_out += vk11ic0oc1 * i10;
        vc2_out += vk11ic0oc2 * i10;
        vc3_out += vk11ic0oc3 * i10;

        const float vk21ic0oc0 = w[48];
        const float vk21ic0oc1 = w[49];
        const float vk21ic0oc2 = w[50];
        const float vk21ic0oc3 = w[51];

        const float i20 = i2[0];

        vc0_out += vk21ic0oc0 * i20;
        vc1_out += vk21ic0oc1 * i20;
        vc2_out += vk21ic0oc2 * i20;
        vc3_out += vk21ic0oc3 * i20;

        const float vk01ic1oc0 = w[52];
        const float vk01ic1oc1 = w[53];
        const float vk01ic1oc2 = w[54];
        const float vk01ic1oc3 = w[55];

        const float i01 = i0[1];

        vc0_out += vk01ic1oc0 * i01;
        vc1_out += vk01ic1oc1 * i01;
        vc2_out += vk01ic1oc2 * i01;
        vc3_out += vk01ic1oc3 * i01;

        const float vk11ic1oc0 = w[56];
        const float vk11ic1oc1 = w[57];
        const float vk11ic1oc2 = w[58];
        const float vk11ic1oc3 = w[59];

        const float i11 = i1[1];

        vc0_out += vk11ic1oc0 * i11;
        vc1_out += vk11ic1oc1 * i11;
        vc2_out += vk11ic1oc2 * i11;
        vc3_out += vk11ic1oc3 * i11;

        const float vk21ic1oc0 = w[60];
        const float vk21ic1oc1 = w[61];
        const float vk21ic1oc2 = w[62];
        const float vk21ic1oc3 = w[63];

        const float i21 = i2[1];

        vc0_out += vk21ic1oc0 * i21;
        vc1_out += vk21ic1oc1 * i21;
        vc2_out += vk21ic1oc2 * i21;
        vc3_out += vk21ic1oc3 * i21;

        const float vk01ic2oc0 = w[64];
        const float vk01ic2oc1 = w[65];
        const float vk01ic2oc2 = w[66];
        const float vk01ic2oc3 = w[67];

        const float i02 = i0[2];

        vc0_out += vk01ic2oc0 * i02;
        vc1_out += vk01ic2oc1 * i02;
        vc2_out += vk01ic2oc2 * i02;
        vc3_out += vk01ic2oc3 * i02;

        const float vk11ic2oc0 = w[68];
        const float vk11ic2oc1 = w[69];
        const float vk11ic2oc2 = w[70];
        const float vk11ic2oc3 = w[71];

        const float i12 = i1[2];

        vc0_out += vk11ic2oc0 * i12;
        vc1_out += vk11ic2oc1 * i12;
        vc2_out += vk11ic2oc2 * i12;
        vc3_out += vk11ic2oc3 * i12;

        const float vk21ic2oc0 = w[72];
        const float vk21ic2oc1 = w[73];
        const float vk21ic2oc2 = w[74];
        const float vk21ic2oc3 = w[75];

        const float i22 = i2[2];

        vc0_out += vk21ic2oc0 * i22;
        vc1_out += vk21ic2oc1 * i22;
        vc2_out += vk21ic2oc2 * i22;
        vc3_out += vk21ic2oc3 * i22;

        vc0_out = math_min_f32(vc0_out, voutput_max);
        vc0_out = math_max_f32(vc0_out, voutput_min);
        vc1_out = math_min_f32(vc1_out, voutput_max);
        vc1_out = math_max_f32(vc1_out, voutput_min);
        vc2_out = math_min_f32(vc2_out, voutput_max);
        vc2_out = math_max_f32(vc2_out, voutput_min);
        vc3_out = math_min_f32(vc3_out, voutput_max);
        vc3_out = math_max_f32(vc3_out, voutput_min);

        *o0c0 = vc0_out; o0c0 += 1;
        *o0c1 = vc1_out; o0c1 += 1;
        *o0c2 = vc2_out; o0c2 += 1;
        *o0c3 = vc3_out; o0c3 += 1;
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (float*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (float*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (float*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (float*) ((uintptr_t) o0c3 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_increment);
      i1 = (const float*) ((uintptr_t) i1 - input_width_increment);
      i2 = (const float*) ((uintptr_t) i2 - input_width_increment);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next two rows
    output0 = (float*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next four rows
    i0 = i2;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  }
}
