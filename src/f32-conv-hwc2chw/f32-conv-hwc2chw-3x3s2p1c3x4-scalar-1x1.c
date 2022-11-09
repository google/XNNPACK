// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/conv.h>
#include <xnnpack/math.h>


void xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__scalar_1x1(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_decrement = round_down_po2(input_width, 2) * 3 /* channels */ * sizeof(float);
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
      float vi00c0 = 0.0f;
      float vi00c1 = 0.0f;
      float vi00c2 = 0.0f;
      float vi10c0 = 0.0f;
      float vi10c1 = 0.0f;
      float vi10c2 = 0.0f;
      float vi20c0 = 0.0f;
      float vi20c1 = 0.0f;
      float vi20c2 = 0.0f;

      size_t iw = input_width;
      for (; iw >= 2; iw -= 2) {
        float voc0 = w[0];
        float voc1 = w[1];
        float voc2 = w[2];
        float voc3 = w[3];

        const float vk00c0x0 = w[4];
        const float vk00c0x1 = w[5];
        const float vk00c0x2 = w[6];
        const float vk00c0x3 = w[7];

        voc0 += vk00c0x0 * vi00c0;
        voc1 += vk00c0x1 * vi00c0;
        voc2 += vk00c0x2 * vi00c0;
        voc3 += vk00c0x3 * vi00c0;

        const float vk10c0x0 = w[8];
        const float vk10c0x1 = w[9];
        const float vk10c0x2 = w[10];
        const float vk10c0x3 = w[11];

        voc0 += vk10c0x0 * vi10c0;
        voc1 += vk10c0x1 * vi10c0;
        voc2 += vk10c0x2 * vi10c0;
        voc3 += vk10c0x3 * vi10c0;

        const float vk20c0x0 = w[12];
        const float vk20c0x1 = w[13];
        const float vk20c0x2 = w[14];
        const float vk20c0x3 = w[15];

        voc0 += vk20c0x0 * vi20c0;
        voc1 += vk20c0x1 * vi20c0;
        voc2 += vk20c0x2 * vi20c0;
        voc3 += vk20c0x3 * vi20c0;

        const float vk00c1x0 = w[16];
        const float vk00c1x1 = w[17];
        const float vk00c1x2 = w[18];
        const float vk00c1x3 = w[19];

        voc0 += vk00c1x0 * vi00c1;
        voc1 += vk00c1x1 * vi00c1;
        voc2 += vk00c1x2 * vi00c1;
        voc3 += vk00c1x3 * vi00c1;

        const float vk10c1x0 = w[20];
        const float vk10c1x1 = w[21];
        const float vk10c1x2 = w[22];
        const float vk10c1x3 = w[23];

        voc0 += vk10c1x0 * vi10c1;
        voc1 += vk10c1x1 * vi10c1;
        voc2 += vk10c1x2 * vi10c1;
        voc3 += vk10c1x3 * vi10c1;

        const float vk20c1x0 = w[24];
        const float vk20c1x1 = w[25];
        const float vk20c1x2 = w[26];
        const float vk20c1x3 = w[27];

        voc0 += vk20c1x0 * vi20c1;
        voc1 += vk20c1x1 * vi20c1;
        voc2 += vk20c1x2 * vi20c1;
        voc3 += vk20c1x3 * vi20c1;

        const float vk00c2x0 = w[28];
        const float vk00c2x1 = w[29];
        const float vk00c2x2 = w[30];
        const float vk00c2x3 = w[31];

        voc0 += vk00c2x0 * vi00c2;
        voc1 += vk00c2x1 * vi00c2;
        voc2 += vk00c2x2 * vi00c2;
        voc3 += vk00c2x3 * vi00c2;

        const float vk10c2x0 = w[32];
        const float vk10c2x1 = w[33];
        const float vk10c2x2 = w[34];
        const float vk10c2x3 = w[35];

        voc0 += vk10c2x0 * vi10c2;
        voc1 += vk10c2x1 * vi10c2;
        voc2 += vk10c2x2 * vi10c2;
        voc3 += vk10c2x3 * vi10c2;

        const float vk20c2x0 = w[36];
        const float vk20c2x1 = w[37];
        const float vk20c2x2 = w[38];
        const float vk20c2x3 = w[39];

        voc0 += vk20c2x0 * vi20c2;
        voc1 += vk20c2x1 * vi20c2;
        voc2 += vk20c2x2 * vi20c2;
        voc3 += vk20c2x3 * vi20c2;

        const float vk01c0x0 = w[40];
        const float vk01c0x1 = w[41];
        const float vk01c0x2 = w[42];
        const float vk01c0x3 = w[43];

        const float vi01c0 = i0[0];

        voc0 += vk01c0x0 * vi01c0;
        voc1 += vk01c0x1 * vi01c0;
        voc2 += vk01c0x2 * vi01c0;
        voc3 += vk01c0x3 * vi01c0;

        const float vk11c0x0 = w[44];
        const float vk11c0x1 = w[45];
        const float vk11c0x2 = w[46];
        const float vk11c0x3 = w[47];

        const float vi11c0 = i1[0];

        voc0 += vk11c0x0 * vi11c0;
        voc1 += vk11c0x1 * vi11c0;
        voc2 += vk11c0x2 * vi11c0;
        voc3 += vk11c0x3 * vi11c0;

        const float vk21c0x0 = w[48];
        const float vk21c0x1 = w[49];
        const float vk21c0x2 = w[50];
        const float vk21c0x3 = w[51];

        const float vi21c0 = i2[0];

        voc0 += vk21c0x0 * vi21c0;
        voc1 += vk21c0x1 * vi21c0;
        voc2 += vk21c0x2 * vi21c0;
        voc3 += vk21c0x3 * vi21c0;

        const float vk01c1x0 = w[52];
        const float vk01c1x1 = w[53];
        const float vk01c1x2 = w[54];
        const float vk01c1x3 = w[55];

        const float vi01c1 = i0[1];

        voc0 += vk01c1x0 * vi01c1;
        voc1 += vk01c1x1 * vi01c1;
        voc2 += vk01c1x2 * vi01c1;
        voc3 += vk01c1x3 * vi01c1;

        const float vk11c1x0 = w[56];
        const float vk11c1x1 = w[57];
        const float vk11c1x2 = w[58];
        const float vk11c1x3 = w[59];

        const float vi11c1 = i1[1];

        voc0 += vk11c1x0 * vi11c1;
        voc1 += vk11c1x1 * vi11c1;
        voc2 += vk11c1x2 * vi11c1;
        voc3 += vk11c1x3 * vi11c1;

        const float vk21c1x0 = w[60];
        const float vk21c1x1 = w[61];
        const float vk21c1x2 = w[62];
        const float vk21c1x3 = w[63];

        const float vi21c1 = i2[1];

        voc0 += vk21c1x0 * vi21c1;
        voc1 += vk21c1x1 * vi21c1;
        voc2 += vk21c1x2 * vi21c1;
        voc3 += vk21c1x3 * vi21c1;

        const float vk01c2x0 = w[64];
        const float vk01c2x1 = w[65];
        const float vk01c2x2 = w[66];
        const float vk01c2x3 = w[67];

        const float vi01c2 = i0[2];

        voc0 += vk01c2x0 * vi01c2;
        voc1 += vk01c2x1 * vi01c2;
        voc2 += vk01c2x2 * vi01c2;
        voc3 += vk01c2x3 * vi01c2;

        const float vk11c2x0 = w[68];
        const float vk11c2x1 = w[69];
        const float vk11c2x2 = w[70];
        const float vk11c2x3 = w[71];

        const float vi11c2 = i1[2];

        voc0 += vk11c2x0 * vi11c2;
        voc1 += vk11c2x1 * vi11c2;
        voc2 += vk11c2x2 * vi11c2;
        voc3 += vk11c2x3 * vi11c2;

        const float vk21c2x0 = w[72];
        const float vk21c2x1 = w[73];
        const float vk21c2x2 = w[74];
        const float vk21c2x3 = w[75];

        const float vi21c2 = i2[2];

        voc0 += vk21c2x0 * vi21c2;
        voc1 += vk21c2x1 * vi21c2;
        voc2 += vk21c2x2 * vi21c2;
        voc3 += vk21c2x3 * vi21c2;

        const float vk02c0x0 = w[76];
        const float vk02c0x1 = w[77];
        const float vk02c0x2 = w[78];
        const float vk02c0x3 = w[79];

        const float vi02c0 = i0[3];

        voc0 += vk02c0x0 * vi02c0;
        voc1 += vk02c0x1 * vi02c0;
        voc2 += vk02c0x2 * vi02c0;
        voc3 += vk02c0x3 * vi02c0;

        const float vk12c0x0 = w[80];
        const float vk12c0x1 = w[81];
        const float vk12c0x2 = w[82];
        const float vk12c0x3 = w[83];

        const float vi12c0 = i1[3];

        voc0 += vk12c0x0 * vi12c0;
        voc1 += vk12c0x1 * vi12c0;
        voc2 += vk12c0x2 * vi12c0;
        voc3 += vk12c0x3 * vi12c0;

        const float vk22c0x0 = w[84];
        const float vk22c0x1 = w[85];
        const float vk22c0x2 = w[86];
        const float vk22c0x3 = w[87];

        const float vi22c0 = i2[3];

        voc0 += vk22c0x0 * vi22c0;
        voc1 += vk22c0x1 * vi22c0;
        voc2 += vk22c0x2 * vi22c0;
        voc3 += vk22c0x3 * vi22c0;

        vi00c0 = vi02c0;
        vi10c0 = vi12c0;
        vi20c0 = vi22c0;

        const float vk02c1x0 = w[88];
        const float vk02c1x1 = w[89];
        const float vk02c1x2 = w[90];
        const float vk02c1x3 = w[91];

        const float vi02c1 = i0[4];

        voc0 += vk02c1x0 * vi02c1;
        voc1 += vk02c1x1 * vi02c1;
        voc2 += vk02c1x2 * vi02c1;
        voc3 += vk02c1x3 * vi02c1;

        const float vk12c1x0 = w[92];
        const float vk12c1x1 = w[93];
        const float vk12c1x2 = w[94];
        const float vk12c1x3 = w[95];

        const float vi12c1 = i1[4];

        voc0 += vk12c1x0 * vi12c1;
        voc1 += vk12c1x1 * vi12c1;
        voc2 += vk12c1x2 * vi12c1;
        voc3 += vk12c1x3 * vi12c1;

        const float vk22c1x0 = w[96];
        const float vk22c1x1 = w[97];
        const float vk22c1x2 = w[98];
        const float vk22c1x3 = w[99];

        const float vi22c1 = i2[4];

        voc0 += vk22c1x0 * vi22c1;
        voc1 += vk22c1x1 * vi22c1;
        voc2 += vk22c1x2 * vi22c1;
        voc3 += vk22c1x3 * vi22c1;

        vi00c1 = vi02c1;
        vi10c1 = vi12c1;
        vi20c1 = vi22c1;

        const float vk02c2x0 = w[100];
        const float vk02c2x1 = w[101];
        const float vk02c2x2 = w[102];
        const float vk02c2x3 = w[103];

        const float vi02c2 = i0[5];

        voc0 += vk02c2x0 * vi02c2;
        voc1 += vk02c2x1 * vi02c2;
        voc2 += vk02c2x2 * vi02c2;
        voc3 += vk02c2x3 * vi02c2;

        const float vk12c2x0 = w[104];
        const float vk12c2x1 = w[105];
        const float vk12c2x2 = w[106];
        const float vk12c2x3 = w[107];

        const float vi12c2 = i1[5];

        voc0 += vk12c2x0 * vi12c2;
        voc1 += vk12c2x1 * vi12c2;
        voc2 += vk12c2x2 * vi12c2;
        voc3 += vk12c2x3 * vi12c2;

        const float vk22c2x0 = w[108];
        const float vk22c2x1 = w[109];
        const float vk22c2x2 = w[110];
        const float vk22c2x3 = w[111];

        const float vi22c2 = i2[5];

        voc0 += vk22c2x0 * vi22c2;
        voc1 += vk22c2x1 * vi22c2;
        voc2 += vk22c2x2 * vi22c2;
        voc3 += vk22c2x3 * vi22c2;

        vi00c2 = vi02c2;
        vi10c2 = vi12c2;
        vi20c2 = vi22c2;

        voc0 = math_min_f32(voc0, voutput_max);
        voc1 = math_min_f32(voc1, voutput_max);
        voc2 = math_min_f32(voc2, voutput_max);
        voc3 = math_min_f32(voc3, voutput_max);

        voc0 = math_max_f32(voc0, voutput_min);
        voc1 = math_max_f32(voc1, voutput_min);
        voc2 = math_max_f32(voc2, voutput_min);
        voc3 = math_max_f32(voc3, voutput_min);

        *o0c0++ = voc0;
        *o0c1++ = voc1;
        *o0c2++ = voc2;
        *o0c3++ = voc3;

        i0 += 6;
        i1 += 6;
        i2 += 6;
      }
      assert(iw < 2);
      if XNN_UNLIKELY(iw != 0) {
        float voc0 = w[0];
        float voc1 = w[1];
        float voc2 = w[2];
        float voc3 = w[3];

        const float vk00c0x0 = w[4];
        const float vk00c0x1 = w[5];
        const float vk00c0x2 = w[6];
        const float vk00c0x3 = w[7];

        voc0 += vk00c0x0 * vi00c0;
        voc1 += vk00c0x1 * vi00c0;
        voc2 += vk00c0x2 * vi00c0;
        voc3 += vk00c0x3 * vi00c0;

        const float vk10c0x0 = w[8];
        const float vk10c0x1 = w[9];
        const float vk10c0x2 = w[10];
        const float vk10c0x3 = w[11];

        voc0 += vk10c0x0 * vi10c0;
        voc1 += vk10c0x1 * vi10c0;
        voc2 += vk10c0x2 * vi10c0;
        voc3 += vk10c0x3 * vi10c0;

        const float vk20c0x0 = w[12];
        const float vk20c0x1 = w[13];
        const float vk20c0x2 = w[14];
        const float vk20c0x3 = w[15];

        voc0 += vk20c0x0 * vi20c0;
        voc1 += vk20c0x1 * vi20c0;
        voc2 += vk20c0x2 * vi20c0;
        voc3 += vk20c0x3 * vi20c0;

        const float vk00c1x0 = w[16];
        const float vk00c1x1 = w[17];
        const float vk00c1x2 = w[18];
        const float vk00c1x3 = w[19];

        voc0 += vk00c1x0 * vi00c1;
        voc1 += vk00c1x1 * vi00c1;
        voc2 += vk00c1x2 * vi00c1;
        voc3 += vk00c1x3 * vi00c1;

        const float vk10c1x0 = w[20];
        const float vk10c1x1 = w[21];
        const float vk10c1x2 = w[22];
        const float vk10c1x3 = w[23];

        voc0 += vk10c1x0 * vi10c1;
        voc1 += vk10c1x1 * vi10c1;
        voc2 += vk10c1x2 * vi10c1;
        voc3 += vk10c1x3 * vi10c1;

        const float vk20c1x0 = w[24];
        const float vk20c1x1 = w[25];
        const float vk20c1x2 = w[26];
        const float vk20c1x3 = w[27];

        voc0 += vk20c1x0 * vi20c1;
        voc1 += vk20c1x1 * vi20c1;
        voc2 += vk20c1x2 * vi20c1;
        voc3 += vk20c1x3 * vi20c1;

        const float vk00c2x0 = w[28];
        const float vk00c2x1 = w[29];
        const float vk00c2x2 = w[30];
        const float vk00c2x3 = w[31];

        voc0 += vk00c2x0 * vi00c2;
        voc1 += vk00c2x1 * vi00c2;
        voc2 += vk00c2x2 * vi00c2;
        voc3 += vk00c2x3 * vi00c2;

        const float vk10c2x0 = w[32];
        const float vk10c2x1 = w[33];
        const float vk10c2x2 = w[34];
        const float vk10c2x3 = w[35];

        voc0 += vk10c2x0 * vi10c2;
        voc1 += vk10c2x1 * vi10c2;
        voc2 += vk10c2x2 * vi10c2;
        voc3 += vk10c2x3 * vi10c2;

        const float vk20c2x0 = w[36];
        const float vk20c2x1 = w[37];
        const float vk20c2x2 = w[38];
        const float vk20c2x3 = w[39];

        voc0 += vk20c2x0 * vi20c2;
        voc1 += vk20c2x1 * vi20c2;
        voc2 += vk20c2x2 * vi20c2;
        voc3 += vk20c2x3 * vi20c2;

        const float vk01c0x0 = w[40];
        const float vk01c0x1 = w[41];
        const float vk01c0x2 = w[42];
        const float vk01c0x3 = w[43];

        const float vi01c0 = i0[0];

        voc0 += vk01c0x0 * vi01c0;
        voc1 += vk01c0x1 * vi01c0;
        voc2 += vk01c0x2 * vi01c0;
        voc3 += vk01c0x3 * vi01c0;

        const float vk11c0x0 = w[44];
        const float vk11c0x1 = w[45];
        const float vk11c0x2 = w[46];
        const float vk11c0x3 = w[47];

        const float vi11c0 = i1[0];

        voc0 += vk11c0x0 * vi11c0;
        voc1 += vk11c0x1 * vi11c0;
        voc2 += vk11c0x2 * vi11c0;
        voc3 += vk11c0x3 * vi11c0;

        const float vk21c0x0 = w[48];
        const float vk21c0x1 = w[49];
        const float vk21c0x2 = w[50];
        const float vk21c0x3 = w[51];

        const float vi21c0 = i2[0];

        voc0 += vk21c0x0 * vi21c0;
        voc1 += vk21c0x1 * vi21c0;
        voc2 += vk21c0x2 * vi21c0;
        voc3 += vk21c0x3 * vi21c0;

        const float vk01c1x0 = w[52];
        const float vk01c1x1 = w[53];
        const float vk01c1x2 = w[54];
        const float vk01c1x3 = w[55];

        const float vi01c1 = i0[1];

        voc0 += vk01c1x0 * vi01c1;
        voc1 += vk01c1x1 * vi01c1;
        voc2 += vk01c1x2 * vi01c1;
        voc3 += vk01c1x3 * vi01c1;

        const float vk11c1x0 = w[56];
        const float vk11c1x1 = w[57];
        const float vk11c1x2 = w[58];
        const float vk11c1x3 = w[59];

        const float vi11c1 = i1[1];

        voc0 += vk11c1x0 * vi11c1;
        voc1 += vk11c1x1 * vi11c1;
        voc2 += vk11c1x2 * vi11c1;
        voc3 += vk11c1x3 * vi11c1;

        const float vk21c1x0 = w[60];
        const float vk21c1x1 = w[61];
        const float vk21c1x2 = w[62];
        const float vk21c1x3 = w[63];

        const float vi21c1 = i2[1];

        voc0 += vk21c1x0 * vi21c1;
        voc1 += vk21c1x1 * vi21c1;
        voc2 += vk21c1x2 * vi21c1;
        voc3 += vk21c1x3 * vi21c1;

        const float vk01c2x0 = w[64];
        const float vk01c2x1 = w[65];
        const float vk01c2x2 = w[66];
        const float vk01c2x3 = w[67];

        const float vi01c2 = i0[2];

        voc0 += vk01c2x0 * vi01c2;
        voc1 += vk01c2x1 * vi01c2;
        voc2 += vk01c2x2 * vi01c2;
        voc3 += vk01c2x3 * vi01c2;

        const float vk11c2x0 = w[68];
        const float vk11c2x1 = w[69];
        const float vk11c2x2 = w[70];
        const float vk11c2x3 = w[71];

        const float vi11c2 = i1[2];

        voc0 += vk11c2x0 * vi11c2;
        voc1 += vk11c2x1 * vi11c2;
        voc2 += vk11c2x2 * vi11c2;
        voc3 += vk11c2x3 * vi11c2;

        const float vk21c2x0 = w[72];
        const float vk21c2x1 = w[73];
        const float vk21c2x2 = w[74];
        const float vk21c2x3 = w[75];

        const float vi21c2 = i2[2];

        voc0 += vk21c2x0 * vi21c2;
        voc1 += vk21c2x1 * vi21c2;
        voc2 += vk21c2x2 * vi21c2;
        voc3 += vk21c2x3 * vi21c2;

        voc0 = math_min_f32(voc0, voutput_max);
        voc1 = math_min_f32(voc1, voutput_max);
        voc2 = math_min_f32(voc2, voutput_max);
        voc3 = math_min_f32(voc3, voutput_max);

        voc0 = math_max_f32(voc0, voutput_min);
        voc1 = math_max_f32(voc1, voutput_min);
        voc2 = math_max_f32(voc2, voutput_min);
        voc3 = math_max_f32(voc3, voutput_min);

        *o0c0++ = voc0;
        *o0c1++ = voc1;
        *o0c2++ = voc2;
        *o0c3++ = voc3;
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (float*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (float*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (float*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (float*) ((uintptr_t) o0c3 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_decrement);
      i1 = (const float*) ((uintptr_t) i1 - input_width_decrement);
      i2 = (const float*) ((uintptr_t) i2 - input_width_decrement);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next row
    output0 = (float*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next row
    i0 = i2;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  }
}
