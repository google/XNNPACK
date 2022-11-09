// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/conv.h>
#include <xnnpack/math.h>


void xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__wasmsimd_2x2(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_increment = round_down_po2(input_width, 4) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  float* output0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);
  float* output1 = (float*) ((uintptr_t) output0 + output_height_stride);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const v128_t vmax = wasm_v128_load64_splat(params->wasmsimd.max);
  const v128_t vmin = wasm_v128_load64_splat(params->wasmsimd.min);

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    const size_t input_y4 = input_y2 + 2;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 > input_height) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 >= input_height) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(output_y + 2 > output_y_end) {
      output1 = output0;
    }

    const float* w = weights;
    size_t c = output_channels;
    float* o0c0 = output0;
    float* o1c0 = output1;
    float* o0c1 = (float*) ((uintptr_t) o0c0 + output_channel_stride);
    float* o1c1 = (float*) ((uintptr_t) o1c0 + output_channel_stride);
    float* o0c2 = (float*) ((uintptr_t) o0c1 + output_channel_stride);
    float* o1c2 = (float*) ((uintptr_t) o1c1 + output_channel_stride);
    float* o0c3 = (float*) ((uintptr_t) o0c2 + output_channel_stride);
    float* o1c3 = (float*) ((uintptr_t) o1c2 + output_channel_stride);
    do {
      if XNN_UNPREDICTABLE(c < 2) {
        o0c1 = o0c0;
        o1c1 = o1c0;
      }
      if XNN_UNPREDICTABLE(c <= 2) {
        o0c2 = o0c1;
        o1c2 = o1c1;
      }
      if XNN_UNPREDICTABLE(c < 4) {
        o0c3 = o0c2;
        o1c3 = o1c2;
      }

      // viMx0 = ( iM0c2, iM0c1, iM0c0, --- )
      v128_t vi0x0 = wasm_f32x4_const_splat(0.0f);
      v128_t vi1x0 = wasm_f32x4_const_splat(0.0f);
      v128_t vi2x0 = wasm_f32x4_const_splat(0.0f);
      v128_t vi3x0 = wasm_f32x4_const_splat(0.0f);
      v128_t vi4x0 = wasm_f32x4_const_splat(0.0f);

      size_t iw = input_width;
      for (; iw >= 4; iw -= 4) {
        v128_t vo0x0 = wasm_v128_load(w);
        v128_t vo1x0 = vo0x0;
        v128_t vo0x1 = vo0x0;
        v128_t vo1x1 = vo0x0;

        const v128_t vk00c0 = wasm_v128_load(w + 4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        const v128_t vi0x1 = wasm_v128_load(i0); i0 += 4;
        const v128_t vi1x1 = wasm_v128_load(i1); i1 += 4;
        const v128_t vi2x1 = wasm_v128_load(i2); i2 += 4;
        const v128_t vi3x1 = wasm_v128_load(i3); i3 += 4;
        const v128_t vi4x1 = wasm_v128_load(i4); i4 += 4;

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk00c0, wasm_v32x4_shuffle(vi0x0, vi0x0, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk00c0, wasm_v32x4_shuffle(vi2x0, vi2x0, 1, 1, 1, 1)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk00c0, wasm_v32x4_shuffle(vi0x1, vi0x1, 3, 3, 3, 3)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk00c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 3, 3, 3, 3)));

        const v128_t vk10c0 = wasm_v128_load(w + 8);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk10c0, wasm_v32x4_shuffle(vi1x0, vi1x0, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk10c0, wasm_v32x4_shuffle(vi3x0, vi3x0, 1, 1, 1, 1)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk10c0, wasm_v32x4_shuffle(vi1x1, vi1x1, 3, 3, 3, 3)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk10c0, wasm_v32x4_shuffle(vi3x1, vi3x1, 3, 3, 3, 3)));

        const v128_t vk20c0 = wasm_v128_load(w + 12);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk20c0, wasm_v32x4_shuffle(vi2x0, vi2x0, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk20c0, wasm_v32x4_shuffle(vi4x0, vi4x0, 1, 1, 1, 1)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk20c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 3, 3, 3, 3)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk20c0, wasm_v32x4_shuffle(vi4x1, vi4x1, 3, 3, 3, 3)));

        const v128_t vk00c1 = wasm_v128_load(w + 16);

        // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
        const v128_t vi0x2 = wasm_v128_load(i0); i0 += 4;
        const v128_t vi1x2 = wasm_v128_load(i1); i1 += 4;
        const v128_t vi2x2 = wasm_v128_load(i2); i2 += 4;
        const v128_t vi3x2 = wasm_v128_load(i3); i3 += 4;
        const v128_t vi4x2 = wasm_v128_load(i4); i4 += 4;

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk00c1, wasm_v32x4_shuffle(vi0x0, vi0x0, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk00c1, wasm_v32x4_shuffle(vi2x0, vi2x0, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk00c1, wasm_v32x4_shuffle(vi0x2, vi0x2, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk00c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 0, 0, 0, 0)));

        const v128_t vk10c1 = wasm_v128_load(w + 20);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk10c1, wasm_v32x4_shuffle(vi1x0, vi1x0, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk10c1, wasm_v32x4_shuffle(vi3x0, vi3x0, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk10c1, wasm_v32x4_shuffle(vi1x2, vi1x2, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk10c1, wasm_v32x4_shuffle(vi3x2, vi3x2, 0, 0, 0, 0)));

        const v128_t vk20c1 = wasm_v128_load(w + 24);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk20c1, wasm_v32x4_shuffle(vi2x0, vi2x0, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk20c1, wasm_v32x4_shuffle(vi4x0, vi4x0, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk20c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk20c1, wasm_v32x4_shuffle(vi4x2, vi4x2, 0, 0, 0, 0)));

        const v128_t vk00c2 = wasm_v128_load(w + 28);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk00c2, wasm_v32x4_shuffle(vi0x0, vi0x0, 3, 3, 3, 3)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk00c2, wasm_v32x4_shuffle(vi2x0, vi2x0, 3, 3, 3, 3)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk00c2, wasm_v32x4_shuffle(vi0x2, vi0x2, 1, 1, 1, 1)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk00c2, wasm_v32x4_shuffle(vi2x2, vi2x2, 1, 1, 1, 1)));

        const v128_t vk10c2 = wasm_v128_load(w + 32);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk10c2, wasm_v32x4_shuffle(vi1x0, vi1x0, 3, 3, 3, 3)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk10c2, wasm_v32x4_shuffle(vi3x0, vi3x0, 3, 3, 3, 3)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk10c2, wasm_v32x4_shuffle(vi1x2, vi1x2, 1, 1, 1, 1)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk10c2, wasm_v32x4_shuffle(vi3x2, vi3x2, 1, 1, 1, 1)));

        const v128_t vk20c2 = wasm_v128_load(w + 36);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk20c2, wasm_v32x4_shuffle(vi2x0, vi2x0, 3, 3, 3, 3)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk20c2, wasm_v32x4_shuffle(vi4x0, vi4x0, 3, 3, 3, 3)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk20c2, wasm_v32x4_shuffle(vi2x2, vi2x2, 1, 1, 1, 1)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk20c2, wasm_v32x4_shuffle(vi4x2, vi4x2, 1, 1, 1, 1)));

        const v128_t vk01c0 = wasm_v128_load(w + 40);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk01c0, wasm_v32x4_shuffle(vi0x1, vi0x1, 0, 0, 0, 0)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk01c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 0, 0, 0, 0)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk01c0, wasm_v32x4_shuffle(vi0x2, vi0x2, 2, 2, 2, 2)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk01c0, wasm_v32x4_shuffle(vi2x2, vi2x2, 2, 2, 2, 2)));

        const v128_t vk11c0 = wasm_v128_load(w + 44);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk11c0, wasm_v32x4_shuffle(vi1x1, vi1x1, 0, 0, 0, 0)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk11c0, wasm_v32x4_shuffle(vi3x1, vi3x1, 0, 0, 0, 0)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk11c0, wasm_v32x4_shuffle(vi1x2, vi1x2, 2, 2, 2, 2)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk11c0, wasm_v32x4_shuffle(vi3x2, vi3x2, 2, 2, 2, 2)));

        const v128_t vk21c0 = wasm_v128_load(w + 48);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk21c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 0, 0, 0, 0)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk21c0, wasm_v32x4_shuffle(vi4x1, vi4x1, 0, 0, 0, 0)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk21c0, wasm_v32x4_shuffle(vi2x2, vi2x2, 2, 2, 2, 2)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk21c0, wasm_v32x4_shuffle(vi4x2, vi4x2, 2, 2, 2, 2)));

        const v128_t vk01c1 = wasm_v128_load(w + 52);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk01c1, wasm_v32x4_shuffle(vi0x1, vi0x1, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk01c1, wasm_v32x4_shuffle(vi2x1, vi2x1, 1, 1, 1, 1)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk01c1, wasm_v32x4_shuffle(vi0x2, vi0x2, 3, 3, 3, 3)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk01c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 3, 3, 3, 3)));

        const v128_t vk11c1 = wasm_v128_load(w + 56);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk11c1, wasm_v32x4_shuffle(vi1x1, vi1x1, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk11c1, wasm_v32x4_shuffle(vi3x1, vi3x1, 1, 1, 1, 1)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk11c1, wasm_v32x4_shuffle(vi1x2, vi1x2, 3, 3, 3, 3)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk11c1, wasm_v32x4_shuffle(vi3x2, vi3x2, 3, 3, 3, 3)));

        const v128_t vk21c1 = wasm_v128_load(w + 60);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk21c1, wasm_v32x4_shuffle(vi2x1, vi2x1, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk21c1, wasm_v32x4_shuffle(vi4x1, vi4x1, 1, 1, 1, 1)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk21c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 3, 3, 3, 3)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk21c1, wasm_v32x4_shuffle(vi4x2, vi4x2, 3, 3, 3, 3)));

        const v128_t vk01c2 = wasm_v128_load(w + 64);

        // viMx3 = ( iM4c2, iM4c1, iM4c0, iM3c2 )
        const v128_t vi0x3 = wasm_v128_load(i0); i0 += 4;
        const v128_t vi1x3 = wasm_v128_load(i1); i1 += 4;
        const v128_t vi2x3 = wasm_v128_load(i2); i2 += 4;
        const v128_t vi3x3 = wasm_v128_load(i3); i3 += 4;
        const v128_t vi4x3 = wasm_v128_load(i4); i4 += 4;

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk01c2, wasm_v32x4_shuffle(vi0x1, vi0x1, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk01c2, wasm_v32x4_shuffle(vi2x1, vi2x1, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk01c2, wasm_v32x4_shuffle(vi0x3, vi0x3, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk01c2, wasm_v32x4_shuffle(vi2x3, vi2x3, 0, 0, 0, 0)));

        const v128_t vk11c2 = wasm_v128_load(w + 68);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk11c2, wasm_v32x4_shuffle(vi1x1, vi1x1, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk11c2, wasm_v32x4_shuffle(vi3x1, vi3x1, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk11c2, wasm_v32x4_shuffle(vi1x3, vi1x3, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk11c2, wasm_v32x4_shuffle(vi3x3, vi3x3, 0, 0, 0, 0)));

        const v128_t vk21c2 = wasm_v128_load(w + 72);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk21c2, wasm_v32x4_shuffle(vi2x1, vi2x1, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk21c2, wasm_v32x4_shuffle(vi4x1, vi4x1, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk21c2, wasm_v32x4_shuffle(vi2x3, vi2x3, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk21c2, wasm_v32x4_shuffle(vi4x3, vi4x3, 0, 0, 0, 0)));

        const v128_t vk02c0 = wasm_v128_load(w + 76);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk02c0, wasm_v32x4_shuffle(vi0x1, vi0x1, 3, 3, 3, 3)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk02c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 3, 3, 3, 3)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk02c0, wasm_v32x4_shuffle(vi0x3, vi0x3, 1, 1, 1, 1)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk02c0, wasm_v32x4_shuffle(vi2x3, vi2x3, 1, 1, 1, 1)));

        const v128_t vk12c0 = wasm_v128_load(w + 80);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk12c0, wasm_v32x4_shuffle(vi1x1, vi1x1, 3, 3, 3, 3)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk12c0, wasm_v32x4_shuffle(vi3x1, vi3x1, 3, 3, 3, 3)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk12c0, wasm_v32x4_shuffle(vi1x3, vi1x3, 1, 1, 1, 1)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk12c0, wasm_v32x4_shuffle(vi3x3, vi3x3, 1, 1, 1, 1)));

        const v128_t vk22c0 = wasm_v128_load(w + 84);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk22c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 3, 3, 3, 3)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk22c0, wasm_v32x4_shuffle(vi4x1, vi4x1, 3, 3, 3, 3)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk22c0, wasm_v32x4_shuffle(vi2x3, vi2x3, 1, 1, 1, 1)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk22c0, wasm_v32x4_shuffle(vi4x3, vi4x3, 1, 1, 1, 1)));

        const v128_t vk02c1 = wasm_v128_load(w + 88);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk02c1, wasm_v32x4_shuffle(vi0x2, vi0x2, 0, 0, 0, 0)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk02c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 0, 0, 0, 0)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk02c1, wasm_v32x4_shuffle(vi0x3, vi0x3, 2, 2, 2, 2)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk02c1, wasm_v32x4_shuffle(vi2x3, vi2x3, 2, 2, 2, 2)));

        const v128_t vk12c1 = wasm_v128_load(w + 92);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk12c1, wasm_v32x4_shuffle(vi1x2, vi1x2, 0, 0, 0, 0)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk12c1, wasm_v32x4_shuffle(vi3x2, vi3x2, 0, 0, 0, 0)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk12c1, wasm_v32x4_shuffle(vi1x3, vi1x3, 2, 2, 2, 2)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk12c1, wasm_v32x4_shuffle(vi3x3, vi3x3, 2, 2, 2, 2)));

        const v128_t vk22c1 = wasm_v128_load(w + 96);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk22c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 0, 0, 0, 0)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk22c1, wasm_v32x4_shuffle(vi4x2, vi4x2, 0, 0, 0, 0)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk22c1, wasm_v32x4_shuffle(vi2x3, vi2x3, 2, 2, 2, 2)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk22c1, wasm_v32x4_shuffle(vi4x3, vi4x3, 2, 2, 2, 2)));

        const v128_t vk02c2 = wasm_v128_load(w + 100);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk02c2, wasm_v32x4_shuffle(vi0x2, vi0x2, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk02c2, wasm_v32x4_shuffle(vi2x2, vi2x2, 1, 1, 1, 1)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk02c2, wasm_v32x4_shuffle(vi0x3, vi0x3, 3, 3, 3, 3)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk02c2, wasm_v32x4_shuffle(vi2x3, vi2x3, 3, 3, 3, 3)));

        const v128_t vk12c2 = wasm_v128_load(w + 104);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk12c2, wasm_v32x4_shuffle(vi1x2, vi1x2, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk12c2, wasm_v32x4_shuffle(vi3x2, vi3x2, 1, 1, 1, 1)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk12c2, wasm_v32x4_shuffle(vi1x3, vi1x3, 3, 3, 3, 3)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk12c2, wasm_v32x4_shuffle(vi3x3, vi3x3, 3, 3, 3, 3)));

        const v128_t vk22c2 = wasm_v128_load(w + 108);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk22c2, wasm_v32x4_shuffle(vi2x2, vi2x2, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk22c2, wasm_v32x4_shuffle(vi4x2, vi4x2, 1, 1, 1, 1)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk22c2, wasm_v32x4_shuffle(vi2x3, vi2x3, 3, 3, 3, 3)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk22c2, wasm_v32x4_shuffle(vi4x3, vi4x3, 3, 3, 3, 3)));

        vi0x0 = vi0x3;
        vi1x0 = vi1x3;
        vi2x0 = vi2x3;
        vi3x0 = vi3x3;
        vi4x0 = vi4x3;

        vo0x0 = wasm_f32x4_pmax(vmin, vo0x0);
        vo1x0 = wasm_f32x4_pmax(vmin, vo1x0);
        vo0x1 = wasm_f32x4_pmax(vmin, vo0x1);
        vo1x1 = wasm_f32x4_pmax(vmin, vo1x1);

        vo0x0 = wasm_f32x4_pmin(vmax, vo0x0);
        vo1x0 = wasm_f32x4_pmin(vmax, vo1x0);
        vo0x1 = wasm_f32x4_pmin(vmax, vo0x1);
        vo1x1 = wasm_f32x4_pmin(vmax, vo1x1);

        const v128_t vo0c01 = wasm_v32x4_shuffle(vo0x0, vo0x1, 0, 4, 1, 5);
        const v128_t vo0c23 = wasm_v32x4_shuffle(vo0x0, vo0x1, 2, 6, 3, 7);
        const v128_t vo1c01 = wasm_v32x4_shuffle(vo1x0, vo1x1, 0, 4, 1, 5);
        const v128_t vo1c23 = wasm_v32x4_shuffle(vo1x0, vo1x1, 2, 6, 3, 7);

        // Always 2+ output width elements remaining
        wasm_v128_store64_lane(o1c0, vo1c01, 0); o1c0 += 2;
        wasm_v128_store64_lane(o1c1, vo1c01, 1); o1c1 += 2;
        wasm_v128_store64_lane(o1c2, vo1c23, 0); o1c2 += 2;
        wasm_v128_store64_lane(o1c3, vo1c23, 1); o1c3 += 2;
        wasm_v128_store64_lane(o0c0, vo0c01, 0); o0c0 += 2;
        wasm_v128_store64_lane(o0c1, vo0c01, 1); o0c1 += 2;
        wasm_v128_store64_lane(o0c2, vo0c23, 0); o0c2 += 2;
        wasm_v128_store64_lane(o0c3, vo0c23, 1); o0c3 += 2;
      }
      assert(iw < 4);
      if XNN_UNLIKELY(iw != 0) {
        v128_t vo0x0 = wasm_v128_load(w);
        v128_t vo1x0 = vo0x0;
        v128_t vo0x1 = vo0x0;
        v128_t vo1x1 = vo0x0;

        const v128_t vk00c0 = wasm_v128_load(w + 4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        v128_t vi0x1 = wasm_v128_load(i0);
        v128_t vi1x1 = wasm_v128_load(i1);
        v128_t vi2x1 = wasm_v128_load(i2);
        v128_t vi3x1 = wasm_v128_load(i3);
        v128_t vi4x1 = wasm_v128_load(i4);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk00c0, wasm_v32x4_shuffle(vi0x0, vi0x0, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk00c0, wasm_v32x4_shuffle(vi2x0, vi2x0, 1, 1, 1, 1)));
        if (iw > 2) {
          vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk00c0, wasm_v32x4_shuffle(vi0x1, vi0x1, 3, 3, 3, 3)));
          vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk00c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 3, 3, 3, 3)));
        }

        const v128_t vk10c0 = wasm_v128_load(w + 8);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk10c0, wasm_v32x4_shuffle(vi1x0, vi1x0, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk10c0, wasm_v32x4_shuffle(vi3x0, vi3x0, 1, 1, 1, 1)));
        if (iw > 2) {
          vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk10c0, wasm_v32x4_shuffle(vi1x1, vi1x1, 3, 3, 3, 3)));
          vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk10c0, wasm_v32x4_shuffle(vi3x1, vi3x1, 3, 3, 3, 3)));
        }

        const v128_t vk20c0 = wasm_v128_load(w + 12);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk20c0, wasm_v32x4_shuffle(vi2x0, vi2x0, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk20c0, wasm_v32x4_shuffle(vi4x0, vi4x0, 1, 1, 1, 1)));
        if (iw > 2) {
          vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk20c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 3, 3, 3, 3)));
          vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk20c0, wasm_v32x4_shuffle(vi4x1, vi4x1, 3, 3, 3, 3)));
        }

        const v128_t vk00c1 = wasm_v128_load(w + 16);

        v128_t vi0x2 = wasm_f32x4_const_splat(0.0f);
        v128_t vi1x2 = wasm_f32x4_const_splat(0.0f);
        v128_t vi2x2 = wasm_f32x4_const_splat(0.0f);
        v128_t vi3x2 = wasm_f32x4_const_splat(0.0f);
        v128_t vi4x2 = wasm_f32x4_const_splat(0.0f);
        if (iw >= 2) {
          // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
          vi0x2 = wasm_v128_load(i0 + 4);
          vi1x2 = wasm_v128_load(i1 + 4);
          vi2x2 = wasm_v128_load(i2 + 4);
          vi3x2 = wasm_v128_load(i3 + 4);
          vi4x2 = wasm_v128_load(i4 + 4);
        }

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk00c1, wasm_v32x4_shuffle(vi0x0, vi0x0, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk00c1, wasm_v32x4_shuffle(vi2x0, vi2x0, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk00c1, wasm_v32x4_shuffle(vi0x2, vi0x2, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk00c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 0, 0, 0, 0)));

        const v128_t vk10c1 = wasm_v128_load(w + 20);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk10c1, wasm_v32x4_shuffle(vi1x0, vi1x0, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk10c1, wasm_v32x4_shuffle(vi3x0, vi3x0, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk10c1, wasm_v32x4_shuffle(vi1x2, vi1x2, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk10c1, wasm_v32x4_shuffle(vi3x2, vi3x2, 0, 0, 0, 0)));

        const v128_t vk20c1 = wasm_v128_load(w + 24);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk20c1, wasm_v32x4_shuffle(vi2x0, vi2x0, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk20c1, wasm_v32x4_shuffle(vi4x0, vi4x0, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk20c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk20c1, wasm_v32x4_shuffle(vi4x2, vi4x2, 0, 0, 0, 0)));

        const v128_t vk00c2 = wasm_v128_load(w + 28);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk00c2, wasm_v32x4_shuffle(vi0x0, vi0x0, 3, 3, 3, 3)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk00c2, wasm_v32x4_shuffle(vi2x0, vi2x0, 3, 3, 3, 3)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk00c2, wasm_v32x4_shuffle(vi0x2, vi0x2, 1, 1, 1, 1)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk00c2, wasm_v32x4_shuffle(vi2x2, vi2x2, 1, 1, 1, 1)));

        const v128_t vk10c2 = wasm_v128_load(w + 32);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk10c2, wasm_v32x4_shuffle(vi1x0, vi1x0, 3, 3, 3, 3)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk10c2, wasm_v32x4_shuffle(vi3x0, vi3x0, 3, 3, 3, 3)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk10c2, wasm_v32x4_shuffle(vi1x2, vi1x2, 1, 1, 1, 1)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk10c2, wasm_v32x4_shuffle(vi3x2, vi3x2, 1, 1, 1, 1)));

        const v128_t vk20c2 = wasm_v128_load(w + 36);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk20c2, wasm_v32x4_shuffle(vi2x0, vi2x0, 3, 3, 3, 3)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk20c2, wasm_v32x4_shuffle(vi4x0, vi4x0, 3, 3, 3, 3)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk20c2, wasm_v32x4_shuffle(vi2x2, vi2x2, 1, 1, 1, 1)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk20c2, wasm_v32x4_shuffle(vi4x2, vi4x2, 1, 1, 1, 1)));

        const v128_t vk01c0 = wasm_v128_load(w + 40);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk01c0, wasm_v32x4_shuffle(vi0x1, vi0x1, 0, 0, 0, 0)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk01c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 0, 0, 0, 0)));
        if (iw > 2) {
          vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk01c0, wasm_v32x4_shuffle(vi0x2, vi0x2, 2, 2, 2, 2)));
          vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk01c0, wasm_v32x4_shuffle(vi2x2, vi2x2, 2, 2, 2, 2)));
        }

        const v128_t vk11c0 = wasm_v128_load(w + 44);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk11c0, wasm_v32x4_shuffle(vi1x1, vi1x1, 0, 0, 0, 0)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk11c0, wasm_v32x4_shuffle(vi3x1, vi3x1, 0, 0, 0, 0)));
        if (iw > 2) {
          vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk11c0, wasm_v32x4_shuffle(vi1x2, vi1x2, 2, 2, 2, 2)));
          vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk11c0, wasm_v32x4_shuffle(vi3x2, vi3x2, 2, 2, 2, 2)));
        }

        const v128_t vk21c0 = wasm_v128_load(w + 48);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk21c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 0, 0, 0, 0)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk21c0, wasm_v32x4_shuffle(vi4x1, vi4x1, 0, 0, 0, 0)));
        if (iw > 2) {
          vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk21c0, wasm_v32x4_shuffle(vi2x2, vi2x2, 2, 2, 2, 2)));
          vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk21c0, wasm_v32x4_shuffle(vi4x2, vi4x2, 2, 2, 2, 2)));
        }

        const v128_t vk01c1 = wasm_v128_load(w + 52);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk01c1, wasm_v32x4_shuffle(vi0x1, vi0x1, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk01c1, wasm_v32x4_shuffle(vi2x1, vi2x1, 1, 1, 1, 1)));
        if (iw > 2) {
          vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk01c1, wasm_v32x4_shuffle(vi0x2, vi0x2, 3, 3, 3, 3)));
          vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk01c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 3, 3, 3, 3)));
        }

        const v128_t vk11c1 = wasm_v128_load(w + 56);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk11c1, wasm_v32x4_shuffle(vi1x1, vi1x1, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk11c1, wasm_v32x4_shuffle(vi3x1, vi3x1, 1, 1, 1, 1)));
        if (iw > 2) {
          vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk11c1, wasm_v32x4_shuffle(vi1x2, vi1x2, 3, 3, 3, 3)));
          vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk11c1, wasm_v32x4_shuffle(vi3x2, vi3x2, 3, 3, 3, 3)));
        }

        const v128_t vk21c1 = wasm_v128_load(w + 60);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk21c1, wasm_v32x4_shuffle(vi2x1, vi2x1, 1, 1, 1, 1)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk21c1, wasm_v32x4_shuffle(vi4x1, vi4x1, 1, 1, 1, 1)));
        if (iw > 2) {
          vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk21c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 3, 3, 3, 3)));
          vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk21c1, wasm_v32x4_shuffle(vi4x2, vi4x2, 3, 3, 3, 3)));
        }

        const v128_t vk01c2 = wasm_v128_load(w + 64);

        v128_t vi0x3 = wasm_f32x4_const_splat(0.0f);
        v128_t vi1x3 = wasm_f32x4_const_splat(0.0f);
        v128_t vi2x3 = wasm_f32x4_const_splat(0.0f);
        v128_t vi3x3 = wasm_f32x4_const_splat(0.0f);
        v128_t vi4x3 = wasm_f32x4_const_splat(0.0f);
        if (iw > 2) {
          // viMx3 = ( 0.0, 0.0, 0.0, iM3c2 )
          vi0x3 = wasm_v128_load32_splat(i0 + 8);
          vi1x3 = wasm_v128_load32_splat(i1 + 8);
          vi2x3 = wasm_v128_load32_splat(i2 + 8);
          vi3x3 = wasm_v128_load32_splat(i3 + 8);
          vi4x3 = wasm_v128_load32_splat(i4 + 8);
        }

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk01c2, wasm_v32x4_shuffle(vi0x1, vi0x1, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk01c2, wasm_v32x4_shuffle(vi2x1, vi2x1, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk01c2, wasm_v32x4_shuffle(vi0x3, vi0x3, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk01c2, wasm_v32x4_shuffle(vi2x3, vi2x3, 0, 0, 0, 0)));

        const v128_t vk11c2 = wasm_v128_load(w + 68);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk11c2, wasm_v32x4_shuffle(vi1x1, vi1x1, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk11c2, wasm_v32x4_shuffle(vi3x1, vi3x1, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk11c2, wasm_v32x4_shuffle(vi1x3, vi1x3, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk11c2, wasm_v32x4_shuffle(vi3x3, vi3x3, 0, 0, 0, 0)));

        const v128_t vk21c2 = wasm_v128_load(w + 72);

        vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk21c2, wasm_v32x4_shuffle(vi2x1, vi2x1, 2, 2, 2, 2)));
        vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk21c2, wasm_v32x4_shuffle(vi4x1, vi4x1, 2, 2, 2, 2)));
        vo0x1 = wasm_f32x4_add(vo0x1, wasm_f32x4_mul(vk21c2, wasm_v32x4_shuffle(vi2x3, vi2x3, 0, 0, 0, 0)));
        vo1x1 = wasm_f32x4_add(vo1x1, wasm_f32x4_mul(vk21c2, wasm_v32x4_shuffle(vi4x3, vi4x3, 0, 0, 0, 0)));

        if (iw >= 2) {
          const v128_t vk02c0 = wasm_v128_load(w + 76);

          vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk02c0, wasm_v32x4_shuffle(vi0x1, vi0x1, 3, 3, 3, 3)));
          vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk02c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 3, 3, 3, 3)));

          const v128_t vk12c0 = wasm_v128_load(w + 80);

          vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk12c0, wasm_v32x4_shuffle(vi1x1, vi1x1, 3, 3, 3, 3)));
          vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk12c0, wasm_v32x4_shuffle(vi3x1, vi3x1, 3, 3, 3, 3)));

          const v128_t vk22c0 = wasm_v128_load(w + 84);

          vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk22c0, wasm_v32x4_shuffle(vi2x1, vi2x1, 3, 3, 3, 3)));
          vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk22c0, wasm_v32x4_shuffle(vi4x1, vi4x1, 3, 3, 3, 3)));

          const v128_t vk02c1 = wasm_v128_load(w + 88);

          vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk02c1, wasm_v32x4_shuffle(vi0x2, vi0x2, 0, 0, 0, 0)));
          vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk02c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 0, 0, 0, 0)));

          const v128_t vk12c1 = wasm_v128_load(w + 92);

          vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk12c1, wasm_v32x4_shuffle(vi1x2, vi1x2, 0, 0, 0, 0)));
          vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk12c1, wasm_v32x4_shuffle(vi3x2, vi3x2, 0, 0, 0, 0)));

          const v128_t vk22c1 = wasm_v128_load(w + 96);

          vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk22c1, wasm_v32x4_shuffle(vi2x2, vi2x2, 0, 0, 0, 0)));
          vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk22c1, wasm_v32x4_shuffle(vi4x2, vi4x2, 0, 0, 0, 0)));

          const v128_t vk02c2 = wasm_v128_load(w + 100);

          vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk02c2, wasm_v32x4_shuffle(vi0x2, vi0x2, 1, 1, 1, 1)));
          vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk02c2, wasm_v32x4_shuffle(vi2x2, vi2x2, 1, 1, 1, 1)));

          const v128_t vk12c2 = wasm_v128_load(w + 104);

          vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk12c2, wasm_v32x4_shuffle(vi1x2, vi1x2, 1, 1, 1, 1)));
          vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk12c2, wasm_v32x4_shuffle(vi3x2, vi3x2, 1, 1, 1, 1)));

          const v128_t vk22c2 = wasm_v128_load(w + 108);

          vo0x0 = wasm_f32x4_add(vo0x0, wasm_f32x4_mul(vk22c2, wasm_v32x4_shuffle(vi2x2, vi2x2, 1, 1, 1, 1)));
          vo1x0 = wasm_f32x4_add(vo1x0, wasm_f32x4_mul(vk22c2, wasm_v32x4_shuffle(vi4x2, vi4x2, 1, 1, 1, 1)));
        }

        vo0x0 = wasm_f32x4_pmax(vmin, vo0x0);
        vo1x0 = wasm_f32x4_pmax(vmin, vo1x0);
        vo0x1 = wasm_f32x4_pmax(vmin, vo0x1);
        vo1x1 = wasm_f32x4_pmax(vmin, vo1x1);

        vo0x0 = wasm_f32x4_pmin(vmax, vo0x0);
        vo1x0 = wasm_f32x4_pmin(vmax, vo1x0);
        vo0x1 = wasm_f32x4_pmin(vmax, vo0x1);
        vo1x1 = wasm_f32x4_pmin(vmax, vo1x1);

        if (iw == 3) {
          // Exactly 2 output width elements remaining
          const v128_t vo0c01 = wasm_v32x4_shuffle(vo0x0, vo0x1, 0, 4, 1, 5);
          const v128_t vo0c23 = wasm_v32x4_shuffle(vo0x0, vo0x1, 2, 6, 3, 7);
          const v128_t vo1c01 = wasm_v32x4_shuffle(vo1x0, vo1x1, 0, 4, 1, 5);
          const v128_t vo1c23 = wasm_v32x4_shuffle(vo1x0, vo1x1, 2, 6, 3, 7);

          wasm_v128_store64_lane(o1c0, vo1c01, 0); o1c0 += 2;
          wasm_v128_store64_lane(o1c1, vo1c01, 1); o1c1 += 2;
          wasm_v128_store64_lane(o1c2, vo1c23, 0); o1c2 += 2;
          wasm_v128_store64_lane(o1c3, vo1c23, 1); o1c3 += 2;

          wasm_v128_store64_lane(o0c0, vo0c01, 0); o0c0 += 2;
          wasm_v128_store64_lane(o0c1, vo0c01, 1); o0c1 += 2;
          wasm_v128_store64_lane(o0c2, vo0c23, 0); o0c2 += 2;
          wasm_v128_store64_lane(o0c3, vo0c23, 1); o0c3 += 2;
        } else {
          // Exactly 1 output width element remaining

          wasm_v128_store32_lane(o1c0, vo1x0, 0); o1c0 += 1;
          wasm_v128_store32_lane(o1c1, vo1x0, 1); o1c1 += 1;
          wasm_v128_store32_lane(o1c2, vo1x0, 2); o1c2 += 1;
          wasm_v128_store32_lane(o1c3, vo1x0, 3); o1c3 += 1;
          wasm_v128_store32_lane(o0c0, vo0x0, 0); o0c0 += 1;
          wasm_v128_store32_lane(o0c1, vo0x0, 1); o0c1 += 1;
          wasm_v128_store32_lane(o0c2, vo0x0, 2); o0c2 += 1;
          wasm_v128_store32_lane(o0c3, vo0x0, 3); o0c3 += 1;
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (float*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (float*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (float*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (float*) ((uintptr_t) o0c3 + output_channel_increment);
      o1c0 = (float*) ((uintptr_t) o1c0 + output_channel_increment);
      o1c1 = (float*) ((uintptr_t) o1c1 + output_channel_increment);
      o1c2 = (float*) ((uintptr_t) o1c2 + output_channel_increment);
      o1c3 = (float*) ((uintptr_t) o1c3 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_increment);
      i1 = (const float*) ((uintptr_t) i1 - input_width_increment);
      i2 = (const float*) ((uintptr_t) i2 - input_width_increment);
      i3 = (const float*) ((uintptr_t) i3 - input_width_increment);
      i4 = (const float*) ((uintptr_t) i4 - input_width_increment);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next two rows
    output0 = (float*) ((uintptr_t) output1 + output_height_stride);
    output1 = (float*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next four rows
    i0 = i4;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  }
}
