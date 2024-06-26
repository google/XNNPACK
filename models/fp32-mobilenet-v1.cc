// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!

#include "xnnpack.h"

#include <array>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "xnnpack/cache.h"
#include "xnnpack/common.h"
#include "xnnpack/models.h"

namespace models {

ExecutionPlan FP32MobileNetV1(pthreadpool_t threadpool) {
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(150528, float)> v0;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(401408, float)> v1;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(401408, float)> v2;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(802816, float)> v3;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(200704, float)> v4;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(401408, float)> v5;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(401408, float)> v6;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(401408, float)> v7;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v8;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(200704, float)> v9;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(200704, float)> v10;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(200704, float)> v11;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(50176, float)> v12;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v13;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v14;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v15;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v16;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v17;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v18;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v19;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v20;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v21;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v22;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(100352, float)> v23;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(25088, float)> v24;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(50176, float)> v25;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(50176, float)> v26;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(50176, float)> v27;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1024, float)> v28;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1001, float)> v29;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(864, float)> w30;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(32, float)> w31;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(288, float)> w32;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(32, float)> w33;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(2048, float)> w34;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(64, float)> w35;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(576, float)> w36;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(64, float)> w37;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(8192, float)> w38;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(128, float)> w39;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1152, float)> w40;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(128, float)> w41;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(16384, float)> w42;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(128, float)> w43;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1152, float)> w44;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(128, float)> w45;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(32768, float)> w46;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(256, float)> w47;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(2304, float)> w48;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(256, float)> w49;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(65536, float)> w50;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(256, float)> w51;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(2304, float)> w52;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(256, float)> w53;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(131072, float)> w54;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w55;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w56;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w57;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(262144, float)> w58;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w59;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w60;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w61;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(262144, float)> w62;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w63;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w64;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w65;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(262144, float)> w66;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w67;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w68;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w69;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(262144, float)> w70;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w71;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w72;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w73;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(262144, float)> w74;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w75;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(4608, float)> w76;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(512, float)> w77;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(524288, float)> w78;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1024, float)> w79;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(9216, float)> w80;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1024, float)> w81;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1048576, float)> w82;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1024, float)> w83;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1025024, float)> w84;
  alignas(16) static std::array<float, XNN_PAD_EXTRA_BYTES(1001, float)> w85;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  std::generate(v0.begin(), v0.end(), std::ref(f32rng));
  std::generate(v1.begin(), v1.end(), std::ref(f32rng));
  std::generate(v2.begin(), v2.end(), std::ref(f32rng));
  std::generate(v3.begin(), v3.end(), std::ref(f32rng));
  std::generate(v4.begin(), v4.end(), std::ref(f32rng));
  std::generate(v5.begin(), v5.end(), std::ref(f32rng));
  std::generate(v6.begin(), v6.end(), std::ref(f32rng));
  std::generate(v7.begin(), v7.end(), std::ref(f32rng));
  std::generate(v8.begin(), v8.end(), std::ref(f32rng));
  std::generate(v9.begin(), v9.end(), std::ref(f32rng));
  std::generate(v10.begin(), v10.end(), std::ref(f32rng));
  std::generate(v11.begin(), v11.end(), std::ref(f32rng));
  std::generate(v12.begin(), v12.end(), std::ref(f32rng));
  std::generate(v13.begin(), v13.end(), std::ref(f32rng));
  std::generate(v14.begin(), v14.end(), std::ref(f32rng));
  std::generate(v15.begin(), v15.end(), std::ref(f32rng));
  std::generate(v16.begin(), v16.end(), std::ref(f32rng));
  std::generate(v17.begin(), v17.end(), std::ref(f32rng));
  std::generate(v18.begin(), v18.end(), std::ref(f32rng));
  std::generate(v19.begin(), v19.end(), std::ref(f32rng));
  std::generate(v20.begin(), v20.end(), std::ref(f32rng));
  std::generate(v21.begin(), v21.end(), std::ref(f32rng));
  std::generate(v22.begin(), v22.end(), std::ref(f32rng));
  std::generate(v23.begin(), v23.end(), std::ref(f32rng));
  std::generate(v24.begin(), v24.end(), std::ref(f32rng));
  std::generate(v25.begin(), v25.end(), std::ref(f32rng));
  std::generate(v26.begin(), v26.end(), std::ref(f32rng));
  std::generate(v27.begin(), v27.end(), std::ref(f32rng));
  std::generate(v28.begin(), v28.end(), std::ref(f32rng));
  std::generate(v29.begin(), v29.end(), std::ref(f32rng));
  std::generate(w30.begin(), w30.end(), std::ref(f32rng));
  std::generate(w31.begin(), w31.end(), std::ref(f32rng));
  std::generate(w32.begin(), w32.end(), std::ref(f32rng));
  std::generate(w33.begin(), w33.end(), std::ref(f32rng));
  std::generate(w34.begin(), w34.end(), std::ref(f32rng));
  std::generate(w35.begin(), w35.end(), std::ref(f32rng));
  std::generate(w36.begin(), w36.end(), std::ref(f32rng));
  std::generate(w37.begin(), w37.end(), std::ref(f32rng));
  std::generate(w38.begin(), w38.end(), std::ref(f32rng));
  std::generate(w39.begin(), w39.end(), std::ref(f32rng));
  std::generate(w40.begin(), w40.end(), std::ref(f32rng));
  std::generate(w41.begin(), w41.end(), std::ref(f32rng));
  std::generate(w42.begin(), w42.end(), std::ref(f32rng));
  std::generate(w43.begin(), w43.end(), std::ref(f32rng));
  std::generate(w44.begin(), w44.end(), std::ref(f32rng));
  std::generate(w45.begin(), w45.end(), std::ref(f32rng));
  std::generate(w46.begin(), w46.end(), std::ref(f32rng));
  std::generate(w47.begin(), w47.end(), std::ref(f32rng));
  std::generate(w48.begin(), w48.end(), std::ref(f32rng));
  std::generate(w49.begin(), w49.end(), std::ref(f32rng));
  std::generate(w50.begin(), w50.end(), std::ref(f32rng));
  std::generate(w51.begin(), w51.end(), std::ref(f32rng));
  std::generate(w52.begin(), w52.end(), std::ref(f32rng));
  std::generate(w53.begin(), w53.end(), std::ref(f32rng));
  std::generate(w54.begin(), w54.end(), std::ref(f32rng));
  std::generate(w55.begin(), w55.end(), std::ref(f32rng));
  std::generate(w56.begin(), w56.end(), std::ref(f32rng));
  std::generate(w57.begin(), w57.end(), std::ref(f32rng));
  std::generate(w58.begin(), w58.end(), std::ref(f32rng));
  std::generate(w59.begin(), w59.end(), std::ref(f32rng));
  std::generate(w60.begin(), w60.end(), std::ref(f32rng));
  std::generate(w61.begin(), w61.end(), std::ref(f32rng));
  std::generate(w62.begin(), w62.end(), std::ref(f32rng));
  std::generate(w63.begin(), w63.end(), std::ref(f32rng));
  std::generate(w64.begin(), w64.end(), std::ref(f32rng));
  std::generate(w65.begin(), w65.end(), std::ref(f32rng));
  std::generate(w66.begin(), w66.end(), std::ref(f32rng));
  std::generate(w67.begin(), w67.end(), std::ref(f32rng));
  std::generate(w68.begin(), w68.end(), std::ref(f32rng));
  std::generate(w69.begin(), w69.end(), std::ref(f32rng));
  std::generate(w70.begin(), w70.end(), std::ref(f32rng));
  std::generate(w71.begin(), w71.end(), std::ref(f32rng));
  std::generate(w72.begin(), w72.end(), std::ref(f32rng));
  std::generate(w73.begin(), w73.end(), std::ref(f32rng));
  std::generate(w74.begin(), w74.end(), std::ref(f32rng));
  std::generate(w75.begin(), w75.end(), std::ref(f32rng));
  std::generate(w76.begin(), w76.end(), std::ref(f32rng));
  std::generate(w77.begin(), w77.end(), std::ref(f32rng));
  std::generate(w78.begin(), w78.end(), std::ref(f32rng));
  std::generate(w79.begin(), w79.end(), std::ref(f32rng));
  std::generate(w80.begin(), w80.end(), std::ref(f32rng));
  std::generate(w81.begin(), w81.end(), std::ref(f32rng));
  std::generate(w82.begin(), w82.end(), std::ref(f32rng));
  std::generate(w83.begin(), w83.end(), std::ref(f32rng));
  std::generate(w84.begin(), w84.end(), std::ref(f32rng));
  std::generate(w85.begin(), w85.end(), std::ref(f32rng));

  Operators operators;
  xnn_status status;
  xnn_code_cache* code_cache_ptr = nullptr;
  size_t max_workspace_size = 0;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/3,
    /*group_output_channels=*/32,
    /*input_channel_stride=*/3,
    /*output_channel_stride=*/32,
    /*kernel=*/w30.data(), /*bias=*/w31.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/32,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/32,
    /*output_channel_stride=*/32,
    /*kernel=*/w32.data(), /*bias=*/w33.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/64,
    /*input_channel_stride=*/32,
    /*output_channel_stride=*/64,
    /*kernel=*/w34.data(), /*bias=*/w35.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/64,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/64,
    /*output_channel_stride=*/64,
    /*kernel=*/w36.data(), /*bias=*/w37.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/128,
    /*input_channel_stride=*/64,
    /*output_channel_stride=*/128,
    /*kernel=*/w38.data(), /*bias=*/w39.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/128,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/128,
    /*output_channel_stride=*/128,
    /*kernel=*/w40.data(), /*bias=*/w41.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/128,
    /*group_output_channels=*/128,
    /*input_channel_stride=*/128,
    /*output_channel_stride=*/128,
    /*kernel=*/w42.data(), /*bias=*/w43.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/128,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/128,
    /*output_channel_stride=*/128,
    /*kernel=*/w44.data(), /*bias=*/w45.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/128,
    /*group_output_channels=*/256,
    /*input_channel_stride=*/128,
    /*output_channel_stride=*/256,
    /*kernel=*/w46.data(), /*bias=*/w47.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/256,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/256,
    /*output_channel_stride=*/256,
    /*kernel=*/w48.data(), /*bias=*/w49.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/256,
    /*group_output_channels=*/256,
    /*input_channel_stride=*/256,
    /*output_channel_stride=*/256,
    /*kernel=*/w50.data(), /*bias=*/w51.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/256,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/256,
    /*output_channel_stride=*/256,
    /*kernel=*/w52.data(), /*bias=*/w53.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/256,
    /*group_output_channels=*/512,
    /*input_channel_stride=*/256,
    /*output_channel_stride=*/512,
    /*kernel=*/w54.data(), /*bias=*/w55.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/512,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w56.data(), /*bias=*/w57.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/512,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w58.data(), /*bias=*/w59.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/512,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w60.data(), /*bias=*/w61.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/512,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w62.data(), /*bias=*/w63.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/512,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w64.data(), /*bias=*/w65.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/512,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w66.data(), /*bias=*/w67.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/512,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w68.data(), /*bias=*/w69.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/512,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w70.data(), /*bias=*/w71.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/512,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w72.data(), /*bias=*/w73.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/512,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w74.data(), /*bias=*/w75.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/512,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/512,
    /*kernel=*/w76.data(), /*bias=*/w77.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/512,
    /*group_output_channels=*/1024,
    /*input_channel_stride=*/512,
    /*output_channel_stride=*/1024,
    /*kernel=*/w78.data(), /*bias=*/w79.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1024,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/1024,
    /*output_channel_stride=*/1024,
    /*kernel=*/w80.data(), /*bias=*/w81.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/1024,
    /*group_output_channels=*/1024,
    /*input_channel_stride=*/1024,
    /*output_channel_stride=*/1024,
    /*kernel=*/w82.data(), /*bias=*/w83.data(),
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/1024,
    /*group_output_channels=*/1001,
    /*input_channel_stride=*/1024,
    /*output_channel_stride=*/1001,
    /*kernel=*/w84.data(), /*bias=*/w85.data(),
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  size_t op0_workspace_size = 0;
  size_t op0_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op0,
    /*batch_size=*/1, /*input_height=*/224, /*input_width=*/224,
    &op0_workspace_size, &op0_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op0_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #0" << std::endl;
    return ExecutionPlan();
  }

  size_t op1_workspace_size = 0;
  size_t op1_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op1,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    &op1_workspace_size, &op1_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op1_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #1" << std::endl;
    return ExecutionPlan();
  }

  size_t op2_workspace_size = 0;
  size_t op2_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op2,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    &op2_workspace_size, &op2_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op2_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #2" << std::endl;
    return ExecutionPlan();
  }

  size_t op3_workspace_size = 0;
  size_t op3_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op3,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    &op3_workspace_size, &op3_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op3_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #3" << std::endl;
    return ExecutionPlan();
  }

  size_t op4_workspace_size = 0;
  size_t op4_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op4,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op4_workspace_size, &op4_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op4_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #4" << std::endl;
    return ExecutionPlan();
  }

  size_t op5_workspace_size = 0;
  size_t op5_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op5,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op5_workspace_size, &op5_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op5_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #5" << std::endl;
    return ExecutionPlan();
  }

  size_t op6_workspace_size = 0;
  size_t op6_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op6,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op6_workspace_size, &op6_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op6_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #6" << std::endl;
    return ExecutionPlan();
  }

  size_t op7_workspace_size = 0;
  size_t op7_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op7,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op7_workspace_size, &op7_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op7_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #7" << std::endl;
    return ExecutionPlan();
  }

  size_t op8_workspace_size = 0;
  size_t op8_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op8,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op8_workspace_size, &op8_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op8_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #8" << std::endl;
    return ExecutionPlan();
  }

  size_t op9_workspace_size = 0;
  size_t op9_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op9,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op9_workspace_size, &op9_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op9_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #9" << std::endl;
    return ExecutionPlan();
  }

  size_t op10_workspace_size = 0;
  size_t op10_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op10,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op10_workspace_size, &op10_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op10_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #10" << std::endl;
    return ExecutionPlan();
  }

  size_t op11_workspace_size = 0;
  size_t op11_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op11,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op11_workspace_size, &op11_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op11_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #11" << std::endl;
    return ExecutionPlan();
  }

  size_t op12_workspace_size = 0;
  size_t op12_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op12,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op12_workspace_size, &op12_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op12_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #12" << std::endl;
    return ExecutionPlan();
  }

  size_t op13_workspace_size = 0;
  size_t op13_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op13,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op13_workspace_size, &op13_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op13_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #13" << std::endl;
    return ExecutionPlan();
  }

  size_t op14_workspace_size = 0;
  size_t op14_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op14,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op14_workspace_size, &op14_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op14_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #14" << std::endl;
    return ExecutionPlan();
  }

  size_t op15_workspace_size = 0;
  size_t op15_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op15,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op15_workspace_size, &op15_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op15_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #15" << std::endl;
    return ExecutionPlan();
  }

  size_t op16_workspace_size = 0;
  size_t op16_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op16,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op16_workspace_size, &op16_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op16_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #16" << std::endl;
    return ExecutionPlan();
  }

  size_t op17_workspace_size = 0;
  size_t op17_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op17,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op17_workspace_size, &op17_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op17_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #17" << std::endl;
    return ExecutionPlan();
  }

  size_t op18_workspace_size = 0;
  size_t op18_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op18,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op18_workspace_size, &op18_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op18_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #18" << std::endl;
    return ExecutionPlan();
  }

  size_t op19_workspace_size = 0;
  size_t op19_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op19,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op19_workspace_size, &op19_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op19_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #19" << std::endl;
    return ExecutionPlan();
  }

  size_t op20_workspace_size = 0;
  size_t op20_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op20,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op20_workspace_size, &op20_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op20_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #20" << std::endl;
    return ExecutionPlan();
  }

  size_t op21_workspace_size = 0;
  size_t op21_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op21,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op21_workspace_size, &op21_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op21_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #21" << std::endl;
    return ExecutionPlan();
  }

  size_t op22_workspace_size = 0;
  size_t op22_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op22,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op22_workspace_size, &op22_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op22_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #22" << std::endl;
    return ExecutionPlan();
  }

  size_t op23_workspace_size = 0;
  size_t op23_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op23,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op23_workspace_size, &op23_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op23_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #23" << std::endl;
    return ExecutionPlan();
  }

  size_t op24_workspace_size = 0;
  size_t op24_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op24,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op24_workspace_size, &op24_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op24_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #24" << std::endl;
    return ExecutionPlan();
  }

  size_t op25_workspace_size = 0;
  size_t op25_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op25,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op25_workspace_size, &op25_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op25_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #25" << std::endl;
    return ExecutionPlan();
  }

  size_t op26_workspace_size = 0;
  size_t op26_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op26,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op26_workspace_size, &op26_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op26_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #26" << std::endl;
    return ExecutionPlan();
  }

  size_t op27_workspace_size = 0;
  size_t op27_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op27,
    /*batch_size=*/1, 49 /* width */,
    1024 /* channels */, 1024 /* input stride */, 1024 /* output stride */,
    &op27_workspace_size, &op27_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op27_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #27" << std::endl;
    return ExecutionPlan();
  }

  size_t op28_workspace_size = 0;
  size_t op28_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op28,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op28_workspace_size, &op28_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op28_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #28" << std::endl;
    return ExecutionPlan();
  }

  Workspace workspace(max_workspace_size);

  status = xnn_setup_convolution2d_nhwc_f32(
    op0,
    workspace.data(), /*input=*/v0.data(), /*output=*/v1.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op1,
    workspace.data(), /*input=*/v1.data(), /*output=*/v2.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op2,
    workspace.data(), /*input=*/v2.data(), /*output=*/v3.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op3,
    workspace.data(), /*input=*/v3.data(), /*output=*/v4.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op4,
    workspace.data(), /*input=*/v4.data(), /*output=*/v5.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op5,
    workspace.data(), /*input=*/v5.data(), /*output=*/v6.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op6,
    workspace.data(), /*input=*/v6.data(), /*output=*/v7.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op7,
    workspace.data(), /*input=*/v7.data(), /*output=*/v8.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op8,
    workspace.data(), /*input=*/v8.data(), /*output=*/v9.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op9,
    workspace.data(), /*input=*/v9.data(), /*output=*/v10.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op10,
    workspace.data(), /*input=*/v10.data(), /*output=*/v11.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op11,
    workspace.data(), /*input=*/v11.data(), /*output=*/v12.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op12,
    workspace.data(), /*input=*/v12.data(), /*output=*/v13.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op13,
    workspace.data(), /*input=*/v13.data(), /*output=*/v14.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op14,
    workspace.data(), /*input=*/v14.data(), /*output=*/v15.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op15,
    workspace.data(), /*input=*/v15.data(), /*output=*/v16.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op16,
    workspace.data(), /*input=*/v16.data(), /*output=*/v17.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op17,
    workspace.data(), /*input=*/v17.data(), /*output=*/v18.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op18,
    workspace.data(), /*input=*/v18.data(), /*output=*/v19.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op19,
    workspace.data(), /*input=*/v19.data(), /*output=*/v20.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op20,
    workspace.data(), /*input=*/v20.data(), /*output=*/v21.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op21,
    workspace.data(), /*input=*/v21.data(), /*output=*/v22.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op22,
    workspace.data(), /*input=*/v22.data(), /*output=*/v23.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op23,
    workspace.data(), /*input=*/v23.data(), /*output=*/v24.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op24,
    workspace.data(), /*input=*/v24.data(), /*output=*/v25.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op25,
    workspace.data(), /*input=*/v25.data(), /*output=*/v26.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op26,
    workspace.data(), /*input=*/v26.data(), /*output=*/v27.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op27,
    workspace.data(),
    /*input=*/v27.data(), /*output=*/v28.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op28,
    workspace.data(), /*input=*/v28.data(), /*output=*/v29.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  XNN_PRAGMA_CLANG("clang diagnostic push")
  XNN_PRAGMA_CLANG("clang diagnostic ignored \"-Wpessimizing-move\"")
  return ExecutionPlan{operators, workspace};
  XNN_PRAGMA_CLANG("clang diagnostic pop")
}

}  // namespace models
