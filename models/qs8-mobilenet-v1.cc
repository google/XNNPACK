// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <array>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "models/models.h"

namespace models {

ExecutionPlan QS8MobileNetV1(pthreadpool_t threadpool) {
  alignas(16) static std::array<int8_t, 150528 + XNN_EXTRA_BYTES / sizeof(int8_t)> v0;
  alignas(16) static std::array<int8_t, 401408> v1;
  alignas(16) static std::array<int8_t, 401408> v2;
  alignas(16) static std::array<int8_t, 802816> v3;
  alignas(16) static std::array<int8_t, 200704> v4;
  alignas(16) static std::array<int8_t, 401408> v5;
  alignas(16) static std::array<int8_t, 401408> v6;
  alignas(16) static std::array<int8_t, 401408> v7;
  alignas(16) static std::array<int8_t, 100352> v8;
  alignas(16) static std::array<int8_t, 200704> v9;
  alignas(16) static std::array<int8_t, 200704> v10;
  alignas(16) static std::array<int8_t, 200704> v11;
  alignas(16) static std::array<int8_t, 50176> v12;
  alignas(16) static std::array<int8_t, 100352> v13;
  alignas(16) static std::array<int8_t, 100352> v14;
  alignas(16) static std::array<int8_t, 100352> v15;
  alignas(16) static std::array<int8_t, 100352> v16;
  alignas(16) static std::array<int8_t, 100352> v17;
  alignas(16) static std::array<int8_t, 100352> v18;
  alignas(16) static std::array<int8_t, 100352> v19;
  alignas(16) static std::array<int8_t, 100352> v20;
  alignas(16) static std::array<int8_t, 100352> v21;
  alignas(16) static std::array<int8_t, 100352> v22;
  alignas(16) static std::array<int8_t, 100352> v23;
  alignas(16) static std::array<int8_t, 25088> v24;
  alignas(16) static std::array<int8_t, 50176> v25;
  alignas(16) static std::array<int8_t, 50176> v26;
  alignas(16) static std::array<int8_t, 50176> v27;
  alignas(16) static std::array<int8_t, 1024> v28;
  alignas(16) static std::array<int8_t, 1001> v29;
  alignas(16) static std::array<int8_t, 864> w30;
  alignas(16) static std::array<int32_t, 32> w31;
  alignas(16) static std::array<int8_t, 288> w32;
  alignas(16) static std::array<int32_t, 32> w33;
  alignas(16) static std::array<int8_t, 2048> w34;
  alignas(16) static std::array<int32_t, 64> w35;
  alignas(16) static std::array<int8_t, 576> w36;
  alignas(16) static std::array<int32_t, 64> w37;
  alignas(16) static std::array<int8_t, 8192> w38;
  alignas(16) static std::array<int32_t, 128> w39;
  alignas(16) static std::array<int8_t, 1152> w40;
  alignas(16) static std::array<int32_t, 128> w41;
  alignas(16) static std::array<int8_t, 16384> w42;
  alignas(16) static std::array<int32_t, 128> w43;
  alignas(16) static std::array<int8_t, 1152> w44;
  alignas(16) static std::array<int32_t, 128> w45;
  alignas(16) static std::array<int8_t, 32768> w46;
  alignas(16) static std::array<int32_t, 256> w47;
  alignas(16) static std::array<int8_t, 2304> w48;
  alignas(16) static std::array<int32_t, 256> w49;
  alignas(16) static std::array<int8_t, 65536> w50;
  alignas(16) static std::array<int32_t, 256> w51;
  alignas(16) static std::array<int8_t, 2304> w52;
  alignas(16) static std::array<int32_t, 256> w53;
  alignas(16) static std::array<int8_t, 131072> w54;
  alignas(16) static std::array<int32_t, 512> w55;
  alignas(16) static std::array<int8_t, 4608> w56;
  alignas(16) static std::array<int32_t, 512> w57;
  alignas(16) static std::array<int8_t, 262144> w58;
  alignas(16) static std::array<int32_t, 512> w59;
  alignas(16) static std::array<int8_t, 4608> w60;
  alignas(16) static std::array<int32_t, 512> w61;
  alignas(16) static std::array<int8_t, 262144> w62;
  alignas(16) static std::array<int32_t, 512> w63;
  alignas(16) static std::array<int8_t, 4608> w64;
  alignas(16) static std::array<int32_t, 512> w65;
  alignas(16) static std::array<int8_t, 262144> w66;
  alignas(16) static std::array<int32_t, 512> w67;
  alignas(16) static std::array<int8_t, 4608> w68;
  alignas(16) static std::array<int32_t, 512> w69;
  alignas(16) static std::array<int8_t, 262144> w70;
  alignas(16) static std::array<int32_t, 512> w71;
  alignas(16) static std::array<int8_t, 4608> w72;
  alignas(16) static std::array<int32_t, 512> w73;
  alignas(16) static std::array<int8_t, 262144> w74;
  alignas(16) static std::array<int32_t, 512> w75;
  alignas(16) static std::array<int8_t, 4608> w76;
  alignas(16) static std::array<int32_t, 512> w77;
  alignas(16) static std::array<int8_t, 524288> w78;
  alignas(16) static std::array<int32_t, 1024> w79;
  alignas(16) static std::array<int8_t, 9216> w80;
  alignas(16) static std::array<int32_t, 1024> w81;
  alignas(16) static std::array<int8_t, 1048576> w82;
  alignas(16) static std::array<int32_t, 1024> w83;
  alignas(16) static std::array<int8_t, 1025024> w84;
  alignas(16) static std::array<int32_t, 1001> w85;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(-127, 127), std::ref(rng));
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  std::generate(v0.begin(), v0.end(), std::ref(i8rng));
  std::generate(v1.begin(), v1.end(), std::ref(i8rng));
  std::generate(v2.begin(), v2.end(), std::ref(i8rng));
  std::generate(v3.begin(), v3.end(), std::ref(i8rng));
  std::generate(v4.begin(), v4.end(), std::ref(i8rng));
  std::generate(v5.begin(), v5.end(), std::ref(i8rng));
  std::generate(v6.begin(), v6.end(), std::ref(i8rng));
  std::generate(v7.begin(), v7.end(), std::ref(i8rng));
  std::generate(v8.begin(), v8.end(), std::ref(i8rng));
  std::generate(v9.begin(), v9.end(), std::ref(i8rng));
  std::generate(v10.begin(), v10.end(), std::ref(i8rng));
  std::generate(v11.begin(), v11.end(), std::ref(i8rng));
  std::generate(v12.begin(), v12.end(), std::ref(i8rng));
  std::generate(v13.begin(), v13.end(), std::ref(i8rng));
  std::generate(v14.begin(), v14.end(), std::ref(i8rng));
  std::generate(v15.begin(), v15.end(), std::ref(i8rng));
  std::generate(v16.begin(), v16.end(), std::ref(i8rng));
  std::generate(v17.begin(), v17.end(), std::ref(i8rng));
  std::generate(v18.begin(), v18.end(), std::ref(i8rng));
  std::generate(v19.begin(), v19.end(), std::ref(i8rng));
  std::generate(v20.begin(), v20.end(), std::ref(i8rng));
  std::generate(v21.begin(), v21.end(), std::ref(i8rng));
  std::generate(v22.begin(), v22.end(), std::ref(i8rng));
  std::generate(v23.begin(), v23.end(), std::ref(i8rng));
  std::generate(v24.begin(), v24.end(), std::ref(i8rng));
  std::generate(v25.begin(), v25.end(), std::ref(i8rng));
  std::generate(v26.begin(), v26.end(), std::ref(i8rng));
  std::generate(v27.begin(), v27.end(), std::ref(i8rng));
  std::generate(v28.begin(), v28.end(), std::ref(i8rng));
  std::generate(v29.begin(), v29.end(), std::ref(i8rng));
  std::generate(w30.begin(), w30.end(), std::ref(i8rng));
  std::generate(w31.begin(), w31.end(), std::ref(i32rng));
  std::generate(w32.begin(), w32.end(), std::ref(i8rng));
  std::generate(w33.begin(), w33.end(), std::ref(i32rng));
  std::generate(w34.begin(), w34.end(), std::ref(i8rng));
  std::generate(w35.begin(), w35.end(), std::ref(i32rng));
  std::generate(w36.begin(), w36.end(), std::ref(i8rng));
  std::generate(w37.begin(), w37.end(), std::ref(i32rng));
  std::generate(w38.begin(), w38.end(), std::ref(i8rng));
  std::generate(w39.begin(), w39.end(), std::ref(i32rng));
  std::generate(w40.begin(), w40.end(), std::ref(i8rng));
  std::generate(w41.begin(), w41.end(), std::ref(i32rng));
  std::generate(w42.begin(), w42.end(), std::ref(i8rng));
  std::generate(w43.begin(), w43.end(), std::ref(i32rng));
  std::generate(w44.begin(), w44.end(), std::ref(i8rng));
  std::generate(w45.begin(), w45.end(), std::ref(i32rng));
  std::generate(w46.begin(), w46.end(), std::ref(i8rng));
  std::generate(w47.begin(), w47.end(), std::ref(i32rng));
  std::generate(w48.begin(), w48.end(), std::ref(i8rng));
  std::generate(w49.begin(), w49.end(), std::ref(i32rng));
  std::generate(w50.begin(), w50.end(), std::ref(i8rng));
  std::generate(w51.begin(), w51.end(), std::ref(i32rng));
  std::generate(w52.begin(), w52.end(), std::ref(i8rng));
  std::generate(w53.begin(), w53.end(), std::ref(i32rng));
  std::generate(w54.begin(), w54.end(), std::ref(i8rng));
  std::generate(w55.begin(), w55.end(), std::ref(i32rng));
  std::generate(w56.begin(), w56.end(), std::ref(i8rng));
  std::generate(w57.begin(), w57.end(), std::ref(i32rng));
  std::generate(w58.begin(), w58.end(), std::ref(i8rng));
  std::generate(w59.begin(), w59.end(), std::ref(i32rng));
  std::generate(w60.begin(), w60.end(), std::ref(i8rng));
  std::generate(w61.begin(), w61.end(), std::ref(i32rng));
  std::generate(w62.begin(), w62.end(), std::ref(i8rng));
  std::generate(w63.begin(), w63.end(), std::ref(i32rng));
  std::generate(w64.begin(), w64.end(), std::ref(i8rng));
  std::generate(w65.begin(), w65.end(), std::ref(i32rng));
  std::generate(w66.begin(), w66.end(), std::ref(i8rng));
  std::generate(w67.begin(), w67.end(), std::ref(i32rng));
  std::generate(w68.begin(), w68.end(), std::ref(i8rng));
  std::generate(w69.begin(), w69.end(), std::ref(i32rng));
  std::generate(w70.begin(), w70.end(), std::ref(i8rng));
  std::generate(w71.begin(), w71.end(), std::ref(i32rng));
  std::generate(w72.begin(), w72.end(), std::ref(i8rng));
  std::generate(w73.begin(), w73.end(), std::ref(i32rng));
  std::generate(w74.begin(), w74.end(), std::ref(i8rng));
  std::generate(w75.begin(), w75.end(), std::ref(i32rng));
  std::generate(w76.begin(), w76.end(), std::ref(i8rng));
  std::generate(w77.begin(), w77.end(), std::ref(i32rng));
  std::generate(w78.begin(), w78.end(), std::ref(i8rng));
  std::generate(w79.begin(), w79.end(), std::ref(i32rng));
  std::generate(w80.begin(), w80.end(), std::ref(i8rng));
  std::generate(w81.begin(), w81.end(), std::ref(i32rng));
  std::generate(w82.begin(), w82.end(), std::ref(i8rng));
  std::generate(w83.begin(), w83.end(), std::ref(i32rng));
  std::generate(w84.begin(), w84.end(), std::ref(i8rng));
  std::generate(w85.begin(), w85.end(), std::ref(i32rng));

  ExecutionPlan operators;
  xnn_status status;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    3 /* input channels per group */,
    32 /* output_channels_per_group */,
    3 /* input pixel stride */,
    32 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w30.data(), w31.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    32 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    32 /* input pixel stride */,
    32 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w32.data(), w33.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    64 /* output_channels_per_group */,
    32 /* input pixel stride */,
    64 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w34.data(), w35.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    64 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    64 /* input pixel stride */,
    64 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w36.data(), w37.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    128 /* output_channels_per_group */,
    64 /* input pixel stride */,
    128 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w38.data(), w39.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    128 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    128 /* input pixel stride */,
    128 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w40.data(), w41.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    128 /* input channels per group */,
    128 /* output_channels_per_group */,
    128 /* input pixel stride */,
    128 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w42.data(), w43.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    128 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    128 /* input pixel stride */,
    128 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w44.data(), w45.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    128 /* input channels per group */,
    256 /* output_channels_per_group */,
    128 /* input pixel stride */,
    256 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w46.data(), w47.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    256 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    256 /* input pixel stride */,
    256 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w48.data(), w49.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    256 /* input channels per group */,
    256 /* output_channels_per_group */,
    256 /* input pixel stride */,
    256 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w50.data(), w51.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    256 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    256 /* input pixel stride */,
    256 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w52.data(), w53.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    256 /* input channels per group */,
    512 /* output_channels_per_group */,
    256 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w54.data(), w55.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w56.data(), w57.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    512 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w58.data(), w59.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w60.data(), w61.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    512 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w62.data(), w63.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w64.data(), w65.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    512 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w66.data(), w67.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w68.data(), w69.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    512 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w70.data(), w71.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w72.data(), w73.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    512 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w74.data(), w75.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w76.data(), w77.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    512 /* input channels per group */,
    1024 /* output_channels_per_group */,
    512 /* input pixel stride */,
    1024 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w78.data(), w79.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1024 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    1024 /* input pixel stride */,
    1024 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w80.data(), w81.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    1024 /* input channels per group */,
    1024 /* output_channels_per_group */,
    1024 /* input pixel stride */,
    1024 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w82.data(), w83.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qs8(
    1024 /* channels */, 1024 /* input stride */, 1024 /* output stride */,
    -1 /* input zero point */, 0.5f /* input scale */,
    -1 /* output zero point */, 0.5f /* output scale */,
    -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nhwc_qs8(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    1024 /* input channels per group */,
    1001 /* output_channels_per_group */,
    1024 /* input pixel stride */,
    1001 /* output pixel stride */,
    -1 /* input zero point */, 0.5f /* input scale */, 0.5f /* kernel scale */,
    w84.data(), w85.data(),
    -1 /* output zero point */, 0.5f /* output scale */, -126 /* output min */, 126 /* output max */,
    0 /* flags */,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);



  status = xnn_setup_convolution2d_nhwc_qs8(
    op0,
    1 /* batch size */, 224 /* input height */, 224 /* input width */,
    v0.data() /* input */, v1.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op1,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v1.data() /* input */, v2.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op2,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v2.data() /* input */, v3.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op3,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v3.data() /* input */, v4.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op4,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v4.data() /* input */, v5.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op5,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v5.data() /* input */, v6.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op6,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v6.data() /* input */, v7.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op7,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v7.data() /* input */, v8.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op8,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v8.data() /* input */, v9.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op9,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v9.data() /* input */, v10.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op10,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v10.data() /* input */, v11.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op11,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v11.data() /* input */, v12.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op12,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v12.data() /* input */, v13.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op13,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v13.data() /* input */, v14.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op14,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v14.data() /* input */, v15.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op15,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v15.data() /* input */, v16.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op16,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v16.data() /* input */, v17.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op17,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v17.data() /* input */, v18.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op18,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v18.data() /* input */, v19.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op19,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v19.data() /* input */, v20.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op20,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v20.data() /* input */, v21.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op21,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v21.data() /* input */, v22.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op22,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v22.data() /* input */, v23.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op23,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v23.data() /* input */, v24.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op24,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v24.data() /* input */, v25.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op25,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v25.data() /* input */, v26.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op26,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v26.data() /* input */, v27.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qs8(
    op27,
    1 /* batch size */, 49 /* width */,
    v27.data() /* input */, v28.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qs8(
    op28,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v28.data() /* input */, v29.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpessimizing-move"
  return operators;
  #pragma clang diagnostic pop
}

}  // namespace models
