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

#include <xnnpack/cache.h>
#include <xnnpack/common.h>
#include <xnnpack/models.h>

#include <fp16/fp16.h>

namespace models {

ExecutionPlan FP16SparseMobileNetV1(float sparsity, pthreadpool_t threadpool) {
  alignas(16) static std::array<uint16_t, 150528 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v0;
  alignas(16) static std::array<uint16_t, 401408 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v1;
  alignas(16) static std::array<uint16_t, 401408 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v2;
  alignas(16) static std::array<uint16_t, 802816 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v3;
  alignas(16) static std::array<uint16_t, 200704 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v4;
  alignas(16) static std::array<uint16_t, 401408 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v5;
  alignas(16) static std::array<uint16_t, 401408 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v6;
  alignas(16) static std::array<uint16_t, 401408 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v7;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v8;
  alignas(16) static std::array<uint16_t, 200704 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v9;
  alignas(16) static std::array<uint16_t, 200704 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v10;
  alignas(16) static std::array<uint16_t, 200704 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v11;
  alignas(16) static std::array<uint16_t, 50176 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v12;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v13;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v14;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v15;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v16;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v17;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v18;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v19;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v20;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v21;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v22;
  alignas(16) static std::array<uint16_t, 100352 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v23;
  alignas(16) static std::array<uint16_t, 25088 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v24;
  alignas(16) static std::array<uint16_t, 50176 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v25;
  alignas(16) static std::array<uint16_t, 50176 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v26;
  alignas(16) static std::array<uint16_t, 50176 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v27;
  alignas(16) static std::array<uint16_t, 1024 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v28;
  alignas(16) static std::array<uint16_t, 1001 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v29;
  alignas(16) static std::array<uint16_t, 864 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w30;
  alignas(16) static std::array<uint16_t, 32 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w31;
  alignas(16) static std::array<uint16_t, 288 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w32;
  alignas(16) static std::array<uint16_t, 32 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w33;
  alignas(16) static std::array<uint16_t, 2048 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w34;
  alignas(16) static std::array<uint16_t, 64 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w35;
  alignas(16) static std::array<uint16_t, 576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w36;
  alignas(16) static std::array<uint16_t, 64 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w37;
  alignas(16) static std::array<uint16_t, 8192 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w38;
  alignas(16) static std::array<uint16_t, 128 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w39;
  alignas(16) static std::array<uint16_t, 1152 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w40;
  alignas(16) static std::array<uint16_t, 128 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w41;
  alignas(16) static std::array<uint16_t, 16384 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w42;
  alignas(16) static std::array<uint16_t, 128 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w43;
  alignas(16) static std::array<uint16_t, 1152 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w44;
  alignas(16) static std::array<uint16_t, 128 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w45;
  alignas(16) static std::array<uint16_t, 32768 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w46;
  alignas(16) static std::array<uint16_t, 256 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w47;
  alignas(16) static std::array<uint16_t, 2304 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w48;
  alignas(16) static std::array<uint16_t, 256 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w49;
  alignas(16) static std::array<uint16_t, 65536 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w50;
  alignas(16) static std::array<uint16_t, 256 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w51;
  alignas(16) static std::array<uint16_t, 2304 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w52;
  alignas(16) static std::array<uint16_t, 256 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w53;
  alignas(16) static std::array<uint16_t, 131072 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w54;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w55;
  alignas(16) static std::array<uint16_t, 4608 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w56;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w57;
  alignas(16) static std::array<uint16_t, 262144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w58;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w59;
  alignas(16) static std::array<uint16_t, 4608 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w60;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w61;
  alignas(16) static std::array<uint16_t, 262144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w62;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w63;
  alignas(16) static std::array<uint16_t, 4608 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w64;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w65;
  alignas(16) static std::array<uint16_t, 262144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w66;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w67;
  alignas(16) static std::array<uint16_t, 4608 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w68;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w69;
  alignas(16) static std::array<uint16_t, 262144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w70;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w71;
  alignas(16) static std::array<uint16_t, 4608 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w72;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w73;
  alignas(16) static std::array<uint16_t, 262144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w74;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w75;
  alignas(16) static std::array<uint16_t, 4608 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w76;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w77;
  alignas(16) static std::array<uint16_t, 524288 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w78;
  alignas(16) static std::array<uint16_t, 1024 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w79;
  alignas(16) static std::array<uint16_t, 9216 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w80;
  alignas(16) static std::array<uint16_t, 1024 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w81;
  alignas(16) static std::array<uint16_t, 1048576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w82;
  alignas(16) static std::array<uint16_t, 1024 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w83;
  alignas(16) static std::array<uint16_t, 1025024 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w84;
  alignas(16) static std::array<uint16_t, 1001 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w85;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);
  std::generate(v0.begin(), v0.end(), std::ref(f16rng));
  std::generate(v1.begin(), v1.end(), std::ref(f16rng));
  std::generate(v2.begin(), v2.end(), std::ref(f16rng));
  std::generate(v3.begin(), v3.end(), std::ref(f16rng));
  std::generate(v4.begin(), v4.end(), std::ref(f16rng));
  std::generate(v5.begin(), v5.end(), std::ref(f16rng));
  std::generate(v6.begin(), v6.end(), std::ref(f16rng));
  std::generate(v7.begin(), v7.end(), std::ref(f16rng));
  std::generate(v8.begin(), v8.end(), std::ref(f16rng));
  std::generate(v9.begin(), v9.end(), std::ref(f16rng));
  std::generate(v10.begin(), v10.end(), std::ref(f16rng));
  std::generate(v11.begin(), v11.end(), std::ref(f16rng));
  std::generate(v12.begin(), v12.end(), std::ref(f16rng));
  std::generate(v13.begin(), v13.end(), std::ref(f16rng));
  std::generate(v14.begin(), v14.end(), std::ref(f16rng));
  std::generate(v15.begin(), v15.end(), std::ref(f16rng));
  std::generate(v16.begin(), v16.end(), std::ref(f16rng));
  std::generate(v17.begin(), v17.end(), std::ref(f16rng));
  std::generate(v18.begin(), v18.end(), std::ref(f16rng));
  std::generate(v19.begin(), v19.end(), std::ref(f16rng));
  std::generate(v20.begin(), v20.end(), std::ref(f16rng));
  std::generate(v21.begin(), v21.end(), std::ref(f16rng));
  std::generate(v22.begin(), v22.end(), std::ref(f16rng));
  std::generate(v23.begin(), v23.end(), std::ref(f16rng));
  std::generate(v24.begin(), v24.end(), std::ref(f16rng));
  std::generate(v25.begin(), v25.end(), std::ref(f16rng));
  std::generate(v26.begin(), v26.end(), std::ref(f16rng));
  std::generate(v27.begin(), v27.end(), std::ref(f16rng));
  std::generate(v28.begin(), v28.end(), std::ref(f16rng));
  std::generate(v29.begin(), v29.end(), std::ref(f16rng));
  std::generate(w30.begin(), w30.end(), std::ref(f16rng));
  std::generate(w31.begin(), w31.end(), std::ref(f16rng));
  std::generate(w32.begin(), w32.end(), std::ref(f16rng));
  std::generate(w33.begin(), w33.end(), std::ref(f16rng));
  std::fill(w34.begin(), w34.end(), 0);
  std::generate(w34.begin(), w34.end() - size_t(sparsity * w34.size()), std::ref(f16rng));
  std::shuffle(w34.begin(), w34.end(), rng);
  std::generate(w35.begin(), w35.end(), std::ref(f16rng));
  std::generate(w36.begin(), w36.end(), std::ref(f16rng));
  std::generate(w37.begin(), w37.end(), std::ref(f16rng));
  std::fill(w38.begin(), w38.end(), 0);
  std::generate(w38.begin(), w38.end() - size_t(sparsity * w38.size()), std::ref(f16rng));
  std::shuffle(w38.begin(), w38.end(), rng);
  std::generate(w39.begin(), w39.end(), std::ref(f16rng));
  std::generate(w40.begin(), w40.end(), std::ref(f16rng));
  std::generate(w41.begin(), w41.end(), std::ref(f16rng));
  std::fill(w42.begin(), w42.end(), 0);
  std::generate(w42.begin(), w42.end() - size_t(sparsity * w42.size()), std::ref(f16rng));
  std::shuffle(w42.begin(), w42.end(), rng);
  std::generate(w43.begin(), w43.end(), std::ref(f16rng));
  std::generate(w44.begin(), w44.end(), std::ref(f16rng));
  std::generate(w45.begin(), w45.end(), std::ref(f16rng));
  std::fill(w46.begin(), w46.end(), 0);
  std::generate(w46.begin(), w46.end() - size_t(sparsity * w46.size()), std::ref(f16rng));
  std::shuffle(w46.begin(), w46.end(), rng);
  std::generate(w47.begin(), w47.end(), std::ref(f16rng));
  std::generate(w48.begin(), w48.end(), std::ref(f16rng));
  std::generate(w49.begin(), w49.end(), std::ref(f16rng));
  std::fill(w50.begin(), w50.end(), 0);
  std::generate(w50.begin(), w50.end() - size_t(sparsity * w50.size()), std::ref(f16rng));
  std::shuffle(w50.begin(), w50.end(), rng);
  std::generate(w51.begin(), w51.end(), std::ref(f16rng));
  std::generate(w52.begin(), w52.end(), std::ref(f16rng));
  std::generate(w53.begin(), w53.end(), std::ref(f16rng));
  std::fill(w54.begin(), w54.end(), 0);
  std::generate(w54.begin(), w54.end() - size_t(sparsity * w54.size()), std::ref(f16rng));
  std::shuffle(w54.begin(), w54.end(), rng);
  std::generate(w55.begin(), w55.end(), std::ref(f16rng));
  std::generate(w56.begin(), w56.end(), std::ref(f16rng));
  std::generate(w57.begin(), w57.end(), std::ref(f16rng));
  std::fill(w58.begin(), w58.end(), 0);
  std::generate(w58.begin(), w58.end() - size_t(sparsity * w58.size()), std::ref(f16rng));
  std::shuffle(w58.begin(), w58.end(), rng);
  std::generate(w59.begin(), w59.end(), std::ref(f16rng));
  std::generate(w60.begin(), w60.end(), std::ref(f16rng));
  std::generate(w61.begin(), w61.end(), std::ref(f16rng));
  std::fill(w62.begin(), w62.end(), 0);
  std::generate(w62.begin(), w62.end() - size_t(sparsity * w62.size()), std::ref(f16rng));
  std::shuffle(w62.begin(), w62.end(), rng);
  std::generate(w63.begin(), w63.end(), std::ref(f16rng));
  std::generate(w64.begin(), w64.end(), std::ref(f16rng));
  std::generate(w65.begin(), w65.end(), std::ref(f16rng));
  std::fill(w66.begin(), w66.end(), 0);
  std::generate(w66.begin(), w66.end() - size_t(sparsity * w66.size()), std::ref(f16rng));
  std::shuffle(w66.begin(), w66.end(), rng);
  std::generate(w67.begin(), w67.end(), std::ref(f16rng));
  std::generate(w68.begin(), w68.end(), std::ref(f16rng));
  std::generate(w69.begin(), w69.end(), std::ref(f16rng));
  std::fill(w70.begin(), w70.end(), 0);
  std::generate(w70.begin(), w70.end() - size_t(sparsity * w70.size()), std::ref(f16rng));
  std::shuffle(w70.begin(), w70.end(), rng);
  std::generate(w71.begin(), w71.end(), std::ref(f16rng));
  std::generate(w72.begin(), w72.end(), std::ref(f16rng));
  std::generate(w73.begin(), w73.end(), std::ref(f16rng));
  std::fill(w74.begin(), w74.end(), 0);
  std::generate(w74.begin(), w74.end() - size_t(sparsity * w74.size()), std::ref(f16rng));
  std::shuffle(w74.begin(), w74.end(), rng);
  std::generate(w75.begin(), w75.end(), std::ref(f16rng));
  std::generate(w76.begin(), w76.end(), std::ref(f16rng));
  std::generate(w77.begin(), w77.end(), std::ref(f16rng));
  std::fill(w78.begin(), w78.end(), 0);
  std::generate(w78.begin(), w78.end() - size_t(sparsity * w78.size()), std::ref(f16rng));
  std::shuffle(w78.begin(), w78.end(), rng);
  std::generate(w79.begin(), w79.end(), std::ref(f16rng));
  std::generate(w80.begin(), w80.end(), std::ref(f16rng));
  std::generate(w81.begin(), w81.end(), std::ref(f16rng));
  std::fill(w82.begin(), w82.end(), 0);
  std::generate(w82.begin(), w82.end() - size_t(sparsity * w82.size()), std::ref(f16rng));
  std::shuffle(w82.begin(), w82.end(), rng);
  std::generate(w83.begin(), w83.end(), std::ref(f16rng));
  std::generate(w84.begin(), w84.end(), std::ref(f16rng));
  std::generate(w85.begin(), w85.end(), std::ref(f16rng));

  Operators operators;
  xnn_status status;
  size_t max_workspace_size = 0;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    3 /* input channels per group */,
    32 /* output_channels_per_group */,
    3 /* input pixel stride */,
    32 /* output pixel stride */,
    w30.data(), w31.data(),
    0.0f /* output min */, 6.0f /* output max */,
    XNN_FLAG_INPUT_NHWC /* flags */,
    nullptr,
    nullptr,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w32.data(), w33.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w34.data(), w35.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    64 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    64 /* input pixel stride */,
    64 /* output pixel stride */,
    w36.data(), w37.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w38.data(), w39.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w40.data(), w41.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w42.data(), w43.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    128 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    128 /* input pixel stride */,
    128 /* output pixel stride */,
    w44.data(), w45.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w46.data(), w47.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w48.data(), w49.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w50.data(), w51.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    256 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    256 /* input pixel stride */,
    256 /* output pixel stride */,
    w52.data(), w53.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w54.data(), w55.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w56.data(), w57.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w58.data(), w59.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w60.data(), w61.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w62.data(), w63.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w64.data(), w65.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w66.data(), w67.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w68.data(), w69.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w70.data(), w71.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w72.data(), w73.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w74.data(), w75.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    512 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    512 /* input pixel stride */,
    512 /* output pixel stride */,
    w76.data(), w77.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w78.data(), w79.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w80.data(), w81.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nchw_f16(
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
    w82.data(), w83.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f16(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
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
    w84.data(), w85.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    nullptr,
    nullptr,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  status = xnn_reshape_convolution2d_nchw_f16(
    op0,
    /*batch_size=*/1, /*input_height=*/224, /*input_width=*/224,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op1,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op2,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op3,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op4,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op5,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op6,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op7,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op8,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op9,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op10,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op11,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op12,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op13,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op14,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op15,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op16,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op17,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op18,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op19,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op20,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op21,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op22,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op23,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op24,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op25,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f16(
    op26,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f16(
    op27,
    /*batch_size=*/1, 49 /* width */,
    1024 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #27" << std::endl;
    return ExecutionPlan();
  }

  size_t op28_workspace_size = 0;
  size_t op28_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f16(
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

  status = xnn_setup_convolution2d_nchw_f16(
    op0,
    /*input=*/v0.data(), /*output=*/v1.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op1,
    /*input=*/v1.data(), /*output=*/v2.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op2,
    /*input=*/v2.data(), /*output=*/v3.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op3,
    /*input=*/v3.data(), /*output=*/v4.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op4,
    /*input=*/v4.data(), /*output=*/v5.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op5,
    /*input=*/v5.data(), /*output=*/v6.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op6,
    /*input=*/v6.data(), /*output=*/v7.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op7,
    /*input=*/v7.data(), /*output=*/v8.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op8,
    /*input=*/v8.data(), /*output=*/v9.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op9,
    /*input=*/v9.data(), /*output=*/v10.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op10,
    /*input=*/v10.data(), /*output=*/v11.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op11,
    /*input=*/v11.data(), /*output=*/v12.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op12,
    /*input=*/v12.data(), /*output=*/v13.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op13,
    /*input=*/v13.data(), /*output=*/v14.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op14,
    /*input=*/v14.data(), /*output=*/v15.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op15,
    /*input=*/v15.data(), /*output=*/v16.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op16,
    /*input=*/v16.data(), /*output=*/v17.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op17,
    /*input=*/v17.data(), /*output=*/v18.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op18,
    /*input=*/v18.data(), /*output=*/v19.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op19,
    /*input=*/v19.data(), /*output=*/v20.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op20,
    /*input=*/v20.data(), /*output=*/v21.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op21,
    /*input=*/v21.data(), /*output=*/v22.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op22,
    /*input=*/v22.data(), /*output=*/v23.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op23,
    /*input=*/v23.data(), /*output=*/v24.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op24,
    /*input=*/v24.data(), /*output=*/v25.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op25,
    /*input=*/v25.data(), /*output=*/v26.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f16(
    op26,
    /*input=*/v26.data(), /*output=*/v27.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f16(
    op27,
    /*input=*/v27.data(), /*output=*/v28.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op28,
    workspace.data(),
    /*input=*/v28.data(), /*output=*/v29.data());
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
