// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "bench/subgraph/benchmark.h"
#include "include/xnnpack.h"
#include "test/replicable_random_device.h"
#include <benchmark/benchmark.h>

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) \
  (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

struct Weights {
  std::vector<float> w0;
  std::vector<float> w1;
};

// Depthwise convolution with kernel size kw x kw, followed by a 1x1 conv with
// ci -> co channels. This is a common pattern in imaging models.
xnn_subgraph_t FP32Convolution(size_t w, size_t h, size_t kw, size_t ci,
                               size_t co, uint32_t op_flags,
                               Weights& weights) {
  xnn_status status;
  auto subgraph = xnnpack::CreateUniqueSubgraph(/*num_external_values=*/2, 0);
  if (!subgraph) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  xnnpack::ReplicableRandomDevice rng;

  uint32_t v0 = 0;
  std::array<size_t, 4> v0_dims = {{1, h, w, ci}};
  status = xnn_define_tensor_value(
      subgraph.get(), xnn_datatype_fp32, v0_dims.size(), v0_dims.data(),
      /*data=*/nullptr, v0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = 1;
  std::array<size_t, 4> v1_dims = {{1, h - 2, w - 2, co}};
  status = xnn_define_tensor_value(
      subgraph.get(), xnn_datatype_fp32, v1_dims.size(), v1_dims.data(),
      /*data=*/nullptr, v1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  weights.w0 =
      std::vector<float>(co * kw * kw * ci + XNN_EXTRA_BYTES / sizeof(float));
  weights.w1 = std::vector<float>(co + XNN_EXTRA_BYTES / sizeof(float));

  uint32_t w0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w0_dims = {{co, kw, kw, ci}};
  status = xnn_define_tensor_value(
      subgraph.get(), xnn_datatype_fp32, w0_dims.size(), w0_dims.data(),
      /*data=*/weights.w0.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w0" << std::endl;
    return nullptr;
  }

  uint32_t w1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w1_dims = {{co}};
  status = xnn_define_tensor_value(
      subgraph.get(), xnn_datatype_fp32, w1_dims.size(), w1_dims.data(),
      /*data=*/weights.w1.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w1" << std::endl;
    return nullptr;
  }

  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f),
                          std::ref(rng));
  std::generate(weights.w0.begin(), weights.w0.end(), std::ref(f32rng));
  std::generate(weights.w1.begin(), weights.w1.end(), std::ref(f32rng));

  status = xnn_define_convolution_2d(
      subgraph.get(),
      /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0,
      /*padding_left=*/0,
      /*kernel_height=*/kw, /*kernel_width=*/kw,
      /*subsampling_height=*/1, /*subsampling_width=*/1,
      /*dilation_height=*/1, /*dilation_width=*/1,
      /*groups=*/1,
      /*group_input_channels=*/ci,
      /*group_output_channels=*/co,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(), v0, w0, w1, v1,
      op_flags);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #2" << std::endl;
    return nullptr;
  }

  return subgraph.release();
}

struct QS8Weights {
  std::vector<int8_t> w0;
  std::vector<float> w0_scale;
  std::vector<int32_t> w1;
  std::vector<float> w1_scale;
};

xnn_subgraph_t QS8Convolution(size_t n, size_t h, size_t w, size_t kh,
                              size_t kw, size_t ph, size_t pw, size_t s,
                              size_t d, size_t g, size_t g_cin, size_t g_cout,
                              QS8Weights& weights) {
  xnn_status status;
  auto subgraph = xnnpack::CreateUniqueSubgraph(/*num_external_values=*/2, 0);
  if (!subgraph) {
    std::cerr << "failed to create subgraph" << std::endl;
    return nullptr;
  }

  xnnpack::ReplicableRandomDevice rng;
  const size_t effective_kh = (kh - 1) * d + 1;
  const size_t effective_kw = (kw - 1) * d + 1;
  const size_t pad_left = pw / 2;
  const size_t pad_top = ph / 2;
  const size_t pad_right = pw - pad_left;
  const size_t pad_bottom = ph - pad_top;
  const size_t out_h = (h + ph - effective_kh) / s + 1;
  const size_t out_w = (w + pw - effective_kw) / s + 1;
  const size_t cin = g * g_cin;
  const size_t cout = g * g_cout;

  const float input_scale = 0.05f;
  const float output_scale = 0.5f;

  uint32_t v0 = 0;
  std::array<size_t, 4> v0_dims = {{n, h, w, cin}};
  status = xnn_define_quantized_tensor_value(
      subgraph.get(), xnn_datatype_qint8, /*zero_point=*/-10, input_scale,
      v0_dims.size(), v0_dims.data(), /*data=*/nullptr, v0,
      XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = 1;
  std::array<size_t, 4> v1_dims = {{n, out_h, out_w, cout}};
  status = xnn_define_quantized_tensor_value(
      subgraph.get(), xnn_datatype_qint8, /*zero_point=*/5, output_scale,
      v1_dims.size(), v1_dims.data(), /*data=*/nullptr, v1,
      XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  weights.w0.resize(XNN_PAD_EXTRA_BYTES(cout * kh * kw * g_cin, int8_t));
  weights.w0_scale.resize(cout);
  weights.w1.resize(XNN_PAD_EXTRA_BYTES(cout, int32_t));
  weights.w1_scale.resize(cout);

  auto i8rng = std::bind(std::uniform_int_distribution<int>(-127, 127),
                         std::ref(rng));
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-1000, 1000),
                          std::ref(rng));
  auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 0.1f),
                            std::ref(rng));
  std::generate(weights.w0.begin(), weights.w0.end(),
                [&]() { return static_cast<int8_t>(i8rng()); });
  std::generate(weights.w1.begin(), weights.w1.end(), std::ref(i32rng));
  for (size_t c = 0; c < cout; ++c) {
    weights.w0_scale[c] = scalerng();
    weights.w1_scale[c] = input_scale * weights.w0_scale[c];
  }

  uint32_t w0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w0_dims = {{cout, kh, kw, g_cin}};
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph.get(), xnn_datatype_qcint8, weights.w0_scale.data(),
      w0_dims.size(), /*channel_dim=*/0, w0_dims.data(), weights.w0.data(),
      XNN_INVALID_VALUE_ID, /*flags=*/0, &w0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w0" << std::endl;
    return nullptr;
  }

  uint32_t w1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w1_dims = {{cout}};
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph.get(), xnn_datatype_qcint32, weights.w1_scale.data(),
      w1_dims.size(), /*channel_dim=*/0, w1_dims.data(), weights.w1.data(),
      XNN_INVALID_VALUE_ID, /*flags=*/0, &w1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w1" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
      subgraph.get(), pad_top, pad_right, pad_bottom, pad_left, kh, kw, s, s, d,
      d, g, g_cin, g_cout, -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::infinity(), v0, w0, w1, v1, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create convolution_2d node" << std::endl;
    return nullptr;
  }

  return subgraph.release();
}

}  // namespace models

static void FP32Convolution(benchmark::State& state) {
  models::Weights weights;
  xnnpack::RunBenchmark(state, [&state, &weights]() {
    uint32_t op_flags = 0;
    if (state.range(5)) {
      op_flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
    }
    return models::FP32Convolution(state.range(0), state.range(1),
                                   state.range(2), state.range(3),
                                   state.range(4), op_flags, weights);
  });
}

static void ConvolutionArguments(benchmark::Benchmark* b) {
  b->ArgNames({"W", "H", "KW", "CI", "CO", "IB"});

  for (bool transient_indirection_buffer : {false, true}) {
    // Mobilenet v1-ish
    b->Args({112, 112, 3, 32, 16, transient_indirection_buffer});
    b->Args({56, 56, 3, 96, 24, transient_indirection_buffer});
    b->Args({28, 28, 3, 144, 32, transient_indirection_buffer});
    b->Args({14, 14, 3, 192, 64, transient_indirection_buffer});
    b->Args({14, 14, 3, 384, 96, transient_indirection_buffer});
    b->Args({14, 14, 3, 576, 160, transient_indirection_buffer});
    b->Args({7, 7, 3, 960, 320, transient_indirection_buffer});

    // Bigger
    b->Args({512, 512, 3, 128, 128, transient_indirection_buffer});
  }
}

BENCHMARK(FP32Convolution)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(ConvolutionArguments);

static void QS8Convolution(benchmark::State& state) {
  models::QS8Weights weights;
  xnnpack::RunBenchmark(state, [&state, &weights]() {
    return models::QS8Convolution(
        state.range(0), state.range(1), state.range(2), state.range(3),
        state.range(4), state.range(5), state.range(6), state.range(7),
        state.range(8), state.range(9), state.range(10), state.range(11),
        weights);
  });
}

static void ResNet50Arguments(benchmark::Benchmark* b) {
  b->ArgNames(
      {"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});
  b->Args({1, 224, 224, 7, 7, 6, 6, 2, 1, 1, 3, 64});
  b->Args({1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 64, 64});
  b->Args({1, 56, 56, 3, 3, 2, 2, 1, 1, 1, 64, 64});
  b->Args({1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 64, 256});
  b->Args({1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 256, 64});
  b->Args({1, 56, 56, 1, 1, 0, 0, 1, 1, 1, 256, 128});
  b->Args({1, 56, 56, 3, 3, 2, 2, 2, 1, 1, 128, 128});
  b->Args({1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 128, 512});
  b->Args({1, 56, 56, 1, 1, 0, 0, 2, 1, 1, 256, 512});
  b->Args({1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 512, 128});
  b->Args({1, 28, 28, 3, 3, 2, 2, 1, 1, 1, 128, 128});
  b->Args({1, 28, 28, 1, 1, 0, 0, 1, 1, 1, 512, 256});
  b->Args({1, 28, 28, 3, 3, 2, 2, 2, 1, 1, 256, 256});
  b->Args({1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 256, 1024});
  b->Args({1, 28, 28, 1, 1, 0, 0, 2, 1, 1, 512, 1024});
  b->Args({1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1024, 256});
  b->Args({1, 14, 14, 3, 3, 2, 2, 1, 1, 1, 256, 256});
  b->Args({1, 14, 14, 1, 1, 0, 0, 1, 1, 1, 1024, 512});
  b->Args({1, 14, 14, 3, 3, 2, 2, 2, 1, 1, 512, 512});
  b->Args({1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 512, 2048});
  b->Args({1, 14, 14, 1, 1, 0, 0, 2, 1, 1, 1024, 2048});
  b->Args({1, 7, 7, 1, 1, 0, 0, 1, 1, 1, 2048, 512});
  b->Args({1, 7, 7, 3, 3, 2, 2, 1, 1, 1, 512, 512});
}

BENCHMARK(QS8Convolution)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(ResNet50Arguments);
