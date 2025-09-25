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
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  xnnpack::ReplicableRandomDevice rng;

  uint32_t v0 = 0;
  std::array<size_t, 4> v0_dims = {{1, h, w, ci}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v0_dims.size(), v0_dims.data(),
      /*data=*/nullptr, v0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = 1;
  std::array<size_t, 4> v1_dims = {{1, h - 2, w - 2, co}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v1_dims.size(), v1_dims.data(),
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
      subgraph, xnn_datatype_fp32, w0_dims.size(), w0_dims.data(),
      /*data=*/weights.w0.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w0" << std::endl;
    return nullptr;
  }

  uint32_t w1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w1_dims = {{co}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w1_dims.size(), w1_dims.data(),
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
      subgraph,
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

  return subgraph;
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

static void ConvolutionArguments(benchmark::internal::Benchmark* b) {
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
