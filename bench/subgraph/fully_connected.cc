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

#include "bench/subgraph/model_runtime.h"
#include "bench/utils.h"
#include "include/xnnpack.h"
#include "test/next_prime.h"
#include <benchmark/benchmark.h>

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) \
  (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

xnn_subgraph_t FP32FullyConnected(size_t batch_size, size_t input_channels,
                                  size_t output_channels) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::random_device random_device;  // NOLINT(runtime/random_device)
  auto rng = std::mt19937(random_device());

  uint32_t v0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> v0_dims = {{batch_size, input_channels}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v0_dims.size(), v0_dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v38 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> v38_dims = {{batch_size, output_channels}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v38_dims.size(), v38_dims.data(),
      /*data=*/nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v38" << std::endl;
    return nullptr;
  }

  static std::vector<float> w42_data;
  w42_data.resize(XNN_PAD_EXTRA_BYTES(input_channels * output_channels, float));
  uint32_t w42 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> w42_dims = {{output_channels, input_channels}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, w42_dims.size(), w42_dims.data(),
      /*data=*/w42_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w42" << std::endl;
    return nullptr;
  }

  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f),
                          std::ref(rng));
  std::generate(w42_data.begin(), w42_data.end(), std::ref(f32rng));

  status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(),
      /*input_id=*/v0,
      /*filter_id=*/w42,
      /*bias_id=*/XNN_INVALID_VALUE_ID,
      /*output_id=*/v38,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #6" << std::endl;
    return nullptr;
  }

  return subgraph;
}  // NOLINT(readability/fn_size)

xnn_subgraph_t QD8FullyConnected(size_t batch_size, size_t input_channels,
                                 size_t output_channels) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::random_device random_device;  // NOLINT(runtime/random_device)
  auto rng = std::mt19937(random_device());

  uint32_t v0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> v0_dims = {{batch_size, input_channels}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v0_dims.size(), v0_dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> v1_dims = {{batch_size, input_channels}};
  status = xnn_define_dynamically_quantized_tensor_value(
      subgraph, xnn_datatype_qdint8, /*num_dims=*/v1_dims.size(),
      /*num_non_batch_dims=*/1, /*dims=*/v1_dims.data(),
      /*external_id=*/XNN_INVALID_VALUE_ID,
      /*flags=*/0, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  uint32_t v38 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> v38_dims = {{batch_size, output_channels}};
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, v38_dims.size(), v38_dims.data(),
      /*data=*/nullptr, 1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v38" << std::endl;
    return nullptr;
  }

  static std::vector<int8_t> w42_data;
  w42_data.resize(
      XNN_PAD_EXTRA_BYTES(input_channels * output_channels, int8_t));
  uint32_t w42 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> w42_dims = {{output_channels, input_channels}};
  static std::vector<float> w42_scale;
  w42_scale.resize(output_channels);
  {
    auto scalerng = std::bind(
        std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w42_scale.begin(), w42_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
      subgraph, xnn_datatype_qcint8,
      /*scale=*/w42_scale.data(), w42_dims.size(), 0, w42_dims.data(),
      /*data=*/w42_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0, &w42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w42" << std::endl;
    return nullptr;
  }

  auto qc8rng = std::bind(
      std::uniform_int_distribution<int>(std::numeric_limits<int8_t>::min(),
                                         std::numeric_limits<int8_t>::max()),
      std::ref(rng));
  std::generate(w42_data.begin(), w42_data.end(), std::ref(qc8rng));

  status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                            /*input_id=*/v0, /*output_id=*/v1,
                            /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create create convert " << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(),
      /*input_id=*/v1,
      /*filter_id=*/w42,
      /*bias_id=*/XNN_INVALID_VALUE_ID,
      /*output_id=*/v38,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #6" << std::endl;
    return nullptr;
  }

  return subgraph;
}  // NOLINT(readability/fn_size)

}  // namespace models

static void QD8FullyConnected(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state]() {
    return models::QD8FullyConnected(
        /*batch_size=*/state.range(0),
        /*input_channels=*/state.range(2), /*output_channels=*/state.range(1));
  });
}

static void FP32FullyConnected(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(state, [&state]() {
    return models::FP32FullyConnected(
        /*batch_size=*/state.range(0),
        /*input_channels=*/state.range(2), /*output_channels=*/state.range(1));
  });
}

inline void FullyConnectedArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "K", "N"});

  // Set a reasonable minimum/maximum size for the products we want to test.
  const uint64_t max_dim_size = 10 * 1000;
  const uint64_t min_flops = 100 * 1000;
  const uint64_t max_flops = 10ULL * 1000 * 1000 * 1000;  // m * k * n.

  // Loop over the matrix dimensions.
  for (uint64_t m = 1; m < max_dim_size; m = xnnpack::NextPrime(4 * m)) {
    for (uint64_t k = 16; k < max_dim_size && m * k < max_flops;
         k = xnnpack::NextPrime(4 * k)) {
      for (uint64_t n = 10; n < max_dim_size && m * k * n < max_flops;
           n = xnnpack::NextPrime(4 * n)) {
        if (m * k * n < min_flops) {
          continue;
        }
        b->Args(
            {static_cast<int>(m), static_cast<int>(k), static_cast<int>(n)});
      }
    }
  }
}

static void FP16FullyConnected(benchmark::State& state) {
  xnnpack::ModelRuntime::BenchmarkInvoke(
      state,
      [&state]() {
        return models::FP32FullyConnected(
            /*batch_size=*/state.range(0),
            /*input_channels=*/state.range(2),
            /*output_channels=*/state.range(1));
      },
      XNN_FLAG_FORCE_FP16_INFERENCE);
}

BENCHMARK(QD8FullyConnected)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(FullyConnectedArguments);

BENCHMARK(FP32FullyConnected)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(FullyConnectedArguments);

BENCHMARK(FP16FullyConnected)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Apply(FullyConnectedArguments);

XNN_BENCHMARK_MAIN();
