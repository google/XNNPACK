// Copyright 2020-2025 Google LLC
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
#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include <benchmark/benchmark.h>

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) \
  (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

template <typename T, typename W = T, typename O = T>
xnn_subgraph_t FullyConnected(size_t batch_size, size_t m, size_t k, size_t n,
                              bool dynamically_quantize_lhs = false,
                              bool static_rhs = true) {
  const xnn_datatype datatype_in = xnn_datatype_of<T>();
  const xnn_datatype datatype_w = xnn_datatype_of<W>();
  const xnn_datatype datatype_out = xnn_datatype_of<O>();

  xnnpack::ReplicableRandomDevice rng;

  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/static_rhs ? 2 : 3, 0,
                               &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::array<size_t, 3> dims_in = {{batch_size, m, k}};
  std::array<size_t, 2> dims_w = {{n, k}};
  std::array<size_t, 3> dims_out = {{batch_size, m, n}};

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, datatype_in, dims_in.size(),
                                   dims_in.data(),
                                   /*data=*/nullptr, /*external_id=*/0,
                                   XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create input tensor" << std::endl;
    return nullptr;
  }

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, datatype_out, dims_out.size(), dims_out.data(),
      /*data=*/nullptr, /*external_id=*/1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create output tensor" << std::endl;
    return nullptr;
  }

  uint32_t weights_id = XNN_INVALID_VALUE_ID;
  if (static_rhs) {
    using Wunpacked = typename xnnpack::unwrap_quantized<W>::type;
    static std::vector<Wunpacked> w1_data;
    w1_data.resize(XNN_PAD_EXTRA_BYTES(k * n, Wunpacked));
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f),
                            std::ref(rng));
    std::generate(w1_data.begin(), w1_data.end(), std::ref(f32rng));
    if (xnn_datatype_is_channelwise_quantized(datatype_w)) {
      static std::vector<float> w1_scale;
      w1_scale.resize(n);
      std::fill(w1_scale.begin(), w1_scale.end(), 1.0f);
      status = xnn_define_channelwise_quantized_tensor_value(
          subgraph, datatype_w, /*scale=*/w1_scale.data(), dims_w.size(),
          /*channel_dim=*/0, dims_w.data(),
          /*data=*/w1_data.data(), /*external_id=*/XNN_INVALID_VALUE_ID,
          /*flags=*/0, &weights_id);
    } else if (xnn_datatype_is_quantized(datatype_w)) {
      status = xnn_define_quantized_tensor_value(
          subgraph, datatype_w, /*zero_point=*/0, /*scale=*/1.0f, dims_w.size(),
          dims_w.data(),
          /*data=*/w1_data.data(), /*external_id=*/XNN_INVALID_VALUE_ID,
          /*flags=*/0, &weights_id);
    } else {
      status = xnn_define_tensor_value(
          subgraph, datatype_w, dims_w.size(), dims_w.data(),
          /*data=*/w1_data.data(), XNN_INVALID_VALUE_ID, /*flags=*/0,
          &weights_id);
    }
  } else {
    status = xnn_define_tensor_value(
        subgraph, datatype_w, dims_w.size(), dims_w.data(),
        /*data=*/nullptr, /*external_id=*/2, XNN_VALUE_FLAG_EXTERNAL_INPUT,
        &weights_id);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to create weights tensor" << std::endl;
    return nullptr;
  }

  if (dynamically_quantize_lhs) {
    uint32_t quantized_input_id = XNN_INVALID_VALUE_ID;
    status = xnn_define_dynamically_quantized_tensor_value(
        subgraph, xnn_datatype_qdint8, /*num_dims=*/dims_in.size(),
        /*num_non_batch_dims=*/1, /*dims=*/dims_in.data(),
        /*external_id=*/XNN_INVALID_VALUE_ID,
        /*flags=*/0, &quantized_input_id);
    if (status != xnn_status_success) {
      std::cerr << "failed to create dynamically quantized input tensor "
                << std::endl;
      return nullptr;
    }

    status = xnn_define_unary(subgraph, xnn_unary_convert, /*params=*/nullptr,
                              input_id, /*output_id=*/quantized_input_id,
                              /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create create convert " << std::endl;
      return nullptr;
    }
    input_id = quantized_input_id;
  }

  status = xnn_define_fully_connected(
      subgraph,
      /*output_min=*/-std::numeric_limits<float>::infinity(),
      /*output_max=*/std::numeric_limits<float>::infinity(),
      /*input_id=*/input_id,
      /*filter_id=*/weights_id,
      /*bias_id=*/XNN_INVALID_VALUE_ID,
      /*output_id=*/output_id,
      /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models

static void FP32FullyConnected(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FullyConnected<float>(
        FLAGS_batch_size,
        /*m=*/state.range(0), /*k=*/state.range(1), /*n=*/state.range(2));
  });
}

static void FP16FullyConnected(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FullyConnected<xnn_float16>(
        FLAGS_batch_size,
        /*m=*/state.range(0), /*k=*/state.range(1), /*n=*/state.range(2));
  });
}

static void QD8FullyConnected(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FullyConnected<
        float, xnnpack::quantized<int8_t, xnnpack::channelwise>>(
        FLAGS_batch_size,
        /*m=*/state.range(0), /*k=*/state.range(1),
        /*n=*/state.range(2),
        /*dynamically_quantize_lhs=*/true);
  });
}

static void FullyConnectedArgs(benchmark::internal::Benchmark* b);

BENCHMARK(FP32FullyConnected)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(FullyConnectedArgs);

BENCHMARK(FP16FullyConnected)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(FullyConnectedArgs);

BENCHMARK(QD8FullyConnected)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(FullyConnectedArgs);

static void FullyConnectedArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "K", "N"});

  static const std::array<int64_t, 17> kDims = {
      1,   2,   4,    8,    16,   32,   64,    128,
      256, 512, 1024, 2048, 4096, 8192, 16384, 65536};
  const int64_t kMinK = 8;
  const int64_t kMaxSmall = 16;
  const int64_t kMinHuge = 1024;
  const int64_t kMinFLOPs = (int64_t)1 << 16;
  const int64_t kMaxFLOPs = (int64_t)1 << 30;

  for (int64_t m : kDims) {
    for (int64_t k : kDims) {
      if (k < kMinK) {
        continue;
      }
      for (int64_t n : kDims) {
        const int num_small = static_cast<int>(m <= kMaxSmall) +
                              static_cast<int>(k <= kMaxSmall) +
                              static_cast<int>(n <= kMaxSmall);
        const int num_huge = static_cast<int>(m >= kMinHuge) +
                             static_cast<int>(k >= kMinHuge) +
                             static_cast<int>(n >= kMinHuge);
        const int64_t flops = m * k * n;
        if (num_small < 2 && num_huge < 2 && flops >= kMinFLOPs &&
            flops <= kMaxFLOPs) {
          b->Args({m, k, n});
        }
      }
    }
  }
}
