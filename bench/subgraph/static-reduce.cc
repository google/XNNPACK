// Copyright 2020-2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "bench/subgraph/benchmark.h"
#include "include/xnnpack.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/operator-utils.h"
#include <benchmark/benchmark.h>

namespace models {

enum custom_reduce_operator {
  custom_reduce_mean = xnn_reduce_mean,
  custom_reduce_max = xnn_reduce_max,
  custom_reduce_sum = xnn_reduce_sum,
  custom_reduce_sum_squared = xnn_reduce_sum_squared,

  custom_reduce_sum_square_differences = 100,
  custom_reduce_sum_absolute_differences,
};

template <typename T>
xnn_subgraph_t StaticReduce(custom_reduce_operator op_type,
                            const std::vector<size_t>& dims,
                            const std::vector<size_t>& axes) {
  const xnn_datatype datatype = xnn_datatype_of<T>();

  bool is_binary = (op_type == custom_reduce_sum_square_differences) ||
                   (op_type == custom_reduce_sum_absolute_differences);

  xnn_status status;
  auto subgraph = xnnpack::CreateUniqueSubgraph(is_binary ? 3 : 2, 0);
  if (!subgraph) {
    std::cerr << "failed to create subgraph" << std::endl;
    return nullptr;
  }

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), datatype, dims.size(),
                                   dims.data(),
                                   /*data=*/nullptr, /*external_id=*/0,
                                   XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create input tensor" << std::endl;
    return nullptr;
  }

  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  if (is_binary) {
    status = xnn_define_tensor_value(subgraph.get(), datatype, dims.size(),
                                     dims.data(),
                                     /*data=*/nullptr, /*external_id=*/1,
                                     XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id);
    if (status != xnn_status_success) {
      std::cerr << "failed to create input2 tensor" << std::endl;
      return nullptr;
    }
  }

  std::vector<size_t> output_dims(dims);
  for (size_t axis : axes) {
    output_dims[axis] = 1;
  }

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const uint32_t output_external_id = is_binary ? 2 : 1;
  status = xnn_define_tensor_value(
      subgraph.get(), datatype, output_dims.size(), output_dims.data(),
      /*data=*/nullptr, /*external_id=*/output_external_id,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create output tensor" << std::endl;
    return nullptr;
  }

  if (op_type == custom_reduce_sum_square_differences) {
    uint32_t diff_id = XNN_INVALID_VALUE_ID;
    status = xnn_define_tensor_value(
        subgraph.get(), datatype, dims.size(), dims.data(),
        /*data=*/nullptr, /*external_id=*/XNN_INVALID_VALUE_ID,
        /*flags=*/0, &diff_id);
    if (status != xnn_status_success) {
      std::cerr << "failed to create diff tensor" << std::endl;
      return nullptr;
    }

    xnn_binary_params params;
    params.output_min = -std::numeric_limits<T>::infinity();
    params.output_max = std::numeric_limits<T>::infinity();
    status =
        xnn_define_binary(subgraph.get(), xnn_binary_squared_difference,
                          &params, input_id, input2_id, diff_id, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create node #0" << std::endl;
      return nullptr;
    }

    input_id = diff_id;
    op_type = custom_reduce_sum;
  } else if (op_type == custom_reduce_sum_absolute_differences) {
    uint32_t diff_id = XNN_INVALID_VALUE_ID;
    status = xnn_define_tensor_value(
        subgraph.get(), datatype, dims.size(), dims.data(),
        /*data=*/nullptr, /*external_id=*/XNN_INVALID_VALUE_ID,
        /*flags=*/0, &diff_id);
    if (status != xnn_status_success) {
      std::cerr << "failed to create diff tensor" << std::endl;
      return nullptr;
    }

    uint32_t abs_diff_id = XNN_INVALID_VALUE_ID;
    status = xnn_define_tensor_value(
        subgraph.get(), datatype, dims.size(), dims.data(),
        /*data=*/nullptr, /*external_id=*/XNN_INVALID_VALUE_ID,
        /*flags=*/0, &abs_diff_id);
    if (status != xnn_status_success) {
      std::cerr << "failed to create diff tensor" << std::endl;
      return nullptr;
    }

    xnn_binary_params params;
    params.output_min = -std::numeric_limits<T>::infinity();
    params.output_max = std::numeric_limits<T>::infinity();
    status = xnn_define_binary(subgraph.get(), xnn_binary_subtract, &params,
                               input_id, input2_id, diff_id, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create node #0" << std::endl;
      return nullptr;
    }

    status = xnn_define_unary(subgraph.get(), xnn_unary_abs, /*params=*/nullptr,
                              diff_id, abs_diff_id, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create node #1" << std::endl;
      return nullptr;
    }

    input_id = abs_diff_id;
    op_type = custom_reduce_sum;
  }

  status = xnn_define_static_reduce(
      subgraph.get(), static_cast<xnn_reduce_operator>(op_type), axes.size(),
    axes.data(), input_id, output_id, /*flags=*/0);

  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  return subgraph.release();
}

}  // namespace models

static void FP32Reduce(benchmark::State& state,
                       models::custom_reduce_operator op_type) {
  const size_t d0 = state.range(0);
  const size_t d1 = state.range(1);
  const size_t d2 = state.range(2);
  const size_t norm_mask = state.range(3);
  std::vector<size_t> dims = {d0, d1, d2};
  std::vector<size_t> axes;
  axes.reserve(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) {
    if ((norm_mask & (1 << i)) != 0) {
      axes.push_back(i);
    }
  }

  xnnpack::RunBenchmark(
      state,
      [=]() { return models::StaticReduce<float>(op_type, dims, axes); });
}

static void ReduceArguments(benchmark::Benchmark* b) {
  b->ArgNames({"K1", "K2", "K3", "NormMask"});
  for (int norm_mask = 1; norm_mask < 8; ++norm_mask) {
    if ((norm_mask & (1 << 0)) == 0) {
      b->Args({1, 1024, 64, norm_mask});
      b->Args({1, 4096, 1024, norm_mask});
      if ((norm_mask & (1 << 2)) == 0) {
        b->Args({1, 16384, 1, norm_mask});
      }
    }
    if ((norm_mask & (1 << 1)) == 0) {
      b->Args({64, 1, 256, norm_mask});
    }
    if ((norm_mask & (1 << 2)) == 0) {
      b->Args({64, 1024, 1, norm_mask});
    }
  }
}

static void FP32Sum(benchmark::State& state) {
  FP32Reduce(state, models::custom_reduce_sum);
}
static void FP32SumSquared(benchmark::State& state) {
  FP32Reduce(state, models::custom_reduce_sum_squared);
}
static void FP32SumSquareDifferences(benchmark::State& state) {
  FP32Reduce(state, models::custom_reduce_sum_square_differences);
}
static void FP32SumAbsoluteDifferences(benchmark::State& state) {
  FP32Reduce(state, models::custom_reduce_sum_absolute_differences);
}
static void FP32Mean(benchmark::State& state) {
  FP32Reduce(state, models::custom_reduce_mean);
}
static void FP32Max(benchmark::State& state) {
  FP32Reduce(state, models::custom_reduce_max);
}

BENCHMARK(FP32Sum)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(ReduceArguments);

BENCHMARK(FP32SumSquared)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(ReduceArguments);

BENCHMARK(FP32SumSquareDifferences)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(ReduceArguments);

BENCHMARK(FP32SumAbsoluteDifferences)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(ReduceArguments);

BENCHMARK(FP32Mean)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(ReduceArguments);

BENCHMARK(FP32Max)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(ReduceArguments);
