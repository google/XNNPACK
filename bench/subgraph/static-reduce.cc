// Copyright 2020-2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "bench/subgraph/benchmark.h"
#include "include/xnnpack.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/operator-utils.h"
#include <benchmark/benchmark.h>

namespace models {

template <typename T>
xnn_subgraph_t StaticReduce(xnn_reduce_operator op_type,
                            const std::vector<size_t>& dims,
                            const std::vector<size_t>& axes) {
  const xnn_datatype datatype = xnn_datatype_of<T>();

  xnn_status status;
  auto subgraph = xnnpack::CreateUniqueSubgraph(/*num_external_values=*/2, 0);
  if (!subgraph) {
    std::cerr << "failed to create subgrpah" << std::endl;
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

  std::vector<size_t> output_dims(dims);
  for (size_t axis : axes) {
    output_dims[axis] = 1;
  }

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph.get(), datatype, output_dims.size(), output_dims.data(),
      /*data=*/nullptr, /*external_id=*/1,
      /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create output tensor" << std::endl;
    return nullptr;
  }

  status =
      xnn_define_static_reduce(subgraph.get(), op_type, axes.size(),
                               axes.data(), input_id, output_id, /*flags=*/0);

  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  return subgraph.release();
}

}  // namespace models

static void FP32Reduce(benchmark::State& state) {
  const xnn_reduce_operator op_type =
      static_cast<xnn_reduce_operator>(state.range(0));
  state.SetLabel(xnn_reduce_operator_to_string(op_type));
  const size_t d0 = state.range(1);
  const size_t d1 = state.range(2);
  const size_t d2 = state.range(3);
  const size_t norm_mask = state.range(4);
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
      [&]() { return models::StaticReduce<float>(op_type, dims, axes); });
}

static void ReduceArguments(benchmark::Benchmark* b) {
  static const xnn_reduce_operator reduce_ops[] = {
      xnn_reduce_mean,
      xnn_reduce_max,
      xnn_reduce_sum,
      xnn_reduce_sum_squared,
  };

  b->ArgNames({"op", "K1", "K2", "K3", "NormMask"});
  for (auto op : reduce_ops) {
    for (int norm_mask = 1; norm_mask < 8; ++norm_mask) {
      b->Args({op, 1, 1024, 64, norm_mask});
      b->Args({op, 1, 4096, 64, norm_mask});
      b->Args({op, 64, 1024, 1, norm_mask});
      b->Args({op, 64, 1, 256, norm_mask});
      b->Args({op, 1, 16384, 1, norm_mask});
    }
  }
}

BENCHMARK(FP32Reduce)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(ReduceArguments);
