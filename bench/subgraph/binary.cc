// Copyright 2020-2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "bench/subgraph/benchmark.h"
#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/operator-utils.h"
#include <benchmark/benchmark.h>

namespace models {

template <typename T>
xnn_subgraph_t Binary(int64_t op, size_t d0, size_t d1, size_t d2,
                      bool a_broadcasted, bool b_broadcasted) {
  const xnn_datatype datatype = xnn_datatype_of<T>();

  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/3, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgraph" << std::endl;
    return nullptr;
  }

  std::array<size_t, 3> dims_input1 = {{d0, d1, a_broadcasted ? 1 : d2}};
  uint32_t input1_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, datatype, dims_input1.size(),
                                   dims_input1.data(),
                                   /*data=*/nullptr, /*external_id=*/0,
                                   XNN_VALUE_FLAG_EXTERNAL_INPUT, &input1_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create input1 tensor" << std::endl;
    return nullptr;
  }

  std::array<size_t, 3> dims_input2 = {{d0, d1, b_broadcasted ? 1 : d2}};
  uint32_t input2_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, datatype, dims_input2.size(),
                                   dims_input2.data(),
                                   /*data=*/nullptr, /*external_id=*/1,
                                   XNN_VALUE_FLAG_EXTERNAL_INPUT, &input2_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create input2 tensor" << std::endl;
    return nullptr;
  }

  std::array<size_t, 3> dims = {{d0, d1, d2}};
  uint32_t output_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, datatype, dims.size(), dims.data(),
                                   /*data=*/nullptr, /*external_id=*/2,
                                   /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                   &output_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create output tensor" << std::endl;
    return nullptr;
  }

  xnn_binary_params params;
  params.output_min = -std::numeric_limits<T>::infinity();
  params.output_max = std::numeric_limits<T>::infinity();
  status = xnn_define_binary(subgraph, static_cast<xnn_binary_operator>(op),
                             /*params=*/&params, input1_id, input2_id,
                             /*output_id=*/output_id,
                             /*flags=*/0);

  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models

static void FP32Binary(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::Binary<float>(state.range(0), FLAGS_batch_size,
                                 state.range(1), state.range(2), state.range(3),
                                 state.range(4));
  });
}

static void RegisterBenchmarks() {
  // Use a static variable inside the function to ensure this block runs only
  // once.
  static bool registered = [] {
    static const xnn_binary_operator real_ops[] = {
        xnn_binary_add,      xnn_binary_subtract,
        xnn_binary_multiply, xnn_binary_divide,
        xnn_binary_maximum,  xnn_binary_minimum,
        xnn_binary_copysign, xnn_binary_squared_difference,
        xnn_binary_prelu,    xnn_binary_atan2,
        xnn_binary_pow,
    };

    for (auto op : real_ops) {
      std::string op_name = xnn_binary_operator_to_string(op);
      auto b =
          benchmark::RegisterBenchmark("FP32Binary/" + op_name, FP32Binary);
      b->Unit(benchmark::kMicrosecond)->MeasureProcessCPUTime()->UseRealTime();
      b->ArgNames({"Op", "M", "N", "BroadcastA", "BroadcastB"});

      const std::pair<int, int> shapes[] = {{1, 256}, {1, 4096}, {256, 4096}};

      const std::pair<bool, bool> broadcasts[] = {
          {false, false}, {true, false}, {false, true}};

      for (auto shape : shapes) {
        for (auto bc : broadcasts) {
          b->Args({op, shape.first, shape.second, bc.first, bc.second});
        }
      }
    }
    return true;
  }();
  (void)registered;  // Avoid unused variable warning.
}

struct BenchmarkRegisterer {
  BenchmarkRegisterer() { RegisterBenchmarks(); }
};
static BenchmarkRegisterer br;
