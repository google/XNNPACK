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
xnn_subgraph_t Unary(int64_t op, size_t d0, size_t d1, size_t d2) {
  const xnn_datatype datatype = xnn_datatype_of<T>();

  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::array<size_t, 3> dims = {{d0, d1, d2}};
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, datatype, dims.size(), dims.data(),
                                   /*data=*/nullptr, /*external_id=*/0,
                                   XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create input tensor" << std::endl;
    return nullptr;
  }

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, datatype, dims.size(), dims.data(),
                                   /*data=*/nullptr, /*external_id=*/1,
                                   /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                   &output_id);
  if (status != xnn_status_success) {
    std::cerr << "failed to create output tensor" << std::endl;
    return nullptr;
  }

  if (status != xnn_status_success) {
    std::cerr << "failed to create weights tensor" << std::endl;
    return nullptr;
  }

  xnn_unary_params params;
  memset(&params, 0, sizeof(params));
  status =
      xnn_define_unary(subgraph, static_cast<xnn_unary_operator>(op),
                       /*params=*/&params, input_id, /*output_id=*/output_id,
                       /*flags=*/0);

  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models

static void FP32Unary(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::Unary<float>(state.range(0), FLAGS_batch_size,
                                state.range(1), state.range(2));
  });
}

// Initialize a static global variable to trigger the registration.
// The lambda will be executed only once during static initialization.
static bool benchmarks_registered = [] {
  static const xnn_unary_operator real_ops[] = {
      xnn_unary_abs,
      xnn_unary_approxgelu,
      xnn_unary_bankers_rounding,
      xnn_unary_ceiling,
      xnn_unary_clamp,
      xnn_unary_convert,
      xnn_unary_cosine,
      xnn_unary_elu,
      xnn_unary_exp,
      xnn_unary_floor,
      xnn_unary_gelu,
      xnn_unary_hardswish,
      xnn_unary_leaky_relu,
      xnn_unary_log,
      xnn_unary_negate,
      xnn_unary_reciprocal_square_root,
      xnn_unary_sigmoid,
      xnn_unary_sine,
      xnn_unary_square_root,
      xnn_unary_square,
      xnn_unary_tanh,
      xnn_unary_cube_root,
      xnn_unary_sign,
  };

  for (auto op : real_ops) {
    std::string op_name = xnn_unary_operator_to_string(op);
    auto b = benchmark::RegisterBenchmark("FP32Unary/" + op_name, FP32Unary);
    b->Unit(benchmark::kMicrosecond)->MeasureProcessCPUTime()->UseRealTime();
    b->ArgNames({"Op", "M", "N"});

    b->Args({op, 1, 256});
    b->Args({op, 1, 4096});
    b->Args({op, 256, 4096});
  }
  return true;
}();

