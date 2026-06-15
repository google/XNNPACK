// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "bench/subgraph/benchmark.h"
#include "include/xnnpack.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include <benchmark/benchmark.h>

namespace models {

// Compute the L2 norm of [m x n x k] tensors, where the mean
// is computed over the dimensions in `norm_mask`. This computation is
// equivalent to: input_mean = mean(input, norm_mask) (input - input_mean) /
// sqrt(mean(squared_difference(input, input_mean), norm_mask) + epsilon) *
// weight + bias
//
// Where `mean(x, norm_mask)` means computing the mean of the dimensions in the
// `norm_mask`.
template <typename T>
xnn_subgraph_t L2NormImpl(size_t m, size_t n, size_t k, uint32_t norm_mask) {
  xnn_status status;
  const xnn_datatype datatype = xnn_datatype_of<T>();
  const xnn_datatype math_datatype = xnn_datatype_fp32;
  auto subgraph = xnnpack::CreateUniqueSubgraph(/*num_external_values=*/2, 0);
  if (!subgraph) {
    std::cerr << "failed to create subgraph" << std::endl;
    return nullptr;
  }

  std::array<size_t, 3> dims = {{m, n, k}};
  std::array<size_t, 3> reduction_dims = dims;
  for (size_t i = 0; i < reduction_dims.size(); ++i) {
    if ((norm_mask & (1 << i)) != 0) {
      reduction_dims[i] = 1;
    }
  }

  uint32_t input = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph.get(), datatype, dims.size(), dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor input" << std::endl;
    return nullptr;
  }

  uint32_t output = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph.get(), datatype, dims.size(), dims.data(),
      /*data=*/nullptr, 1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor output" << std::endl;
    return nullptr;
  }

  std::vector<size_t> reduction_axes;
  reduction_axes.reserve(reduction_dims.size());
  for (size_t i = 0; i < reduction_dims.size(); ++i) {
    if ((norm_mask & (1 << i)) != 0) {
      reduction_axes.push_back(i);
    }
  }

  uint32_t math_input = input;
  if (datatype != math_datatype) {
    status = xnn_define_tensor_value(
        subgraph.get(), math_datatype, dims.size(), dims.data(),
        /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0, &math_input);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor math_input" << std::endl;
      return nullptr;
    }
    status = xnn_define_unary(subgraph.get(), xnn_unary_convert, nullptr, input,
                              math_input, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create unary convert input->math" << std::endl;
      return nullptr;
    }
  }

  uint32_t input_sq = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), math_datatype, dims.size(),
                                   dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &input_sq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor input_sq" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph.get(), xnn_unary_square, nullptr,
                            math_input, input_sq, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create unary square" << std::endl;
    return nullptr;
  }

  uint32_t sum_sq = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), math_datatype,
                                   reduction_dims.size(), reduction_dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &sum_sq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor sum_sq" << std::endl;
    return nullptr;
  }

  status = xnn_define_static_reduce(subgraph.get(), xnn_reduce_sum,
                                    reduction_axes.size(),
                                    reduction_axes.data(), input_sq, sum_sq,
                                    /*flags=*/XNN_FLAG_KEEP_DIMS);
  if (status != xnn_status_success) {
    std::cerr << "failed to create reduce sum" << std::endl;
    return nullptr;
  }

  // 4. Compute inv_sqrt_sum_sq = rsqrt(sum_sq) in math_datatype (FP32)
  uint32_t inv_sqrt_sum_sq_math = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), math_datatype, dims.size(),
                                   dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &inv_sqrt_sum_sq_math);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor inv_sqrt_sum_sq_math" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph.get(), xnn_unary_reciprocal_square_root,
                            nullptr, sum_sq, inv_sqrt_sum_sq_math, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create unary reciprocal square root" << std::endl;
    return nullptr;
  }

  // Transition to datatype (FP16 or FP32)

  // 5. Convert inv_sqrt_sum_sq_math to datatype if necessary
  uint32_t inv_sqrt_sum_sq = inv_sqrt_sum_sq_math;
  if (datatype != math_datatype) {
    status = xnn_define_tensor_value(
        subgraph.get(), datatype, dims.size(), dims.data(),
        /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0, &inv_sqrt_sum_sq);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor inv_sqrt_sum_sq" << std::endl;
      return nullptr;
    }
    status =
        xnn_define_unary(subgraph.get(), xnn_unary_convert, nullptr,
                         inv_sqrt_sum_sq_math, inv_sqrt_sum_sq, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create unary convert inv_sqrt_sum_sq->datatype"
                << std::endl;
      return nullptr;
    }
  }

  // 6. Compute output = input * inv_sqrt_sum_sq in datatype
  xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph.get(), xnn_binary_multiply, &params,
                             input, inv_sqrt_sum_sq, output,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary multiply" << std::endl;
    return nullptr;
  }

  return subgraph.release();
}
xnn_subgraph_t FP32L2Norm(size_t m, size_t n, size_t k, uint32_t norm_mask) {
  return L2NormImpl<float>(m, n, k, norm_mask);
}

xnn_subgraph_t FP16L2Norm(size_t m, size_t n, size_t k, uint32_t norm_mask) {
  return L2NormImpl<xnn_float16>(m, n, k, norm_mask);
}

}  // namespace models

static void FP32L2Norm(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FP32L2Norm(state.range(0), state.range(1), state.range(2),
                              state.range(3));
  });
}
static void FP16L2Norm(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FP16L2Norm(state.range(0), state.range(1), state.range(2),
                              state.range(3));
  });
}

static void L2NormArguments(benchmark::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "NormMask"});
  for (int norm_mask = 1; norm_mask < 8; norm_mask++) {
    b->Args({128, 256, 512, norm_mask});
  }
}

BENCHMARK(FP32L2Norm)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(L2NormArguments);
BENCHMARK(FP16L2Norm)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(L2NormArguments);
