// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <utility>

#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/tensor.h"
#include <benchmark/benchmark.h>
#include "slinky/runtime/buffer.h"

namespace ynn {
namespace {

constexpr std::size_t max_rank = 5;

template <int NumInnerDims, std::size_t NumInputs, std::size_t... I>
void call_fuse_and_slice_leading_dims(
    slinky::dim* x_dims, slinky::raw_buffer& x,
    std::array<slinky::raw_buffer, NumInputs>& inputs,
    slinky::dim (&in_peeled)[NumInputs][NumInnerDims],
    std::index_sequence<I...>) {
  std::apply(
      [&](auto&&... args) {
        fuse_and_slice_leading_dims<NumInnerDims>(
            x_dims, x, std::forward<decltype(args)>(args)...);
      },
      std::tuple_cat(std::forward_as_tuple(&in_peeled[I][0], inputs[I])...));
}

template <std::size_t NumInnerDims, std::size_t NumInputs>
void BM_fuse_and_slice_leading_dims(benchmark::State& state) {
  const int rank = state.range(0);
  assert(rank <= max_rank);
  const size_t dims[5] = {10, 10, 10, 10, 10};

  slinky::buffer<void, max_rank> x;
  std::array<slinky::buffer<void, max_rank>, NumInputs> inputs;

  init_buffer(x, 4, rank, dims, nullptr);
  for (std::size_t i = 0; i < NumInputs; ++i) {
    init_buffer(inputs[i], 4, rank, dims, reinterpret_cast<void*>(0x1000));
  }
  for (auto _ : state) {
    std::array<slinky::raw_buffer, NumInputs> raw_inputs;
    for (std::size_t i = 0; i < NumInputs; ++i) {
      raw_inputs[i] = inputs[i];
    }

    slinky::raw_buffer raw_x = x;
    slinky::dim x_dims[NumInnerDims];
    slinky::dim in_dims[NumInputs][NumInnerDims];
    call_fuse_and_slice_leading_dims<NumInnerDims>(
        x_dims, raw_x, raw_inputs, in_dims,
        std::make_index_sequence<NumInputs>{});

    benchmark::DoNotOptimize(x_dims);
    benchmark::DoNotOptimize(in_dims);
  }
}

BENCHMARK_TEMPLATE(BM_fuse_and_slice_leading_dims, 2, 1)->DenseRange(0, 4);
BENCHMARK_TEMPLATE(BM_fuse_and_slice_leading_dims, 2, 2)->DenseRange(0, 4);
BENCHMARK_TEMPLATE(BM_fuse_and_slice_leading_dims, 2, 3)->DenseRange(0, 4);

}  // namespace
}  // namespace ynn

BENCHMARK_MAIN();
