// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/dot/dot.h"
#include <benchmark/benchmark.h>

namespace ynn {

struct Shape {
  int m, n, k;

  static Shape parse(std::string str) {
    std::replace(str.begin(), str.end(), 'x', ' ');
    std::stringstream ss(str);
    Shape result;
    ss >> result.m >> result.n >> result.k;
    return result;
  }
};

template <typename T>
void fill(T* data, size_t n, int value) {
  for (size_t i = 0; i < n; ++i) {
    type_info<T>::set(data, i, value);
  }
}

Shape shape = {240, 240, 240};

template <typename TA, typename TB, typename TC>
void dot(benchmark::State& state, uint64_t arch_flags, dot_kernel_fn kernel,
         size_t block_m, size_t block_n, size_t tile_n, size_t tile_k,
         uint32_t flags, TA, TB, TC) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  const size_t a_elem_count = type_element_count(type_of<TA>());
  const size_t b_elem_count = type_element_count(type_of<TB>());

  const size_t m = shape.m;
  const size_t n = shape.n;
  // If k gets aligned up, that means this kernel would have needed to pad the
  // input with zeros up to a multiple of tile_k. To make checking correctness
  // easier, let's just round k up, it should be computationally equivalent.
  const size_t k = align_up<size_t>(shape.k, tile_k);
  state.SetLabel(std::to_string(m) + "x" + std::to_string(n) + "x" +
                 std::to_string(k));

  const bool transpose_a = flags & dot_flag::transpose_a;

  Tensor<TA> a({m, k / a_elem_count});
  Tensor<TB> b({k, align_up(n, tile_n) / b_elem_count},
               Alignment{.bytes = tile_n * tile_k * sizeof(TB)});
  Tensor<TC> c({m, n});
  fill(a.data(), a.size() * a_elem_count, 1);
  fill(b.data(), b.size() * b_elem_count, 1);
  c.fill(0);
  b = b.crop_padding({0, 0}, {b.extent(0) - k, b.extent(1) - n});

  if (transpose_a) {
    // This mangles the data, but we don't care here.
    a = a.reshape({k / tile_k, m * tile_k});
  }

  for (auto _ : state) {
    for (size_t i = 0; i < m; i += block_m) {
      size_t m_i = std::min(block_m, m - i);
      const void* a_i = transpose_a ? &a(0, i * tile_k) : &a(i, 0);
      kernel(m_i, n, 1, 1, k, a.stride(0) * sizeof(TA), 0, 0, a_i, 0, 0,
             b.stride(0) * sizeof(TB), b.base(), /*init_c_stride_m=*/0, nullptr,
             c.stride(0) * sizeof(TC), &c(i, 0));
    }
  }

  // Check that the kernel didn't compute the wrong thing. We assume the kernel
  // is correct, but we have some logic here that needs validation too. We
  // filled a and b with 1, so the result should be k everywhere.
  if (!std::all_of(c.begin(), c.end(), [=](TC x) { return x == k; })) {
    state.SkipWithError("Incorrect result");
  }

  const size_t ops = m * n * k * 2;
  state.counters["FLOP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

#define YNN_DOT_KERNEL(arch_flags, kernel, block_m, block_n, block_k, tile_n,  \
                       tile_k, flags, a_type, b_type, c_type)                  \
  BENCHMARK_CAPTURE(dot, kernel, arch_flags, kernel, block_m, block_n, tile_n, \
                    tile_k, flags, a_type(), b_type(), c_type())               \
      ->UseRealTime();
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL

template <typename A, typename B, typename C>
void get_dot_kernel(benchmark::State& state, A, B, C) {
  constexpr uint64_t all_archs = -1;
  const size_t m = state.range(0);
  const size_t n = state.range(1);
  const size_t k = state.range(2);

  dot_packed_shape packed_shape;
  packed_shape.tile_k = state.range(3);
  packed_shape.block_n = state.range(4);

  dot_type type = {type_of<A>(), type_of<B>(), type_of<C>()};
  for (auto _ : state) {
    get_dot_kernel(type, {m, n, k}, &packed_shape, all_archs);
  }
}

void get_dot_kernel_args(benchmark::internal::Benchmark* b) {
  // We use a large highly composite value when we want to test large shapes, so
  // it is unlikely that block shapes do not divide this extent.
  const int large_shape = 3 * 5 * 7 * 64;

  b->ArgNames({"M", "N", "K", "TileK", "BlockN"});
  b->Args({large_shape, large_shape, large_shape, 0, 0});
  b->Args({large_shape, large_shape, large_shape, 1, 16});
  b->Args({large_shape, large_shape, large_shape, 2, 16});
  b->Args({large_shape, large_shape, large_shape, 4, 16});
}

BENCHMARK_CAPTURE(get_dot_kernel, fp32_fp32_fp32, float{}, float{}, float{})
    ->Apply(get_dot_kernel_args);
BENCHMARK_CAPTURE(get_dot_kernel, bf16_bf16_fp32, bfloat16{}, bfloat16{},
                  float{})
    ->Apply(get_dot_kernel_args);
BENCHMARK_CAPTURE(get_dot_kernel, fp16_fp16_fp32, half{}, half{}, float{})
    ->Apply(get_dot_kernel_args);
BENCHMARK_CAPTURE(get_dot_kernel, int8_int8_int32, int8_t{}, int8_t{},
                  int32_t{})
    ->Apply(get_dot_kernel_args);
BENCHMARK_CAPTURE(get_dot_kernel, int8_int4_int32, int8_t{}, int4x2{},
                  int32_t{})
    ->Apply(get_dot_kernel_args);

}  // namespace ynn

int main(int argc, char** argv) {
  for (int i = 1; i < argc;) {
    if (strncmp(argv[i], "--shape=", 8) == 0) {
      ynn::shape = ynn::Shape::parse(argv[i] + 8);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else {
      ++i;
    }
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
