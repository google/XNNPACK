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
#include <optional>
#include <sstream>
#include <string>
#include <vector>

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
         size_t block_m, size_t block_n, size_t tile_m, size_t tile_n,
         size_t tile_k, uint32_t flags, TA, TB, TC) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  const size_t a_elem_count = type_element_count(type_of<TA>());
  const size_t b_elem_count = type_element_count(type_of<TB>());

  const size_t m = shape.m;
  const size_t n = shape.n;
  const size_t k = align_up<size_t>(shape.k, tile_k);
  state.SetLabel(std::to_string(m) + "x" + std::to_string(n) + "x" +
                 std::to_string(k));

  const bool transpose_a = flags & dot_flag::transpose_a;

  // Initial setup
  Tensor<TA> a({align_up(m, tile_m), k / a_elem_count});
  Tensor<TB> b({k, align_up(n, tile_n) / b_elem_count},
               Alignment{.bytes = tile_n * tile_k * sizeof(TB)});
  Tensor<TC> c({m, n});
  fill(a.data(), a.size() * a_elem_count, 1);
  fill(b.data(), b.size() * b_elem_count, 1);
  c.fill(0);
  b = b.crop_padding({0, 0}, {b.extent(0) - k, b.extent(1) - n});

  // Panel Packing for A if transpose_a is requested.
  // This simulates the optimal memory layout where A is stored in vertical strips of width `tile_m`.
  // This ensures that the stride within a strip is exactly `tile_m`, enabling svld1_x4.
  std::vector<TA> a_packed_buffer;
  if (transpose_a) {
    size_t m_aligned = align_up(m, tile_m);
    a_packed_buffer.resize(m_aligned * k);
    
    // Pack into panels: [Number_of_Panels, K, Tile_M]
    // But flattened, so we access it via pointer arithmetic.
    for (size_t panel_start = 0; panel_start < m; panel_start += tile_m) {
        size_t current_tile_m = std::min(tile_m, m - panel_start);
        TA* panel_ptr = a_packed_buffer.data() + panel_start * k;
        
        for (size_t row = 0; row < k; ++row) {
            for (size_t col = 0; col < current_tile_m; ++col) {
                // Determine source index. Original a is [M, K].
                // a(row, col) accesses usually [row, col] -> M, K?
                // Tensor `a` was init as {M, K}.
                // a(i, k) is row i, col k.
                size_t src_m = panel_start + col;
                size_t src_k = row;
                
                // Destination: Panel is dense [K, Tile_M].
                // Offset = row * tile_m + col
                panel_ptr[row * tile_m + col] = a(src_m, src_k);
            }
            // Pad the rest of the tile width with 0 if boundary
            for (size_t col = current_tile_m; col < tile_m; ++col) {
                panel_ptr[row * tile_m + col] = 0;
            }
        }
    }
  }

  for (auto _ : state) {
    for (size_t i = 0; i < m; i += block_m) {
      size_t m_i = std::min(block_m, m - i);
      
      const void* a_i = nullptr;
      size_t a_stride = 0;

      if (transpose_a) {
          // Point to the start of the panel corresponding to row `i`.
          // Since we packed in blocks of `tile_m` (which matches `block_m` usually),
          // we just offset by K * i.
          // Note: i must be aligned to block_m/tile_m for this to work perfectly 
          // without partial panel logic, but dot kernel loops aligned.
          a_i = a_packed_buffer.data() + i * k;
          // Stride is exactly tile_m because we packed it that way!
          a_stride = tile_m * sizeof(TA);
      } else {
          a_i = &a(i, 0);
          a_stride = a.stride(0) * sizeof(TA);
      }

      kernel(m_i, n, 1, 1, k, a_stride, 0, 0, a_i, 0, 0,
             b.stride(0) * sizeof(TB), b.base(), /*init_c_stride_m=*/0, nullptr,
             c.stride(0) * sizeof(TC), &c(i, 0));
    }
  }

  if (!std::all_of(c.begin(), c.end(), [=](TC x) { return x == k; })) {
    state.SkipWithError("Incorrect result");
  }

  const size_t ops = shape.m * shape.n * shape.k * 2;
  state.counters["OP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

#define YNN_DOT_KERNEL(arch_flags, kernel, block_m, block_n, block_k, tile_m,  \
                       tile_n, tile_k, flags, a_type, b_type, c_type)          \
  BENCHMARK_CAPTURE(dot, kernel, arch_flags, kernel, block_m, block_n, tile_m, \
                    tile_n, tile_k, flags, a_type(), b_type(), c_type())       \
      ->UseRealTime();
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL

template <typename A, typename B, typename C>
void get_dot_kernel(benchmark::State& state, A, B, C) {
  const size_t m = state.range(0);
  const size_t n = state.range(1);
  const size_t k = state.range(2);

  dot_packed_shape packed_shape;
  packed_shape.tile_k = state.range(3);
  packed_shape.block_n = state.range(4);

  dot_type type = {type_of<A>(), type_of<B>(), type_of<C>()};
  for (auto _ : state) {
    get_dot_kernel(type, {m, n, k}, &packed_shape,
                   /*consistent_arithmetic=*/false,
                   /*transpose_a=*/std::nullopt);
  }
}

void get_dot_kernel_args(benchmark::internal::Benchmark* b) {
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