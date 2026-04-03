// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/span.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/kernels/dot/pack_test_tensor.h"
#include "ynnpack/kernels/dot/schedule.h"

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

int parse_dot_loop_dim(char c) {
  switch (c) {
    case 'm':
      return dot_loop::m;
    case 'n':
      return dot_loop::n;
    case 'k':
      return dot_loop::k;
    default:
      return -1;
  }
}

dot_loop parse_dot_loop(std::string str) {
  if (str.empty()) return {};
  return {parse_dot_loop_dim(str[0]), std::stoul(str.substr(1))};
}

struct kernel_info {
  const char* name;
  dot_kernel_fn kernel;
  size_t block_m, block_n, block_k, tile_m, tile_n, tile_k;
  uint32_t flags;
  multi_type type;
};

kernel_info get_kernel(const std::string& kernel_name) {
  std::vector<kernel_info> kernels;
#define YNN_DOT_KERNEL(arch, name, block_m, block_n, block_k, tile_m, tile_n, \
                       tile_k, flags, a_type, b_type, c_type)                 \
  if (#name == kernel_name) {                                                 \
    if (!is_arch_supported(arch)) {                                           \
      std::cerr << "Kernel architecture not supported by this CPU\n";         \
      return kernel_info{};                                                   \
    }                                                                         \
    return {#name,   name,                                                    \
            block_m, block_n,                                                 \
            block_k, tile_m,                                                  \
            tile_n,  tile_k,                                                  \
            flags,   multi_type_of(a_type{}, b_type{}, c_type{})};            \
  }
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL
  return kernel_info{};
}

// Benchmark a call.
template <class F>
double benchmark(F op) {
  const int max_trials = 10;
  const double min_time_s = 0.5;
  double time_per_iteration_s = 0;
  int iterations = 1;
  for (int trials = 0; trials < max_trials; trials++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < iterations; j++) {
      op();
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    time_per_iteration_s =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() /
        (iterations * 1e9);
    if (time_per_iteration_s * iterations > min_time_s) {
      break;
    }

    int next_iterations = std::ceil((min_time_s * 2) / time_per_iteration_s);
    iterations = std::clamp(next_iterations, iterations, iterations * 10);
  }
  return time_per_iteration_s;
}

template <typename T>
void fill(T* data, size_t n, int value) {
  for (size_t i = 0; i < n; ++i) {
    type_info<T>::set(data, i, value);
  }
}

template <typename TA, typename TB, typename TC>
double run_benchmark(TA, TB, TC, const kernel_info& kernel, size_t m, size_t n,
                     size_t k, span<dot_loop> loops) {
  const bool pack_a = kernel.flags & dot_flag::transpose_a;

  const size_t tile_m = kernel.tile_m;
  const size_t tile_n = kernel.tile_n;
  const size_t tile_k = kernel.tile_k;

  // Align k to tile_k as in bench.cc
  k = align_up(k, kernel.tile_k);

  const size_t a_elem_count = type_element_count(type_of<TA>());
  const size_t b_elem_count = type_element_count(type_of<TB>());

  // If k gets aligned up, that means this kernel would have needed to pad the
  // input with zeros up to a multiple of tile_k. To make checking correctness
  // easier, let's just round k up, it should be computationally equivalent.

  Tensor<TA> a({align_up(m, tile_m), k / a_elem_count});
  Tensor<TB> b({k, align_up(n, tile_n) / b_elem_count},
               Alignment{.bytes = tile_n * kernel.tile_k * sizeof(TB)});
  Tensor<TC> c({m, n});
  fill(a.data(), a.size() * a_elem_count, 1);
  fill(b.data(), b.size() * b_elem_count, 1);
  c.fill(0);
  b = b.crop_padding({0, 0}, {b.extent(0) - k, b.extent(1) - n});

  a = pack_a ? transpose_a(a, tile_m, tile_k) : a;

  auto kernel_wrapper = [&](size_t m, size_t n, size_t k, const void* a_ptr,
                            const void* b_ptr, size_t init_c_stride_m,
                            const void* init_c, void* c_ptr) {
    kernel.kernel(m, n, 1, 1, k,
                  a.stride(0) * sizeof(TA) / (pack_a ? tile_k : 1), 0, 0, a_ptr,
                  0, 0, b.stride(0) * sizeof(TB), b_ptr, init_c_stride_m,
                  init_c, c.stride(0) * sizeof(TC), c_ptr);
  };

  const size_t a_stride_m = pack_a ? kernel.tile_k * sizeof(TA) / a_elem_count
                                   : a.stride(0) * sizeof(TA);
  const size_t a_stride_k = pack_a ? a.stride(0) * sizeof(TA) / kernel.tile_k
                                   : a.stride(1) * sizeof(TA) / a_elem_count;
  const size_t b_stride_k = b.stride(0) * sizeof(TB);
  const size_t b_stride_n = b.stride(1) * sizeof(TB) / b_elem_count;
  const size_t c_stride_m = c.stride(0) * sizeof(TC);
  const size_t c_stride_n = c.stride(1) * sizeof(TC);

  double t = benchmark([&]() {
    run_dot(loops, m, n, k, kernel.block_m, kernel.block_n, kernel.block_k,
            a_stride_m, a_stride_k, a.base(), b_stride_k, b_stride_n, b.base(),
            0, nullptr, c_stride_m, c_stride_n, c.base(), kernel_wrapper);
  });
  // Check that the kernel didn't compute the wrong thing. We assume the kernel
  // is correct, but we have some logic here that needs validation too. We
  // filled a and b with 1, so the result should be k everywhere.
  if (!std::all_of(c.begin(), c.end(), [=](TC x) { return x == k; })) {
    std::cerr << "Incorrect result" << std::endl;
    return -1;
  }

  return t;
}

}  // namespace ynn

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <kernel_name> <MxNxK> [<loop1> <loop2> ...]" << std::endl;
    return 1;
  }

  std::string kernel_name = argv[1];
  ynn::Shape shape = ynn::Shape::parse(argv[2]);
  if (shape.m <= 0 || shape.n <= 0 || shape.k <= 0) {
    std::cerr << "Error parsing shape: " << argv[2] << std::endl;
    return 1;
  }

  std::vector<ynn::dot_loop> loops;
  for (int i = 3; i < argc; ++i) {
    ynn::dot_loop loop = ynn::parse_dot_loop(argv[i]);
    if (loop.dim < 0 || loop.blocks == 0) {
      std::cerr << "Error parsing loop specifier: " << argv[i] << std::endl;
      return 1;
    }
    loops.push_back(loop);
  }

  // Find the kernel
  auto kernel = ynn::get_kernel(kernel_name);
  if (!kernel.kernel) {
    std::cerr << "Unknown kernel: " << kernel_name << std::endl;
    return 1;
  }

  // Kernels require an outer loop for m, make sure we have one.
  size_t min_block_m = -1;
  // The dot loops are interpreted as a multiple of blocks. To make this CLI
  // easier to use, we convert the input to blocks from elements.
  for (ynn::dot_loop& loop : loops) {
    switch (loop.dim) {
      case ynn::dot_loop::m:
        loop.blocks = ynn::ceil_div(loop.blocks, kernel.block_m);
        min_block_m = std::min(min_block_m, loop.blocks);
        break;
      case ynn::dot_loop::n:
        loop.blocks = ynn::ceil_div(loop.blocks, kernel.block_n);
        break;
      case ynn::dot_loop::k:
        loop.blocks = ynn::ceil_div(loop.blocks, kernel.block_k);
        break;
    }
  }
  if (min_block_m > 1) loops.push_back({ynn::dot_loop::m, 1});

  double t = ynn::SwitchThreeTypes(kernel.type, [&](auto a, auto b, auto c) {
    return ynn::run_benchmark(a, b, c, kernel, shape.m, shape.n, shape.k,
                              loops);
  });

  if (t < 0) {
    return 1;
  }

  double gops = (double)shape.m * shape.n * shape.k * 2 / t / 1e9;
  std::cout << kernel.name << " " << shape.m << "x" << shape.n << "x" << shape.k
            << ": " << t * 1e3 << " ms, " << gops << " GFLOPS" << std::endl;

  return 0;
}
