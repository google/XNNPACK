// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/switch_element_size.h"
#include "ynnpack/kernels/transpose/transpose.h"

namespace ynn {

// Transposes of large elements represent the elements as std::array<uint8_t, N>
// This type_info implementation lets us work with these types.
template <size_t N>
class type_info<std::array<uint8_t, N>> {
 public:
  static constexpr size_t element_count() { return 1; }
  static void set(std::array<uint8_t, N>* ptr, size_t i, uint8_t value) {
    std::fill(ptr[i].begin(), ptr[i].end(), value);
  }
  static std::array<uint8_t, N> get(const std::array<uint8_t, N>* ptr,
                                    size_t i) {
    return ptr[i];
  }
};

template <typename T>
void fill_ramp(T* x, size_t n, size_t begin = 0, size_t stride = 1) {
  for (size_t i = 0; i < n; ++i) {
    type_info<T>::set(x, i, begin + i * stride);
  }
}

template <typename T>
void TestTranspose(T, transpose_fn kernel, std::vector<size_t> ms,
                   std::vector<size_t> ns) {
  constexpr size_t element_count = type_info<T>::element_count();
  const size_t max_m = *std::max_element(ms.begin(), ms.end());
  const size_t max_n = *std::max_element(ns.begin(), ns.end());
  Tensor<T> input({max_n, max_m / element_count});
  Tensor<T> output({max_m, max_n / element_count});
  Tensor<T> expected({max_m, max_n / element_count});
  for (size_t m : ms) {
    for (size_t i = 0; i < max_n; ++i) {
      fill_ramp(&input(i, 0), m, i, m);
    }

    for (size_t n : ns) {
      if (m % element_count != 0 || n % element_count != 0) continue;

      // TODO(dsharlet): Find a better way to test the padding capability.
      for (size_t n_input : {m, m - 1, m / 2}) {
        if (n_input % element_count != 0) continue;

        kernel(m, n, n_input * sizeof(T) / element_count,
               input.stride(0) * sizeof(T), input.base(),
               output.stride(0) * sizeof(T), output.base());

        // Verify results.
        for (size_t i = 0; i < m; ++i) {
          const T* output_i = &output(i, 0);
          for (size_t j = 0; j < n; ++j) {
            T expected;
            type_info<T>::set(&expected, 0, i < n_input ? i * m + j : 0);
            ASSERT_EQ(type_info<T>::get(output_i, j),
                      type_info<T>::get(&expected, 0));
          }
        }
      }
    }
  }
}

template <typename T>
void TestInterleave(T type, size_t factor, interleave_kernel_fn kernel) {
  constexpr size_t element_count = type_info<T>::element_count();
  constexpr int max_n = 64;
  Tensor<T> input({factor, max_n / element_count});
  Tensor<T> output({factor * max_n / element_count});
  for (size_t i = 0; i < factor; ++i) {
    fill_ramp(&input(i, 0), max_n, i, factor);
  }
  for (size_t m = 1; m <= factor; ++m) {
    for (size_t n : simd_sizes_up_to(max_n, element_count)) {
      kernel(factor, m, n, input.stride(0) * sizeof(T), input.base(),
             output.base());

      for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < factor; ++i) {
          // Store the expected value in an instance of T so we get any
          // expected truncation.
          T expected;
          if (i < m) {
            type_info<T>::set(&expected, 0, j * factor + i);
          } else {
            type_info<T>::set(&expected, 0, 0);
          }
          ASSERT_EQ(type_info<T>::get(output.base(), j * factor + i),
                    type_info<T>::get(&expected, 0));
        }
      }
    }
  }
}

struct TransposeParam {
  uint64_t arch_flags;
  transpose_kernel_fn kernel;
  size_t element_size_bits;
};

const char* to_string(const TransposeParam& param) { return ""; }

class Transpose : public ::testing::TestWithParam<TransposeParam> {};

std::vector<size_t> aligned_sizes(size_t element_size_bits) {
  return {std::max<size_t>(4, 256 * 8 / element_size_bits)};
};
std::vector<size_t> unaligned_sizes(size_t element_size_bits) {
  return simd_sizes_up_to(std::max<size_t>(4, 32 * 8 / element_size_bits));
}

TEST_P(Transpose, aligned) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const size_t element_size_bits = kernel.element_size_bits;
  switch_element_size(element_size_bits, [&](auto type) {
    auto sizes = aligned_sizes(element_size_bits);
    TestTranspose(type, kernel.kernel, sizes, sizes);
  });
}

TEST_P(Transpose, mx1) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const size_t element_size_bits = kernel.element_size_bits;
  switch_element_size(element_size_bits, [&](auto type) {
    auto sizes = unaligned_sizes(element_size_bits);
    TestTranspose(type, kernel.kernel, sizes, {1});
  });
}

TEST_P(Transpose, 1xn) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const size_t element_size_bits = kernel.element_size_bits;
  switch_element_size(element_size_bits, [&](auto type) {
    auto sizes = unaligned_sizes(element_size_bits);
    TestTranspose(type, kernel.kernel, {1}, sizes);
  });
}

TEST_P(Transpose, aligned_m) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const size_t element_size_bits = kernel.element_size_bits;
  switch_element_size(element_size_bits, [&](auto type) {
    TestTranspose(type, kernel.kernel, aligned_sizes(element_size_bits),
                  unaligned_sizes(element_size_bits));
  });
}

TEST_P(Transpose, aligned_n) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const size_t element_size_bits = kernel.element_size_bits;
  switch_element_size(element_size_bits, [&](auto type) {
    TestTranspose(type, kernel.kernel, unaligned_sizes(element_size_bits),
                  aligned_sizes(element_size_bits));
  });
}

TEST_P(Transpose, unaligned) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const size_t element_size_bits = kernel.element_size_bits;
  switch_element_size(element_size_bits, [&](auto type) {
    auto sizes = unaligned_sizes(element_size_bits);
    TestTranspose(type, kernel.kernel, sizes, sizes);
  });
}

TEST_P(Transpose, tiled) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const size_t element_size_bits = kernel.element_size_bits;
  switch_element_size(element_size_bits, [&](auto type) {
    auto tiled = make_tiled_transpose(element_size_bits, kernel.kernel);
    auto sizes = aligned_sizes(element_size_bits);
    TestTranspose(type, tiled, sizes, sizes);
  });
}

#define YNN_TRANSPOSE_KERNEL(arch_flags, name, elem_size_bits) \
  INSTANTIATE_TEST_SUITE_P(                                    \
      name, Transpose,                                         \
      testing::Values(TransposeParam{arch_flags, name, elem_size_bits}));
#include "ynnpack/kernels/transpose/transpose.inc"
#undef YNN_TRANSPOSE_KERNEL

struct InterleaveParam {
  uint64_t arch_flags;
  interleave_kernel_fn kernel;
  size_t element_size_bits;
};

std::string to_string(const InterleaveParam& param) {
  return std::to_string(param.element_size_bits);
}

class Interleave
    : public ::testing::TestWithParam<std::tuple<InterleaveParam, int>> {};

TEST_P(Interleave, test) {
  InterleaveParam kernel = std::get<0>(GetParam());
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  int factor = std::get<1>(GetParam());
  switch_element_size(kernel.element_size_bits, [&](auto type) {
    TestInterleave(type, factor, kernel.kernel);
  });
}

auto AllFactors(int factor, int elem_size_bits) {
  std::vector<int> result;
  const int element_count = std::max(elem_size_bits, 8) / elem_size_bits;
  // If the kernel doesn't require a specific factor, test factors up to this.
  constexpr int max_factor = 8;
  return factor == 0
             ? testing::Range(element_count, max_factor + 1, element_count)
             : testing::Values(factor);
}

#define YNN_INTERLEAVE_KERNEL(arch_flags, name, factor, elem_size_bits)       \
  INSTANTIATE_TEST_SUITE_P(                                                   \
      name, Interleave,                                                       \
      testing::Combine(                                                       \
          testing::Values(InterleaveParam{arch_flags, name, elem_size_bits}), \
          AllFactors(factor, elem_size_bits)),                                \
      test_param_to_string<Interleave::ParamType>);
#include "ynnpack/kernels/transpose/interleave.inc"
#undef YNN_INTERLEAVE_KERNEL

}  // namespace ynn
