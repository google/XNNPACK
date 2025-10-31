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
#include "ynnpack/base/base.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/transpose/interleave.h"
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
void TestTranspose(T, transpose_kernel_fn kernel, std::vector<size_t> ms,
                   std::vector<size_t> ns) {
  constexpr size_t element_count = type_info<T>::element_count();
  const size_t max_m = *std::max_element(ms.begin(), ms.end());
  const size_t max_n = *std::max_element(ns.begin(), ns.end());
  Tensor<T> input({max_n, max_m / element_count});
  Tensor<T> output({max_m, max_n / element_count});
  Tensor<T> expected({max_m, max_n / element_count});
  for (size_t m : ms) {
    for (size_t n : ns) {
      if (m % element_count != 0 || n % element_count != 0) continue;

      // TODO(dsharlet): Find a better way to test the padding capability.
      for (size_t n_input : {m, m - 1, m / 2}) {
        if (n_input % element_count != 0) continue;

        for (size_t i = 0; i < n; ++i) {
          fill_ramp(&input(i, 0), n_input, i, m);
        }

        kernel(m, n, n_input * sizeof(T) / element_count,
               input.stride(0) * sizeof(T), input.base(),
               output.stride(0) * sizeof(T), output.base());

        // Verify results.
        for (size_t i = 0; i < m; ++i) {
          for (size_t j = 0; j < n; ++j) {
            T expected;
            type_info<T>::set(&expected, 0, i < n_input ? i * m + j : 0);
            ASSERT_EQ(type_info<T>::get(output.base(), i * max_n + j),
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
  for (size_t m = 1; m <= factor; ++m) {
    for (size_t n : simd_sizes_up_to(max_n, element_count)) {
      for (size_t i = 0; i < m; ++i) {
        fill_ramp(&input(i, 0), n, i, factor);
      }

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

template <typename F>
constexpr decltype(auto) SwitchElementType(size_t element_size_bits, F&& f) {
  switch (element_size_bits) {
    case 4:
      return std::forward<F>(f)(uint4x2());
    case 8:
      return std::forward<F>(f)(uint8_t());
    case 16:
      return std::forward<F>(f)(uint16_t());
    case 32:
      return std::forward<F>(f)(uint32_t());
    case 64:
      return std::forward<F>(f)(uint64_t());
    case 128:
      return std::forward<F>(f)(x128_t());
    case 256:
      return std::forward<F>(f)(x256_t());
    case 512:
      return std::forward<F>(f)(x512_t());
    case 1024:
      return std::forward<F>(f)(x1024_t());
    case 2048:
      return std::forward<F>(f)(x2048_t());
    default:
      YNN_UNREACHABLE;
  }
}

struct TransposeParam {
  uint64_t arch_flags;
  transpose_kernel_fn kernel;
  size_t element_size_bits;
};

const char* to_string(const TransposeParam& param) { return ""; }

class Transpose : public ::testing::TestWithParam<TransposeParam> {};

std::vector<size_t> aligned_sizes = {512};
std::vector<size_t> unaligned_sizes = simd_sizes_up_to(64);

TEST_P(Transpose, aligned) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchElementType(kernel.element_size_bits, [&](auto type) {
    TestTranspose(type, kernel.kernel, aligned_sizes, aligned_sizes);
  });
}

TEST_P(Transpose, mx1) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchElementType(kernel.element_size_bits, [&](auto type) {
    TestTranspose(type, kernel.kernel, unaligned_sizes, {1});
  });
}

TEST_P(Transpose, 1xn) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchElementType(kernel.element_size_bits, [&](auto type) {
    TestTranspose(type, kernel.kernel, {1}, unaligned_sizes);
  });
}

TEST_P(Transpose, aligned_m) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchElementType(kernel.element_size_bits, [&](auto type) {
    TestTranspose(type, kernel.kernel, aligned_sizes, unaligned_sizes);
  });
}

TEST_P(Transpose, aligned_n) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchElementType(kernel.element_size_bits, [&](auto type) {
    TestTranspose(type, kernel.kernel, unaligned_sizes, aligned_sizes);
  });
}

TEST_P(Transpose, unaligned) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchElementType(kernel.element_size_bits, [&](auto type) {
    TestTranspose(type, kernel.kernel, unaligned_sizes, unaligned_sizes);
  });
}

#define YNN_TRANSPOSE_KERNEL(arch_flags, name, type)       \
  INSTANTIATE_TEST_SUITE_P(name, Transpose,                \
                           testing::Values(TransposeParam{ \
                               arch_flags, name, elem_size_of(type{})}));
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
  SwitchElementType(kernel.element_size_bits, [&](auto type) {
    TestInterleave(type, factor, kernel.kernel);
  });
}

template <typename T>
auto AllFactors(int factor) {
  std::vector<int> result;
  constexpr int element_count = type_info<T>::element_count();
  return factor == 0 ? testing::Range(element_count, 17, element_count)
                     : testing::Values(factor);
}

#define YNN_INTERLEAVE_KERNEL(arch_flags, name, factor, type)                  \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      name, Interleave,                                                        \
      testing::Combine(testing::Values(InterleaveParam{arch_flags, name,       \
                                                       elem_size_of(type{})}), \
                       AllFactors<type>(factor)),                              \
      test_param_to_string<Interleave::ParamType>);
#include "ynnpack/kernels/transpose/interleave.inc"
#undef YNN_INTERLEAVE_KERNEL

}  // namespace ynn
