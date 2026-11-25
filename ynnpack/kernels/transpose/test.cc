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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/test/buffer.h"
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
  static std::array<uint8_t, N>& ref(std::array<uint8_t, N>* ptr, size_t i) {
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
  const size_t max_m =
      align_up(*std::max_element(ms.begin(), ms.end()), element_count);
  const size_t max_n =
      align_up(*std::max_element(ns.begin(), ns.end()), element_count);
  Tensor<T> input({max_n, max_m});
  Tensor<T> output({max_m, max_n});
  for (size_t m : ms) {
    for (size_t i = 0; i < max_n; ++i) {
      fill_ramp(address_of(input(i, 0)), m, i, m);
    }

    for (size_t n : ns) {
      if (m % element_count != 0) continue;

      // TODO(dsharlet): Find a better way to test the padding capability.
      for (size_t n_input : {m, m - 1, m / 2}) {
        size_t n_bytes = ceil_div(n_input * sizeof(T), element_count);
        kernel(m, n, n_bytes, input.stride_bytes(0), input.base(),
               output.stride_bytes(0), output.base());

        // Verify results.
        for (size_t i = 0; i < m; ++i) {
          const T* output_i = address_of(output(i, 0));
          if (element_count > 1 && i >= n_input) continue;
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
  Tensor<T> input({factor, max_n});
  Buffer<T> output(factor * max_n);
  for (size_t i = 0; i < factor; ++i) {
    fill_ramp(address_of(input(i, 0)), max_n, i, factor);
  }
  for (size_t m = 1; m <= factor; ++m) {
    for (size_t n : simd_sizes_up_to(max_n, element_count)) {
      kernel(factor, m, n, input.stride_bytes(0), input.base(), output.data());

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
          ASSERT_EQ(output[j * factor + i], type_info<T>::get(&expected, 0));
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
  const size_t element_count = std::max<size_t>(1, 8 / element_size_bits);
  switch_element_size(element_size_bits, [&](auto type) {
    auto sizes = unaligned_sizes(element_size_bits);
    TestTranspose(type, kernel.kernel, {element_count}, sizes);
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

TEST_P(Transpose, sub_element_n_bytes) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  if (kernel.element_size_bits != 32) GTEST_SKIP();

  constexpr size_t kElemBytes = 4;
  // Test shapes spanning partial SVLs, svl, 2*svl, and >4*svl (up to SVL=128B),
  // with byte-granularity n_bytes_a including 0 and non-multiples of 4.
  const std::vector<std::pair<size_t, size_t>> shapes = {
      {1, 1},   {3, 5},   {15, 16}, {16, 15}, {17, 17},
      {33, 35}, {37, 65}, {70, 19}, {70, 70}, {135, 70},
  };
  for (const auto& [m, n] : shapes) {
    const size_t stride_a = m * kElemBytes + 13;
    const size_t stride_x = n * kElemBytes + 17;
    // Offset buffers by 1 byte to also exercise unaligned pointers under UBSan.
    std::vector<uint8_t> input_storage(n * stride_a + 1);
    std::vector<uint8_t> output_storage(m * stride_x + 1, 0xCD);
    uint8_t* input = input_storage.data() + 1;
    uint8_t* output = output_storage.data() + 1;

    for (size_t j = 0; j < n; ++j) {
      for (size_t b = 0; b < m * kElemBytes; ++b) {
        input[j * stride_a + b] =
            static_cast<uint8_t>((j * 31 + b * 7 + 1) & 0xFF);
      }
    }

    const size_t full_bytes = m * kElemBytes;
    std::vector<size_t> n_bytes_cases = {0, 1, 2, 3, full_bytes};
    if (full_bytes > 5) n_bytes_cases.push_back(5);
    if (full_bytes > 7) n_bytes_cases.push_back(7);
    if (full_bytes > 3) n_bytes_cases.push_back(full_bytes - 1);
    if (full_bytes > 3) n_bytes_cases.push_back(full_bytes - 3);
    if (full_bytes > 8) n_bytes_cases.push_back((full_bytes / 2) | 1);

    for (size_t n_bytes_a : n_bytes_cases) {
      std::fill(output_storage.begin(), output_storage.end(), 0xCD);
      kernel.kernel(m, n, n_bytes_a, stride_a, input, stride_x, output);

      ASSERT_EQ(output_storage[0], 0xCD)
          << "m=" << m << " n=" << n << " n_bytes_a=" << n_bytes_a;
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
          for (size_t b = 0; b < kElemBytes; ++b) {
            const size_t src_byte = i * kElemBytes + b;
            const uint8_t expected =
                src_byte < n_bytes_a ? input[j * stride_a + src_byte] : 0;
            ASSERT_EQ(output[i * stride_x + j * kElemBytes + b], expected)
                << "m=" << m << " n=" << n << " n_bytes_a=" << n_bytes_a
                << " i=" << i << " j=" << j << " b=" << b;
          }
        }
        for (size_t b = n * kElemBytes; b < stride_x; ++b) {
          ASSERT_EQ(output[i * stride_x + b], 0xCD)
              << "m=" << m << " n=" << n << " n_bytes_a=" << n_bytes_a
              << " i=" << i << " pad_b=" << b;
        }
      }
    }
  }
}

TEST_P(Transpose, large_and_pow2_stride) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  if (kernel.element_size_bits != 32) GTEST_SKIP();

  // Exercises kMBlock (512), 4*svl/2*svl/svl loops (up to SVL=128B), narrow
  // strip-tiled fallbacks, bottom tails, and power-of-2 row strides.
  const std::vector<std::tuple<size_t, size_t, size_t, size_t>> cases = {
      {520, 40, 2080, 160},
      {515, 70, 2064, 296},
      {12, 135, 64, 560},
      {135, 70, 1024, 512},
      {70, 135, 512, 1024},
      {130, 16, 1024, 64},
      {130, 32, 1024, 128},
      {64, 64, 2048, 2048},
  };
  for (const auto& [m, n, stride_a, stride_x] : cases) {
    std::vector<uint8_t> input(n * stride_a, 0);
    std::vector<uint8_t> output(m * stride_x, 0xAB);
    for (size_t j = 0; j < n; ++j) {
      uint32_t* row = reinterpret_cast<uint32_t*>(input.data() + j * stride_a);
      for (size_t i = 0; i < m; ++i) {
        row[i] = static_cast<uint32_t>(j * 10007 + i * 3 + 1);
      }
    }
    kernel.kernel(m, n, m * 4, stride_a, input.data(), stride_x, output.data());
    for (size_t i = 0; i < m; ++i) {
      const uint32_t* row =
          reinterpret_cast<const uint32_t*>(output.data() + i * stride_x);
      for (size_t j = 0; j < n; ++j) {
        ASSERT_EQ(row[j], static_cast<uint32_t>(j * 10007 + i * 3 + 1))
            << "m=" << m << " n=" << n << " i=" << i << " j=" << j;
      }
      for (size_t b = n * 4; b < stride_x; ++b) {
        ASSERT_EQ(output[i * stride_x + b], 0xAB)
            << "m=" << m << " n=" << n << " i=" << i << " pad_b=" << b;
      }
    }
  }
}

TEST_P(Transpose, zero_dimensions) {
  TransposeParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  if (kernel.element_size_bits != 32) GTEST_SKIP();

  std::vector<uint8_t> input(64, 0x3C);
  std::vector<uint8_t> output(64, 0x5A);
  kernel.kernel(0, 16, 0, 64, input.data(), 64, output.data());
  kernel.kernel(16, 0, 64, 64, input.data(), 64, output.data());
  kernel.kernel(0, 0, 0, 0, input.data(), 0, output.data());
  for (uint8_t byte : output) {
    ASSERT_EQ(byte, 0x5A);
  }
}

TEST(GetTiledTranspose, x32) {
  transpose_fn tiled = get_tiled_transpose(32);
  ASSERT_NE(tiled, nullptr);
  TestTranspose(uint32_t{}, tiled, {1, 12, 15, 16, 64, 520},
                {1, 15, 16, 40, 135});
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
