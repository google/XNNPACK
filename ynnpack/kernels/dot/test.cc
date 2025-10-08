// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <ostream>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/build_config.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/dot.h"

namespace ynn {

struct DotShape {
  size_t m, n, k;

  DotShape with_m(size_t m) const { return DotShape{m, n, k}; }
  DotShape with_n(size_t n) const { return DotShape{m, n, k}; }
  DotShape with_k(size_t k) const { return DotShape{m, n, k}; }
};

std::ostream& operator<<(std::ostream& os, const DotShape& shape) {
  return os << shape.m << "x" << shape.n << "x" << shape.k;
}

// Align the last two dimensions of x up to a multiple of a2, a1, with zero
// padding.
template <typename T>
Tensor<T> AlignUp(Tensor<T> x, size_t a2, size_t a1) {
  std::vector<size_t> extents = x.extents();
  extents[extents.size() - 2] = align_up(extents[extents.size() - 2], a2);
  extents[extents.size() - 1] = align_up(extents[extents.size() - 1], a1);
  if (extents == x.extents()) {
    return x;
  }

  Tensor<T> aligned(extents, Alignment({.bytes = a1 * a2 * sizeof(T)}));

  aligned.fill(0);
  Tensor<T> cropped = aligned;
  cropped.set_shape(x.extents(), aligned.strides());
  cropped.assign(x);

  return aligned;
}

// If `tile_k > 1`, we need to transpose b such that `tile_k` values of the k
// dimension are contiguous in memory.
template <typename T>
Tensor<T> PackB(Tensor<T> b, size_t tile_k, size_t tile_n) {
  Tensor<T> aligned_b = AlignUp(b, tile_k, tile_n);
  if (tile_k == 1) {
    return aligned_b;
  }

  // The following is basically:
  //   b.reshape(..., k/tile_k, tile_k, n).transpose(..., 1, 0)
  size_t k = aligned_b.extent(b.rank() - 2);
  std::vector<int32_t> perm(aligned_b.rank() - 2);
  std::iota(perm.begin(), perm.end(), 0);
  perm.push_back(aligned_b.rank() - 2);
  perm.push_back(aligned_b.rank());
  perm.push_back(aligned_b.rank() - 1);
  Tensor<T> packed_b =
      aligned_b.split(aligned_b.rank() - 2, {k / tile_k, tile_k})
          .transpose(perm)
          .deep_copy(Alignment{.bytes = YNN_ALLOCATION_ALIGNMENT});
  packed_b.set_shape(aligned_b.extents(), aligned_b.strides());
  return packed_b;
}

template <typename T>
Tensor<T> TransposeA(Tensor<T> a, size_t tile_k) {
  Tensor<T> aligned_a = AlignUp(a, 1, tile_k);

  // The following is basically:
  //   b.reshape(..., m, k/tile_k, tile_k).transpose(..., 2, 1, 0)
  size_t k = aligned_a.extent(a.rank() - 1);
  std::vector<int32_t> perm(aligned_a.rank() - 2);
  std::iota(perm.begin(), perm.end(), 0);
  perm.push_back(aligned_a.rank() - 1);
  perm.push_back(aligned_a.rank() - 2);
  perm.push_back(aligned_a.rank());
  Tensor<T> packed_a =
      aligned_a.split(aligned_a.rank() - 1, {k / tile_k, tile_k})
          .transpose(perm)
          .deep_copy(Alignment{.bytes = YNN_ALLOCATION_ALIGNMENT});
  packed_a = packed_a.fuse({aligned_a.rank() - 1, aligned_a.rank()});
  return packed_a;
}

template <typename AT, typename BT, typename CT>
void Reference(Tensor<AT> a, Tensor<BT> b, Tensor<CT> c) {
  using B_info = type_info<BT>;

  // This helper allows omitting 2 of the 3 k dimensions. Canonicalize to 3 k
  // dimensions here.
  while (a.rank() < 4 && b.rank() < 4) {
    a = a.expand_dims({1});
    b = b.expand_dims({0});
  }

  ASSERT_EQ(c.rank(), 2);
  const size_t K3 = a.extent(1);
  const size_t K2 = a.extent(2);
  const size_t K1 = a.extent(3);
  ASSERT_EQ(c.extent(0), a.extent(0));
  ASSERT_EQ(c.extent(1), b.extent(3) * B_info::element_count());
  ASSERT_EQ(K3, b.extent(0));
  ASSERT_EQ(K2, b.extent(1));
  ASSERT_EQ(K1, b.extent(2));
  for (size_t i = 0; i < c.extent(0); ++i) {
    CT* c_i = &c(i, 0);
    for (size_t k3 = 0; k3 < K3; ++k3) {
      for (size_t k2 = 0; k2 < K2; ++k2) {
        for (size_t k1 = 0; k1 < K1; ++k1) {
          const CT a_ik = static_cast<CT>(a(i, k3, k2, k1));
          const BT* b_k1 = &b(k3, k2, k1, 0);
          for (size_t j = 0; j < c.extent(1); ++j) {
            c_i[j] = c_i[j] + a_ik * static_cast<CT>(B_info::get(b_k1, j));
          }
        }
      }
    }
  }
}

struct KernelInfo {
  uint64_t arch_flags;
  dot_kernel_fn kernel;
  DotShape block_shape;
  size_t tile_n;
  size_t tile_k;
  uint32_t flags;
  multi_type type;
};

template <typename AT, typename BT, typename CT>
void TestMatMul(AT, BT, CT, const DotShape& shape, const KernelInfo& kernel,
                bool init_zero = false) {
  using B_info = type_info<BT>;

  ReplicableRandomDevice rng;

  const size_t tile_k = kernel.tile_k;
  const size_t tile_n = kernel.tile_n;
  const size_t m = shape.m;
  const size_t n = shape.n;
  const size_t k = shape.k;
  const bool transpose_a = kernel.flags & dot_flag::transpose_a;

  const float max_abs_value = 10.0f;
  TypeGenerator<AT> a_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<BT> b_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<CT> c_gen(-max_abs_value, max_abs_value, quantization_params{});

  Tensor<AT> a({m, k});
  Tensor<BT> b({k, n / B_info::element_count()},
               Alignment{.bytes = tile_n * sizeof(BT)});
  Tensor<CT> c({m, n});
  Tensor<CT> expected;

  a.generate([&]() { return a_gen(rng); });
  b.generate([&]() { return b_gen(rng); });
  if (init_zero) {
    expected = Tensor<CT>({m, n});
    expected.fill(0);
  } else {
    c.generate([&]() { return c_gen(rng); });
    expected = c.deep_copy();
  }

  // dot kernels require B's k and n dimensions to be aligned to tile_k,
  // tile_n. The kernel might also require b to be packed (tile_k > 1).
  Tensor<BT> packed_b = PackB(b, tile_k, tile_n);
  Tensor<AT> packed_a = transpose_a ? TransposeA(a, tile_k) : a;

  kernel.kernel(m, n, 1, 1, k, packed_a.stride(0) * sizeof(AT), 0, 0,
                packed_a.base(), 0, 0, packed_b.stride(0) * sizeof(BT),
                packed_b.base(), c.stride(0) * sizeof(CT),
                init_zero ? nullptr : c.base(), c.stride(0) * sizeof(CT),
                c.base());

  // Verify results.
  Reference(a, b, expected);
  for (const auto& i : EnumerateIndices({m, n})) {
    if (std::is_integral<CT>::value) {
      ASSERT_EQ(c(i), expected(i)) << shape;
    } else {
      const float tolerance = epsilon(type_of<CT>()) * (k + 1) * max_abs_value *
                              max_abs_value * 2.0f;
      ASSERT_NEAR(c(i), expected(i), tolerance) << shape;
    }
  }
}

template <typename AT, typename BT, typename CT>
void TestConv2D(AT, BT, CT, const KernelInfo& kernel) {
  using B_info = type_info<BT>;

  ReplicableRandomDevice rng;

  const DotShape& block_shape = kernel.block_shape;
  const size_t tile_k = kernel.tile_k;
  const size_t tile_n = kernel.tile_n;
  const bool transpose_a = kernel.flags & dot_flag::transpose_a;

  // We always have m = 1, because that would just be a batch dimension here,
  // it does not exercise the dot kernel API.
  // TODO: Try to parameterize the test on this.
  struct Conv2DShape {
    size_t w, kw, kh, co, ci;
  };

  const auto cos = simd_sizes_up_to(block_shape.n, B_info::element_count());
  std::vector<Conv2DShape> shapes;
  for (size_t w = 1; w <= block_shape.m; ++w) {
    for (size_t kw : {1, 3}) {
      for (size_t kh : {2}) {
        for (size_t co : cos) {
          for (size_t ci = tile_k; ci <= tile_k * 3; ci += tile_k) {
            shapes.push_back({w, kw, kh, co, ci});
          }
        }
      }
    }
  }

  const float max_abs_value = 10.0f;
  TypeGenerator<AT> a_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<BT> b_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<CT> c_gen(-max_abs_value, max_abs_value, quantization_params{});

  for (const Conv2DShape& shape : shapes) {
    const size_t w = shape.w;
    const size_t kw = shape.kw;
    const size_t kh = shape.kh;
    const size_t co = shape.co;
    const size_t ci = shape.ci;

    Tensor<AT> a({kh, w + kw - 1, ci});
    // dot kernels assume that the rows of b are aligned to a multiple of
    // block_n.
    Tensor<BT> b({kh, kw, ci, co / B_info::element_count()},
                 Alignment{.bytes = tile_n * sizeof(BT)});
    Tensor<CT> c({w, co});

    a.generate([&]() { return a_gen(rng); });
    b.generate([&]() { return b_gen(rng); });

    // Fill the output with some random data, and copy it for the reference
    // result before running the kernel, which updates it in place.
    c.generate([&]() { return c_gen(rng); });
    Tensor<CT> expected = c.deep_copy();

    // We need to transpose before making the stencil, otherwise we "realize"
    // the im2col in memory.
    Tensor<AT> packed_a = transpose_a ? TransposeA(a, tile_k) : a;

    if (transpose_a) {
      // When we transpose, we tile_k, making the kw dimension tile_k times
      // bigger, which also dilates the kernel.
      // [kh, ci/tile_k, wkw*tile_k] -> [kh, ci/tile_k, kw*tile_k, w]
      packed_a = make_stencil_dim(packed_a, 2, kw * tile_k, /*stride=*/1,
                                  /*dilation=*/tile_k);

      // [kh, ci/tile_k, kw*tile_k, w] -> [ci/tile_k, kh, kw*tile_k, w]
      packed_a = packed_a.transpose({1, 0, 2, 3});
    } else {
      // [kh, wkw, ci] -> [kh, kw, w, ci]
      packed_a = make_stencil_dim(packed_a, 1, kw);

      // [kh, kw, w, ci] -> [w, kh, kw, ci]
      packed_a = packed_a.transpose({2, 0, 1, 3});
    }

    // dot kernels require B's k and n dimensions to be aligned to tile_k,
    // tile_n. The kernel might also require b to be packed (tile_k > 1).
    Tensor<BT> packed_b = PackB(b, tile_k, tile_n);

    b = b.crop_padding({0, 0, 0, 0},
                       {0, 0, 0, b.extent(3) - co / B_info::element_count()});

    kernel.kernel(
        w, co, kh, kw, ci, packed_a.stride(0) * sizeof(AT),
        packed_a.stride(1) * sizeof(AT), packed_a.stride(2) * sizeof(AT),
        packed_a.base(), packed_b.stride(0) * sizeof(BT),
        packed_b.stride(1) * sizeof(BT), packed_b.stride(2) * sizeof(BT),
        packed_b.base(), c.stride(0) * sizeof(CT), c.base(),
        c.stride(0) * sizeof(CT), c.base());

    // Verify results.
    a = make_stencil_dim(a, 1, kw).transpose({2, 0, 1, 3});
    // a dimensions are now {n, kh, kw, ci}
    Reference(a, b, expected);
    for (const auto& i : EnumerateIndices({w, co})) {
      if (std::is_integral<CT>::value) {
        ASSERT_EQ(c(i), expected(i));
      } else {
        const float tolerance = epsilon(type_of<CT>()) * (ci * kh * kw + 1) *
                                max_abs_value * max_abs_value * 2.0f;
        ASSERT_NEAR(c(i), expected(i), tolerance);
      }
    }
  }
}

const char* to_string(const KernelInfo& param) { return ""; }

class Dot : public ::testing::TestWithParam<KernelInfo> {};

TEST_P(Dot, Block) {
  KernelInfo kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const DotShape& block_shape = kernel.block_shape;
  SwitchThreeTypes(kernel.type, [&](auto a_type, auto b_type, auto c_type) {
    TestMatMul(a_type, b_type, c_type, block_shape, kernel);
  });
}

TEST_P(Dot, BlockInitZero) {
  KernelInfo kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const DotShape& block_shape = kernel.block_shape;
  SwitchThreeTypes(kernel.type, [&](auto a_type, auto b_type, auto c_type) {
    TestMatMul(a_type, b_type, c_type, block_shape, kernel, /*init_zero=*/true);
  });
}

TEST_P(Dot, BlockAlignedN) {
  KernelInfo kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const DotShape& block_shape = kernel.block_shape;
  SwitchThreeTypes(kernel.type, [&](auto a_type, auto b_type, auto c_type) {
    for (size_t n = 2; n <= 4; ++n) {
      TestMatMul(a_type, b_type, c_type, block_shape.with_n(block_shape.n * n),
                 kernel);
    }
  });
}

TEST_P(Dot, BlockAlignedK) {
  KernelInfo kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const DotShape& block_shape = kernel.block_shape;
  SwitchThreeTypes(kernel.type, [&](auto a_type, auto b_type, auto c_type) {
    for (size_t k = 2; k <= 4; ++k) {
      TestMatMul(a_type, b_type, c_type, block_shape.with_k(block_shape.k * k),
                 kernel);
    }
  });
}

TEST_P(Dot, TileAlignedN) {
  KernelInfo kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const DotShape& block_shape = kernel.block_shape;
  SwitchThreeTypes(kernel.type, [&](auto a_type, auto b_type, auto c_type) {
    for (size_t n = kernel.tile_n; n < block_shape.n; n += kernel.tile_n) {
      TestMatMul(a_type, b_type, c_type, block_shape.with_n(block_shape.n * n),
                 kernel);
    }
  });
}

TEST_P(Dot, TileAlignedK) {
  KernelInfo kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const DotShape& block_shape = kernel.block_shape;
  SwitchThreeTypes(kernel.type, [&](auto a_type, auto b_type, auto c_type) {
    for (size_t k = kernel.tile_k; k < block_shape.k; k += kernel.tile_k) {
      TestMatMul(a_type, b_type, c_type, block_shape.with_k(block_shape.k * k),
                 kernel);
    }
  });
}

TEST_P(Dot, UnalignedM) {
  KernelInfo kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const DotShape& block_shape = kernel.block_shape;
  SwitchThreeTypes(kernel.type, [&](auto a_type, auto b_type, auto c_type) {
    for (size_t m = 1; m < block_shape.m; ++m) {
      TestMatMul(a_type, b_type, c_type, block_shape.with_m(m), kernel);
    }
  });
}

TEST_P(Dot, UnalignedN) {
  KernelInfo kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const DotShape& block_shape = kernel.block_shape;
  SwitchThreeTypes(kernel.type, [&](auto a_type, auto b_type, auto c_type) {
    const size_t align_n = type_element_count(type_of<decltype(b_type)>());
    for (size_t n = align_n; n < 2 * block_shape.n; n += align_n) {
      TestMatMul(a_type, b_type, c_type, block_shape.with_n(n), kernel);
    }
  });
}

TEST_P(Dot, UnalignedNInitZero) {
  KernelInfo kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  const DotShape& block_shape = kernel.block_shape;
  SwitchThreeTypes(kernel.type, [&](auto a_type, auto b_type, auto c_type) {
    const size_t align_n = type_element_count(type_of<decltype(b_type)>());
    for (size_t n = align_n; n < 2 * block_shape.n; n += align_n) {
      TestMatMul(a_type, b_type, c_type, block_shape.with_n(n), kernel,
                 /*init_zero=*/true);
    }
  });
}

TEST_P(Dot, Conv2D) {
  KernelInfo kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchThreeTypes(kernel.type, [&](auto a_type, auto b_type, auto c_type) {
    TestConv2D(a_type, b_type, c_type, kernel);
  });
}

#define YNN_DOT_KERNEL(arch_flags, name, block_m, block_n, block_k, tile_n, \
                       tile_k, flags, a_type, b_type, c_type)               \
  INSTANTIATE_TEST_SUITE_P(name, Dot,                                       \
                           testing::Values(KernelInfo{                      \
                               arch_flags,                                  \
                               name,                                        \
                               {block_m, block_n, block_k},                 \
                               tile_n,                                      \
                               tile_k,                                      \
                               flags,                                       \
                               multi_type_of(a_type(), b_type(), c_type())}));
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL

}  // namespace ynn
