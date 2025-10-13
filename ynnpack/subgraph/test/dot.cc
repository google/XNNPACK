// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

constexpr int max_k_dims = 3;

std::uniform_int_distribution<size_t> rank_dist(2, max_k_dims + 2);
std::uniform_int_distribution<size_t> dim_dist(1, 512);
// Large k2, k3 dimensions are unrealistic and very slow to test.
std::uniform_int_distribution<size_t> k23_dim_dist(1, 5);
std::bernoulli_distribution random_bool(0.5);

std::vector<int32_t> iota(int32_t begin, int32_t size) {
  std::vector<int32_t> iota(size);
  std::iota(iota.begin(), iota.end(), begin);
  return iota;
}

// Returns {x[i] for i in perm}
template <typename T>
std::vector<T> permute(const std::vector<int>& perm, const std::vector<T>& x) {
  std::vector<T> result(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    result[i] = x[perm[i]];
  }
  return result;
}

template <typename T>
std::vector<size_t> align_logical_shape(std::vector<size_t> shape) {
  shape.back() = align_up(shape.back(), type_element_count(type_of<T>()));
  return shape;
}

template <typename T>
std::vector<size_t> to_physical_shape(std::vector<size_t> shape) {
  assert(shape.back() % type_element_count(type_of<T>()) == 0);
  shape.back() /= type_element_count(type_of<T>());
  return shape;
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

  const size_t M = c.extent(0);
  const size_t N = c.extent(1);
  ASSERT_EQ(c.rank(), 2);
  ASSERT_EQ(M, a.extent(0));
  ASSERT_EQ(N, b.extent(3) * B_info::element_count());
  const size_t K3 = a.extent(1);
  const size_t K2 = a.extent(2);
  const size_t K1 = a.extent(3);
  ASSERT_EQ(b.extent(0), K3);
  ASSERT_EQ(b.extent(1), K2);
  ASSERT_EQ(b.extent(2), K1);

  for (size_t i = 0; i < M; ++i) {
    CT* c_i = &c(i, 0);
    for (size_t k3 = 0; k3 < K3; ++k3) {
      for (size_t k2 = 0; k2 < K2; ++k2) {
        for (size_t k1 = 0; k1 < K1; ++k1) {
          const CT a_ik = static_cast<CT>(a(i, k3, k2, k1));
          const BT* b_k1 = &b(k3, k2, k1, 0);
          for (size_t j = 0; j < N; ++j) {
            c_i[j] = c_i[j] + a_ik * static_cast<CT>(B_info::get(b_k1, j));
          }
        }
      }
    }
  }
}

// Remove the leading `at` dimensions of `tensor`
template <typename T>
Tensor<T> slice_batches(Tensor<T> tensor, std::vector<size_t> at) {
  tensor = tensor.slice_leading(at);
  std::vector<size_t> shape = tensor.shape();
  std::vector<size_t> strides = tensor.strides();
  shape.erase(shape.begin(), shape.begin() + at.size());
  strides.erase(strides.begin(), strides.begin() + at.size());
  tensor.set_shape(shape, strides);
  return tensor;
}

template <typename A, typename B, typename C>
void TestStaticB(A, B, C) {
  ReplicableRandomDevice rng;

  const float max_abs_value = 10.0f;
  TypeGenerator<A> a_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<B> b_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<C> c_gen(-max_abs_value, max_abs_value, quantization_params{});

  TestScheduler scheduler(3);

  for (auto _ : FuzzTest(std::chrono::seconds(5))) {
    const size_t input_rank = rank_dist(rng);
    const size_t max_k_dims = std::min<size_t>(2, input_rank - 2) + 1;
    const size_t num_k_dims =
        std::uniform_int_distribution<size_t>(1, max_k_dims)(rng);
    const size_t output_rank = input_rank - num_k_dims + 1;

    // We want the total number of output elements to be reasonable, so choose
    // `max_dim` such that a random shape of rank `output_rank` produces this
    // max size.
    constexpr size_t max_size = 20;
    const size_t max_output_dim = static_cast<size_t>(std::ceil(
        std::pow(static_cast<double>(max_size),
                 1.0 / static_cast<double>(std::max<size_t>(1, output_rank)))));

    // Make the dot dimensions for b (k..., n)
    std::vector<size_t> b_shape(num_k_dims + 1);
    std::generate(b_shape.begin(), b_shape.end(),
                  [&]() { return dim_dist(rng); });
    for (int d = 1; d < num_k_dims; ++d) {
      b_shape[d] = k23_dim_dist(rng);
    }

    // Align the shape as required by the type of B.
    b_shape = align_logical_shape<B>(b_shape);

    Tensor<B> b(to_physical_shape<B>(b_shape));
    b.generate([&]() { return b_gen(rng); });

    SubgraphBuilder subgraph(4);
    const uint32_t a_id = 0;
    const uint32_t b_id = 1;
    const uint32_t output_id = 3;
    subgraph.AddInput(type_of<A>(), input_rank, a_id)
        .AddTensor(b, b_id)
        .AddOutput(type_of<C>(), output_rank, output_id);

    uint32_t c_id = 2;
    const bool init_c = random_bool(rng);
    const C init_value = random_bool(rng) ? c_gen(rng) : static_cast<C>(0);
    if (init_c) {
      if (init_value == 0 && random_bool(rng)) {
        c_id = YNN_INVALID_VALUE_ID;
      } else {
        subgraph.AddScalar<C>(init_value, c_id);
      }
    } else {
      subgraph.AddInput(type_of<C>(), output_rank, c_id);
    }

    subgraph.AddDot(num_k_dims, a_id, b_id, c_id, output_id);

    Runtime runtime(subgraph.GetSubgraph(),
                    random_bool(rng) ? &scheduler : nullptr);
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> batch_dims =
          random_shape(rng, output_rank - 1, 1, max_output_dim);

      // Start with the batch dimensions.
      std::vector<size_t> c_shape = batch_dims;
      std::vector<size_t> a_shape = batch_dims;
      // After this, don't treat the m dimension as a batch dimension.
      batch_dims.pop_back();

      // The output gets n.
      c_shape.push_back(b_shape.back());
      // A gets the k dims.
      for (size_t i = 0; i < num_k_dims; ++i) {
        a_shape.push_back(b_shape[i]);
      }

      Tensor<A> a(a_shape);
      a.generate([&]() { return a_gen(rng); });

      runtime.ReshapeExternalTensor(a_shape, a.data(), a_id);

      Tensor<C> c(c_shape);
      if (!init_c) {
        c.generate([&]() { return c_gen(rng); });
        runtime.ReshapeExternalTensor(c_shape, c.data(), c_id);
      }
      runtime.ReshapeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      ASSERT_EQ(runtime.GetExternalTensorShape(output_id), c_shape);

      Tensor<C> expected(c_shape);
      if (init_c) {
        expected.fill(init_value);
      } else {
        // Copy the expected output before running the pipeline, since it is
        // updated in place by the test and reference implementations.
        expected.assign(c);
      }

      runtime.SetupExternalTensor(c.data(), output_id).InvokeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      Tensor<B> broadcasted_b = b;
      while (broadcasted_b.rank() < a.rank()) {
        broadcasted_b = broadcasted_b.expand_dims({0});
      }
      for (const auto& i : EnumerateIndices(batch_dims)) {
        Reference(slice_batches(a, i), slice_batches(broadcasted_b, i),
                  slice_batches(expected, i));
      }
      size_t num_k_elements = 1;
      for (size_t i = 0; i < num_k_dims; ++i) {
        num_k_elements *= b_shape[i];
      }
      for (const auto& i : EnumerateIndices(c_shape)) {
        if (std::is_integral<C>::value) {
          ASSERT_EQ(c(i), expected(i))
              << "i=" << index_to_string(i) << " num_k_dims=" << num_k_dims
              << " a_shape=" << index_to_string(a_shape)
              << " b_shape=" << index_to_string(b_shape);
        } else {
          const float tolerance = epsilon(type_of<C>()) * (num_k_elements + 1) *
                                  max_abs_value * max_abs_value * 2.0f;
          ASSERT_NEAR(c(i), expected(i), tolerance)
              << "i=" << index_to_string(i) << " num_k_dims=" << num_k_dims
              << " a_shape=" << index_to_string(a_shape)
              << " b_shape=" << index_to_string(b_shape);
        }
      }
    }
  }
}

template <typename A, typename B, typename C>
void TestDynamicB(A, B, C) {
  ReplicableRandomDevice rng;

  const float max_abs_value = 10.0f;
  TypeGenerator<A> a_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<B> b_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<C> c_gen(-max_abs_value, max_abs_value, quantization_params{});

  TestScheduler scheduler(3);

  for (auto _ : FuzzTest(std::chrono::seconds(5))) {
    const size_t a_rank = rank_dist(rng);
    const size_t b_rank = rank_dist(rng);
    const size_t max_k_dims =
        std::min<size_t>(2, std::min(a_rank, b_rank) - 2) + 1;
    const size_t num_k_dims =
        std::uniform_int_distribution<size_t>(1, max_k_dims)(rng);
    const size_t output_rank = std::max(a_rank, b_rank) - num_k_dims + 1;

    // We want the total number of output elements to be reasonable, so choose
    // `max_dim` such that a random shape of rank `output_rank` produces this
    // max size.
    constexpr size_t max_size = 20;
    const size_t max_output_dim = static_cast<size_t>(std::ceil(
        std::pow(static_cast<double>(max_size),
                 1.0 / static_cast<double>(std::max<size_t>(1, output_rank)))));

    SubgraphBuilder subgraph(4);
    const uint32_t a_id = 0;
    const uint32_t b_id = 1;
    const uint32_t output_id = 3;
    subgraph.AddInput(type_of<A>(), a_rank, a_id)
        .AddInput(type_of<B>(), b_rank, b_id)
        .AddOutput(type_of<C>(), output_rank, output_id);

    std::vector<int> b_perm;
    uint32_t b_tr_id = b_id;
    // `Tensor::transpose` doesn't support type_element_count != 1, so it's
    // tricky to test that codepath.
    if (type_element_count(type_of<B>()) == 1) {
      b_perm = iota(0, b_rank);
      // Make a random transpose. We can only optimize transposes where the
      // stride 1 dimension is the n or k dimension, so filter out the rest (50%
      // of the time).
      std::shuffle(b_perm.begin(), b_perm.end(), rng);
      if (b_perm[b_perm.size() - 1] == b_perm.size() - 1 ||
          b_perm[b_perm.size() - 2] == b_perm.size() - 1 || random_bool(rng)) {
        b_tr_id = YNN_INVALID_VALUE_ID;
        subgraph.AddTensor(type_of<B>(), b_rank, b_tr_id);
        subgraph.AddTranspose(b_perm, b_id, b_tr_id);
      }
    }

    uint32_t c_id = 2;
    const bool init_c = random_bool(rng);
    const C init_value = random_bool(rng) ? c_gen(rng) : static_cast<C>(0);
    if (init_c) {
      if (init_value == 0 && random_bool(rng)) {
        c_id = YNN_INVALID_VALUE_ID;
      } else {
        subgraph.AddScalar<C>(init_value, c_id);
      }
    } else {
      subgraph.AddInput(type_of<C>(), output_rank, c_id);
    }

    subgraph.AddDot(num_k_dims, a_id, b_tr_id, c_id, output_id);

    Runtime runtime(subgraph.GetSubgraph(),
                    random_bool(rng) ? &scheduler : nullptr);
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> batch_dims =
          random_shape(rng, output_rank - 2, 1, max_output_dim);

      // Start with the batch dimensions.
      std::vector<size_t> c_shape = batch_dims;
      std::vector<size_t> a_shape = batch_dims;
      std::vector<size_t> b_shape = batch_dims;

      // Add the dot dimensions (m, k..., n)
      std::vector<size_t> dot_dims(num_k_dims + 2);
      std::generate(dot_dims.begin(), dot_dims.end(),
                    [&]() { return dim_dist(rng); });
      for (int d = 1; d < num_k_dims; ++d) {
        dot_dims[1 + d] = k23_dim_dist(rng);
      }

      // Align the shape as required by the type of B.
      dot_dims = align_logical_shape<B>(dot_dims);

      // The output gets m, n.
      c_shape.push_back(dot_dims[0]);
      c_shape.push_back(dot_dims.back());
      // A gets m.
      a_shape.push_back(dot_dims[0]);
      // A and B get the k dims.
      for (size_t i = 0; i < num_k_dims; ++i) {
        a_shape.push_back(dot_dims[i + 1]);
        b_shape.push_back(dot_dims[i + 1]);
      }
      // B gets n.
      b_shape.push_back(dot_dims.back());

      // Remove the broadcasted batch dimensions from a and b.
      size_t a_broadcast_dims = a_shape.size() - a_rank;
      size_t b_broadcast_dims = b_shape.size() - b_rank;
      a_shape.erase(a_shape.begin(), a_shape.begin() + a_broadcast_dims);
      b_shape.erase(b_shape.begin(), b_shape.begin() + b_broadcast_dims);

      Tensor<A> a(a_shape);
      Tensor<B> b(to_physical_shape<B>(b_shape));
      a.generate([&]() { return a_gen(rng); });
      b.generate([&]() { return b_gen(rng); });

      Tensor<B> b_tr = b;
      if (b_tr_id != b_id) {
        // Apply the inverse of the permutation, so the shape arrives as
        // expected.
        std::vector<int> inv_b_perm = iota(0, b_perm.size());
        std::sort(inv_b_perm.begin(), inv_b_perm.end(),
                  [&](int i, int j) { return b_perm[i] < b_perm[j]; });
        b_tr = b.transpose(inv_b_perm).deep_copy();
        b_shape = permute(inv_b_perm, b_shape);
      }

      runtime.ReshapeExternalTensor(a_shape, a.data(), a_id)
          .ReshapeExternalTensor(b_shape, b_tr.data(), b_id);

      Tensor<C> c(c_shape);
      if (!init_c) {
        c.generate([&]() { return c_gen(rng); });
        runtime.ReshapeExternalTensor(c_shape, c.data(), c_id);
      }
      runtime.ReshapeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      ASSERT_EQ(runtime.GetExternalTensorShape(output_id), c_shape);

      Tensor<C> expected(c_shape);
      if (init_c) {
        expected.fill(init_value);
      } else {
        // Copy the expected output before running the pipeline, since it is
        // updated in place by the test and reference implementations.
        expected.assign(c);
      }

      runtime.SetupExternalTensor(c.data(), output_id).InvokeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      // Put broadcast dimensions back for the reference computation.
      a = a.expand_dims(iota(0, a_broadcast_dims));
      b = b.expand_dims(iota(0, b_broadcast_dims));
      broadcast_extent_1(a);
      broadcast_extent_1(b);

      for (const auto& i : EnumerateIndices(batch_dims)) {
        Reference(slice_batches(a, i), slice_batches(b, i),
                  slice_batches(expected, i));
      }
      size_t num_k_elements = 1;
      for (size_t i = 0; i < num_k_dims; ++i) {
        num_k_elements *= dot_dims[i + 1];
      }
      for (const auto& i : EnumerateIndices(c_shape)) {
        if (std::is_integral<C>::value) {
          ASSERT_EQ(c(i), expected(i))
              << "i=" << index_to_string(i) << " num_k_dims=" << num_k_dims
              << " a_shape=" << index_to_string(a_shape)
              << " b_shape=" << index_to_string(b_shape);
        } else {
          const float tolerance = epsilon(type_of<C>()) * (num_k_elements + 1) *
                                  max_abs_value * max_abs_value * 2.0f;
          ASSERT_NEAR(c(i), expected(i), tolerance)
              << "i=" << index_to_string(i) << " num_k_dims=" << num_k_dims
              << " a_shape=" << index_to_string(a_shape)
              << " b_shape=" << index_to_string(b_shape);
        }
      }
    }
  }
}

class Dot : public testing::TestWithParam<multi_type> {};

TEST_P(Dot, StaticB) {
  SwitchThreeTypes(GetParam(), [&](auto a_type, auto b_type, auto c_type) {
    TestStaticB(a_type, b_type, c_type);
  });
}

TEST_P(Dot, DynamicB) {
  SwitchThreeTypes(GetParam(), [&](auto a_type, auto b_type, auto c_type) {
    TestDynamicB(a_type, b_type, c_type);
  });
}

INSTANTIATE_TEST_SUITE_P(
    Test, Dot,
    testing::Values(multi_type::fp32, multi_type::fp16_fp16_fp32,
                    multi_type::bf16_bf16_fp32, multi_type::int8_int8_int32,
                    multi_type::int8_int4_int32, multi_type::uint8_int8_int32),
    [](const testing::TestParamInfo<multi_type>& info) {
      return to_string(info.param);
    });

}  // namespace ynn
