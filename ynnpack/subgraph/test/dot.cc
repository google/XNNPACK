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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/fp8.h"
#include "ynnpack/base/half.h"
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
  ASSERT_EQ(N, b.extent(3));
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
          const BT* b_k1 = address_of(b(k3, k2, k1, 0));
          for (size_t j = 0; j < N; ++j) {
            c_i[j] = c_i[j] + a_ik * static_cast<CT>(B_info::get(b_k1, j));
          }
        }
      }
    }
  }
}

void Reference(Tensor<half> a, Tensor<half> b, Tensor<float> c) {
  Reference(a.convert<float>(), b.convert<float>(), c);
}
void Reference(Tensor<bfloat16> a, Tensor<bfloat16> b, Tensor<float> c) {
  Reference(a.convert<float>(), b.convert<float>(), c);
}
void Reference(Tensor<fp8_e4m3> a, Tensor<fp8_e4m3> b, Tensor<float> c) {
  Reference(a.convert<float>(), b.convert<float>(), c);
}
void Reference(Tensor<fp8_e5m2> a, Tensor<fp8_e5m2> b, Tensor<float> c) {
  Reference(a.convert<float>(), b.convert<float>(), c);
}

// If the output type is bf16 or fp16, we want to compute the result in fp32,
// and convert to the output type after.
template <typename AT, typename BT>
void Reference(Tensor<AT> a, Tensor<BT> b, Tensor<bfloat16> c) {
  Tensor<float> c_float(c.extents());
  c_float.assign(c);
  Reference(a, b, c_float);
  c.assign(c_float);
}

template <typename AT, typename BT>
void Reference(Tensor<AT> a, Tensor<BT> b, Tensor<half> c) {
  Tensor<float> c_float(c.extents());
  c_float.assign(c);
  Reference(a, b, c_float);
  c.assign(c_float);
}

template <typename T>
T get_dot_tolerance(size_t num_k_elements, float max_abs_value) {
  double eps = type_info<T>::epsilon();
  // Account for the initial value too.
  num_k_elements += 1;
  return eps * num_k_elements * max_abs_value * max_abs_value * 2.0;
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
void VerifyDotResults(Tensor<A> a, Tensor<B> b, Tensor<C> c, Tensor<C> expected,
                      const std::vector<size_t>& batch_dims,
                      const std::vector<size_t>& a_shape,
                      const std::vector<size_t>& b_shape, size_t num_k_dims,
                      float max_abs_value) {
  // Explicitly broadcast any implicit broadcast dimensions.
  const size_t target_rank_a = batch_dims.size() + 1 + num_k_dims;
  if (a.rank() < target_rank_a) {
    const size_t a_broadcast_dims = target_rank_a - a.rank();
    a = a.expand_dims(iota(0, a_broadcast_dims));
  }
  const size_t target_rank_b = batch_dims.size() + num_k_dims + 1;
  if (b.rank() < target_rank_b) {
    const size_t b_broadcast_dims = target_rank_b - b.rank();
    b = b.expand_dims(iota(0, b_broadcast_dims));
  }
  broadcast_extent_1(a);
  broadcast_extent_1(b);

  for (const auto& i : EnumerateIndices(batch_dims)) {
    Reference(slice_batches(a, i), slice_batches(b, i),
              slice_batches(expected, i));
  }
  size_t num_k_elements = 1;
  for (size_t i = 0; i < num_k_dims; ++i) {
    num_k_elements *= b.extent(batch_dims.size() + i);
  }
  for (const auto& i : EnumerateIndices(expected.shape())) {
    if (is_integral<C>::value) {
      ASSERT_EQ(c(i), expected(i))
          << "i=" << index_to_string(i) << " num_k_dims=" << num_k_dims
          << " a_shape=" << index_to_string(a_shape)
          << " b_shape=" << index_to_string(b_shape);
    } else {
      const auto tolerance =
          get_dot_tolerance<C>(num_k_elements, max_abs_value);
      ASSERT_NEAR(c(i), expected(i), tolerance)
          << "i=" << index_to_string(i) << " num_k_dims=" << num_k_dims
          << " a_shape=" << index_to_string(a_shape)
          << " b_shape=" << index_to_string(b_shape);
    }
  }
}

template <typename A, typename B, typename C>
void TestStaticB(A, B, C) {
  ReplicableRandomDevice rng;

  const float max_abs_value = 10.0f;

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
    for (size_t d = 0; d + 1 < num_k_dims; ++d) {
      b_shape[d] = k23_dim_dist(rng);
    }

    // Align the shape as required by the type of B.
    b_shape = align_logical_shape<B>(b_shape);

    Tensor<B> b(b_shape);
    fill_random(b.data(), b.size(), rng, -max_abs_value, max_abs_value);

    uint32_t subgraph_flags = 0;
    if (random_bool(rng)) {
      subgraph_flags |= YNN_FLAG_CONSISTENT_ARITHMETIC;
    }
    std::vector<size_t> batch_template_shape =
        random_shape(rng, output_rank - 1, 0, 9);
    std::vector<size_t> a_template_shape = batch_template_shape;
    for (size_t i = 0; i < num_k_dims; ++i) {
      a_template_shape.push_back(b_shape[i]);
    }
    SubgraphBuilder subgraph(4, subgraph_flags);
    const uint32_t a_id = 0;
    const uint32_t b_id = 1;
    const uint32_t output_id = 3;
    subgraph.AddInput(type_of<A>(), a_template_shape, a_id)
        .AddTensor(b, b_id)
        .AddOutput(type_of<C>(), output_rank, output_id);

    uint32_t c_id = 2;
    const bool init_c = random_bool(rng);
    const C init_value =
        random_bool(rng) ? random_value<C>(rng, -max_abs_value, max_abs_value)
                         : static_cast<C>(0);
    if (init_c) {
      if (init_value == 0 && random_bool(rng)) {
        c_id = YNN_INVALID_VALUE_ID;
      } else {
        subgraph.AddScalar<C>(init_value, c_id);
      }
    } else {
      subgraph.AddInput(type_of<C>(), output_rank, c_id);
    }

    uint32_t dot_flags = 0;
    if (random_bool(rng)) {
      dot_flags |= YNN_NODE_FLAG_SYMMETRIC_B;
    }
    subgraph.AddDot(num_k_dims, a_id, b_id, c_id, output_id, dot_flags);

    Runtime runtime(subgraph.GetSubgraph(),
                    random_bool(rng) ? &scheduler : nullptr);
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> batch_dims =
          random_shape(rng, batch_template_shape, 1, max_output_dim);

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
      fill_random(a.data(), a.size(), rng, -max_abs_value, max_abs_value);

      runtime.ReshapeExternalTensor(a_shape, a.data(), a_id);

      Tensor<C> c(c_shape);
      if (!init_c) {
        fill_random(c.data(), c.size(), rng, -max_abs_value, max_abs_value);
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

      VerifyDotResults(a, b, c, expected, batch_dims, a_shape, b_shape,
                       num_k_dims, max_abs_value);
    }
  }
}

struct DotShapes {
  std::vector<size_t> a, b, c;
  std::vector<size_t> batch_dims;
  std::vector<size_t> dot_dims;
};

template <typename B, typename Rng>
DotShapes RandomDotShapes(Rng& rng, size_t num_k_dims, size_t a_rank,
                          size_t b_rank, size_t output_rank) {
  // We want the total number of output elements to be reasonable, so choose
  // `max_dim` such that a random shape of rank `output_rank` produces this
  // max size.
  constexpr size_t max_size = 20;
  const size_t max_output_dim = static_cast<size_t>(std::ceil(
      std::pow(static_cast<double>(max_size),
               1.0 / static_cast<double>(std::max<size_t>(1, output_rank)))));

  std::vector<size_t> batch_dims =
      random_shape(rng, output_rank - 2, 1, max_output_dim);

  // Start with the batch dimensions.
  DotShapes shapes = {batch_dims, batch_dims, batch_dims, batch_dims};

  // Add the dot dimensions (m, k..., n)
  std::vector<size_t> dot_dims =
      random_shape(rng, num_k_dims + 2, 1, max_output_dim);
  for (size_t d = 0; d + 1 < num_k_dims; ++d) {
    dot_dims[1 + d] = k23_dim_dist(rng);
  }
  dot_dims[num_k_dims] = dim_dist(rng);

  // Align the shape as required by the type of B.
  dot_dims = align_logical_shape<B>(dot_dims);

  // The output gets m, n.
  shapes.c.push_back(dot_dims[0]);
  shapes.c.push_back(dot_dims.back());
  // A gets m.
  shapes.a.push_back(dot_dims[0]);
  // A and B get the k dims.
  for (size_t i = 0; i < num_k_dims; ++i) {
    shapes.a.push_back(dot_dims[i + 1]);
    shapes.b.push_back(dot_dims[i + 1]);
  }
  // B gets n.
  shapes.b.push_back(dot_dims.back());

  shapes.dot_dims = std::move(dot_dims);

  return shapes;
}

template <typename A, typename B, typename C>
void TestDynamicB(A, B, C) {
  ReplicableRandomDevice rng;

  const float max_abs_value = 10.0f;

  TestScheduler scheduler(3);

  for (auto _ : FuzzTest(std::chrono::seconds(5))) {
    const size_t a_rank = rank_dist(rng);
    const size_t b_rank = rank_dist(rng);
    const size_t max_k_dims =
        std::min<size_t>(2, std::min(a_rank, b_rank) - 2) + 1;
    const size_t num_k_dims =
        std::uniform_int_distribution<size_t>(1, max_k_dims)(rng);
    const size_t output_rank = std::max(a_rank, b_rank) - num_k_dims + 1;

    uint32_t subgraph_flags = 0;
    if (random_bool(rng)) {
      subgraph_flags |= YNN_FLAG_CONSISTENT_ARITHMETIC;
    }
    SubgraphBuilder subgraph(4, subgraph_flags);
    const uint32_t a_id = 0;
    const uint32_t b_id = 1;
    const uint32_t output_id = 3;
    subgraph.AddInput(type_of<A>(), a_rank, a_id)
        .AddInput(type_of<B>(), b_rank, b_id)
        .AddOutput(type_of<C>(), output_rank, output_id);

    std::vector<int> a_perm;
    uint32_t a_tr_id = a_id;
    if (type_element_count(type_of<A>()) == 1) {
      a_perm = iota(0, a_rank);
      std::shuffle(a_perm.begin(), a_perm.end(), rng);
      if (a_perm[a_perm.size() - 1] == a_perm.size() - 1 ||
          a_perm[a_perm.size() - 2] == a_perm.size() - 1 || random_bool(rng)) {
        a_tr_id = YNN_INVALID_VALUE_ID;
        subgraph.AddTensor(type_of<A>(), a_rank, a_tr_id);
        subgraph.AddTranspose(a_perm, a_id, a_tr_id);
      }
    }

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
    const C init_value =
        random_bool(rng) ? random_value<C>(rng, -max_abs_value, max_abs_value)
                         : static_cast<C>(0);
    if (init_c) {
      if (init_value == 0 && random_bool(rng)) {
        c_id = YNN_INVALID_VALUE_ID;
      } else {
        subgraph.AddScalar<C>(init_value, c_id);
      }
    } else {
      subgraph.AddInput(type_of<C>(), output_rank, c_id);
    }

    uint32_t dot_flags_dyn = 0;
    if (random_bool(rng)) {
      dot_flags_dyn |= YNN_NODE_FLAG_SYMMETRIC_B;
    }
    subgraph.AddDot(num_k_dims, a_tr_id, b_tr_id, c_id, output_id,
                    dot_flags_dyn);

    Runtime runtime(subgraph.GetSubgraph(),
                    random_bool(rng) ? &scheduler : nullptr);
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      DotShapes shapes =
          RandomDotShapes<B>(rng, num_k_dims, a_rank, b_rank, output_rank);

      // Remove the broadcasted batch dimensions from a and b.
      size_t a_broadcast_dims = shapes.a.size() - a_rank;
      size_t b_broadcast_dims = shapes.b.size() - b_rank;
      shapes.a.erase(shapes.a.begin(), shapes.a.begin() + a_broadcast_dims);
      shapes.b.erase(shapes.b.begin(), shapes.b.begin() + b_broadcast_dims);

      Tensor<A> a(shapes.a);
      Tensor<B> b(shapes.b);
      fill_random(a.data(), a.size(), rng, -max_abs_value, max_abs_value);
      fill_random(b.data(), b.size(), rng, -max_abs_value, max_abs_value);

      Tensor<A> a_tr = a;
      if (a_tr_id != a_id) {
        // Apply the inverse of the permutation, so the shape arrives as
        // expected.
        std::vector<int> inv_a_perm = iota(0, a_perm.size());
        std::sort(inv_a_perm.begin(), inv_a_perm.end(),
                  [&](int i, int j) { return a_perm[i] < a_perm[j]; });
        a_tr = a.transpose(inv_a_perm).deep_copy();
        shapes.a = permute(inv_a_perm, shapes.a);
      }

      Tensor<B> b_tr = b;
      if (b_tr_id != b_id) {
        // Apply the inverse of the permutation, so the shape arrives as
        // expected.
        std::vector<int> inv_b_perm = iota(0, b_perm.size());
        std::sort(inv_b_perm.begin(), inv_b_perm.end(),
                  [&](int i, int j) { return b_perm[i] < b_perm[j]; });
        b_tr = b.transpose(inv_b_perm).deep_copy();
        shapes.b = permute(inv_b_perm, shapes.b);
      }

      runtime.ReshapeExternalTensor(shapes.a, a_tr.data(), a_id)
          .ReshapeExternalTensor(shapes.b, b_tr.data(), b_id);

      Tensor<C> c(shapes.c);
      if (!init_c) {
        fill_random(c.data(), c.size(), rng, -max_abs_value, max_abs_value);
        runtime.ReshapeExternalTensor(shapes.c, c.data(), c_id);
      }
      runtime.ReshapeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      ASSERT_EQ(runtime.GetExternalTensorShape(output_id), shapes.c);

      Tensor<C> expected(shapes.c);
      if (init_c) {
        expected.fill(init_value);
      } else {
        // Copy the expected output before running the pipeline, since it is
        // updated in place by the test and reference implementations.
        expected.assign(c);
      }

      runtime.SetupExternalTensor(c.data(), output_id).InvokeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      VerifyDotResults(a, b, c, expected, shapes.batch_dims, shapes.a, shapes.b,
                       num_k_dims, max_abs_value);
    }
  }
}

template <typename A, typename B, typename C>
void TestStaticShapeDynamicB(A, B, C) {
  ReplicableRandomDevice rng;

  const float max_abs_value = 10.0f;

  TestScheduler scheduler(3);

  for (auto _ : FuzzTest(std::chrono::seconds(5))) {
    const size_t a_rank = rank_dist(rng);
    const size_t b_rank = rank_dist(rng);
    const size_t max_k_dims =
        std::min<size_t>(2, std::min(a_rank, b_rank) - 2) + 1;
    const size_t num_k_dims =
        std::uniform_int_distribution<size_t>(1, max_k_dims)(rng);
    const size_t output_rank = std::max(a_rank, b_rank) - num_k_dims + 1;

    DotShapes shapes =
        RandomDotShapes<B>(rng, num_k_dims, a_rank, b_rank, output_rank);

    // Remove the broadcasted batch dimensions from a and b.
    size_t a_broadcast_dims = shapes.a.size() - a_rank;
    size_t b_broadcast_dims = shapes.b.size() - b_rank;
    shapes.a.erase(shapes.a.begin(), shapes.a.begin() + a_broadcast_dims);
    shapes.b.erase(shapes.b.begin(), shapes.b.begin() + b_broadcast_dims);

    ASSERT_EQ(shapes.b.size(), b_rank);

    std::vector<int> a_perm = iota(0, a_rank);
    // `Tensor::transpose` doesn't support type_element_count != 1, so it's
    // tricky to test that codepath.
    if (type_element_count(type_of<A>()) == 1) {
      // Make a random transpose.
      std::shuffle(a_perm.begin(), a_perm.end(), rng);
      if (a_perm[a_perm.size() - 1] == a_perm.size() - 1 ||
          a_perm[a_perm.size() - 2] == a_perm.size() - 1) {
        // This transpose affects the innermost dimension.
      } else if (!random_bool(rng)) {
        a_perm = iota(0, a_rank);
      }
    }
    std::vector<int> inv_a_perm = a_perm;
    // Apply the inverse of the permutation, so the shape arrives as
    // expected.
    std::sort(inv_a_perm.begin(), inv_a_perm.end(),
              [&](int i, int j) { return a_perm[i] < a_perm[j]; });

    std::vector<int> b_perm = iota(0, b_rank);
    // `Tensor::transpose` doesn't support type_element_count != 1, so it's
    // tricky to test that codepath.
    if (type_element_count(type_of<B>()) == 1) {
      // Make a random transpose.
      std::shuffle(b_perm.begin(), b_perm.end(), rng);
      if (b_perm[b_perm.size() - 1] == b_perm.size() - 1 ||
          b_perm[b_perm.size() - 2] == b_perm.size() - 1) {
        // This transpose affects the innermost dimension.
      } else if (!random_bool(rng)) {
        // We can only optimize transposes where the stride 1 dimension is the n
        // or k dimension, so filter out the rest (50% of the time).
        b_perm = iota(0, b_rank);
      }
    }
    std::vector<int> inv_b_perm = b_perm;
    // Apply the inverse of the permutation, so the shape arrives as
    // expected.
    std::sort(inv_b_perm.begin(), inv_b_perm.end(),
              [&](int i, int j) { return b_perm[i] < b_perm[j]; });

    uint32_t subgraph_flags = 0;
    if (random_bool(rng)) {
      subgraph_flags |= YNN_FLAG_CONSISTENT_ARITHMETIC;
    }
    SubgraphBuilder subgraph(4, subgraph_flags);
    const uint32_t a_id = 0;
    const uint32_t b_id = 1;
    const uint32_t output_id = 3;
    subgraph.AddInput(type_of<A>(), permute(inv_a_perm, shapes.a), a_id)
        .AddInput(type_of<B>(), permute(inv_b_perm, shapes.b), b_id)
        .AddOutput(type_of<C>(), shapes.c, output_id);

    uint32_t c_id = 2;
    const bool init_c = random_bool(rng);
    const C init_value =
        random_bool(rng) ? random_value<C>(rng, -max_abs_value, max_abs_value)
                         : static_cast<C>(0);
    if (init_c) {
      if (init_value == 0 && random_bool(rng)) {
        c_id = YNN_INVALID_VALUE_ID;
      } else {
        subgraph.AddScalar<C>(init_value, c_id);
      }
    } else {
      subgraph.AddInput(type_of<C>(), shapes.c, c_id);
    }

    uint32_t a_tr_id = a_id;
    if (a_perm != iota(0, a_rank)) {
      a_tr_id = YNN_INVALID_VALUE_ID;
      subgraph.AddTensor(type_of<A>(), a_rank, a_tr_id);
      subgraph.AddTranspose(a_perm, a_id, a_tr_id);
    }

    uint32_t b_tr_id = b_id;
    if (b_perm != iota(0, b_rank)) {
      b_tr_id = YNN_INVALID_VALUE_ID;
      subgraph.AddTensor(type_of<B>(), b_rank, b_tr_id);
      subgraph.AddTranspose(b_perm, b_id, b_tr_id);
    }

    uint32_t dot_flags_static = 0;
    if (random_bool(rng)) {
      dot_flags_static |= YNN_NODE_FLAG_SYMMETRIC_B;
    }
    subgraph.AddDot(num_k_dims, a_tr_id, b_tr_id, c_id, output_id,
                    dot_flags_static);

    Runtime runtime(subgraph.GetSubgraph(),
                    random_bool(rng) ? &scheduler : nullptr);
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int revalue = 0; revalue < 2; ++revalue) {
      Tensor<A> a(permute(inv_a_perm, shapes.a));
      Tensor<B> b(permute(inv_b_perm, shapes.b));
      fill_random(a.data(), a.size(), rng, -max_abs_value, max_abs_value);
      fill_random(b.data(), b.size(), rng, -max_abs_value, max_abs_value);

      runtime.SetupExternalTensor(a.data(), a_id)
          .SetupExternalTensor(b.data(), b_id);

      Tensor<C> c(shapes.c);
      if (!init_c) {
        fill_random(c.data(), c.size(), rng, -max_abs_value, max_abs_value);
        runtime.SetupExternalTensor(c.data(), c_id);
      }

      Tensor<C> expected(shapes.c);
      if (init_c) {
        expected.fill(init_value);
      } else {
        // Copy the expected output before running the pipeline, since it is
        // updated in place by the test and reference implementations.
        expected.assign(c);
      }

      runtime.SetupExternalTensor(c.data(), output_id)
          .ReshapeRuntime()
          .InvokeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      // Apply the transpose so we can compute the reference result.
      if (a_perm != iota(0, a_rank)) {
        a = a.transpose(a_perm).deep_copy();
      }
      if (b_perm != iota(0, b_rank)) {
        b = b.transpose(b_perm).deep_copy();
      }

      VerifyDotResults(a, b, c, expected, shapes.batch_dims, shapes.a, shapes.b,
                       num_k_dims, max_abs_value);
    }
  }
}

template <typename A, typename B, typename C>
void TestBroadcastExtent1(A, B, C) {
  ReplicableRandomDevice rng;
  const float max_abs_value = 10.0f;

  const size_t num_k_dims = 1;
  const std::vector<size_t> batch_dims = {2, 3};
  const size_t m = 4;
  const size_t k = 8;
  const size_t n = 16;

  const std::vector<size_t> a_shape = {2, 1, m, k};
  const std::vector<size_t> b_shape = {1, 3, k, n};
  const std::vector<size_t> output_shape = {2, 3, m, n};

  for (const std::vector<size_t>& c_init_shape :
       std::vector<std::vector<size_t>>{{1, 3, 1, n}, {2, 1, m, 1}}) {
    SubgraphBuilder subgraph(4);
    const uint32_t a_id = 0;
    const uint32_t b_id = 1;
    const uint32_t c_id = 2;
    const uint32_t output_id = 3;
    subgraph.AddInput(type_of<A>(), a_shape, a_id)
        .AddInput(type_of<B>(), b_shape, b_id)
        .AddInput(type_of<C>(), c_init_shape, c_id)
        .AddOutput(type_of<C>(), output_shape, output_id)
        .AddDot(num_k_dims, a_id, b_id, c_id, output_id);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    Tensor<A> a(a_shape);
    Tensor<B> b(b_shape);
    Tensor<C> c_init(c_init_shape);
    fill_random(a.data(), a.size(), rng, -max_abs_value, max_abs_value);
    fill_random(b.data(), b.size(), rng, -max_abs_value, max_abs_value);
    fill_random(c_init.data(), c_init.size(), rng, -max_abs_value,
                max_abs_value);

    Tensor<C> expected(output_shape);
    broadcast_extent_1(c_init);
    expected.assign(c_init);

    Tensor<C> output(output_shape);
    runtime.SetupExternalTensor(a.data(), a_id)
        .SetupExternalTensor(b.data(), b_id)
        .SetupExternalTensor(c_init.data(), c_id)
        .SetupExternalTensor(output.data(), output_id)
        .ReshapeRuntime()
        .InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    VerifyDotResults(a, b, output, expected, batch_dims, a_shape, b_shape,
                     num_k_dims, max_abs_value);
  }
}

template <typename A, typename B, typename C>
void TestBroadcastAM(A, B, C) {
  ReplicableRandomDevice rng;
  const float max_abs_value = 10.0f;
  TestScheduler scheduler(3);

  struct BatchCase {
    std::vector<size_t> a_batches;
    std::vector<size_t> b_batches;
    std::vector<size_t> out_batches;
  };
  const std::vector<BatchCase> batch_cases = {
      {{}, {}, {}},
      {{2, 1}, {1, 3}, {2, 3}},
      {{}, {2, 3}, {2, 3}},
  };
  for (const BatchCase& batch_case : batch_cases) {
    for (size_t k : {size_t{1}, size_t{16}}) {
      for (const std::vector<size_t>& k_dims :
           std::vector<std::vector<size_t>>{{k}, {3, k}}) {
        const size_t num_k_dims = k_dims.size();
        for (bool static_b : {false, true}) {
          for (int transpose_mode : {0, 1, 2}) {
            if (transpose_mode != 0 && type_element_count(type_of<A>()) != 1) {
              continue;
            }
            if (transpose_mode == 2 && (num_k_dims < 2 || k == 1)) {
              continue;
            }
            for (bool init_c : {false, true}) {
              const size_t m = init_c ? 35 : 1;
              const size_t n = 32;

              std::vector<size_t> a_shape = batch_case.a_batches;
              a_shape.push_back(1);
              a_shape.insert(a_shape.end(), k_dims.begin(), k_dims.end());

              std::vector<size_t> b_shape = batch_case.b_batches;
              b_shape.insert(b_shape.end(), k_dims.begin(), k_dims.end());
              b_shape.push_back(n);

              std::vector<size_t> output_shape = batch_case.out_batches;
              output_shape.push_back(m);
              output_shape.push_back(n);

              const size_t a_rank = a_shape.size();
              std::vector<int32_t> a_perm = iota(0, a_rank);
              if (transpose_mode == 1) {
                std::swap(a_perm[batch_case.a_batches.size()],
                          a_perm[a_rank - 1]);
              } else if (transpose_mode == 2) {
                std::swap(a_perm[a_rank - 2], a_perm[a_rank - 1]);
              }
              const std::vector<size_t> a_input_shape =
                  permute(a_perm, a_shape);

              Tensor<A> a(a_input_shape);
              Tensor<B> b(b_shape);
              Tensor<C> c_init(output_shape);
              fill_random(a.data(), a.size(), rng, -max_abs_value,
                          max_abs_value);
              fill_random(b.data(), b.size(), rng, -max_abs_value,
                          max_abs_value);
              if (init_c) {
                fill_random(c_init.data(), c_init.size(), rng, -max_abs_value,
                            max_abs_value);
              }

              SubgraphBuilder subgraph(4);
              const uint32_t a_id = 0;
              const uint32_t b_id = 1;
              const uint32_t c_id = init_c ? 2 : YNN_INVALID_VALUE_ID;
              const uint32_t output_id = 3;
              subgraph.AddInput(type_of<A>(), a_input_shape, a_id);
              uint32_t dot_a_id = a_id;
              if (transpose_mode != 0) {
                dot_a_id = YNN_INVALID_VALUE_ID;
                subgraph.AddTensor(type_of<A>(), a_rank, dot_a_id);
                subgraph.AddTranspose(a_perm, a_id, dot_a_id);
              }
              if (static_b) {
                subgraph.AddTensor(b, b_id);
              } else {
                subgraph.AddInput(type_of<B>(), b_shape, b_id);
              }
              if (init_c) {
                subgraph.AddInput(type_of<C>(), output_shape, c_id);
              }
              subgraph.AddOutput(type_of<C>(), output_shape, output_id)
                  .AddDot(num_k_dims, dot_a_id, b_id, c_id, output_id);

              Runtime runtime(subgraph.GetSubgraph(), &scheduler);
              ASSERT_EQ(runtime.Status(), ynn_status_success);

              Tensor<C> expected(output_shape);
              if (init_c) {
                expected.assign(c_init);
              } else {
                expected.fill(static_cast<C>(0));
              }

              Tensor<C> output(output_shape);
              runtime.SetupExternalTensor(a.data(), a_id)
                  .SetupExternalTensor(output.data(), output_id);
              if (init_c) {
                runtime.SetupExternalTensor(c_init.data(), c_id);
              }
              if (!static_b) {
                runtime.SetupExternalTensor(b.data(), b_id);
              }
              runtime.ReshapeRuntime().InvokeRuntime();
              ASSERT_EQ(runtime.Status(), ynn_status_success);

              Tensor<A> a_logical =
                  transpose_mode != 0 ? a.transpose(a_perm).deep_copy() : a;
              std::vector<size_t> full_a_shape = batch_case.out_batches;
              full_a_shape.push_back(m);
              full_a_shape.insert(full_a_shape.end(), k_dims.begin(),
                                  k_dims.end());
              Tensor<A> a_ref = a_logical.broadcast_like(full_a_shape);

              VerifyDotResults(a_ref, b, output, expected,
                               batch_case.out_batches, a_shape, b_shape,
                               num_k_dims, max_abs_value);
            }
          }
        }
      }
    }
  }
}

class Dot : public testing::TestWithParam<multi_type> {};

// Test a dot where `a` is a scalar.
template <typename A, typename B, typename C>
void TestScalarA(A, B, C) {
  ReplicableRandomDevice rng;
  const float max_abs_value = 10.0f;
  const size_t k = 37;
  const size_t n = 32;
  const std::vector<size_t> b_shape = {k, n};
  const std::vector<size_t> output_shape = {1, n};

  for (bool static_a : {false, true}) {
    for (bool static_b : {false, true}) {
      for (bool dynamic_shape : {false, true}) {
        if (static_b && dynamic_shape) continue;
        Tensor<A> a(std::vector<size_t>{});
        Tensor<B> b(b_shape);
        fill_random(a.data(), a.size(), rng, -max_abs_value, max_abs_value);
        fill_random(b.data(), b.size(), rng, -max_abs_value, max_abs_value);

        SubgraphBuilder subgraph(3);
        const uint32_t a_id = 0;
        const uint32_t b_id = 1;
        const uint32_t output_id = 2;
        if (static_a) {
          subgraph.AddTensor(a, a_id);
        } else {
          subgraph.AddInput(type_of<A>(), {}, a_id);
        }
        if (static_b) {
          subgraph.AddTensor(b, b_id);
        } else if (dynamic_shape) {
          subgraph.AddInput(type_of<B>(), b_shape.size(), b_id);
        } else {
          subgraph.AddInput(type_of<B>(), b_shape, b_id);
        }
        subgraph
            .AddOutput(type_of<C>(),
                       dynamic_shape ? TensorShape(output_shape.size())
                                     : TensorShape(output_shape),
                       output_id)
            .AddDot(/*num_k_dims=*/1, a_id, b_id, YNN_INVALID_VALUE_ID,
                    output_id);

        Runtime runtime(subgraph.GetSubgraph());
        ASSERT_EQ(runtime.Status(), ynn_status_success);

        Tensor<C> output(output_shape);
        if (!static_a) {
          runtime.SetupExternalTensor(a.data(), a_id);
        }
        if (!static_b) {
          if (dynamic_shape) {
            runtime.ReshapeExternalTensor(b_shape, b.data(), b_id);
          } else {
            runtime.SetupExternalTensor(b.data(), b_id);
          }
        }
        runtime.SetupExternalTensor(output.data(), output_id)
            .ReshapeRuntime()
            .InvokeRuntime();
        ASSERT_EQ(runtime.Status(), ynn_status_success);

        Tensor<A> a_ref = a.broadcast_like({1, k});
        Tensor<C> expected(output_shape);
        expected.fill(static_cast<C>(0));
        VerifyDotResults(a_ref, b, output, expected, /*batch_dims=*/{},
                         /*a_shape=*/{}, b_shape, /*num_k_dims=*/1,
                         max_abs_value);
      }
    }
  }
}

TEST_P(Dot, ScalarA) {
  SwitchThreeTypes(GetParam(), [&](auto a_type, auto b_type, auto c_type) {
    TestScalarA(a_type, b_type, c_type);
  });
}

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

TEST_P(Dot, StaticShapeDynamicB) {
  SwitchThreeTypes(GetParam(), [&](auto a_type, auto b_type, auto c_type) {
    TestStaticShapeDynamicB(a_type, b_type, c_type);
  });
}

TEST_P(Dot, BroadcastExtent1) {
  SwitchThreeTypes(GetParam(), [&](auto a_type, auto b_type, auto c_type) {
    TestBroadcastExtent1(a_type, b_type, c_type);
  });
}

TEST_P(Dot, BroadcastAM) {
  SwitchThreeTypes(GetParam(), [&](auto a_type, auto b_type, auto c_type) {
    TestBroadcastAM(a_type, b_type, c_type);
  });
}

multi_type dot_types[] = {
#ifdef YNN_ENABLE_FP64
    multi_type::fp64,
#endif
    multi_type::fp32,
    multi_type::fp16,
    multi_type::bf16,
    multi_type::fp16_fp16_fp32,
    multi_type::bf16_bf16_fp32,
    multi_type::int8_int8_int32,
    multi_type::int8_int4_int32,
    multi_type::uint8_int8_int32,
};

INSTANTIATE_TEST_SUITE_P(Test, Dot, testing::ValuesIn(dot_types),
                         [](const testing::TestParamInfo<multi_type>& info) {
                           return to_string(info.param);
                         });

TEST(Dot, StaticRowBroadcastWith2DInitC) {
  constexpr size_t M = 8;
  constexpr size_t K = 16;
  constexpr size_t N = 16;

  Tensor<bfloat16> a({1, K});
  Tensor<bfloat16> b({K, N});
  Tensor<float> init_c({M, N});
  Tensor<float> out({M, N});
  a.fill(bfloat16(0.5f));
  b.fill(bfloat16(0.25f));
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      init_c(m, n) = static_cast<float>(m * N + n) * 0.1f;
    }
  }

  SubgraphBuilder subgraph(4);
  uint32_t out_id = 3;
  subgraph.AddInput(ynn_type_bf16, {1, K}, 0)
      .AddInput(ynn_type_bf16, {K, N}, 1)
      .AddInput(ynn_type_fp32, {M, N}, 2)
      .AddOutput(ynn_type_fp32, std::vector<size_t>{M, N}, out_id);
  ASSERT_EQ(ynn_define_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0, 1, 2,
                           &out_id, /*flags=*/0),
            ynn_status_success);

  Runtime runtime(subgraph.GetSubgraph());
  runtime.ReshapeExternalTensor({1, K}, a.data(), 0)
      .ReshapeExternalTensor({K, N}, b.data(), 1)
      .ReshapeExternalTensor({M, N}, init_c.data(), 2)
      .SetupExternalTensor(out.data(), out_id)
      .ReshapeRuntime()
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  const float dot_val = static_cast<float>(K) * 0.5f * 0.25f;
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      EXPECT_NEAR(out(m, n), init_c(m, n) + dot_val, 1e-4f);
    }
  }
}

namespace {

enum class RaggedInitCMode {
  kNone = 0,
  kScalar = 1,
  k1DN = 2,
  kRow1N = 3,
  kColM1 = 4,
  kFullMN = 5,
};

template <typename Rng>
void FillSortedRowGroups(Tensor<int32_t>& rg, size_t g, Rng& rng) {
  const size_t m = rg.extent(rg.rank() - 2);
  if (m == 0) return;
  const size_t batches = rg.size() / m;
  int32_t* data = rg.data();
  std::uniform_int_distribution<int32_t> group_dist(
      0, static_cast<int32_t>(g - 1));
  for (size_t b = 0; b < batches; ++b) {
    int32_t* row_groups = data + b * m;
    if (g == 1 || std::uniform_int_distribution<int>(0, 3)(rng) == 0) {
      std::fill_n(row_groups, m, group_dist(rng));
    } else {
      for (size_t i = 0; i < m; ++i) {
        row_groups[i] = group_dist(rng);
      }
      std::sort(row_groups, row_groups + m);
    }
  }
}

template <typename A, typename B, typename C>
void VerifyRaggedDotResults(Tensor<A> a, Tensor<B> b, Tensor<int32_t> rg,
                            Tensor<C> c, Tensor<C> expected,
                            const std::vector<size_t>& batch_dims,
                            size_t num_k_dims, float max_abs_value) {
  const size_t m = c.extent(batch_dims.size());
  const size_t n = c.extent(batch_dims.size() + 1);
  const size_t g = b.extent(b.rank() - num_k_dims - 2);

  std::vector<size_t> target_a = batch_dims;
  target_a.push_back(m);
  std::vector<size_t> target_b = batch_dims;
  target_b.push_back(g);
  size_t num_k_elements = 1;
  for (size_t d = 0; d < num_k_dims; ++d) {
    const size_t k_d = b.extent(b.rank() - num_k_dims - 1 + d);
    target_a.push_back(k_d);
    target_b.push_back(k_d);
    num_k_elements *= k_d;
  }
  target_b.push_back(n);
  std::vector<size_t> target_rg = batch_dims;
  target_rg.push_back(m);
  target_rg.push_back(1);

  a = a.broadcast_like(target_a);
  b = b.broadcast_like(target_b);
  rg = rg.broadcast_like(target_rg);

  for (const auto& batch_idx : EnumerateIndices(batch_dims)) {
    Tensor<A> a_batch = slice_batches(a, batch_idx);
    Tensor<B> b_batch = slice_batches(b, batch_idx);
    Tensor<int32_t> rg_batch = slice_batches(rg, batch_idx);
    Tensor<C> exp_batch = slice_batches(expected, batch_idx);
    for (size_t row = 0; row < m;) {
      const int32_t group = rg_batch(row, 0);
      ASSERT_GE(group, 0);
      ASSERT_LT(static_cast<size_t>(group), g);
      size_t end = row + 1;
      while (end < m && rg_batch(end, 0) == group) {
        ++end;
      }
      Reference(a_batch.slice(0, row, end),
                slice_batches(b_batch, {static_cast<size_t>(group)}),
                exp_batch.slice(0, row, end));
      row = end;
    }
  }

  for (const auto& i : EnumerateIndices(expected.shape())) {
    if (is_integral<C>::value) {
      ASSERT_EQ(c(i), expected(i))
          << "i=" << index_to_string(i) << " num_k_dims=" << num_k_dims;
    } else {
      const auto tolerance =
          get_dot_tolerance<C>(num_k_elements, max_abs_value);
      ASSERT_NEAR(c(i), expected(i), tolerance)
          << "i=" << index_to_string(i) << " num_k_dims=" << num_k_dims;
    }
  }
}

template <typename C>
void DefineRaggedInitC(SubgraphBuilder& subgraph, RaggedInitCMode c_mode,
                       size_t output_rank, uint32_t c_in_id, C scalar_c,
                       uint32_t& c_id) {
  c_id = YNN_INVALID_VALUE_ID;
  if (c_mode == RaggedInitCMode::kScalar) {
    c_id = c_in_id;
    subgraph.AddScalar<C>(scalar_c, c_id);
  } else if (c_mode == RaggedInitCMode::k1DN) {
    c_id = c_in_id;
    subgraph.AddInput(type_of<C>(), TensorShape(1), c_id);
  } else if (c_mode != RaggedInitCMode::kNone) {
    c_id = c_in_id;
    std::vector<size_t> c_template(output_rank, 0);
    if (c_mode == RaggedInitCMode::kRow1N) {
      c_template[output_rank - 2] = 1;
    } else if (c_mode == RaggedInitCMode::kColM1) {
      c_template[output_rank - 1] = 1;
    }
    subgraph.AddInput(type_of<C>(), TensorShape(c_template), c_id);
  }
}

template <typename C, typename Rng>
void ReshapeRaggedInitC(Runtime& runtime, RaggedInitCMode c_mode,
                        uint32_t c_in_id, const std::vector<size_t>& batch_dims,
                        size_t m, size_t n, const std::vector<size_t>& c_shape,
                        C scalar_c, float max_abs_value, Rng& rng,
                        Tensor<C>& c_init, Tensor<C>& expected) {
  if (c_mode == RaggedInitCMode::kNone) {
    expected.fill(static_cast<C>(0));
    return;
  }
  if (c_mode == RaggedInitCMode::kScalar) {
    expected.fill(scalar_c);
    return;
  }
  std::vector<size_t> c_in_shape;
  if (c_mode == RaggedInitCMode::k1DN) {
    c_in_shape = {n};
  } else {
    c_in_shape = batch_dims;
    c_in_shape.push_back(c_mode == RaggedInitCMode::kRow1N ? 1 : m);
    c_in_shape.push_back(c_mode == RaggedInitCMode::kColM1 ? 1 : n);
  }
  c_init = Tensor<C>(c_in_shape);
  fill_random(c_init.data(), c_init.size(), rng, -max_abs_value, max_abs_value);
  runtime.ReshapeExternalTensor(c_in_shape, c_init.data(), c_in_id);
  expected.assign(c_init.broadcast_like(c_shape));
}

template <typename A, typename B, typename C>
void TestRaggedDot(A, B, C, bool static_b) {
  ReplicableRandomDevice rng;
  TestScheduler scheduler(3);
  std::uniform_int_distribution<int> c_mode_dist(0, 5);
  std::uniform_int_distribution<size_t> g_dist(1, 8);
  std::uniform_int_distribution<size_t> n_dist(1, static_b ? 64 : 48);

  for (auto _ : FuzzTest(std::chrono::seconds(1))) {
    const size_t num_batch_dims = random_bool(rng) ? 0 : 1;
    const bool batch_b = !static_b && num_batch_dims > 0 && random_bool(rng);
    const size_t output_rank = num_batch_dims + 2;
    const size_t num_k_dims =
        std::uniform_int_distribution<size_t>(1, max_k_dims)(rng);
    const bool split_k = static_b && num_k_dims == 1 &&
                         std::uniform_int_distribution<int>(0, 15)(rng) == 0;
    const bool sparse_groups =
        static_b && !split_k &&
        std::uniform_int_distribution<int>(0, 7)(rng) == 0;
    const float max_abs_value = split_k ? 2.0f : 10.0f;
    const size_t a_rank = num_batch_dims + 1 + num_k_dims;
    const size_t b_rank = (batch_b ? num_batch_dims : 0) + 1 + num_k_dims + 1;

    uint32_t subgraph_flags = 0;
    if (random_bool(rng)) {
      subgraph_flags |= YNN_FLAG_CONSISTENT_ARITHMETIC;
    }
    SubgraphBuilder subgraph(5, subgraph_flags);
    const uint32_t a_id = 0;
    const uint32_t b_id = 1;
    const uint32_t c_in_id = 2;
    const uint32_t rg_id = 3;
    uint32_t output_id = 4;

    size_t static_g = 0;
    std::vector<size_t> static_b_shape;
    Tensor<B> static_b_tensor;
    if (static_b) {
      static_g = sparse_groups ? size_t{32} : g_dist(rng);
      static_b_shape.resize(1 + num_k_dims + 1);
      static_b_shape[0] = static_g;
      for (size_t d = 0; d + 1 < num_k_dims; ++d) {
        static_b_shape[1 + d] = k23_dim_dist(rng);
      }
      static_b_shape[num_k_dims] =
          split_k ? std::uniform_int_distribution<size_t>(2049, 4200)(rng)
                  : std::uniform_int_distribution<size_t>(1, 128)(rng);
      static_b_shape.back() = split_k ? 16 : n_dist(rng);
      static_b_shape = align_logical_shape<B>(static_b_shape);
      static_b_tensor = Tensor<B>(static_b_shape);
      fill_random(static_b_tensor.data(), static_b_tensor.size(), rng,
                  -max_abs_value, max_abs_value);
      subgraph.AddTensor(static_b_tensor, b_id);
    } else {
      subgraph.AddInput(type_of<B>(), TensorShape(b_rank), b_id);
    }

    const auto c_mode = static_cast<RaggedInitCMode>(c_mode_dist(rng));
    const bool broadcast_a_m = !static_b && random_bool(rng);
    std::vector<size_t> a_template(a_rank, 0);
    if (broadcast_a_m) {
      a_template[num_batch_dims] = 1;
    }
    std::vector<size_t> rg_template(output_rank, 0);
    if (random_bool(rng)) {
      rg_template.back() = 1;
    }
    subgraph
        .AddInput(type_of<A>(),
                  broadcast_a_m ? TensorShape(a_template) : TensorShape(a_rank),
                  a_id)
        .AddInput(ynn_type_int32,
                  rg_template.back() == 1 ? TensorShape(rg_template)
                                          : TensorShape(output_rank),
                  rg_id)
        .AddOutput(type_of<C>(), output_rank, output_id);

    const C scalar_c = random_value<C>(rng, -max_abs_value, max_abs_value);
    uint32_t c_id = YNN_INVALID_VALUE_ID;
    ASSERT_NO_FATAL_FAILURE(DefineRaggedInitC(subgraph, c_mode, output_rank,
                                              c_in_id, scalar_c, c_id));

    uint32_t dot_flags = random_bool(rng) ? YNN_NODE_FLAG_SYMMETRIC_B : 0;
    ASSERT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), num_k_dims, a_id,
                                    b_id, c_id, rg_id, &output_id, dot_flags),
              ynn_status_success);

    Runtime runtime(subgraph.GetSubgraph(),
                    random_bool(rng) ? &scheduler : nullptr);
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> batch_dims;
      if (num_batch_dims == 1) {
        batch_dims.push_back(std::uniform_int_distribution<size_t>(1, 3)(rng));
      }
      const size_t m =
          std::uniform_int_distribution<size_t>(1, split_k ? 8 : 16)(rng);

      size_t g = static_g;
      std::vector<size_t> k_dims(num_k_dims);
      std::vector<size_t> dynamic_b_shape;
      Tensor<B> dynamic_b_tensor;
      if (static_b) {
        for (size_t d = 0; d < num_k_dims; ++d) {
          k_dims[d] = static_b_shape[1 + d];
        }
      } else {
        g = g_dist(rng);
        for (size_t d = 0; d + 1 < num_k_dims; ++d) {
          k_dims[d] = k23_dim_dist(rng);
        }
        k_dims.back() = std::uniform_int_distribution<size_t>(1, 96)(rng);
        if (batch_b) {
          dynamic_b_shape = batch_dims;
        }
        dynamic_b_shape.push_back(g);
        dynamic_b_shape.insert(dynamic_b_shape.end(), k_dims.begin(),
                               k_dims.end());
        dynamic_b_shape.push_back(n_dist(rng));
        dynamic_b_shape = align_logical_shape<B>(dynamic_b_shape);
        dynamic_b_tensor = Tensor<B>(dynamic_b_shape);
        fill_random(dynamic_b_tensor.data(), dynamic_b_tensor.size(), rng,
                    -max_abs_value, max_abs_value);
        runtime.ReshapeExternalTensor(dynamic_b_shape, dynamic_b_tensor.data(),
                                      b_id);
      }
      const Tensor<B>& b_ref = static_b ? static_b_tensor : dynamic_b_tensor;
      const size_t n = b_ref.extent(b_ref.rank() - 1);

      std::vector<size_t> a_shape = batch_dims;
      a_shape.push_back(broadcast_a_m ? size_t{1} : m);
      a_shape.insert(a_shape.end(), k_dims.begin(), k_dims.end());

      std::vector<size_t> rg_shape = batch_dims;
      rg_shape.push_back(m);
      rg_shape.push_back(1);

      std::vector<size_t> c_shape = batch_dims;
      c_shape.push_back(m);
      c_shape.push_back(n);

      Tensor<A> a(a_shape);
      Tensor<int32_t> rg(rg_shape);
      fill_random(a.data(), a.size(), rng, -max_abs_value, max_abs_value);
      FillSortedRowGroups(rg, g, rng);
      runtime.ReshapeExternalTensor(a_shape, a.data(), a_id)
          .ReshapeExternalTensor(rg_shape, rg.data(), rg_id);

      Tensor<C> c_init;
      Tensor<C> expected(c_shape);
      ReshapeRaggedInitC(runtime, c_mode, c_in_id, batch_dims, m, n, c_shape,
                         scalar_c, max_abs_value, rng, c_init, expected);

      runtime.ReshapeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);
      ASSERT_EQ(runtime.GetExternalTensorShape(output_id), c_shape);

      Tensor<C> out(c_shape);
      out.fill(static_cast<C>(7));
      runtime.SetupExternalTensor(out.data(), output_id).InvokeRuntime();
      ASSERT_EQ(runtime.Status(), ynn_status_success);

      ASSERT_NO_FATAL_FAILURE(VerifyRaggedDotResults(
          a, b_ref, rg, out, expected, batch_dims, num_k_dims, max_abs_value));
    }
  }
}

}  // namespace

class RaggedDot : public testing::TestWithParam<multi_type> {};

TEST_P(RaggedDot, StaticB) {
  SwitchThreeTypes(GetParam(), [&](auto a_type, auto b_type, auto c_type) {
    TestRaggedDot(a_type, b_type, c_type, /*static_b=*/true);
  });
}

TEST_P(RaggedDot, DynamicB) {
  SwitchThreeTypes(GetParam(), [&](auto a_type, auto b_type, auto c_type) {
    TestRaggedDot(a_type, b_type, c_type, /*static_b=*/false);
  });
}

INSTANTIATE_TEST_SUITE_P(Test, RaggedDot, testing::ValuesIn(dot_types),
                         [](const testing::TestParamInfo<multi_type>& info) {
                           return to_string(info.param);
                         });

TEST(RaggedDot, EdgeShapesAndBroadcastA) {
  ReplicableRandomDevice rng;
  constexpr size_t G = 3;
  constexpr size_t M = 6;
  constexpr size_t N = 5;

  // Empty reduction K = 0 (with and without init_c) and empty M = 0.
  Tensor<float> init_c({M, N});
  Tensor<float> b_m0({G, 8, N});
  Tensor<int32_t> rg({M, 1});
  fill_random(init_c.data(), init_c.size(), rng, -5.0f, 5.0f);
  b_m0.fill(1.0f);
  for (size_t i = 0; i < M; ++i) rg(i, 0) = static_cast<int32_t>(i / 2);

  for (bool with_c : {false, true}) {
    SubgraphBuilder subgraph(5);
    const uint32_t c_id = with_c ? 2 : YNN_INVALID_VALUE_ID;
    uint32_t out_id = 4;
    subgraph.AddInput(ynn_type_fp32, TensorShape(2), 0)
        .AddInput(ynn_type_fp32, TensorShape(3), 1)
        .AddInput(ynn_type_int32, TensorShape(2), 3)
        .AddOutput(ynn_type_fp32, TensorShape(2), out_id);
    if (with_c) subgraph.AddInput(ynn_type_fp32, TensorShape(2), c_id);
    ASSERT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0,
                                    1, c_id, 3, &out_id, /*flags=*/0),
              ynn_status_success);

    float dummy = 0.0f;
    int32_t dummy_rg = 0;
    Tensor<float> x({M, N});
    x.fill(99.0f);
    Runtime runtime(subgraph.GetSubgraph());
    runtime.ReshapeExternalTensor({M, 0}, &dummy, 0)
        .ReshapeExternalTensor({G, 0, N}, &dummy, 1)
        .ReshapeExternalTensor({M, 1}, rg.data(), 3)
        .SetupExternalTensor(x.data(), out_id);
    if (with_c) runtime.ReshapeExternalTensor({M, N}, init_c.data(), c_id);
    runtime.ReshapeRuntime().InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    for (size_t r = 0; r < M; ++r) {
      for (size_t j = 0; j < N; ++j) {
        ASSERT_EQ(x(r, j), with_c ? init_c(r, j) : 0.0f);
      }
    }

    // M = 0 reshape on the same runtime.
    runtime.ReshapeExternalTensor({0, 8}, &dummy, 0)
        .ReshapeExternalTensor({G, 8, N}, b_m0.data(), 1)
        .ReshapeExternalTensor({0, 1}, &dummy_rg, 3);
    if (with_c) runtime.ReshapeExternalTensor({0, N}, &dummy, c_id);
    runtime.ReshapeRuntime().InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    ASSERT_EQ(runtime.GetExternalTensorShape(out_id),
              (std::vector<size_t>{0, N}));
  }

  // Internal negate -> broadcast_like A ([1, K] -> [M, K]).
  constexpr size_t K = 33;
  Tensor<float> a_in({1, K});
  Tensor<float> b({G, K, N});
  fill_random(a_in.data(), a_in.size(), rng, -5.0f, 5.0f);
  fill_random(b.data(), b.size(), rng, -5.0f, 5.0f);

  SubgraphBuilder subgraph(4);
  uint32_t a_neg_id = YNN_INVALID_VALUE_ID;
  uint32_t a_bcast_id = YNN_INVALID_VALUE_ID;
  uint32_t out_id = 3;
  subgraph.AddInput(ynn_type_fp32, TensorShape(2), 0)
      .AddInput(ynn_type_fp32, b.extents(), 1)
      .AddInput(ynn_type_int32, rg.extents(), 2)
      .AddTensor(ynn_type_fp32, 2, a_neg_id)
      .AddTensor(ynn_type_fp32, 2, a_bcast_id)
      .AddOutput(ynn_type_fp32, std::vector<size_t>{M, N}, out_id)
      .AddUnary(ynn_unary_negate, 0, a_neg_id)
      .AddBroadcastLike({0}, a_neg_id, 2, a_bcast_id);
  ASSERT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1,
                                  a_bcast_id, 1, YNN_INVALID_VALUE_ID, 2,
                                  &out_id, /*flags=*/0),
            ynn_status_success);

  Tensor<float> x({M, N});
  Runtime runtime(subgraph.GetSubgraph());
  runtime.ReshapeExternalTensor({1, K}, a_in.data(), 0)
      .SetupExternalTensor(b.data(), 1)
      .SetupExternalTensor(rg.data(), 2)
      .SetupExternalTensor(x.data(), out_id)
      .ReshapeRuntime()
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  const float tolerance = get_dot_tolerance<float>(K, 5.0f);
  for (size_t row = 0; row < M; ++row) {
    const size_t g = static_cast<size_t>(rg(row, 0));
    for (size_t j = 0; j < N; ++j) {
      double expected = 0.0;
      for (size_t k = 0; k < K; ++k) {
        expected +=
            -static_cast<double>(a_in(0, k)) * static_cast<double>(b(g, k, j));
      }
      ASSERT_NEAR(x(row, j), static_cast<float>(expected), tolerance);
    }
  }
}

TEST(RaggedDot, InvalidParameters) {
  // Static mismatched K, static invalid row_group extent, and static B N=1
  // broadcast mismatch.
  for (const auto& [a_dims, b_dims, rg_dims] :
       std::vector<std::tuple<TensorShape, TensorShape, TensorShape>>{
           {{4, 8}, {3, 16, 8}, {4, 1}},
           {{4, 8}, {3, 8, 8}, {4, 2}},
           {{4, 8}, {3, 8, 1}, {4, 1}},
       }) {
    SubgraphBuilder subgraph(5);
    uint32_t out_id = 4;
    subgraph.AddInput(ynn_type_fp32, a_dims, 0)
        .AddInput(ynn_type_fp32, b_dims, 1)
        .AddInput(ynn_type_fp32, {4, 8}, 2)
        .AddInput(ynn_type_int32, rg_dims, 3)
        .AddOutput(ynn_type_fp32, {4, 8}, out_id);
    EXPECT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0,
                                    1, 2, 3, &out_id, /*flags=*/0),
              ynn_status_invalid_parameter);
  }

  // Dynamic reshape with invalid row_group extent {M, 2} or B N=1 mismatch.
  {
    SubgraphBuilder subgraph(5);
    uint32_t out_id = 4;
    subgraph.AddInput(ynn_type_fp32, TensorShape(2), 0)
        .AddInput(ynn_type_fp32, TensorShape(3), 1)
        .AddInput(ynn_type_fp32, TensorShape(2), 2)
        .AddInput(ynn_type_int32, TensorShape(2), 3)
        .AddOutput(ynn_type_fp32, TensorShape(2), out_id);
    ASSERT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0,
                                    1, 2, 3, &out_id, /*flags=*/0),
              ynn_status_success);
    Tensor<float> a({4, 8}), b_good({3, 8, 8}), b_bad({3, 8, 1}), c({4, 8});
    Tensor<int32_t> rg_good({4, 1}), rg_bad({4, 2});
    rg_good.fill(0);
    rg_bad.fill(0);
    Runtime runtime(subgraph.GetSubgraph());
    runtime.ReshapeExternalTensor({4, 8}, a.data(), 0)
        .ReshapeExternalTensor({3, 8, 8}, b_good.data(), 1)
        .ReshapeExternalTensor({4, 8}, c.data(), 2)
        .ReshapeExternalTensor({4, 2}, rg_bad.data(), 3);
    EXPECT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_invalid_parameter);

    runtime.ReshapeExternalTensor({3, 8, 1}, b_bad.data(), 1)
        .ReshapeExternalTensor({4, 1}, rg_good.data(), 3);
    EXPECT_NE(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  }

  // Unsupported input type combinations fail for both dot and ragged_dot
  // without mutating `out_id`.
  for (auto [a_type, b_type] : {std::pair{ynn_type_fp32, ynn_type_int8},
                                std::pair{ynn_type_int32, ynn_type_int32}}) {
    SubgraphBuilder subgraph(5);
    uint32_t out_id = YNN_INVALID_VALUE_ID;
    subgraph.AddInput(a_type, {4, 16}, 0)
        .AddInput(b_type, {16, 8}, 1)
        .AddInput(b_type, {3, 16, 8}, 2)
        .AddInput(ynn_type_int32, {4, 1}, 3)
        .AddInput(ynn_type_bf16, {4, 8}, 4);
    EXPECT_EQ(ynn_define_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0, 1, 4,
                             &out_id, /*flags=*/0),
              ynn_status_unsupported_parameter);
    EXPECT_EQ(out_id, YNN_INVALID_VALUE_ID);
    EXPECT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0,
                                    2, 4, 3, &out_id, /*flags=*/0),
              ynn_status_unsupported_parameter);
    EXPECT_EQ(out_id, YNN_INVALID_VALUE_ID);
  }

  // Static `N` mismatch on `input_c_id` preserves `out_id`.
  {
    SubgraphBuilder subgraph(4);
    subgraph.AddInput(ynn_type_fp32, {4, 16}, 0)
        .AddInput(ynn_type_fp32, {3, 16, 8}, 1)
        .AddInput(ynn_type_fp32, {4, 4}, 2)
        .AddInput(ynn_type_int32, {4, 1}, 3);
    uint32_t out_id = YNN_INVALID_VALUE_ID;
    EXPECT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0,
                                    1, 2, 3, &out_id, /*flags=*/0),
              ynn_status_invalid_parameter);
    EXPECT_EQ(out_id, YNN_INVALID_VALUE_ID);
  }

  // Unsupported `input_c_id` conversion fails cleanly without mutating
  // `out_id` or poisoning subsequent graph construction.
  for (ynn_type bad_c_type : {ynn_type_uint4, ynn_type_uint2}) {
    SubgraphBuilder subgraph(5);
    uint32_t ok_out_id = 4;
    subgraph.AddInput(ynn_type_fp32, {4, 16}, 0)
        .AddInput(ynn_type_fp32, {3, 16, 8}, 1)
        .AddInput(bad_c_type, {4, 8}, 2)
        .AddInput(ynn_type_int32, {4, 1}, 3)
        .AddOutput(ynn_type_fp32, {4, 8}, ok_out_id);
    uint32_t out_id = YNN_INVALID_VALUE_ID;
    EXPECT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0,
                                    1, 2, 3, &out_id, /*flags=*/0),
              ynn_status_unsupported_parameter);
    EXPECT_EQ(out_id, YNN_INVALID_VALUE_ID);
    EXPECT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0,
                                    1, YNN_INVALID_VALUE_ID, 3, &ok_out_id,
                                    /*flags=*/0),
              ynn_status_success);
    Runtime runtime(subgraph.GetSubgraph());
    EXPECT_EQ(runtime.Status(), ynn_status_success);
  }
}

TEST(RaggedDot, OutOfRangeRowGroupFails) {
  for (size_t G : {size_t{1}, size_t{3}}) {
    for (int32_t bad_group : {int32_t{-1}, static_cast<int32_t>(G)}) {
      constexpr size_t M = 4;
      constexpr size_t K = 16;
      constexpr size_t N = 8;
      Tensor<float> a({M, K});
      Tensor<float> b({G, K, N});
      Tensor<int32_t> rg({M, 1});
      a.fill(1.0f);
      b.fill(1.0f);
      rg.fill(0);
      rg(2, 0) = bad_group;

      SubgraphBuilder subgraph(4);
      uint32_t output_id = 3;
      subgraph.AddInput(ynn_type_fp32, a.extents(), 0)
          .AddInput(ynn_type_fp32, b.extents(), 1)
          .AddInput(ynn_type_int32, rg.extents(), 2)
          .AddOutput(ynn_type_fp32, std::vector<size_t>{M, N}, output_id);
      ASSERT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1,
                                      0, 1, YNN_INVALID_VALUE_ID, 2, &output_id,
                                      /*flags=*/0),
                ynn_status_success);

      Tensor<float> x({M, N});
      Runtime runtime(subgraph.GetSubgraph());
      ASSERT_EQ(runtime.Status(), ynn_status_success);
      runtime.SetupExternalTensor(a.data(), 0)
          .SetupExternalTensor(b.data(), 1)
          .SetupExternalTensor(rg.data(), 2)
          .SetupExternalTensor(x.data(), output_id);
      ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
      EXPECT_NE(ynn_invoke_runtime(runtime.get()), ynn_status_success)
          << "G=" << G << " bad_group=" << bad_group;
    }
  }
}

TEST(RaggedDot, UnsortedRowGroupFails) {
  // Intra-tile unsorted `row_group` fails runtime invocation.
  {
    constexpr size_t G = 4;
    constexpr size_t M = 6;
    constexpr size_t K = 16;
    constexpr size_t N = 8;
    Tensor<float> a({M, K});
    Tensor<float> b({G, K, N});
    Tensor<int32_t> rg({M, 1});
    Tensor<float> x({M, N});
    a.fill(1.0f);
    b.fill(1.0f);
    const int32_t groups[M] = {0, 2, 1, 2, 3, 3};
    for (size_t i = 0; i < M; ++i) rg(i, 0) = groups[i];

    SubgraphBuilder subgraph(4);
    uint32_t output_id = 3;
    subgraph.AddInput(ynn_type_fp32, a.extents(), 0)
        .AddInput(ynn_type_fp32, b.extents(), 1)
        .AddInput(ynn_type_int32, rg.extents(), 2)
        .AddOutput(ynn_type_fp32, std::vector<size_t>{M, N}, output_id);
    ASSERT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0,
                                    1, YNN_INVALID_VALUE_ID, 2, &output_id,
                                    /*flags=*/0),
              ynn_status_success);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    runtime.SetupExternalTensor(a.data(), 0)
        .SetupExternalTensor(b.data(), 1)
        .SetupExternalTensor(rg.data(), 2)
        .SetupExternalTensor(x.data(), output_id);
    ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
    EXPECT_NE(ynn_invoke_runtime(runtime.get()), ynn_status_success);
  }

  // Cross-tile inversion at an outer `split_m` boundary (each M-tile sorted in
  // isolation, descending across the boundary) also fails runtime invocation.
  {
    constexpr size_t G = 8;
    constexpr size_t M = 2048;
    constexpr size_t K = 256;
    constexpr size_t N = 16;
    Tensor<float> a({M, K});
    Tensor<float> b({G, K, N});
    Tensor<int32_t> rg({M, 1});
    Tensor<float> x({M, N});
    a.fill(1.0f);
    b.fill(1.0f);
    // Single step down at `M / 2` so every power-of-two `M` tile (`<= M / 2`)
    // is constant internally and only the cross-tile boundary `M / 2` inverts.
    TestScheduler scheduler(4);
    for (bool produced_in_subgraph : {false, true}) {
      for (size_t i = 0; i < M; ++i) {
        rg(i, 0) = (i < M / 2) ? 3 : 2;
      }
      Tensor<float> rg_f32({M, 1});
      for (size_t i = 0; i < M; ++i) {
        rg_f32(i, 0) = static_cast<float>(rg(i, 0));
      }

      SubgraphBuilder subgraph(4);
      uint32_t output_id = 3;
      subgraph.AddInput(ynn_type_fp32, a.extents(), 0)
          .AddTensor(b, 1)
          .AddInput(produced_in_subgraph ? ynn_type_fp32 : ynn_type_int32,
                    rg.extents(), 2)
          .AddOutput(ynn_type_fp32, std::vector<size_t>{M, N}, output_id);
      uint32_t rg_id = 2;
      if (produced_in_subgraph) {
        rg_id = YNN_INVALID_VALUE_ID;
        ASSERT_EQ(ynn_define_convert(subgraph.GetSubgraph(), 2, ynn_type_int32,
                                     &rg_id, /*flags=*/0),
                  ynn_status_success);
      }
      ASSERT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1,
                                      0, 1, YNN_INVALID_VALUE_ID, rg_id,
                                      &output_id, /*flags=*/0),
                ynn_status_success);

      Runtime runtime(subgraph.GetSubgraph(), &scheduler);
      ASSERT_EQ(runtime.Status(), ynn_status_success);
      runtime.SetupExternalTensor(a.data(), 0)
          .SetupExternalTensor(produced_in_subgraph
                                   ? static_cast<void*>(rg_f32.data())
                                   : static_cast<void*>(rg.data()),
                               2)
          .SetupExternalTensor(x.data(), output_id);
      ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
      EXPECT_NE(ynn_invoke_runtime(runtime.get()), ynn_status_success)
          << "produced_in_subgraph=" << produced_in_subgraph;

      for (size_t i = M / 2; i < M; ++i) {
        rg(i, 0) = 4;
        rg_f32(i, 0) = 4.0f;
      }
      EXPECT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success)
          << "produced_in_subgraph=" << produced_in_subgraph;
    }
  }
}

}  // namespace ynn
