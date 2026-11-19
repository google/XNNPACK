// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
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
namespace {

constexpr int max_k_dims = 3;

// Large k2, k3 dimensions are unrealistic and very slow to test.
std::uniform_int_distribution<size_t> k23_dim_dist(1, 5);
std::bernoulli_distribution random_bool(0.5);

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

    float dummy = NAN;
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

  // Dynamic `G == 0` with `b.base == nullptr` fails cleanly for `M > 0`
  // (including `int8` `try_requantize_rewrite`) and succeeds when `batch == 0`.
  for (auto [in_type, out_type] : {std::pair{ynn_type_fp32, ynn_type_fp32},
                                   std::pair{ynn_type_int8, ynn_type_int32}}) {
    constexpr size_t M = 4;
    constexpr size_t K = 16;
    constexpr size_t N = 16;
    std::vector<uint8_t> a(M * K * 4, 0);
    std::vector<uint8_t> x(M * N * 4, 0);
    Tensor<int32_t> rg({M, 1});
    rg.fill(0);

    SubgraphBuilder subgraph(4);
    uint32_t output_id = 3;
    subgraph.AddInput(in_type, TensorShape(2), 0)
        .AddInput(in_type, TensorShape(3), 1)
        .AddInput(ynn_type_int32, TensorShape(2), 2)
        .AddOutput(out_type, TensorShape(2), output_id);
    ASSERT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0,
                                    1, YNN_INVALID_VALUE_ID, 2, &output_id,
                                    /*flags=*/0),
              ynn_status_success);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    runtime.ReshapeExternalTensor({M, K}, a.data(), 0)
        .ReshapeExternalTensor({0, K, N}, nullptr, 1)
        .ReshapeExternalTensor({M, 1}, rg.data(), 2)
        .SetupExternalTensor(x.data(), output_id);
    ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
    EXPECT_NE(ynn_invoke_runtime(runtime.get()), ynn_status_success)
        << "in_type=" << in_type;
  }

  {
    SubgraphBuilder subgraph(4);
    uint32_t output_id = 3;
    subgraph.AddInput(ynn_type_fp32, TensorShape(3), 0)
        .AddInput(ynn_type_fp32, TensorShape(3), 1)
        .AddInput(ynn_type_int32, TensorShape(3), 2)
        .AddOutput(ynn_type_fp32, TensorShape(3), output_id);
    ASSERT_EQ(ynn_define_ragged_dot(subgraph.GetSubgraph(), /*num_k_dims=*/1, 0,
                                    1, YNN_INVALID_VALUE_ID, 2, &output_id,
                                    /*flags=*/0),
              ynn_status_success);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    runtime.ReshapeExternalTensor({0, 4, 16}, nullptr, 0)
        .ReshapeExternalTensor({0, 16, 8}, nullptr, 1)
        .ReshapeExternalTensor({0, 4, 1}, nullptr, 2)
        .SetupExternalTensor(nullptr, output_id);
    ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
    EXPECT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);
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
