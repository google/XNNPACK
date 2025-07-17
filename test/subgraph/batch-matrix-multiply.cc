// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/subgraph.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/calculate_quantization_params.h"
#include "test/subgraph/fake-dynamic-quantize.h"
#include "test/subgraph/runtime-flags.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

// Make input matrices of the given ranks, with random broadcasting, such that
// the result of batch matrix multiply of the two operands is `output_shape`.
template <typename Rng>
std::pair<std::vector<size_t>, std::vector<size_t>> random_broadcasted_inputs(
    Rng& rng, std::vector<size_t> output_shape, size_t a_rank, size_t b_rank) {
  // The logic here is a lot easier if the innermost dimension is first.
  std::reverse(output_shape.begin(), output_shape.end());
  std::vector<size_t> a_shape = output_shape;
  std::vector<size_t> b_shape = output_shape;
  std::bernoulli_distribution broadcast_dist(0.25);
  for (size_t i = 0; i + 2 < output_shape.size(); i++) {
    // We only want to broadcast one of the two inputs in this dimension,
    // which includes broadcasting due to smaller rank.
    if (broadcast_dist(rng) && i < b_rank) {
      a_shape[i] = 1;
    } else if (broadcast_dist(rng) && i < a_rank) {
      b_shape[i] = 1;
    }
  }
  a_shape.resize(a_rank);
  b_shape.resize(b_rank);
  std::reverse(a_shape.begin(), a_shape.end());
  std::reverse(b_shape.begin(), b_shape.end());
  return {a_shape, b_shape};
}

// Remove the leading `at` dimensions of `tensor`, leaving behind a rank 2
// tensor.
template <typename T>
Tensor<T> slice_batches(Tensor<T> tensor, std::vector<size_t> at) {
  std::reverse(at.begin(), at.end());
  while (at.size() + 2 > tensor.rank()) {
    at.pop_back();
  }
  std::reverse(at.begin(), at.end());
  tensor = tensor.slice_leading(at);
  std::vector<size_t> shape = tensor.shape();
  std::vector<size_t> strides = tensor.strides();
  shape.erase(shape.begin(), shape.begin() + at.size());
  strides.erase(strides.begin(), strides.begin() + at.size());
  tensor.set_shape(shape, strides);
  return tensor;
}

template <typename InputA, typename InputB>
void ReferenceImpl(Tensor<InputA> input_a, Tensor<InputB> input_b,
                   const xnn_quantization_params& input_a_quantization,
                   int32_t input_b_zero_point, Tensor<float> input_b_scale,
                   Tensor<float> output) {
  assert(input_a.rank() == 2);
  assert(input_b.rank() == 2);
  assert(output.rank() == 2);
  assert(input_a.extent(0) == output.extent(0));
  assert(input_b.extent(1) == output.extent(1));
  assert(input_a.extent(1) == input_b.extent(0));

  for (size_t i = 0; i < output.extent(0); ++i) {
    for (size_t j = 0; j < output.extent(1); ++j) {
      double output_ij = 0.0;
      for (size_t k = 0; k < input_a.extent(1); ++k) {
        float a = dequantize(input_a(i, k), input_a_quantization);
        float b = dequantize(input_b(k, j), input_b_scale(k, j),
                             input_b_zero_point);
        output_ij += a * b;
      }
      output(i, j) = output_ij;
    }
  }
}

template <typename InputA, typename InputB>
Tensor<float> ReferenceImpl(Tensor<InputA> input_a, Tensor<InputB> input_b,
                            const xnn_quantization_params& input_a_quantization,
                            int32_t input_b_zero_point,
                            Tensor<float> input_b_scale, uint32_t flags) {
  if (flags & XNN_FLAG_TRANSPOSE_B) {
    input_b = input_b.transpose({input_b.rank() - 1, input_b.rank() - 2});
  }

  size_t a_m = input_a.extent(input_a.rank() - 2);
  size_t b_n = input_b.extent(input_b.rank() - 1);

  std::vector<size_t> a_shape = input_a.shape();
  std::vector<size_t> b_shape = input_b.shape();
  std::reverse(a_shape.begin(), a_shape.end());
  std::reverse(b_shape.begin(), b_shape.end());

  size_t output_rank = std::max(input_a.rank(), input_b.rank());
  std::vector<size_t> output_shape(output_rank);
  for (size_t i = 0; i < output_rank; i++) {
    output_shape[i] = std::max(i < a_shape.size() ? a_shape[i] : 1,
                               i < b_shape.size() ? b_shape[i] : 1);
  }
  std::reverse(output_shape.begin(), output_shape.end());
  output_shape[output_rank - 2] = a_m;
  output_shape[output_rank - 1] = b_n;
  Tensor<float> output(output_shape);
  std::vector<size_t> output_batches = output.shape();
  output_batches.pop_back();
  output_batches.pop_back();

  for (const auto& i : EnumerateIndices(output_batches)) {
    Tensor<InputA> a_i = slice_batches(input_a, i);
    Tensor<InputB> b_i = slice_batches(input_b, i);
    Tensor<float> b_scales_i = slice_batches(input_b_scale, i);
    Tensor<float> output_i = slice_batches(output, i);
    ReferenceImpl(a_i, b_i, input_a_quantization, input_b_zero_point,
                  b_scales_i, output_i);
  }

  return output;
}

// For float types, generate data in [-1, 1]
template <typename T>
DatatypeGenerator<T> MakeDatatypeGenerator(T) {
  return DatatypeGenerator<T>(-1.0f, 1.0f);
}
template <typename T>
T MaxDatatype(T) {
  return 1.0f;
}

// For quantized types, generate the full range of the type.
template <typename T, typename Kind>
DatatypeGenerator<quantized<T, Kind>> MakeDatatypeGenerator(
    quantized<T, Kind>) {
  return DatatypeGenerator<quantized<T, Kind>>();
}
template <typename T, typename Kind>
T MaxDatatype(quantized<T, Kind>) {
  return NumericLimits<quantized<T, Kind>>::max();
}

template <typename Input, typename Output = Input>
void TestDynamicB(uint64_t subgraph_flags = xnn_test_runtime_flags()) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution flag_dist(0.5);
  std::bernoulli_distribution k_big_dist(0.33);
  std::uniform_int_distribution<> rank_dist{2, XNN_MAX_TENSOR_DIMS - 1};
  const bool is_qs8_qs8 = xnn_datatype_is_quantized(xnn_datatype_of<Input>());

  // To get good coverage without excessive cost, pick one of the M, N, K
  // dimensions to make very large.
  std::uniform_int_distribution<size_t> dim_dist{1, 10};
  std::uniform_int_distribution<size_t> big_dim_dist{1, 1000};

  const bool is_k_big = k_big_dist(rng);
  const size_t k = is_k_big ? big_dim_dist(rng) : dim_dist(rng);

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_quantization_params input_a_quantization = {0, 1.0f};
  xnn_quantization_params input_b_quantization = {0, 1.0f};
  xnn_quantization_params output_quantization = {0, 1.0f};
  if (is_qs8_qs8) {
    input_a_quantization =
        random_quantization(xnn_datatype_of<Input>(), rng, 0.001f, 2.0f);
    input_b_quantization =
        random_quantization(xnn_datatype_of<Input>(), rng, 0.001f, 2.0f);
    // The output quantization is computed from the kernel size and input
    // quantization.
    output_quantization =
        CalculateGEMMQuantizationParams<Input, Input, Output>(
            k, input_a_quantization, input_b_quantization,
            /*bias_quantization=*/{0, 1.0f});
  }

  Tensor<float> b_scales({1, 1});
  b_scales.fill(input_b_quantization.scale);
  broadcast_extent_1(b_scales);

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    size_t input_a_rank = rank_dist(rng);
    size_t input_b_rank = rank_dist(rng);
    size_t output_rank = std::max(input_a_rank, input_b_rank);

    uint32_t flags = 0;
    if (flag_dist(rng)) {
      flags |= XNN_FLAG_TRANSPOSE_B;
    }
    if (flag_dist(rng)) {
      flags |= XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC;
    }

    SubgraphTester subgraph(3, subgraph_flags);
    const uint32_t input_a_id = 0;
    const uint32_t input_b_id = 1;
    const uint32_t output_id = 2;

    subgraph.AddInputTensor(input_a_rank, xnn_datatype_of<Input>(),
                            input_a_quantization, input_a_id)
            .AddInputTensor(input_b_rank, xnn_datatype_of<Input>(),
                            input_b_quantization, input_b_id)
            .AddOutputTensor(output_rank, xnn_datatype_of<Output>(),
                             output_quantization, output_id)
            .AddBatchMatrixMultiply(input_a_id, input_b_id, output_id, flags);

    xnn_status status = subgraph.CreateRuntime();
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    // Run the subgraph twice, with a different input/output shape each time.
    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> output_shape = random_shape(rng, output_rank, 1, 4);
      std::vector<size_t> a_shape, b_shape;
      std::tie(a_shape, b_shape) = random_broadcasted_inputs(
          rng, output_shape, input_a_rank, input_b_rank);
      size_t mn[] = {dim_dist(rng), dim_dist(rng)};
      if (!is_k_big) {
        mn[rng() % 2] = big_dim_dist(rng);
      }
      const size_t m = mn[0];
      const size_t n = mn[1];

      a_shape[input_a_rank - 2] = m;
      a_shape[input_a_rank - 1] = k;
      b_shape[input_b_rank - 2] = k;
      b_shape[input_b_rank - 1] = n;

      if (flags & XNN_FLAG_TRANSPOSE_B) {
        std::swap(b_shape[input_b_rank - 1], b_shape[input_b_rank - 2]);
      }

      Tensor<Input> input_a(a_shape, XnnExtraBytes);
      Tensor<Input> input_b(b_shape, XnnExtraBytes);
      auto input_gen = MakeDatatypeGenerator(Input());
      input_a.generate([&]() { return input_gen(rng); });
      input_b.generate([&]() { return input_gen(rng); });
      broadcast_extent_1(input_a);
      broadcast_extent_1(input_b);

      Tensor<float> expected = ReferenceImpl(
          input_a, input_b, input_a_quantization,
          input_b_quantization.zero_point, b_scales, flags);

      subgraph.ReshapeExternalTensor(a_shape, input_a.base(), input_a_id)
          .ReshapeExternalTensor(b_shape, input_b.base(), input_b_id)
          .ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(output_id), expected.shape())
          << "a_shape=" << index_to_string(a_shape)
          << ", b_shape=" << index_to_string(b_shape);

      // Run subgraph
      Tensor<Output> output(expected.shape());
      subgraph.SetupExternalTensor(output.base(), output_id)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      ASSERT_EQ(expected.shape(), output.shape());
      if (xnn_datatype_is_quantized(xnn_datatype_of<Output>())) {
        const float tolerance = 1.0f;
        for (const auto& i : EnumerateIndices(output.shape())) {
          ASSERT_NEAR(
              quantize<Output>(expected(i), output_quantization), output(i),
              tolerance)
              << "i=" << index_to_string(i)
              << ", a_shape=" << index_to_string(a_shape)
              << ", b_shape=" << index_to_string(b_shape)
              << ", output_shape=" << index_to_string(expected.shape());
        }
      } else {
        // In this case, both inputs should be in the range [-1, 1].
        const float tolerance =
            xnnpack::epsilon(xnn_datatype_of<Output>()) * k * 2.0f;

        for (const auto& i : EnumerateIndices(output.shape())) {
          ASSERT_NEAR(
              static_cast<float>(output(i)), expected(i), tolerance)
              << "i=" << index_to_string(i)
              << ", a_shape=" << index_to_string(a_shape)
              << ", b_shape=" << index_to_string(b_shape)
              << ", output_shape=" << index_to_string(expected.shape());
        }
      }
    }
  }
}

TEST(BatchMatrixMultiplyF16, dynamic_b) {
  TestDynamicB<xnn_float16, xnn_float16>();
}
TEST(BatchMatrixMultiplyF32, dynamic_b) { TestDynamicB<float, float>(); }
TEST(BatchMatrixMultiplyBF16F32, dynamic_b) {
  TestDynamicB<xnn_bfloat16, float>();
}
TEST(BatchMatrixMultiplyQS8, dynamic_b) {
  TestDynamicB<xnnpack::quantized<int8_t>, xnnpack::quantized<int8_t>>();
}

TEST(BatchMatrixMultiplyF16, dont_inline_lhs_dynamic_b) {
  TestDynamicB<xnn_float16, xnn_float16>(
      /*subgraph_flags=*/xnn_test_runtime_flags() |
      XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(BatchMatrixMultiplyF32, dont_inline_lhs_dynamic_b) {
  TestDynamicB<float, float>(/*subgraph_flags=*/xnn_test_runtime_flags() |
                             XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(BatchMatrixMultiplyBF16F32, dont_inline_lhs_dynamic_b) {
  TestDynamicB<xnn_bfloat16, float>(
      /*subgraph_flags=*/xnn_test_runtime_flags() |
      XNN_FLAG_NO_INLINED_LHS_PACKING);
}

template <typename InputA, typename InputB, typename Output = InputA>
void TestStaticB(xnn_datatype convert_to = xnn_datatype_invalid,
                 uint64_t subgraph_flags = xnn_test_runtime_flags()) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution flag_dist(0.5);
  std::uniform_int_distribution<> dim_dist{1, 100};
  std::uniform_int_distribution<> rank_dist{2, XNN_MAX_TENSOR_DIMS - 1};

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    size_t input_a_rank = rank_dist(rng);
    size_t input_b_rank = rank_dist(rng);
    size_t output_rank = std::max(input_a_rank, input_b_rank);

    uint32_t flags = 0;
    if (flag_dist(rng)) {
      flags |= XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC;
    }

    const bool is_qs8_qs8 =
        std::is_same<InputA, InputB>::value &&
        std::is_same<InputA, xnnpack::quantized<int8_t>>::value;

    const uint32_t input_a_id = 0;
    const uint32_t input_b_id = 1;
    const uint32_t output_id = 2;
    uint32_t bmm_input_a_id = input_a_id;

    std::vector<size_t> b_shape = random_shape(rng, input_b_rank, 1, 4);
    b_shape[input_b_rank - 2] = dim_dist(rng);
    b_shape[input_b_rank - 1] = dim_dist(rng);
    Tensor<InputB> input_b(b_shape, XnnExtraBytes);
    auto input_b_gen = MakeDatatypeGenerator(InputB());
    input_b.generate([&]() { return input_b_gen(rng); });
    broadcast_extent_1(input_b);
    size_t k = input_b.extent(input_b_rank - 2);

    xnn_quantization_params input_a_quantization = {0, 1.0f};
    xnn_quantization_params input_b_quantization = {0, 1.0f};
    xnn_quantization_params output_quantization = {0, 1.0f};
    if (is_qs8_qs8) {
      input_a_quantization =
          random_quantization(xnn_datatype_of<InputA>(), rng, 0.001f, 2.0f);
      input_b_quantization =
          random_quantization(xnn_datatype_of<InputB>(), rng, 0.001f, 2.0f);
      // The output quantization is computed from the kernel size and input
      // quantization.
      output_quantization =
          CalculateGEMMQuantizationParams<InputA, InputB, Output>(
              k, input_a_quantization, input_b_quantization,
              /*bias_quantization=*/{0, 1.0f});
    }


    SubgraphTester subgraph(3, subgraph_flags);
    subgraph.AddInputTensor(input_a_rank, xnn_datatype_of<InputA>(),
                            input_a_quantization, input_a_id);

    if (convert_to != xnn_datatype_invalid) {
      subgraph.AddInternalDynamicallyQuantizedTensor(
          input_a_rank, convert_to, /*num_nonbatch_dims=*/1, &bmm_input_a_id);
      subgraph.AddConvert(input_a_id, bmm_input_a_id);
    }

    Tensor<float> b_scales;
    if (xnn_datatype_is_channelwise_quantized(xnn_datatype_of<InputB>())) {
      std::vector<size_t> scales_shape = input_b.shape();
      scales_shape[input_b.rank() - 2] = 1;
      std::uniform_real_distribution<float> b_scale_dist(
          0.001f, input_b_quantization.scale);
      b_scales = Tensor<float>({scales_shape});
      b_scales.generate([&]() { return b_scale_dist(rng); });
      subgraph.AddStaticTensorQS8(input_b.shape(), input_b.rank() - 1,
                                  TensorType::kDense, b_scales.base(),
                                  input_b_id, /*flags=*/0,
                                  reinterpret_cast<int8_t*>(input_b.data()));
    } else if (is_qs8_qs8) {
      b_scales = Tensor<float>({1, 1});
      b_scales.fill(input_b_quantization.scale);
      subgraph.AddStaticTensor(input_b.shape(), input_b_id, input_b.base(),
                               input_b_quantization);
    } else {
      b_scales = Tensor<float>({1, 1});
      b_scales.fill(1.0f);
      subgraph.AddStaticTensor(input_b.shape(), input_b_id, input_b.base());
    }
    broadcast_extent_1(b_scales);

    subgraph.AddOutputTensor(output_rank, xnn_datatype_of<Output>(),
                             output_quantization, output_id)
        .AddBatchMatrixMultiply(bmm_input_a_id, input_b_id, output_id, flags);
    xnn_status status = subgraph.CreateRuntime();
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    // Run the subgraph twice, with a different input/output shape each time.
    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> output_shape = random_shape(rng, output_rank, 1, 4);
      std::copy(b_shape.begin(), b_shape.end(),
                output_shape.begin() + (output_shape.size() - b_shape.size()));
      std::vector<size_t> a_shape, dummy;
      std::tie(a_shape, dummy) = random_broadcasted_inputs(
          rng, output_shape, input_a_rank, input_b_rank);
      size_t m = dim_dist(rng);
      a_shape[input_a_rank - 2] = m;
      a_shape[input_a_rank - 1] = k;

      Tensor<InputA> input_a(a_shape, XnnExtraBytes);
      auto input_a_gen = MakeDatatypeGenerator(InputA());
      input_a.generate([&]() { return input_a_gen(rng); });
      if (convert_to != xnn_datatype_invalid) {
        // If we are dynamically quantizing, preprocess the data to have zero
        // error when it will be quantized, which allows us to use a much
        // smaller tolerance for error for testing purposes.
        std::vector<size_t> input_batches = input_a.shape();
        input_batches.pop_back();
        for (const auto& i : EnumerateIndices(input_batches)) {
          FakeDynamicQuantize(input_a.slice_leading(i), convert_to);
        }
      }
      broadcast_extent_1(input_a);

      Tensor<float> expected = ReferenceImpl(
          input_a, input_b, input_a_quantization,
          input_b_quantization.zero_point, b_scales, flags);

      subgraph.ReshapeExternalTensor(a_shape, input_a.base(), input_a_id)
          .ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(output_id), expected.shape())
          << "a_shape=" << index_to_string(a_shape)
          << ", b_shape=" << index_to_string(input_b.shape());

      // Run subgraph
      Tensor<Output> output(expected.shape());
      subgraph.SetupExternalTensor(output.base(), output_id)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      ASSERT_EQ(expected.shape(), output.shape());
      if (xnn_datatype_is_quantized(xnn_datatype_of<Output>())) {
        const float tolerance = 1.0f;

        for (const auto& i : EnumerateIndices(output.shape())) {
          ASSERT_NEAR(quantize<Output>(expected(i), output_quantization),
                      output(i), tolerance)
              << "i=" << index_to_string(i)
              << ", a_shape=" << index_to_string(a_shape)
              << ", b_shape=" << index_to_string(input_b.shape())
              << ", output_shape=" << index_to_string(expected.shape());
        }
      } else {
        const float max_a = MaxDatatype(InputA());
        const float max_b = MaxDatatype(InputB()) * input_b_quantization.scale;
        const float tolerance = xnnpack::epsilon(xnn_datatype_of<Output>()) *
                                k * max_a * max_b * 3.0f;

        for (const auto& i : EnumerateIndices(output.shape())) {
          ASSERT_NEAR(output(i), expected(i), tolerance)
              << "i=" << index_to_string(i)
              << ", a_shape=" << index_to_string(a_shape)
              << ", b_shape=" << index_to_string(input_b.shape())
              << ", output_shape=" << index_to_string(expected.shape());
        }
      }
    }
  }
}

TEST(BatchMatrixMultiplyF16, static_b) {
  TestStaticB<xnn_float16, xnn_float16>();
}
TEST(BatchMatrixMultiplyF32, static_b) { TestStaticB<float, float>(); }
TEST(BatchMatrixMultiplyBF16F32, static_b) {
  TestStaticB<xnn_bfloat16, xnn_bfloat16, float>();
}
TEST(BatchMatrixMultiplyQS8, static_b) {
  TestStaticB<xnnpack::quantized<int8_t>, xnnpack::quantized<int8_t>>();
}

TEST(BatchMatrixMultiplyF16, dont_inline_lhs_static_b) {
  TestStaticB<xnn_float16, xnn_float16>(
      /*convert_to=*/xnn_datatype_invalid,
      /*subgraph_flags=*/XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(BatchMatrixMultiplyF32, dont_inline_lhs_static_b) {
  TestStaticB<float, float>(
      /*convert_to=*/xnn_datatype_invalid,
      /*subgraph_flags=*/XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(BatchMatrixMultiplyBF16F32, dont_inline_lhs_static_b) {
  TestStaticB<xnn_bfloat16, xnn_bfloat16, float>(
      /*convert_to=*/xnn_datatype_invalid,
      /*subgraph_flags=*/XNN_FLAG_NO_INLINED_LHS_PACKING);
}

using qcint8 = quantized<int8_t, channelwise>;

TEST(BatchMatrixMultiplyQD8F32, static_b) {
  TestStaticB<float, qcint8>(/*convert_to=*/xnn_datatype_qdint8);
}

}  // namespace xnnpack
