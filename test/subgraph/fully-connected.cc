// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>
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

// Compute output(oc) = bias(oc) + sum(input'(ic) * filter'(oc, ic)), where the
// input, filter, and bias are potentially quantized.
template <typename Input, typename Filter, typename Bias, typename Scale>
void MatrixVectorMultiply(const Input* input, const Tensor<Filter>& filter,
                          Tensor<Bias> bias,
                          const xnn_quantization_params& input_quantization,
                          int filter_zero_point, Tensor<Scale> filter_scale,
                          size_t filter_ic_block_size, int bias_zero_point,
                          Tensor<Scale> bias_scale, float* output) {
  for (size_t oc = 0; oc < filter.extent(0); ++oc) {
    double output_ic =
        bias.empty() ? 0.0f
                     : dequantize(bias(oc), {bias_zero_point, bias_scale(oc)});
    for (size_t ic = 0; ic < filter.extent(1); ++ic) {
      xnn_quantization_params filter_quantization = {
          filter_zero_point, filter_scale(oc, ic / filter_ic_block_size)};
      float input_ic = dequantize(input[ic], input_quantization);
      float filter_ic = dequantize(filter(oc, ic), filter_quantization);
      output_ic += input_ic * filter_ic;
    }
    output[oc] = output_ic;
  }
}

// It's hard to fit 4 bit types into the above template. We could do it if
// Tensor<T> used strides with a unit of bits instead of bytes...
// As a workaround, this template is equivalent to the above, except it reads
// two values at a time from the filter.
template <typename Input, typename Filter, typename Bias, typename Scale>
void MatrixVectorMultiplyInt4(const Input* input, const Tensor<Filter>& filter,
                              Tensor<Bias> bias,
                              const xnn_quantization_params& input_quantization,
                              int filter_zero_point, Tensor<Scale> filter_scale,
                              size_t filter_ic_block_size, int bias_zero_point,
                              Tensor<Scale> bias_scale, float* output) {
  for (size_t oc = 0; oc < filter.extent(0); ++oc) {
    double output_ic =
        bias.empty() ? 0.0f
                     : dequantize(bias(oc), {bias_zero_point, bias_scale(oc)});
    for (size_t ic = 0; ic < filter.extent(1); ++ic) {
      for (size_t p = 0; p < 2; ++p) {
        xnn_quantization_params filter_quantization = {
            filter_zero_point,
            filter_scale(oc, (ic * 2 + p) / filter_ic_block_size)};
        float input_ic = dequantize(input[ic * 2 + p], input_quantization);
        float filter_ic = dequantize(filter(oc, ic)[p], filter_quantization);
        output_ic += input_ic * filter_ic;
      }
    }
    output[oc] = output_ic;
  }
}

template <typename Input, typename Bias, typename Scale>
void MatrixVectorMultiply(const Input* input, const Tensor<qcint4>& filter,
                          Tensor<Bias> bias,
                          const xnn_quantization_params& input_quantization,
                          int filter_zero_point, Tensor<Scale> filter_scale,
                          size_t filter_ic_block_size, int bias_zero_point,
                          Tensor<Scale> bias_scale, float* output) {
  return MatrixVectorMultiplyInt4(
      input, filter, bias, input_quantization, filter_zero_point, filter_scale,
      filter_ic_block_size, bias_zero_point, bias_scale, output);
}

template <typename Input, typename Bias, typename Scale>
void MatrixVectorMultiply(const Input* input, const Tensor<qcuint4>& filter,
                          Tensor<Bias> bias,
                          const xnn_quantization_params& input_quantization,
                          int filter_zero_point, Tensor<Scale> filter_scale,
                          size_t filter_ic_block_size, int bias_zero_point,
                          Tensor<Scale> bias_scale, float* output) {
  return MatrixVectorMultiplyInt4(
      input, filter, bias, input_quantization, filter_zero_point, filter_scale,
      filter_ic_block_size, bias_zero_point, bias_scale, output);
}

template <typename Input, typename Bias, typename Scale>
void MatrixVectorMultiply(const Input* input, const Tensor<qbint4>& filter,
                          Tensor<Bias> bias,
                          const xnn_quantization_params& input_quantization,
                          int filter_zero_point, Tensor<Scale> filter_scale,
                          size_t filter_ic_block_size, int bias_zero_point,
                          Tensor<Scale> bias_scale, float* output) {
  return MatrixVectorMultiplyInt4(
      input, filter, bias, input_quantization, filter_zero_point, filter_scale,
      filter_ic_block_size, bias_zero_point, bias_scale, output);
}

template <typename Input, typename Bias, typename Scale>
void MatrixVectorMultiply(const Input* input, const Tensor<qbuint4>& filter,
                          Tensor<Bias> bias,
                          const xnn_quantization_params& input_quantization,
                          int filter_zero_point, Tensor<Scale> filter_scale,
                          size_t filter_ic_block_size, int bias_zero_point,
                          Tensor<Scale> bias_scale, float* output) {
  return MatrixVectorMultiplyInt4(
      input, filter, bias, input_quantization, filter_zero_point, filter_scale,
      filter_ic_block_size, bias_zero_point, bias_scale, output);
}

template <typename Input, typename Filter, typename Bias, typename Scale>
Tensor<float> ReferenceImpl(Tensor<Input> input, Tensor<Filter> filter,
                            Tensor<Bias> bias,
                            const xnn_quantization_params& input_quantization,
                            int filter_zero_point, Tensor<Scale> filter_scale,
                            size_t filter_ic_block_size, int bias_zero_point,
                            Tensor<Scale> bias_scale, uint32_t flags) {
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    filter = filter.transpose({1, 0});
  }

  std::vector<size_t> output_shape = input.shape();
  output_shape.back() = filter.extent(0);
  Tensor<float> output(output_shape);

  Tensor<Input> input_batches = input.slice(input.rank() - 1, 0);
  Tensor<float> output_batches = output.slice(output.rank() - 1, 0);

  for (const auto& i : EnumerateIndices(output_batches.shape())) {
    MatrixVectorMultiply(&input_batches(i), filter, bias, input_quantization,
                         filter_zero_point, filter_scale, filter_ic_block_size,
                         bias_zero_point, bias_scale, &output_batches(i));
  }

  return output;
}

// For float types, generate data in [-1, 1]
template <typename T>
DatatypeGenerator<T> MakeDatatypeGenerator(T) {
  return DatatypeGenerator<T>(-1.0f, 1.0f);
}
template <typename T>
T MaxOfDatatype(T) {
  return 1.0f;
}

// For quantized types, generate the full range of the type.
template <typename T, typename Kind>
DatatypeGenerator<quantized<T, Kind>> MakeDatatypeGenerator(
    quantized<T, Kind>) {
  return DatatypeGenerator<quantized<T, Kind>>();
}
template <typename T, typename Kind>
int32_t MaxOfDatatype(quantized<T, Kind>) {
  return NumericLimits<quantized<T, Kind>>::max();
}

// Generate 2 values at once with uint8.
DatatypeGenerator<quint8> MakeDatatypeGenerator(qcint4) {
  return DatatypeGenerator<quint8>();
}
DatatypeGenerator<quint8> MakeDatatypeGenerator(qcuint4) {
  return DatatypeGenerator<quint8>();
}
DatatypeGenerator<quint8> MakeDatatypeGenerator(qbint4) {
  return DatatypeGenerator<quint8>();
}
DatatypeGenerator<quint8> MakeDatatypeGenerator(qbuint4) {
  return DatatypeGenerator<quint8>();
}

// Generate values within an explicit range.
template <typename T>
DatatypeGenerator<T> MakeDatatypeGenerator(T, float min, float max) {
  return DatatypeGenerator<T>(min, max);
}

const size_t no_blockwise = std::numeric_limits<size_t>::max();

template <typename Input, typename Filter, typename Bias,
          typename Output = Input, typename Scale = float>
void TestStaticB(xnn_datatype convert_to = xnn_datatype_invalid,
                 size_t block_size = no_blockwise,
                 uint32_t runtime_flags = xnn_test_runtime_flags()) {
  const bool channelwise_quantization =
      xnn_datatype_is_channelwise_quantized(datatype_of<Filter>());
  const bool per_tensor_quantization =
      xnn_datatype_is_blockwise_quantized(datatype_of<Filter>()) &&
      !xnn_datatype_is_channelwise_quantized(datatype_of<Filter>()) &&
      !xnn_datatype_is_blockwise_quantized(datatype_of<Filter>());
  // If the filter datatype is sub-byte, we have more than one filter element
  // per value.
  const size_t filter_channel_factor =
      divide_round_up(8, xnn_datatype_size_bits(datatype_of<Filter>()));

  ReplicableRandomDevice rng;
  std::bernoulli_distribution flag_dist(0.5);

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  auto input_gen = MakeDatatypeGenerator(Input());
  auto output_gen = MakeDatatypeGenerator(Output());
  std::uniform_int_distribution<> channels_dist{1, 100};
  // TODO(b/408280445): The rank should go down to 1, but hits a bug in QP8
  // code paths that assume the LHS has rank >= 2.
  std::uniform_int_distribution<> rank_dist{2, XNN_MAX_TENSOR_DIMS - 1};

  for (auto _ : FuzzTest(std::chrono::milliseconds(500))) {
    size_t rank = rank_dist(rng);
    size_t input_channels = channels_dist(rng);
    size_t output_channels = channels_dist(rng);

    if (block_size != no_blockwise) {
      // Align the input channels to the block size.
      input_channels = round_up(input_channels, block_size);
    } else {
      // Align the input channels to the number of filter elements per byte.
      input_channels = round_up(input_channels, filter_channel_factor);
    }

    uint32_t flags = 0;
    if (filter_channel_factor > 1) {
      // Sub-byte datatypes don't support transposed weights
    } else if (flag_dist(rng)) {
      flags |= XNN_FLAG_TRANSPOSE_WEIGHTS;
    }
    if (flag_dist(rng)) {
      flags |= XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC;
    }

    // Make a random filter.
    std::vector<size_t> filter_shape = {output_channels,
                                        input_channels / filter_channel_factor};
    if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
      std::swap(filter_shape[0], filter_shape[1]);
    }
    auto filter_gen = MakeDatatypeGenerator(Filter());
    Tensor<Filter> filter(filter_shape, XnnExtraBytes);
    filter.generate([&]() { return filter_gen(rng); });

    xnn_quantization_params input_quantization =
        random_quantization(datatype_of<Input>(), rng, 0.001f, 2.0f);

    // All quantization schemes can be stored in a tensor with the appropriate
    // broadcasting:
    // - Per-tensor quantization is a scalar broadcasted to the filter shape.
    // - Per-channel quantization is a broadcast of the input channels.
    // - Blockwise is an upsampling by the block size in the input channels.
    xnn_quantization_params filter_quantization =
        random_quantization(datatype_of<Filter>(), rng, 0.001f, 2.0f);
    Tensor<Scale> filter_scale({per_tensor_quantization ? 1 : output_channels,
                                divide_round_up(input_channels, block_size)});
    if (filter_scale.size() > 1) {
      // Generate random per-channel scales, in the range of the original scale.
      std::uniform_real_distribution<float> filter_scale_dist(
          0.001f, filter_quantization.scale);
      filter_scale.generate([&]() { return filter_scale_dist(rng); });
    } else {
      filter_scale.fill(filter_quantization.scale);
    }
    broadcast_extent_1(filter_scale);

    // (Maybe) make a random bias.
    Tensor<Bias> bias;
    if (!std::is_same<Bias, invalid_type>::value && flag_dist(rng)) {
      std::vector<size_t> bias_shape = {output_channels};
      DatatypeGenerator<Bias> bias_gen = MakeDatatypeGenerator(
          Bias(), -max_abs_bias<Bias>(), max_abs_bias<Bias>());
      bias = Tensor<Bias>(bias_shape, XnnExtraBytes);
      bias.generate([&]() { return bias_gen(rng); });
    }

    // The output quantization is computed from the kernel size and input
    // quantization.
    xnn_quantization_params bias_quantization =
        xnn_datatype_is_quantized(datatype_of<Bias>())
            ? xnn_quantization_params{0, input_quantization.scale *
                                             filter_quantization.scale}
            : xnn_quantization_params{0, 1.0f};
    xnn_quantization_params output_quantization =
        CalculateGEMMQuantizationParams<Input, Filter, Output, Bias>(
            input_channels, input_quantization, filter_quantization,
            bias_quantization);

    float output_min = dequantize(output_gen(rng), output_quantization);
    float output_max = dequantize(output_gen(rng), output_quantization);
    if (output_min >= output_max) {
      // ~50% of the time, there is no min/max.
      output_min = -std::numeric_limits<float>::infinity();
      output_max = std::numeric_limits<float>::infinity();
    }

    SubgraphTester subgraph(4);
    const uint32_t input_id = 0;
    const uint32_t filter_id = 1;
    const uint32_t bias_id = bias.empty() ? XNN_INVALID_VALUE_ID : 2;
    const uint32_t output_id = 3;
    subgraph.AddInputTensor(rank, datatype_of<Input>(), input_quantization,
                            input_id);
    uint32_t fc_input_id = input_id;

    if (convert_to != xnn_datatype_invalid) {
      subgraph.AddInternalDynamicallyQuantizedTensor(
          rank, convert_to, /*num_nonbatch_dims=*/1, &fc_input_id);
      subgraph.AddConvert(input_id, fc_input_id);
    }

    std::vector<size_t> filter_dims = filter.extents();
    filter_dims[1] *= filter_channel_factor;
    if (block_size != no_blockwise) {
      uint32_t id = 0;
      ASSERT_EQ(
          xnn_status_success,
          xnn_define_blockwise_quantized_tensor_value_v3(
              subgraph.Subgraph(), datatype_of<Filter>(),
              filter_quantization.zero_point, filter_scale.data(),
              filter_dims.size(),
              /*block_dim=*/1, block_size, filter_dims.data(), filter.data(),
              filter_id, /*flags=*/0, datatype_of<Scale>(), &id));
      ASSERT_EQ(id, filter_id);
    } else if (channelwise_quantization) {
      const size_t channel_dim = flags & XNN_FLAG_TRANSPOSE_WEIGHTS ? 1 : 0;
      subgraph.AddStaticChannelwiseQuantizedTensor(
          filter_dims, channel_dim, datatype_of<Filter>(),
          reinterpret_cast<float*>(filter_scale.data()), filter_id,
          /*flags=*/0, filter.base());
    } else {
      subgraph.AddStaticTensor(filter_dims, filter_id, filter.base(),
                               filter_quantization);
    }
    if (bias_id != XNN_INVALID_VALUE_ID) {
      subgraph.AddStaticTensor(bias.extents(), bias_id, bias.base(),
                               bias_quantization);
    }
    subgraph
        .AddOutputTensor(rank, datatype_of<Output>(), output_quantization,
                         output_id)
        .AddFullyConnected(output_min, output_max, fc_input_id, filter_id,
                           bias_id, output_id, flags);
    xnn_status status = subgraph.CreateRuntime(nullptr, runtime_flags);
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    // Run the subgraph twice, with a different input/output shape each time
    // (except for the input/output channels, which are determined by the filter
    // shape).
    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = random_shape(rng, rank, 1, 4);
      std::vector<size_t> output_shape = input_shape;
      input_shape.back() = input_channels;
      output_shape.back() = output_channels;

      Tensor<Input> input(input_shape, XnnExtraBytes);
      input.generate([&]() { return input_gen(rng); });
      if (convert_to != xnn_datatype_invalid) {
        // If we are dynamically quantizing, preprocess the data to have zero
        // error when it will be quantized, which allows us to use a much
        // smaller tolerance for error for testing purposes.
        std::vector<size_t> input_batches = input.extents();
        input_batches.pop_back();
        for (const auto& i : EnumerateIndices(input_batches)) {
          FakeDynamicQuantize(input.slice_leading(i), convert_to);
        }
      }

      subgraph.ReshapeExternalTensor(input_shape, input.base(), input_id)
          .ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(output_id), output_shape)
          << ", input_shape=" << index_to_string(input_shape);

      // Run subgraph
      Tensor<Output> output(output_shape);
      subgraph.SetupExternalTensor(output.base(), output_id)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      int32_t bias_zero_point;
      Tensor<Scale> bias_scale({bias.size()});
      if (xnn_datatype_is_channelwise_quantized(datatype_of<Bias>())) {
        bias_zero_point = filter_quantization.zero_point;
        for (size_t k = 0; k < bias_scale.extent(0); k++) {
          bias_scale(k) = input_quantization.scale * filter_scale(k, 0);
        }
      } else {
        bias_zero_point = bias_quantization.zero_point;
        bias_scale.fill(bias_quantization.scale);
      }
      Tensor<float> expected =
          ReferenceImpl(input, filter, bias, input_quantization,
                        filter_quantization.zero_point, filter_scale,
                        block_size, bias_zero_point, bias_scale, flags);
      for (float& i : expected) {
        i = std::max(i, output_min);
        i = std::min(i, output_max);
      }

      ASSERT_EQ(expected.extents(), output.extents());
      if (xnn_datatype_is_quantized(datatype_of<Output>())) {
        for (const auto& i : EnumerateIndices(output.extents())) {
          ASSERT_NEAR(output(i),
                      quantize<Output>(expected(i), output_quantization), 1)
              << "i=" << index_to_string(i)
              << ", input_shape=" << index_to_string(input_shape)
              << ", output_shape=" << index_to_string(output_shape)
              << ", filter_shape=" << index_to_string(filter_shape);
        }
      } else {
        const float max_a = MaxOfDatatype(Input());
        const float max_b = MaxOfDatatype(Filter()) * filter_quantization.scale;
        const float max_bias =
            bias.empty() ? 0.0f
                         : max_abs_bias<Bias>() * bias_quantization.scale;
        const float tolerance = xnnpack::epsilon(xnn_datatype_of<Output>()) *
                                (input_channels * max_a * max_b + max_bias) *
                                4.0f;
        for (const auto& i : EnumerateIndices(output.extents())) {
          ASSERT_NEAR(static_cast<float>(output(i)), expected(i), tolerance)
              << "i=" << index_to_string(i)
              << ", input_shape=" << index_to_string(input_shape)
              << ", output_shape=" << index_to_string(output_shape)
              << ", filter_shape=" << index_to_string(filter_shape);
        }
      }
    }
  }
}

TEST(FullyConnectedQC8, static_b) { TestStaticB<qint8, qcint8, qcint32>(); }
TEST(FullyConnectedQS8, static_b) { TestStaticB<qint8, qint8, qint32>(); }
TEST(FullyConnectedQU8, static_b) { TestStaticB<quint8, quint8, qint32>(); }

TEST(FullyConnectedQS8QC8W, static_b) { TestStaticB<qint8, qcint8, qcint32>(); }
TEST(FullyConnectedQS8QC4W, static_b) { TestStaticB<qint8, qcint4, qcint32>(); }

TEST(FullyConnectedF16F32F16, static_b) {
  TestStaticB<xnn_float16, float, float>();
}
TEST(FullyConnectedF16, static_b) {
  TestStaticB<xnn_float16, xnn_float16, xnn_float16>();
}
TEST(FullyConnectedF32, static_b) { TestStaticB<float, float, float>(); }

// TODO(b/407771627): Either add xnn_datatype_qcuint4, or remove F32QC4W.
TEST(FullyConnectedF32QC4W, static_b) {
  // It looks like these kernels want the bias to be `float` and scaled by the
  // inverse channelwise weight scale. Setting `Bias` to `invalid_type` to
  // disable testing with a bias vector until we've figured this out.
  TestStaticB<float, qcuint4, invalid_type>();
}
TEST(FullyConnectedF32QC8W, static_b) {
  // It looks like these kernels want the bias to be `float` and scaled by the
  // inverse channelwise weight scale. Setting `Bias` to `invalid_type` to
  // disable testing with a bias vector until we've figured this out.
  TestStaticB<float, qcint8, invalid_type>();
}

TEST(FullyConnectedBF16F32, static_b) {
  TestStaticB<xnn_bfloat16, xnn_bfloat16, float, float>();
}

TEST(FullyConnectedQD8F16QC4W, static_b) {
  TestStaticB<xnn_float16, qcint4, float>(
      /*convert_to=*/xnn_datatype_qdint8);
}
TEST(FullyConnectedQD8F16QC8W, static_b) {
  TestStaticB<xnn_float16, qcint8, float>(
      /*convert_to=*/xnn_datatype_qdint8);
}
TEST(FullyConnectedQD8F32QC4W, static_b) {
  TestStaticB<float, qcint4, float>(/*convert_to=*/xnn_datatype_qdint8);
}
TEST(FullyConnectedQD8F32QC8W, static_b) {
  TestStaticB<float, qcint8, float>(/*convert_to=*/xnn_datatype_qdint8);
}

TEST(FullyConnectedQD8F16QB4UW, static_b) {
  TestStaticB<xnn_float16, qbuint4, float, xnn_float16, xnn_bfloat16>(
      /*convert_to=*/xnn_datatype_qdint8, /*block_size=*/32);
}
TEST(FullyConnectedQD8F16QB4W, static_b) {
  TestStaticB<xnn_float16, qbint4, float, xnn_float16, xnn_bfloat16>(
      /*convert_to=*/xnn_datatype_qdint8, /*block_size=*/32);
}

TEST(FullyConnectedQD8F32QB4UW, static_b) {
  TestStaticB<float, qbuint4, float, float, xnn_bfloat16>(
      /*convert_to=*/xnn_datatype_qdint8, /*block_size=*/32);
}
TEST(FullyConnectedQD8F32QB4W, static_b) {
  TestStaticB<float, qbint4, float, float, xnn_bfloat16>(
      /*convert_to=*/xnn_datatype_qdint8, /*block_size=*/32);
}

TEST(FullyConnectedQC8, dont_inline_pack_static_b) {
  TestStaticB<qint8, qcint8, qcint32>(
      /*convert_to=*/xnn_datatype_invalid, /*block_size=*/no_blockwise,
      /*runtime_flags=*/xnn_test_runtime_flags() |
          XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(FullyConnectedF16, dont_inline_pack_static_b) {
  TestStaticB<xnn_float16, float, float>(
      /*convert_to=*/xnn_datatype_invalid, /*block_size=*/no_blockwise,
      /*runtime_flags=*/xnn_test_runtime_flags() |
          XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(FullyConnectedF32, dont_inline_pack_static_b) {
  TestStaticB<float, float, float>(
      /*convert_to=*/xnn_datatype_invalid, /*block_size=*/no_blockwise,
      /*runtime_flags=*/xnn_test_runtime_flags() |
          XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(FullyConnectedQD8F16QC4W, dont_inline_pack_static_b) {
  TestStaticB<xnn_float16, qcint4, float>(
      /*convert_to=*/xnn_datatype_qdint8, /*block_size=*/no_blockwise,
      /*runtime_flags=*/xnn_test_runtime_flags() |
          XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(FullyConnectedQD8F16QC8W, dont_inline_pack_static_b) {
  TestStaticB<xnn_float16, qcint8, float>(
      /*convert_to=*/xnn_datatype_qdint8, /*block_size=*/no_blockwise,
      /*runtime_flags=*/xnn_test_runtime_flags() |
          XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(FullyConnectedQD8F32QC4W, dont_inline_pack_static_b) {
  TestStaticB<float, qcint4, float>(
      /*convert_to=*/xnn_datatype_qdint8, /*block_size=*/no_blockwise,
      /*runtime_flags=*/xnn_test_runtime_flags() |
          XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(FullyConnectedQD8F32QC8W, dont_inline_pack_static_b) {
  TestStaticB<float, qcint8, float>(
      /*convert_to=*/xnn_datatype_qdint8, /*block_size=*/no_blockwise,
      /*runtime_flags=*/xnn_test_runtime_flags() |
          XNN_FLAG_NO_INLINED_LHS_PACKING);
}

template <typename Input, typename Filter, typename Bias,
          typename Output = Input>
void TestDynamicB(xnn_datatype convert_to = xnn_datatype_invalid,
                  size_t block_size = no_blockwise,
                  uint32_t runtime_flags = xnn_test_runtime_flags()) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution flag_dist(0.5);

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  // There is no quantization in this case.
  xnn_quantization_params filter_quantization = {0, 1.0f};
  xnn_quantization_params input_quantization = {0, 1.0f};
  xnn_quantization_params output_quantization = {0, 1.0f};
  xnn_quantization_params bias_quantization = {0, 1.0f};
  Tensor<float> filter_scale({1, 1});
  filter_scale.fill(1.0f);
  broadcast_extent_1(filter_scale);

  auto input_gen = MakeDatatypeGenerator(Input());
  auto output_gen = MakeDatatypeGenerator(Output());
  std::uniform_int_distribution<> channels_dist{1, 100};
  std::uniform_int_distribution<> rank_dist{1, XNN_MAX_TENSOR_DIMS - 1};

  for (auto _ : FuzzTest(std::chrono::milliseconds(500))) {
    const size_t rank = rank_dist(rng);

    uint32_t flags = 0;
    if (flag_dist(rng)) {
      flags |= XNN_FLAG_TRANSPOSE_WEIGHTS;
    }
    if (flag_dist(rng)) {
      flags |= XNN_FLAG_SLOW_CONSISTENT_ARITHMETIC;
    }

    float output_min = output_gen(rng);
    float output_max = output_gen(rng);
    if (output_min >= output_max) {
      // ~50% of the time, there is no min/max.
      output_min = -std::numeric_limits<float>::infinity();
      output_max = std::numeric_limits<float>::infinity();
    }

    SubgraphTester subgraph(4);
    const uint32_t input_id = 0;
    const uint32_t filter_id = 1;
    const uint32_t bias_id = rng() & 1 ? XNN_INVALID_VALUE_ID : 2;
    const uint32_t output_id = 3;
    subgraph.AddInputTensor(rank, datatype_of<Input>(), input_id);

    subgraph.AddInputTensor(2, xnn_datatype_of<Filter>(), filter_id);
    if (bias_id != XNN_INVALID_VALUE_ID) {
      subgraph.AddInputTensor(1, xnn_datatype_of<Bias>(), bias_id);
    }
    subgraph.AddOutputTensor(rank, datatype_of<Output>(), output_id)
        .AddFullyConnected(output_min, output_max, input_id, filter_id, bias_id,
                           output_id, flags);
    xnn_status status =
        subgraph.CreateRuntime(/*threadpool=*/nullptr, runtime_flags);
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    // Run the subgraph twice, with a different input/output shape each time.
    for (int reshape = 0; reshape < 2; ++reshape) {
      size_t input_channels = channels_dist(rng);
      size_t output_channels = channels_dist(rng);

      // Make a random filter.
      std::vector<size_t> filter_shape = {output_channels, input_channels};
      if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
        std::swap(filter_shape[0], filter_shape[1]);
      }
      auto filter_gen = MakeDatatypeGenerator(Filter());
      Tensor<Filter> filter(filter_shape, XnnExtraBytes);
      filter.generate([&]() { return filter_gen(rng); });

      std::vector<size_t> input_shape = random_shape(rng, rank, 1, 4);
      std::vector<size_t> output_shape = input_shape;
      input_shape.back() = input_channels;
      output_shape.back() = output_channels;

      Tensor<Input> input(input_shape, XnnExtraBytes);
      input.generate([&]() { return input_gen(rng); });

      subgraph.ReshapeExternalTensor(input_shape, input.base(), input_id)
          .ReshapeExternalTensor(filter_shape, filter.base(), filter_id);
      Tensor<Bias> bias;
      if (bias_id != XNN_INVALID_VALUE_ID) {
        std::vector<size_t> bias_shape = {output_channels};
        bias = Tensor<Bias>(bias_shape, XnnExtraBytes);
        DatatypeGenerator<Bias> bias_gen = MakeDatatypeGenerator(
            Bias(), -max_abs_bias<Bias>(), max_abs_bias<Bias>());
        bias.generate([&]() { return bias_gen(rng); });
        subgraph.ReshapeExternalTensor(bias_shape, bias.base(), bias_id);
      }
      subgraph.ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(output_id), output_shape)
          << ", input_shape=" << index_to_string(input_shape);

      // Run subgraph
      Tensor<Output> output(output_shape);
      subgraph.SetupExternalTensor(output.base(), output_id)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      Tensor<float> bias_scale({bias.size()});
      bias_scale.fill(bias_quantization.scale);
      Tensor<float> expected = ReferenceImpl(
          input, filter, bias, input_quantization,
          filter_quantization.zero_point, filter_scale, block_size,
          bias_quantization.zero_point, bias_scale, flags);
      for (float& i : expected) {
        i = std::max(i, output_min);
        i = std::min(i, output_max);
      }

      ASSERT_EQ(expected.extents(), output.extents());
      if (xnn_datatype_is_quantized(datatype_of<Output>())) {
        for (const auto& i : EnumerateIndices(output.extents())) {
          ASSERT_NEAR(output(i),
                      quantize<Output>(expected(i), output_quantization), 1)
              << "input_shape=" << index_to_string(input_shape)
              << ", output_shape=" << index_to_string(output_shape)
              << ", filter_shape=" << index_to_string(filter_shape);
        }
      } else {
        const float max_a = MaxOfDatatype(Input());
        const float max_b = MaxOfDatatype(Filter()) * filter_quantization.scale;
        const float max_bias =
            bias.empty() ? 0.0f
                         : max_abs_bias<Bias>() * bias_quantization.scale;
        const float tolerance = xnnpack::epsilon(xnn_datatype_of<Output>()) *
                                (input_channels * max_a * max_b + max_bias) *
                                4.0f;
        for (const auto& i : EnumerateIndices(output.extents())) {
          ASSERT_NEAR(static_cast<float>(output(i)), expected(i), tolerance)
              << "input_shape=" << index_to_string(input_shape)
              << ", output_shape=" << index_to_string(output_shape)
              << ", filter_shape=" << index_to_string(filter_shape);
        }
      }
    }
  }
}

TEST(FullyConnectedF16, dynamic_b) {
  TestDynamicB<xnn_float16, xnn_float16, xnn_float16, xnn_float16>();
}
TEST(FullyConnectedF32, dynamic_b) {
  TestDynamicB<float, float, float, float>();
}

TEST(FullyConnectedF16, dont_inline_pack_dynamic_b) {
  TestDynamicB<xnn_float16, xnn_float16, xnn_float16, xnn_float16>(
      /*convert_to=*/xnn_datatype_invalid, /*block_size=*/no_blockwise,
      /*runtime_flags=*/xnn_test_runtime_flags() |
          XNN_FLAG_NO_INLINED_LHS_PACKING);
}
TEST(FullyConnectedF32, dont_inline_pack_dynamic_b) {
  TestDynamicB<float, float, float, float>(
      /*convert_to=*/xnn_datatype_invalid, /*block_size=*/no_blockwise,
      /*runtime_flags=*/xnn_test_runtime_flags() |
          XNN_FLAG_NO_INLINED_LHS_PACKING);
}

}  // namespace xnnpack
