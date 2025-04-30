// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

// The next chunk of things enables us to work with int4 data in the
// datatype/Tensor template system. It's not a perfect abstraction, I think with
// a little bit of improvement, this could be a clean mechanism.

// Two int4 values stored in an int8.
struct int4x2 {
  uint8_t value;

  int4x2() = default;
  int4x2(uint8_t value) : value(value) {}  // NOLINT

  int8_t operator[](size_t i) const {
    int8_t result = (value >> (i * 4)) & 0xf;
    // Sign extend
    result = static_cast<int8_t>(result << 4) >> 4;
    return result;
  }
};

struct uint4x2 {
  uint8_t value;

  uint4x2() = default;
  uint4x2(uint8_t value) : value(value) {}  // NOLINT

  uint8_t operator[](size_t i) const { return (value >> (i * 4)) & 0xf; }
};

using quint8 = quantized<uint8_t>;
using qint8 = quantized<int8_t>;
using qcint8 = quantized<int8_t, channelwise>;
using qint32 = quantized<int32_t>;
using qcint4 = quantized<int4x2, channelwise>;

// This is not a "real" XNNPACK datatype, but it is required to match the
// behavior of F32QC4W (b/407771627).
using qcuint4 = quantized<uint4x2, channelwise>;

template <>
class NumericLimits<qcint4> {
 public:
  static int32_t min() { return -8; }
  static int32_t max() { return 7; }
  static int32_t smallest_normal() { return 0; }
  static int32_t min_identity() { return max(); }
  static int32_t max_identity() { return min(); }
};

template <>
class NumericLimits<qcuint4> {
 public:
  static int32_t min() { return -8; }
  static int32_t max() { return 7; }
  static int32_t smallest_normal() { return 0; }
  static int32_t min_identity() { return max(); }
  static int32_t max_identity() { return min(); }
};

template <typename T>
xnn_datatype datatype_of() {
  if (std::is_same<T, qcint4>::value) {
    return xnn_datatype_qcint4;
  } else if (std::is_same<T, qcuint4>::value) {
    return xnn_datatype_qcint4;
  } else {
    return xnn_datatype_of<T>();
  }
}

// Implements XNN_FLAG_TENSORFLOW_RESHAPE_2D
std::vector<size_t> Reshape2D(const std::vector<size_t>& shape) {
  size_t total =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  size_t channels = shape.back();
  return {total / channels, channels};
}

// Compute output(oc) = bias(oc) + sum(input'(ic) * filter'(oc, ic)), where the
// input, filter, and bias are potentially quantized.
template <typename Input, typename Filter, typename Bias, typename Scale>
void MatrixVectorMultiply(const Input* input, const Tensor<Filter>& filter,
                          Tensor<Bias> bias,
                          const xnn_quantization_params& input_quantization,
                          int filter_zero_point, Tensor<Scale> filter_scale,
                          size_t filter_ic_block_size,
                          const xnn_quantization_params& bias_quantization,
                          float* output) {
  for (size_t oc = 0; oc < filter.extent(0); ++oc) {
    double output_ic =
        bias.empty() ? 0.0f : dequantize(bias(oc), bias_quantization);
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
                              size_t filter_ic_block_size,
                              const xnn_quantization_params& bias_quantization,
                              float* output) {
  for (size_t oc = 0; oc < filter.extent(0); ++oc) {
    double output_ic =
        bias.empty() ? 0.0f : dequantize(bias(oc), bias_quantization);
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
                          size_t filter_ic_block_size,
                          const xnn_quantization_params& bias_quantization,
                          float* output) {
  return MatrixVectorMultiplyInt4(
      input, filter, bias, input_quantization, filter_zero_point, filter_scale,
      filter_ic_block_size, bias_quantization, output);
}

template <typename Input, typename Bias, typename Scale>
void MatrixVectorMultiply(const Input* input, const Tensor<qcuint4>& filter,
                          Tensor<Bias> bias,
                          const xnn_quantization_params& input_quantization,
                          int filter_zero_point, Tensor<Scale> filter_scale,
                          size_t filter_ic_block_size,
                          const xnn_quantization_params& bias_quantization,
                          float* output) {
  return MatrixVectorMultiplyInt4(
      input, filter, bias, input_quantization, filter_zero_point, filter_scale,
      filter_ic_block_size, bias_quantization, output);
}

template <typename Input, typename Filter, typename Bias, typename Scale>
Tensor<float> ReferenceImpl(Tensor<Input> input, Tensor<Filter> filter,
                            Tensor<Bias> bias,
                            const xnn_quantization_params& input_quantization,
                            int filter_zero_point, Tensor<Scale> filter_scale,
                            size_t filter_ic_block_size,
                            const xnn_quantization_params& bias_quantization,
                            uint32_t flags) {
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    filter = filter.transpose({1, 0});
  }

  std::vector<size_t> output_shape = input.shape();
  output_shape.back() = filter.extent(0);
  Tensor<float> output(output_shape);

  Tensor<Input> input_batches = input.slice(input.rank() - 1, 0);
  Tensor<float> output_batches = output.slice(output.rank() - 1, 0);

  for (const auto& i : EnumerateIndices(output_batches.extents())) {
    MatrixVectorMultiply(&input_batches(i), filter, bias, input_quantization,
                         filter_zero_point, filter_scale, filter_ic_block_size,
                         bias_quantization, &output_batches(i));
  }

  if (flags & XNN_FLAG_TENSORFLOW_RESHAPE_2D) {
    output = output.reshape(Reshape2D(output.extents()));
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

DatatypeGenerator<qint32> MakeDatatypeGenerator(qint32) {
  return DatatypeGenerator<qint32>(-10000, 10000, {0, 1.0f});
}

// Generate 2 values at once with uint8.
DatatypeGenerator<quint8> MakeDatatypeGenerator(qcint4) {
  return DatatypeGenerator<quint8>();
}
DatatypeGenerator<quint8> MakeDatatypeGenerator(qcuint4) {
  return DatatypeGenerator<quint8>();
}

template <typename T>
xnn_quantization_params quantization_for_range(float min, float max) {
  xnn_quantization_params result;
  result.scale = (max - min) / (static_cast<float>(NumericLimits<T>::max()) -
                                static_cast<float>(NumericLimits<T>::min()));
  result.zero_point = NumericLimits<T>::min() - min / result.scale;
  return result;
}

template <typename Input, typename Filter, typename Output>
xnn_quantization_params CalculateFullyConnectedQuantizationParams(
    size_t reduction_size, xnn_quantization_params input_quantization,
    xnn_quantization_params filter_quantization) {
  if (!xnn_datatype_is_quantized(datatype_of<Output>())) {
    return {0, 1.0f};
  }

  // Get the dequantized input and filter ranges.
  const float input_min =
      dequantize(NumericLimits<Input>::min(), input_quantization);
  const float input_max =
      dequantize(NumericLimits<Input>::max(), input_quantization);
  const float filter_min =
      dequantize(NumericLimits<Filter>::min(), filter_quantization);
  const float filter_max =
      dequantize(NumericLimits<Filter>::max(), filter_quantization);

  // Find the range of the product of an input and a filter value.
  std::array<float, 4> corners = {
      input_min * filter_min,
      input_max * filter_min,
      input_min * filter_max,
      input_max * filter_max,
  };
  auto input_filter_minmax =
      std::minmax_element(corners.begin(), corners.end());

  const float output_min = *input_filter_minmax.first * reduction_size;
  const float output_max = *input_filter_minmax.second * reduction_size;

  // Now we want the output quantization to hold the range of the output.
  return quantization_for_range<Output>(output_min, output_max);
}

// Dynamic quantization looks a lot like a float input/output, but the error is
// hard to quantify and test well. Rather than do that, we can just generate
// input data that has (close to) zero error when dynamically quantized, which
// makes it easier to test.
template <typename Data>
void FakeDynamicQuantize(Tensor<Data> input, float qmin, float qmax) {
  auto minmax = std::minmax_element(input.begin(), input.end());
  const float rmin = *minmax.first;
  const float rmax = *minmax.second;
  const float scale = rmin == rmax ? 1.0f : (qmax - qmin) / (rmax - rmin);
  const float inv_scale = 1.0f / scale;
  for (auto& i : input) {
    i = std::round((i - rmin) * scale) * inv_scale;
  }
}

template <typename Data>
void FakeDynamicQuantize(Tensor<Data> input, xnn_datatype datatype) {
  if (datatype == xnn_datatype_qdint8) {
    FakeDynamicQuantize(input, -128.0f, 127.0f);
  } else if (datatype == xnn_datatype_qduint8) {
    FakeDynamicQuantize(input, 0.0f, 255.0f);
  } else {
    XNN_UNREACHABLE;
  }
}

template <typename Data>
void FakeDynamicQuantize(const Tensor<quantized<Data>>& input, xnn_datatype) {}

const size_t no_blockwise = std::numeric_limits<size_t>::max();

template <typename Input, typename Filter, typename Bias,
          typename Output = Input, typename Scale = float>
void TestStaticB(xnn_datatype convert_to = xnn_datatype_invalid,
                 size_t block_size = no_blockwise) {
  const bool channelwise_quantization =
      xnn_datatype_is_channelwise_quantized(datatype_of<Filter>());
  // If the filter datatype is sub-byte, we have more than one filter element
  // per value.
  const size_t filter_channel_factor =
      divide_round_up(8, xnn_datatype_size_bits(datatype_of<Filter>()));

  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  auto input_gen = MakeDatatypeGenerator(Input());
  auto output_gen = MakeDatatypeGenerator(Output());
  std::uniform_int_distribution<> channels_dist{1, 100};
  // TODO(b/408280445): The rank should go down to 1, but hits a bug in QP8
  // codepaths that assume the LHS has rank >= 2.
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
    } else if (rng() & 1) {
      flags |= XNN_FLAG_TRANSPOSE_WEIGHTS;
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

    // (Maybe) make a random bias.
    Tensor<Bias> bias;
    if (rng() & 1) {
      std::vector<size_t> bias_shape = {output_channels};
      DatatypeGenerator<Bias> bias_gen = MakeDatatypeGenerator(Bias());
      Tensor<Bias> bias(bias_shape, XnnExtraBytes);
      bias.generate([&]() { return bias_gen(rng); });
    }

    xnn_quantization_params input_quantization =
        random_quantization(datatype_of<Input>(), rng, 0.001f, 2.0f);

    // All quantization schemes can be stored in a tensor with the appropriate
    // broadcasting:
    // - Per-tensor quantization is a scalar broadcasted to the filter shape.
    // - Per-channel quantization is a broadcast of the input channels.
    // - Blockwise is an upsampling by the block size in the input channels.
    xnn_quantization_params filter_quantization =
        random_quantization(datatype_of<Filter>(), rng, 0.001f, 2.0f);
    Tensor<Scale> filter_scale({channelwise_quantization ? output_channels : 1,
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

    // The output quantization is computed from the kernel size and input
    // quantization.
    xnn_quantization_params output_quantization =
        CalculateFullyConnectedQuantizationParams<Input, Filter, Output>(
            input_channels, input_quantization, filter_quantization);
    xnn_quantization_params bias_quantization = {
        0, input_quantization.scale * filter_quantization.scale};

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
      filter_quantization.zero_point = 8;
      uint32_t id = 0;
      ASSERT_EQ(xnn_status_success,
                xnn_define_blockwise_quantized_tensor_value(
                    subgraph.Subgraph(), xnn_datatype_qbint4,
                    filter_quantization.zero_point,
                    reinterpret_cast<const uint16_t*>(filter_scale.data()),
                    filter_dims.size(),
                    /*channel_dim=*/0, block_size, filter_dims.data(),
                    filter.data(), filter_id, /*flags=*/0, &id));
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
    xnn_status status = subgraph.CreateRuntime();
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
      if (flags & XNN_FLAG_TENSORFLOW_RESHAPE_2D) {
        output_shape = Reshape2D(output_shape);
      }

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
      Tensor<float> expected =
          ReferenceImpl(input, filter, bias, input_quantization,
                        filter_quantization.zero_point, filter_scale,
                        block_size, bias_quantization, flags);
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
        const float tolerance = xnnpack::epsilon(xnn_datatype_of<Output>()) *
                                input_channels * max_a * max_b * 4.0f;
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

TEST(FullyConnectedQC8, static_b) { TestStaticB<qint8, qcint8, qint32>(); }
TEST(FullyConnectedQU8, static_b) { TestStaticB<quint8, quint8, qint32>(); }
TEST(FullyConnectedQS8QC8W, static_b) { TestStaticB<qint8, qcint8, qint32>(); }
TEST(FullyConnectedQS8QC4W, static_b) { TestStaticB<qint8, qcint4, qint32>(); }
TEST(FullyConnectedF16, static_b) { TestStaticB<xnn_float16, float, float>(); }
TEST(FullyConnectedF32, static_b) { TestStaticB<float, float, float>(); }
// TODO(b/407771627): Either add xnn_datatype_qcuint4, or remove F32QC4W.
TEST(FullyConnectedF32QC4W, static_b) { TestStaticB<float, qcuint4, float>(); }
TEST(FullyConnectedF32QC8W, static_b) { TestStaticB<float, qcint8, float>(); }
TEST(FullyConnectedBF16F32, static_b) {
  TestStaticB<xnn_bfloat16, xnn_bfloat16, float, float>();
}
TEST(FullyConnectedQD8F16QC4W, static_b) {
  TestStaticB<xnn_float16, qcint4, xnn_float16>(
      /*convert_to=*/xnn_datatype_qdint8);
}
TEST(FullyConnectedQD8F16QC8W, static_b) {
  TestStaticB<xnn_float16, qcint8, xnn_float16>(
      /*convert_to=*/xnn_datatype_qdint8);
}
TEST(FullyConnectedQD8F32QC4W, static_b) {
  TestStaticB<float, qcint4, float>(/*convert_to=*/xnn_datatype_qdint8);
}
TEST(FullyConnectedQD8F32QC8W, static_b) {
  TestStaticB<float, qcint8, float>(/*convert_to=*/xnn_datatype_qdint8);
}

TEST(FullyConnectedQD8F16QB4W, static_b) {
  TestStaticB<xnn_float16, qcuint4, xnn_float16, xnn_float16, xnn_bfloat16>(
      /*convert_to=*/xnn_datatype_qdint8, /*block_size=*/32);
}
TEST(FullyConnectedQD8F32QB4W, static_b) {
  TestStaticB<float, qcuint4, float, float, xnn_bfloat16>(
      /*convert_to=*/xnn_datatype_qdint8, /*block_size=*/32);
}

template <typename Input, typename Filter, typename Bias,
          typename Output = Input>
void TestDynamicB(xnn_datatype convert_to = xnn_datatype_invalid,
                  size_t block_size = no_blockwise) {
  ReplicableRandomDevice rng;

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
    if (rng() & 1) {
      flags |= XNN_FLAG_TRANSPOSE_WEIGHTS;
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
    xnn_status status = subgraph.CreateRuntime();
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
      if (flags & XNN_FLAG_TENSORFLOW_RESHAPE_2D) {
        output_shape = Reshape2D(output_shape);
      }

      Tensor<Input> input(input_shape, XnnExtraBytes);
      input.generate([&]() { return input_gen(rng); });

      subgraph.ReshapeExternalTensor(input_shape, input.base(), input_id)
          .ReshapeExternalTensor(filter_shape, filter.base(), filter_id);
      Tensor<Bias> bias;
      if (bias_id != XNN_INVALID_VALUE_ID) {
        std::vector<size_t> bias_shape = {output_channels};
        bias = Tensor<Bias>(bias_shape, XnnExtraBytes);
        DatatypeGenerator<Bias> bias_gen = MakeDatatypeGenerator(Bias());
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
      Tensor<float> expected =
          ReferenceImpl(input, filter, bias, input_quantization,
                        filter_quantization.zero_point, filter_scale,
                        block_size, bias_quantization, flags);
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
        const float tolerance = xnnpack::epsilon(xnn_datatype_of<Output>()) *
                                input_channels * max_a * max_b * 4.0f;
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

}  // namespace xnnpack
