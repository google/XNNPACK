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
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/stencil.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

template <typename Data, typename Filter, typename Bias>
Tensor<float> ReferenceImpl(Tensor<Data> input, Tensor<Filter> filter,
                            Tensor<Bias> bias,
                            const xnn_quantization_params& input_quantization,
                            int filter_zero_point, Tensor<float> filter_scale,
                            const xnn_quantization_params& bias_quantization,
                            size_t groups, size_t group_input_channels,
                            size_t group_output_channels,
                            const StencilParams& kh, const StencilParams& kw) {
  Tensor<float> output({input.extent(0), kh.output_extent(input.extent(1)),
                        kw.output_extent(input.extent(2)),
                        groups * group_output_channels});

  // Pad the input
  size_t h_padding_max =
      std::max(kh.padding_max, kh.dilated_kernel_extent() - 1 - kh.padding_min);
  size_t w_padding_max =
      std::max(kw.padding_max, kw.dilated_kernel_extent() - 1 - kw.padding_min);
  Tensor<Data> padded = input.pad(input_quantization.zero_point,
                                  {0, kh.padding_min, kw.padding_min, 0},
                                  {0, h_padding_max, w_padding_max, 0});

  padded = padded.split(3, {groups, group_input_channels});
  output = output.split(3, {groups, group_output_channels});
  if (!bias.empty()) {
    bias = bias.split(0, {groups, group_output_channels});
  }
  filter = filter.split(0, {groups, group_output_channels});
  if (!filter_scale.empty()) {
    filter_scale = filter_scale.split(0, {groups, group_output_channels});
  }

  padded = make_stencil_dim(padded, 2, kw);
  padded = make_stencil_dim(padded, 1, kh);
  for (size_t n = 0; n < output.extent(0); ++n) {
    for (size_t y = 0; y < output.extent(1); ++y) {
      for (size_t x = 0; x < output.extent(2); ++x) {
        for (size_t g = 0; g < groups; ++g) {
          for (size_t oc = 0; oc < group_output_channels; ++oc) {
            xnn_quantization_params filter_quantization = {filter_zero_point,
                                                           filter_scale(g, oc)};
            double output_nyxc =
                bias.empty() ? 0.0f
                             : dequantize(bias(g, oc), bias_quantization);
            for (size_t dy = 0; dy < kh.size; ++dy) {
              for (size_t dx = 0; dx < kw.size; ++dx) {
                for (size_t ic = 0; ic < group_input_channels; ++ic) {
                  float padded_nyxc = dequantize(padded(n, y, dy, x, dx, g, ic),
                                                 input_quantization);
                  float filter_yxc = dequantize(filter(g, oc, dy, dx, ic),
                                                filter_quantization);
                  output_nyxc += padded_nyxc * filter_yxc;
                }
              }
            }
            output(n, y, x, g, oc) = output_nyxc;
          }
        }
      }
    }
  }

  return output.fuse({3, 4});
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

template <>
DatatypeGenerator<quantized<int32_t>> MakeDatatypeGenerator(
    quantized<int32_t>) {
  return DatatypeGenerator<quantized<int32_t>>(-10000, 10000, {0, 1.0f});
}

ConvolutionParams StencilToConvolutionParams(const StencilParams& kh,
                                             const StencilParams& kw) {
  ConvolutionParams params;
  params.padding.top = kh.padding_min;
  params.padding.left = kw.padding_min;
  params.padding.bottom = kh.padding_max;
  params.padding.right = kw.padding_max;
  params.kernel.height = kh.size;
  params.kernel.width = kw.size;
  params.subsampling.height = kh.stride;
  params.subsampling.width = kw.stride;
  params.dilation.height = kh.dilation;
  params.dilation.width = kw.dilation;
  return params;
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
xnn_quantization_params CalculateConvolutionQuantizationParams(
    size_t reduction_size, xnn_quantization_params input_quantization,
    xnn_quantization_params filter_quantization) {
  if (!xnn_datatype_is_quantized(xnn_datatype_of<Output>())) {
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

template <typename Data, typename Filter, typename Bias>
void TestImpl(xnn_datatype convert_to = xnn_datatype_invalid) {
  const bool channelwise_quantization =
      xnn_datatype_is_channelwise_quantized(xnn_datatype_of<Filter>());
  ReplicableRandomDevice rng;
  std::bernoulli_distribution bool_dist(0.5);

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  auto data_gen = MakeDatatypeGenerator(Data());

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    // Generate some random kernel and shape parameters.
    StencilParams kw = random_stencil_params(rng);
    StencilParams kh = random_stencil_params(rng);

    const bool same_padding = bool_dist(rng);

    uint32_t flags = 0;
    if (same_padding) {
      flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
      kw.padding_min = kw.padding_max = 0;
      kh.padding_min = kh.padding_max = 0;
    }

    ConvolutionParams params = StencilToConvolutionParams(kh, kw);
    std::uniform_int_distribution<> groups_dist{1, 3};
    params.groups = groups_dist(rng);
    // To get coverage of large dimensions without making the test too slow,
    // make one big channel dimension and one small one.
    std::uniform_int_distribution<> big_channels_dist{1, 100};
    std::uniform_int_distribution<> small_channels_dist{1, 10};
    params.group_input_channels = big_channels_dist(rng);
    params.group_output_channels = small_channels_dist(rng);
    if (bool_dist(rng)) {
      std::swap(params.group_input_channels, params.group_output_channels);
    }

    // Make a random filter.
    std::vector<size_t> filter_shape = {
        params.groups * params.group_output_channels,
        params.kernel.height,
        params.kernel.width,
        params.group_input_channels,
    };
    DatatypeGenerator<Filter> filter_gen = MakeDatatypeGenerator(Filter());
    Tensor<Filter> filter(filter_shape, XnnExtraBytes);
    filter.generate([&]() { return filter_gen(rng); });
    const size_t reduction_size =
        filter.extent(1) * filter.extent(2) * filter.extent(3);

    // (Maybe) make a random bias.
    Tensor<Bias> bias;
    if (bool_dist(rng)) {
      std::vector<size_t> bias_shape = {params.groups *
                                        params.group_output_channels};
      DatatypeGenerator<Bias> bias_gen = MakeDatatypeGenerator(Bias());
      Tensor<Bias> bias(bias_shape, XnnExtraBytes);
      bias.generate([&]() { return bias_gen(rng); });
    }

    xnn_quantization_params input_quantization =
        random_quantization(xnn_datatype_of<Data>(), rng, 0.001f, 2.0f);

    // The filter quantization might have a per-channel scale. We always store
    // the scale in a Tensor, but it might be a broadcast of a single value.
    xnn_quantization_params filter_quantization =
        random_quantization(xnn_datatype_of<Filter>(), rng, 0.001f, 2.0f);
    Tensor<float> filter_scale(
        {channelwise_quantization ? filter.extent(0) : 1});
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
        CalculateConvolutionQuantizationParams<Data, Filter, Data>(
            reduction_size, input_quantization, filter_quantization);
    xnn_quantization_params bias_quantization = {
        0, input_quantization.scale * filter_quantization.scale};

    params.output_min = dequantize(data_gen(rng), output_quantization);
    params.output_max = dequantize(data_gen(rng), output_quantization);
    if (params.output_min >= params.output_max) {
      // ~50% of the time, there is no min/max.
      params.output_min = -std::numeric_limits<float>::infinity();
      params.output_max = std::numeric_limits<float>::infinity();
    }

    SubgraphTester subgraph(4);
    const uint32_t input_id = 0;
    const uint32_t filter_id = 1;
    const uint32_t bias_id = bias.empty() ? XNN_INVALID_VALUE_ID : 2;
    const uint32_t output_id = 3;
    subgraph.AddInputTensor(4, xnn_datatype_of<Data>(), input_quantization,
                            input_id);
    uint32_t conv_input_id = input_id;

    if (convert_to != xnn_datatype_invalid) {
      subgraph.AddInternalDynamicallyQuantizedTensor(
          4, convert_to, /*num_nonbatch_dims=*/3, &conv_input_id);
      subgraph.AddConvert(input_id, conv_input_id);
    }

    if (channelwise_quantization) {
      subgraph.AddStaticTensorQS8(
          filter.extents(), /*channel_dim=*/0, TensorType::kDense,
          filter_scale.data(), filter_id,
          /*flags=*/0, reinterpret_cast<int8_t*>(filter.base()));
    } else {
      subgraph.AddStaticTensor(filter.extents(), filter_id, filter.base(),
                               filter_quantization);
    }
    if (bias_id != XNN_INVALID_VALUE_ID) {
      subgraph.AddStaticTensor(bias.extents(), bias_id, bias.base(),
                               bias_quantization);
    }
    subgraph
        .AddOutputTensor(4, xnn_datatype_of<Data>(), output_quantization,
                         output_id)
        .AddConvolution2D(params, conv_input_id, filter_id, bias_id, output_id,
                          flags);
    xnn_status status = subgraph.CreateRuntime();
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    // Run the subgraph twice, with a different input/output shape each time.
    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> output_shape = random_shape(rng, 4);
      std::vector<size_t> input_shape = {
          output_shape[0],
          kh.input_extent(output_shape[1], same_padding),
          kw.input_extent(output_shape[2], same_padding),
          params.groups * params.group_input_channels,
      };
      output_shape[3] = params.groups * params.group_output_channels;

      if (same_padding) {
        kh.compute_tf_same_padding(input_shape[1]);
        kw.compute_tf_same_padding(input_shape[2]);
      }

      Tensor<Data> input(input_shape, XnnExtraBytes);
      input.generate([&]() { return data_gen(rng); });
      if (convert_to != xnn_datatype_invalid) {
        assert(convert_to == xnn_datatype_qdint8);
        // If we are dynamically quantizing, preprocess the data to have zero
        // error when it will be quantized, which makes testing the behavior
        // much easier.
        std::vector<size_t> input_batches = input.extents();
        if (kw.size == 1 && kh.size == 1 && kw.stride == 1 && kh.stride == 1 &&
            kw.padding() == 0 && kh.padding() == 0 && params.groups == 1) {
          // 1x1 conv gets rewritten to fully connected, which
          // dynamically quantizes at a finer granularity.
          input_batches.resize(3);
        } else {
          input_batches.resize(1);
        }
        for (const auto& i : EnumerateIndices(input_batches)) {
          FakeDynamicQuantize(input.slice_leading(i), convert_to);
        }
      }

      subgraph.ReshapeExternalTensor(input_shape, input.base(), input_id)
          .ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(output_id), output_shape)
          << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
          << ", kw=" << kw;

      // Run subgraph
      Tensor<Data> output(output_shape);
      subgraph.SetupExternalTensor(output.base(), output_id)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      Tensor<float> expected = ReferenceImpl(
          input, filter, bias, input_quantization,
          filter_quantization.zero_point, filter_scale, bias_quantization,
          params.groups, params.group_input_channels,
          params.group_output_channels, kh, kw);
      for (float& i : expected) {
        i = std::max(i, params.output_min);
        i = std::min(i, params.output_max);
      }

      ASSERT_EQ(expected.extents(), output.extents());
      for (const auto& i : EnumerateIndices(output.extents())) {
        if (xnn_datatype_is_quantized(xnn_datatype_of<Data>())) {
          ASSERT_NEAR(output(i),
                      quantize<Data>(expected(i), output_quantization), 1)
              << "input_shape=" << index_to_string(input_shape)
              << ", output_shape=" << index_to_string(output_shape)
              << ", filter_shape=" << index_to_string(filter_shape)
              << ", kh=" << kh << ", kw=" << kw;
        } else {
          const float max_a = MaxDatatype(Data()) * input_quantization.scale;
          const float max_b = MaxDatatype(Filter()) * filter_quantization.scale;
          const float tolerance = xnnpack::epsilon(xnn_datatype_of<Data>()) *
                                  reduction_size * max_a * max_b * 4.0f;
          ASSERT_NEAR(static_cast<float>(output(i)), expected(i), tolerance)
              << "input_shape=" << index_to_string(input_shape)
              << ", output_shape=" << index_to_string(output_shape)
              << ", filter_shape=" << index_to_string(filter_shape)
              << ", kh=" << kh << ", kw=" << kw;
        }
      }
    }
  }
}

using quint8 = quantized<uint8_t>;
using qint8 = quantized<int8_t>;
using qcint8 = quantized<int8_t, channelwise>;
using qint32 = quantized<int32_t>;

TEST(Convolution2DQC8, test) { TestImpl<qint8, qcint8, qint32>(); }
TEST(Convolution2DQU8, test) { TestImpl<quint8, quint8, qint32>(); }
TEST(Convolution2DQS8, test) { TestImpl<qint8, qint8, qint32>(); }
TEST(Convolution2DF16, test) { TestImpl<xnn_float16, float, float>(); }
TEST(Convolution2DF32, test) { TestImpl<float, float, float>(); }
TEST(Convolution2DQD8F16QC8W, test) {
  TestImpl<xnn_float16, qcint8, xnn_float16>(
      /*convert_to=*/xnn_datatype_qdint8);
}
TEST(Convolution2DQD8F32QC8W, test) {
  TestImpl<float, qcint8, float>(/*convert_to=*/xnn_datatype_qdint8);
}

}  // namespace xnnpack
