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
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/quantization-helpers.h"
#include "test/subgraph/stencil.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

template <typename Data, typename Filter, typename Bias, typename Scale>
Tensor<float> ReferenceImpl(Tensor<Data> input, Tensor<Filter> filter,
                            Tensor<Bias> bias,
                            const xnn_quantization_params& input_quantization,
                            int filter_zero_point, Tensor<float> filter_scale,
                            int bias_zero_point, Tensor<Scale> bias_scale,
                            size_t groups, size_t group_input_channels,
                            size_t group_output_channels,
                            const StencilParams& kh, const StencilParams& kw,
                            const Adjustment& adjustment) {
  Tensor<float> output({input.extent(0), kh.input_extent(input.extent(1), false,
                                                         adjustment.height),
                        kw.input_extent(input.extent(2), false,
                                        adjustment.width),
                        groups * group_output_channels});

  input = input.split(3, {groups, group_input_channels});
  output = output.split(3, {groups, group_output_channels});
  if (!bias.empty()) {
    bias = bias.split(0, {groups, group_output_channels});
  }
  filter = filter.split(0, {groups, group_output_channels});
  if (!filter_scale.empty()) {
    filter_scale = filter_scale.split(0, {groups, group_output_channels});
  }

  for (size_t n = 0; n < output.extent(0); ++n) {
    for (size_t oy = 0; oy < output.extent(1); ++oy) {
      for (size_t ox = 0; ox < output.extent(2); ++ox) {
        for (size_t g = 0; g < groups; ++g) {
          for (size_t oc = 0; oc < group_output_channels; ++oc) {
            xnn_quantization_params filter_quantization = {filter_zero_point,
                                                           filter_scale(g, oc)};
            double output_nyxc =
                bias.empty() ? 0.0f
                             : dequantize(bias(g, oc),
                                          {bias_zero_point, bias_scale(g, oc)});
            for (size_t dy = 0; dy < kh.size; ++dy) {
              const size_t y = oy + kh.padding_min - dy * kh.dilation;
              const size_t iy = y / kh.stride;
              if (iy * kh.stride != y || iy >= input.extent(1)) continue;
              for (size_t dx = 0; dx < kw.size; ++dx) {
                const size_t x = ox + kw.padding_min - dx * kw.dilation;
                const size_t ix = x / kw.stride;
                if (ix * kw.stride != x || ix >= input.extent(2)) continue;
                for (size_t ic = 0; ic < group_input_channels; ++ic) {
                  float input_nyxc =
                      dequantize(input(n, iy, ix, g, ic), input_quantization);
                  float filter_yxc = dequantize(filter(g, oc, dy, dx, ic),
                                                filter_quantization);
                  output_nyxc += input_nyxc * filter_yxc;
                }
              }
            }
            output(n, oy, ox, g, oc) = output_nyxc;
          }
        }
      }
    }
  }

  return output.fuse({3, 4});
}

DeconvolutionParams StencilToDeconvolutionParams(const StencilParams& kh,
                                                 const StencilParams& kw) {
  DeconvolutionParams params;
  params.padding.top = kh.padding_min;
  params.padding.left = kw.padding_min;
  params.padding.bottom = kh.padding_max;
  params.padding.right = kw.padding_max;
  params.kernel.height = kh.size;
  params.kernel.width = kw.size;
  params.upsampling.height = kh.stride;
  params.upsampling.width = kw.stride;
  params.dilation.height = kh.dilation;
  params.dilation.width = kw.dilation;
  return params;
}

template <typename Data, typename Filter, typename Bias, typename Scale = float>
void TestImpl(xnn_datatype convert_to = xnn_datatype_invalid) {
  const bool channelwise_quantization =
      xnn_datatype_is_channelwise_quantized(xnn_datatype_of<Filter>());
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  auto data_gen = MakeDatatypeGenerator(Data());

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    // Generate some random kernel and shape parameters.
    StencilParams kw = random_stencil_params(rng);
    StencilParams kh = random_stencil_params(rng);

    DeconvolutionParams params = StencilToDeconvolutionParams(kh, kw);
    std::uniform_int_distribution<> channels_dist{1, 10};
    std::uniform_int_distribution<> groups_dist{1, 3};
    std::uniform_int_distribution<>
        height_adjustment_dist{0, static_cast<int>(kh.stride) - 1};
    std::uniform_int_distribution<>
        width_adjustment_dist{0, static_cast<int>(kw.stride) - 1};
    params.groups = groups_dist(rng);
    params.group_input_channels = channels_dist(rng);
    params.group_output_channels = channels_dist(rng);
    params.adjustment.height = 0;
    params.adjustment.width = 0;

    // Padding mode is `Valid`.
    if (kw.padding_min == 0 && kw.padding_max == 0 &&
        kh.padding_min == 0 && kh.padding_max == 0) {
      params.adjustment.height = height_adjustment_dist(rng);
      params.adjustment.width = width_adjustment_dist(rng);
    }

    // Make a random filter.
    std::vector<size_t> filter_shape = {
        params.groups * params.group_output_channels,
        params.kernel.height,
        params.kernel.width,
        params.group_input_channels,
    };
    DatatypeGenerator<Filter> filter_gen =
        MakeDatatypeGenerator(Filter(), /*symmetric_range=*/true);
    Tensor<Filter> filter(filter_shape, XnnExtraBytes);
    filter.generate([&]() { return filter_gen(rng); });
    const size_t reduction_size =
        filter.extent(1) * filter.extent(2) * filter.extent(3);

    // (Maybe) make a random bias.
    Tensor<Bias> bias;
    if (rng() & 1) {
      std::vector<size_t> bias_shape = {params.groups *
                                        params.group_output_channels};
      DatatypeGenerator<Bias> bias_gen = MakeDatatypeGenerator(
          Bias(), -max_abs_bias<Bias>(), max_abs_bias<Bias>());
      bias = Tensor<Bias>(bias_shape, XnnExtraBytes);
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
    xnn_quantization_params bias_quantization =
        xnn_datatype_is_quantized(datatype_of<Bias>())
            ? xnn_quantization_params{0, input_quantization.scale *
                                             filter_quantization.scale}
            : xnn_quantization_params{0, 1.0f};
    xnn_quantization_params output_quantization =
        CalculateGEMMQuantizationParams<Data, Filter, Data>(
            reduction_size, input_quantization, filter_quantization,
            bias_quantization);

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
        .AddDeconvolution2D(params, conv_input_id, filter_id, bias_id,
                            output_id);
    xnn_status status = subgraph.CreateRuntime();
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    // Run the subgraph twice, with a different input/output shape each time.
    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = random_shape(rng, 4);
      std::vector<size_t> output_shape = {
          input_shape[0],
          kh.input_extent(input_shape[1], false, params.adjustment.height),
          kw.input_extent(input_shape[2], false, params.adjustment.width),
          params.groups * params.group_output_channels,
      };
      input_shape[3] = params.groups * params.group_input_channels;

      Tensor<Data> input(input_shape, XnnExtraBytes);
      input.generate([&]() { return data_gen(rng); });
      if (convert_to != xnn_datatype_invalid) {
        assert(convert_to == xnn_datatype_qdint8);
        // If we are dynamically quantizing, preprocess the data to have zero
        // error when it will be quantized, which makes testing the behavior
        // much easier.
        for (size_t b = 0; b < input.extent(0); ++b) {
          FakeDynamicQuantize(input.slice(0, b), convert_to);
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
      Tensor<float> expected = ReferenceImpl(
          input, filter, bias, input_quantization,
          filter_quantization.zero_point, filter_scale, bias_zero_point,
          bias_scale, params.groups, params.group_input_channels,
          params.group_output_channels, kh, kw, params.adjustment);
      for (float& i : expected) {
        i = std::max(i, params.output_min);
        i = std::min(i, params.output_max);
      }

      ASSERT_EQ(expected.extents(), output.extents());
      if (xnn_datatype_is_quantized(xnn_datatype_of<Data>())) {
        for (const auto& i : EnumerateIndices(output.extents())) {
          ASSERT_NEAR(output(i),
                      quantize<Data>(expected(i), output_quantization), 1)
              << "input_shape=" << index_to_string(input_shape)
              << ", output_shape=" << index_to_string(output_shape)
              << ", filter_shape=" << index_to_string(filter_shape)
              << ", kh=" << kh << ", kw=" << kw;
        }
      } else {
        const float max_a = MaxDatatype(Data()) * input_quantization.scale;
        const float max_b = MaxDatatype(Filter()) * filter_quantization.scale;
        const float tolerance = xnnpack::epsilon(xnn_datatype_of<Data>()) *
                                reduction_size * max_a * max_b * 4.0f;
        for (const auto& i : EnumerateIndices(output.extents())) {
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

TEST(Deconvolution2DQC8, test) { TestImpl<qint8, qcint8, qcint32>(); }
TEST(Deconvolution2DQU8, test) { TestImpl<quint8, quint8, qint32>(); }
TEST(Deconvolution2DQS8, test) { TestImpl<qint8, qint8, qint32>(); }
TEST(Deconvolution2DF16, test) { TestImpl<xnn_float16, float, float>(); }
TEST(Deconvolution2DF32, test) { TestImpl<float, float, float>(); }
TEST(Deconvolution2DQD8F32QC8W, test) {
  TestImpl<float, qcint8, float>(/*convert_to=*/xnn_datatype_qdint8);
}

TEST(Deconvolution2D, reshape_rejects_input_channel_mismatch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  DeconvolutionParams params;
  params.padding = {0, 0, 0, 0};
  params.adjustment = {0, 0};
  params.kernel = {3, 3};
  params.upsampling = {1, 1};
  params.dilation = {1, 1};
  params.groups = 1;
  params.group_input_channels = 4;
  params.group_output_channels = 8;

  const uint32_t input_id = 0;
  const uint32_t filter_id = 1;
  const uint32_t bias_id = 2;
  const uint32_t output_id = 3;

  Tensor<float> filter(std::vector<size_t>{8, 3, 3, 4});
  filter.fill(0.0f);
  Tensor<float> bias(std::vector<size_t>{8});
  bias.fill(0.0f);

  SubgraphTester subgraph(4);
  subgraph.AddInputTensor(4, xnn_datatype_fp32, input_id)
      .AddStaticTensor(filter.extents(), filter_id, filter.base())
      .AddStaticTensor(bias.extents(), bias_id, bias.base())
      .AddOutputTensor(4, xnn_datatype_fp32, output_id)
      .AddDeconvolution2D(params, input_id, filter_id, bias_id, output_id);
  if (subgraph.CreateRuntime() == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  // Create-time input channels are groups*group_input_channels == 4; a
  // single-channel runtime input must be rejected before the indirection buffer
  // strides past the end of the smaller input.
  std::vector<float> input(1 * 8 * 8 * 1);
  subgraph.ReshapeExternalTensor({1, 8, 8, 1}, input.data(), input_id)
      .ReshapeRuntime();
  EXPECT_EQ(subgraph.Status(), xnn_status_invalid_parameter);
}

#ifndef XNNPACK_USE_YNNPACK
TEST(Deconvolution2D, ReshapeOverflowInputElements) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[4] = {1, 2, 2, 1};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t filter_id = XNN_INVALID_VALUE_ID;
  const size_t filter_dims[4] = {1, 1, 1, 1};
  const float filter_data[1] = {1.0f};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, filter_dims, filter_data,
          /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  const size_t bias_dims[1] = {1};
  const float bias_data[1] = {0.0f};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 1, bias_dims, bias_data,
          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const size_t output_dims[4] = {1, 2, 2, 1};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, output_dims, nullptr,
          /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_deconvolution_2d(
          subgraph, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -INFINITY,
          INFINITY, input_id, filter_id, bias_id, output_id, 0));

  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  runtime->values[input_id].shape.num_dims = 4;
  runtime->values[input_id].shape.dim[0] = SIZE_MAX;
  runtime->values[input_id].shape.dim[1] = 2;
  runtime->values[input_id].shape.dim[2] = 2;
  runtime->values[input_id].shape.dim[3] = 1;

  const enum xnn_status reshape_status = xnn_reshape_runtime(runtime);
  EXPECT_EQ(xnn_status_invalid_parameter, reshape_status);
}

TEST(Deconvolution2D, ReshapeOverflowOutputSize) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(4, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[4] = {1, 2, 2, 1};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t filter_id = XNN_INVALID_VALUE_ID;
  const size_t filter_dims[4] = {1, 1, 1, 1};
  const float filter_data[1] = {1.0f};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, filter_dims, filter_data,
          /*external_id=*/1, /*flags=*/0, &filter_id));

  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  const size_t bias_dims[1] = {1};
  const float bias_data[1] = {0.0f};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 1, bias_dims, bias_data,
          /*external_id=*/2, /*flags=*/0, &bias_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const size_t output_dims[4] = {1, 2, 2, 1};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 4, output_dims, nullptr,
          /*external_id=*/3, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_deconvolution_2d(
          subgraph, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -INFINITY,
          INFINITY, input_id, filter_id, bias_id, output_id, 0));

  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  runtime->values[input_id].shape.num_dims = 4;
  runtime->values[input_id].shape.dim[0] = (SIZE_MAX / 4) + 1;
  runtime->values[input_id].shape.dim[1] = 1;
  runtime->values[input_id].shape.dim[2] = 1;
  runtime->values[input_id].shape.dim[3] = 1;

  const enum xnn_status reshape_status = xnn_reshape_runtime(runtime);
  EXPECT_TRUE(reshape_status == xnn_status_out_of_memory ||
              reshape_status == xnn_status_invalid_parameter);
}

namespace {

template <typename Data = float>
void TestTransientIndirectionBuffer(uint32_t padding, uint32_t kernel_size,
                                    uint32_t stride, uint32_t groups) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  constexpr size_t batch_size = 2, input_height = 6, input_width = 5;
  constexpr size_t group_input_channels = 8, group_output_channels = 12;
  const size_t input_channels = groups * group_input_channels;
  const size_t output_channels = groups * group_output_channels;

  DeconvolutionParams params;
  params.padding = {padding, padding, padding, padding};
  params.adjustment = {0, 0};
  params.kernel = {kernel_size, kernel_size};
  params.upsampling = {stride, stride};
  params.dilation = {1, 1};
  params.groups = groups;
  params.group_input_channels = group_input_channels;
  params.group_output_channels = group_output_channels;

  ReplicableRandomDevice rng;
  auto f32_gen = MakeDatatypeGenerator(float());
  auto data_gen = MakeDatatypeGenerator(Data());
  Tensor<float> filter({output_channels, kernel_size, kernel_size,
                        group_input_channels},
                       XnnExtraBytes);
  Tensor<float> bias({output_channels}, XnnExtraBytes);
  filter.generate([&]() { return f32_gen(rng); });
  bias.generate([&]() { return f32_gen(rng); });

  auto build = [&](SubgraphTester& tester) {
    tester.AddInputTensor(4, xnn_datatype_of<Data>(), 0)
        .AddStaticTensor(filter.extents(), 1, filter.base())
        .AddStaticTensor(bias.extents(), 2, bias.base())
        .AddOutputTensor(4, xnn_datatype_of<Data>(), 3)
        .AddDeconvolution2D(params, 0, 1, 2, 3);
  };

  SubgraphTester ref_tester(4);
  build(ref_tester);
  if (ref_tester.CreateRuntime(/*threadpool=*/nullptr, /*flags=*/0) ==
      xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }

  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool(
      pthreadpool_create(4), pthreadpool_destroy);
  xnn_workspace_t workspace = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_workspace(&workspace));
  std::unique_ptr<xnn_workspace, decltype(&xnn_release_workspace)>
      auto_workspace(workspace, xnn_release_workspace);

  SubgraphTester tester(4);
  build(tester);
  ASSERT_EQ(xnn_status_success,
            tester.CreateRuntime(/*weights_cache=*/nullptr, workspace,
                                 auto_threadpool.get(),
                                 XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER));

  size_t prev_workspace_size = 0;
  for (size_t grow : {size_t{0}, size_t{2}}) {
    const std::vector<size_t> in_shape = {batch_size, input_height + grow,
                                          input_width + grow, input_channels};
    Tensor<Data> in1(in_shape, XnnExtraBytes), in2(in_shape, XnnExtraBytes);
    in1.generate([&]() { return data_gen(rng); });
    in2.generate([&]() { return data_gen(rng); });

    ref_tester.ReshapeExternalTensor(in_shape, in1.base(), 0).ReshapeRuntime();
    tester.ReshapeExternalTensor(in_shape, in1.base(), 0).ReshapeRuntime();
    EXPECT_GT(workspace->size, prev_workspace_size);
    prev_workspace_size = workspace->size;

    const std::vector<size_t> out_shape = tester.GetExternalTensorShape(3);
    for (Tensor<Data>* in : {&in1, &in2}) {
      Tensor<Data> ref_out(out_shape), out(out_shape);
      ref_tester.SetupExternalTensor(in->base(), 0)
          .SetupExternalTensor(ref_out.base(), 3)
          .SetupRuntime()
          .InvokeRuntime();
      tester.SetupExternalTensor(in->base(), 0)
          .SetupExternalTensor(out.base(), 3)
          .SetupRuntime()
          .InvokeRuntime();
      for (size_t i = 0; i < out.size(); ++i) {
        ASSERT_NEAR(static_cast<float>(out.base()[i]),
                    static_cast<float>(ref_out.base()[i]), 1.0e-4f);
      }
    }
  }
}

}  // namespace

TEST(Deconvolution2D, TransientIndirectionBufferIgemm) {
  TestTransientIndirectionBuffer(/*padding=*/1, /*kernel_size=*/3, /*stride=*/1,
                                 /*groups=*/2);
}

TEST(Deconvolution2D, TransientIndirectionBufferSubconv2D) {
  TestTransientIndirectionBuffer(/*padding=*/1, /*kernel_size=*/4, /*stride=*/2,
                                 /*groups=*/2);
}

TEST(Deconvolution2D, TransientIndirectionBufferIgemmF16Fp32Weights) {
  TestTransientIndirectionBuffer<xnn_float16>(/*padding=*/1, /*kernel_size=*/3,
                                              /*stride=*/1, /*groups=*/2);
}

TEST(Deconvolution2D, TransientIndirectionBufferSubconv2DF16Fp32Weights) {
  TestTransientIndirectionBuffer<xnn_float16>(/*padding=*/1, /*kernel_size=*/4,
                                              /*stride=*/2, /*groups=*/2);
}

// Checks the status returned by `xnn_define_deconvolution_2d` for a 3x3
// deconvolution with 4 input channels and 8 output channels.
class Deconvolution2DDefine : public ::testing::Test {
 protected:
  uint32_t DenseTensor(xnn_datatype datatype, std::vector<size_t> dims,
                       bool is_static = false) {
    uint32_t id = XNN_INVALID_VALUE_ID;
    if (is_static) {
      tester_.AddInternalStaticTensor(dims, datatype, &id, data_.data());
    } else {
      tester_.AddInternalDynamicTensor(dims, datatype, &id);
    }
    return id;
  }

  uint32_t QuantizedTensor(xnn_datatype datatype, int32_t zero_point,
                           std::vector<size_t> dims, bool is_static = false) {
    uint32_t id = XNN_INVALID_VALUE_ID;
    if (is_static) {
      tester_.AddInternalStaticTensor(dims, datatype, {zero_point, 1.0f}, &id,
                                      data_.data());
    } else {
      tester_.AddInternalDynamicTensor(dims, datatype, {zero_point, 1.0f}, &id);
    }
    return id;
  }

  uint32_t ChannelwiseTensor(xnn_datatype datatype, std::vector<size_t> dims,
                             size_t channel_dim,
                             const float* zero_points = nullptr) {
    uint32_t id = XNN_INVALID_VALUE_ID;
    tester_.AddInternalChannelwiseQuantizedTensor(
        dims, channel_dim, datatype, /*zero_point=*/0, scales_.data(),
        zero_points, &id, data_.data());
    return id;
  }

  xnn_status Define(uint32_t input_id, uint32_t filter_id, uint32_t bias_id,
                    uint32_t output_id) {
    return xnn_define_deconvolution_2d(
        tester_.Subgraph(), /*padding_top=*/0, /*padding_right=*/0,
        /*padding_bottom=*/0, /*padding_left=*/0, /*adjustment_height=*/0,
        /*adjustment_width=*/0, /*kernel_height=*/3, /*kernel_width=*/3,
        /*upsampling_height=*/1, /*upsampling_width=*/1,
        /*dilation_height=*/1, /*dilation_width=*/1, /*groups=*/1,
        /*group_input_channels=*/4, /*group_output_channels=*/8, -INFINITY,
        INFINITY, input_id, filter_id, bias_id, output_id, /*flags=*/0);
  }

  const std::vector<size_t> input_dims_ = {1, 5, 5, 4};
  const std::vector<size_t> filter_dims_ = {8, 3, 3, 4};
  const std::vector<size_t> bias_dims_ = {8};
  const std::vector<size_t> output_dims_ = {1, 7, 7, 8};

 private:
  SubgraphTester tester_{/*external_value_ids=*/0};
  std::vector<float> data_ = std::vector<float>(8 * 3 * 3 * 4, 0.0f);
  std::vector<float> scales_ = std::vector<float>(8, 1.0f);
};

TEST_F(Deconvolution2DDefine, UnsupportedInputDatatype) {
  EXPECT_EQ(
      Define(DenseTensor(xnn_datatype_int32, input_dims_),
             DenseTensor(xnn_datatype_fp32, filter_dims_, /*is_static=*/true),
             DenseTensor(xnn_datatype_fp32, bias_dims_, /*is_static=*/true),
             DenseTensor(xnn_datatype_fp32, output_dims_)),
      xnn_status_unsupported_parameter);
}

TEST_F(Deconvolution2DDefine, UnsupportedFilterDatatype) {
  EXPECT_EQ(
      Define(DenseTensor(xnn_datatype_fp32, input_dims_),
             DenseTensor(xnn_datatype_int32, filter_dims_, /*is_static=*/true),
             DenseTensor(xnn_datatype_fp32, bias_dims_, /*is_static=*/true),
             DenseTensor(xnn_datatype_fp32, output_dims_)),
      xnn_status_unsupported_parameter);
}

TEST_F(Deconvolution2DDefine, UnsupportedBiasDatatype) {
  EXPECT_EQ(
      Define(DenseTensor(xnn_datatype_fp32, input_dims_),
             DenseTensor(xnn_datatype_fp32, filter_dims_, /*is_static=*/true),
             DenseTensor(xnn_datatype_int32, bias_dims_, /*is_static=*/true),
             DenseTensor(xnn_datatype_fp32, output_dims_)),
      xnn_status_unsupported_parameter);
}

TEST_F(Deconvolution2DDefine, UnsupportedOutputDatatype) {
  EXPECT_EQ(
      Define(DenseTensor(xnn_datatype_fp32, input_dims_),
             DenseTensor(xnn_datatype_fp32, filter_dims_, /*is_static=*/true),
             DenseTensor(xnn_datatype_fp32, bias_dims_, /*is_static=*/true),
             DenseTensor(xnn_datatype_int32, output_dims_)),
      xnn_status_unsupported_parameter);
}

TEST_F(Deconvolution2DDefine, UnsupportedDynamicFilter) {
  EXPECT_EQ(
      Define(DenseTensor(xnn_datatype_fp32, input_dims_),
             DenseTensor(xnn_datatype_fp32, filter_dims_),
             DenseTensor(xnn_datatype_fp32, bias_dims_, /*is_static=*/true),
             DenseTensor(xnn_datatype_fp32, output_dims_)),
      xnn_status_unsupported_parameter);
}

TEST_F(Deconvolution2DDefine, UnsupportedDynamicBias) {
  EXPECT_EQ(
      Define(DenseTensor(xnn_datatype_fp32, input_dims_),
             DenseTensor(xnn_datatype_fp32, filter_dims_, /*is_static=*/true),
             DenseTensor(xnn_datatype_fp32, bias_dims_),
             DenseTensor(xnn_datatype_fp32, output_dims_)),
      xnn_status_unsupported_parameter);
}

TEST_F(Deconvolution2DDefine, UnsupportedDatatypeCombinationWithBias) {
  EXPECT_EQ(
      Define(DenseTensor(xnn_datatype_fp32, input_dims_),
             DenseTensor(xnn_datatype_fp32, filter_dims_, /*is_static=*/true),
             QuantizedTensor(xnn_datatype_qint32, 0, bias_dims_,
                             /*is_static=*/true),
             DenseTensor(xnn_datatype_fp32, output_dims_)),
      xnn_status_unsupported_parameter);
}

TEST_F(Deconvolution2DDefine, UnsupportedDatatypeCombinationWithoutBias) {
  EXPECT_EQ(Define(DenseTensor(xnn_datatype_fp32, input_dims_),
                   QuantizedTensor(xnn_datatype_qint8, 0, filter_dims_,
                                   /*is_static=*/true),
                   XNN_INVALID_VALUE_ID,
                   DenseTensor(xnn_datatype_fp32, output_dims_)),
            xnn_status_unsupported_parameter);
}

TEST_F(Deconvolution2DDefine, UnsupportedFilterChannelDimension) {
  EXPECT_EQ(Define(QuantizedTensor(xnn_datatype_qint8, 0, input_dims_),
                   ChannelwiseTensor(xnn_datatype_qcint8, filter_dims_,
                                     /*channel_dim=*/3),
                   XNN_INVALID_VALUE_ID,
                   QuantizedTensor(xnn_datatype_qint8, 0, output_dims_)),
            xnn_status_unsupported_parameter);
}

TEST_F(Deconvolution2DDefine, InvalidFilterId) {
  EXPECT_EQ(Define(DenseTensor(xnn_datatype_fp32, input_dims_),
                   /*filter_id=*/1000,
                   DenseTensor(xnn_datatype_fp32, bias_dims_,
                               /*is_static=*/true),
                   DenseTensor(xnn_datatype_fp32, output_dims_)),
            xnn_status_invalid_parameter);
}

TEST_F(Deconvolution2DDefine, ValidQC8) {
  EXPECT_EQ(Define(QuantizedTensor(xnn_datatype_qint8, 0, input_dims_),
                   ChannelwiseTensor(xnn_datatype_qcint8, filter_dims_,
                                     /*channel_dim=*/0),
                   XNN_INVALID_VALUE_ID,
                   QuantizedTensor(xnn_datatype_qint8, 0, output_dims_)),
            xnn_status_success);
}
#endif  // XNNPACK_USE_YNNPACK

}  // namespace xnnpack
