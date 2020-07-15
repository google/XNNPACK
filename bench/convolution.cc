// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include <xnnpack.h>

#ifdef BENCHMARK_ARM_COMPUTE_LIBRARY
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#include "arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h"
#include "arm_compute/runtime/NEON/functions/NEConvolutionLayer.h"
#endif  // BENCHMARK_ARM_COMPUTE_LIBRARY
#include <benchmark/benchmark.h>
#include <fp16.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE
#include "bench/utils.h"


void xnnpack_convolution_qu8(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding_height = state.range(5);
  const size_t padding_width = state.range(6);
  const size_t subsampling = state.range(7);
  const size_t dilation = state.range(8);
  const size_t groups = state.range(9);
  const size_t group_input_channels = state.range(10);
  const size_t group_output_channels = state.range(11);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
  auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

  const size_t output_pixel_stride = groups * group_output_channels;
  const size_t input_pixel_stride = groups * group_input_channels;
  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t padding_right = padding_width - padding_left;
  const size_t padding_bottom = padding_height - padding_top;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;

  std::vector<uint8_t> input(batch_size * input_height * input_width * input_pixel_stride);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> kernel(groups * group_output_channels * kernel_height * kernel_width * group_input_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
  std::vector<int32_t> bias(groups * group_output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(s32rng));
  const size_t output_elements = batch_size * output_height * output_width * output_pixel_stride;

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(uint8_t) * kernel.size() + sizeof(int32_t) * bias.size() + sizeof(uint8_t) * output_elements);
  std::vector<uint8_t> output(output_elements * num_buffers);

  std::vector<xnn_operator_t> convolution_operators(num_buffers);
  for (xnn_operator_t& convolution_op : convolution_operators) {
    status = xnn_create_convolution2d_nhwc_qu8(
      padding_top, padding_right, padding_bottom, padding_left,
      kernel_height, kernel_width,
      subsampling, subsampling,
      dilation, dilation,
      groups, group_input_channels, group_output_channels,
      input_pixel_stride, output_pixel_stride,
      127, 0.5f,
      127, 0.5f,
      kernel.data(), bias.data(),
      127, 0.5f, 0, 255,
      0 /* flags */, &convolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create QINT8 Convolution operator");
      return;
    }
  }

  for (size_t i = 0; i < convolution_operators.size(); i++) {
    status = xnn_setup_convolution2d_nhwc_qu8(
      convolution_operators[i],
      batch_size, input_height, input_width,
      input.data(), output.data() + i * output_elements,
      nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup QINT8 Convolution operator");
      return;
    }
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(input.data(), input.size() * sizeof(uint8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    status = xnn_run_operator(convolution_operators[buffer_index],
      nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run QINT8 Convolution operator");
      return;
    }
  }

  for (xnn_operator_t& convolution_op : convolution_operators) {
    status = xnn_delete_operator(convolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to delete QINT8 Convolution operator");
      return;
    }
    convolution_op = nullptr;
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * output_height * output_width *
      groups * group_input_channels * group_output_channels *
      kernel_height * kernel_width,
    benchmark::Counter::kIsRate);
}

void xnnpack_convolution_f16(benchmark::State& state, const char* net) {
  if (!benchmark::utils::CheckNEONFP16ARITH(state)) {
    return;
  }
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding_height = state.range(5);
  const size_t padding_width = state.range(6);
  const size_t subsampling = state.range(7);
  const size_t dilation = state.range(8);
  const size_t groups = state.range(9);
  const size_t group_input_channels = state.range(10);
  const size_t group_output_channels = state.range(11);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.1f, 1.0f), rng);
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  const size_t output_pixel_stride = groups * group_output_channels;
  const size_t input_pixel_stride = groups * group_input_channels;
  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t padding_right = padding_width - padding_left;
  const size_t padding_bottom = padding_height - padding_top;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;

  std::vector<uint16_t> input(batch_size * input_height * input_width * input_pixel_stride + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::generate(input.begin(), input.end(), std::ref(f16rng));
  std::vector<uint16_t> kernel(groups * group_output_channels * kernel_height * kernel_width * group_input_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f16rng));
  std::vector<uint16_t> bias(groups * group_output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(f16rng));
  const size_t output_elements = batch_size * output_height * output_width * output_pixel_stride;

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(uint16_t) * (kernel.size() + bias.size() + output_elements));
  std::vector<uint16_t> output(output_elements * num_buffers);

  std::vector<xnn_operator_t> convolution_operators(num_buffers);
  for (xnn_operator_t& convolution_op : convolution_operators) {
    status = xnn_create_convolution2d_nhwc_f16(
      padding_top, padding_right, padding_bottom, padding_left,
      kernel_height, kernel_width,
      subsampling, subsampling,
      dilation, dilation,
      groups, group_input_channels, group_output_channels,
      input_pixel_stride, output_pixel_stride,
      kernel.data(), bias.data(),
      -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
      0 /* flags */, &convolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create FP16 Convolution operator");
      return;
    }
  }

  for (size_t i = 0; i < convolution_operators.size(); i++) {
    status = xnn_setup_convolution2d_nhwc_f16(
      convolution_operators[i],
      batch_size, input_height, input_width,
      input.data(), output.data() + i * output_elements,
      nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup FP16 Convolution operator");
      return;
    }
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(input.data(), input.size() * sizeof(uint16_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    status = xnn_run_operator(convolution_operators[buffer_index], nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run FP16 Convolution operator");
      return;
    }
  }

  for (xnn_operator_t& convolution_op : convolution_operators) {
    status = xnn_delete_operator(convolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to delete FP16 Convolution operator");
      return;
    }
    convolution_op = nullptr;
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * output_height * output_width *
      groups * group_input_channels * group_output_channels *
      kernel_height * kernel_width,
    benchmark::Counter::kIsRate);
}

void xnnpack_convolution_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding_height = state.range(5);
  const size_t padding_width = state.range(6);
  const size_t subsampling = state.range(7);
  const size_t dilation = state.range(8);
  const size_t groups = state.range(9);
  const size_t group_input_channels = state.range(10);
  const size_t group_output_channels = state.range(11);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

  const size_t output_pixel_stride = groups * group_output_channels;
  const size_t input_pixel_stride = groups * group_input_channels;
  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t padding_right = padding_width - padding_left;
  const size_t padding_bottom = padding_height - padding_top;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;

  std::vector<float> input(batch_size * input_height * input_width * input_pixel_stride + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> kernel(groups * group_output_channels * kernel_height * kernel_width * group_input_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
  std::vector<float> bias(groups * group_output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));
  const size_t output_elements = batch_size * output_height * output_width * output_pixel_stride;

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (kernel.size() + bias.size() + output_elements));
  std::vector<float> output(output_elements * num_buffers);

  std::vector<xnn_operator_t> convolution_operators(num_buffers);
  for (xnn_operator_t& convolution_op : convolution_operators) {
    status = xnn_create_convolution2d_nhwc_f32(
      padding_top, padding_right, padding_bottom, padding_left,
      kernel_height, kernel_width,
      subsampling, subsampling,
      dilation, dilation,
      groups, group_input_channels, group_output_channels,
      input_pixel_stride, output_pixel_stride,
      kernel.data(), bias.data(),
      -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
      0 /* flags */, &convolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create FP32 Convolution operator");
      return;
    }
  }

  for (size_t i = 0; i < convolution_operators.size(); i++) {
    status = xnn_setup_convolution2d_nhwc_f32(
      convolution_operators[i],
      batch_size, input_height, input_width,
      input.data(), output.data() + i * output_elements,
      nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup FP32 Convolution operator");
      return;
    }
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(input.data(), input.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    status = xnn_run_operator(convolution_operators[buffer_index], nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run FP32 Convolution operator");
      return;
    }
  }

  for (xnn_operator_t& convolution_op : convolution_operators) {
    status = xnn_delete_operator(convolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to delete FP32 Convolution operator");
      return;
    }
    convolution_op = nullptr;
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * output_height * output_width *
      groups * group_input_channels * group_output_channels *
      kernel_height * kernel_width,
    benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_TENSORFLOW_LITE
void tflite_convolution_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding_height = state.range(5);
  const size_t padding_width = state.range(6);
  const size_t subsampling = state.range(7);
  const size_t dilation = state.range(8);
  const size_t groups = state.range(9);
  const size_t group_input_channels = state.range(10);
  const size_t group_output_channels = state.range(11);

  bool is_depthwise = false;
  if (groups != 1) {
    if (group_input_channels == 1) {
      is_depthwise = true;
    } else {
      state.SkipWithError("grouped convolution is not supported");
      return;
    }
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;

  tflite::Padding padding = tflite::Padding_VALID;
  if (padding_width == (effective_kernel_width - 1) && padding_height == (effective_kernel_height - 1)) {
    padding = tflite::Padding_SAME;
  } else if (padding_width == 0 && padding_height == 0) {
    padding = tflite::Padding_VALID;
  } else {
    state.SkipWithError("unsupported padding");
    return;
  }

  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;

  std::vector<float> kernel(groups * group_output_channels * kernel_height * kernel_width * group_input_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
  std::vector<float> bias(groups * group_output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(
        builder,
        is_depthwise ? tflite::BuiltinOperator_DEPTHWISE_CONV_2D : tflite::BuiltinOperator_CONV_2D,
        0);

  flatbuffers::Offset<tflite::Conv2DOptions> conv2d_options = CreateConv2DOptions(
      builder,
      padding,
      static_cast<int32_t>(subsampling), static_cast<int32_t>(subsampling),
      tflite::ActivationFunctionType_NONE,
      static_cast<int32_t>(dilation), static_cast<int32_t>(dilation));

  flatbuffers::Offset<tflite::DepthwiseConv2DOptions> dwconv2d_options = CreateDepthwiseConv2DOptions(
      builder,
      padding,
      static_cast<int32_t>(subsampling), static_cast<int32_t>(subsampling),
      static_cast<int32_t>(group_output_channels),
      tflite::ActivationFunctionType_NONE,
      static_cast<int32_t>(dilation), static_cast<int32_t>(dilation));

  flatbuffers::Offset<tflite::Buffer> buffers[3] = {
    tflite::CreateBuffer(builder, builder.CreateVector({})),
    tflite::CreateBuffer(builder, builder.CreateVector(
      reinterpret_cast<const uint8_t*>(kernel.data()),
      sizeof(float) * kernel.size())),
    tflite::CreateBuffer(builder, builder.CreateVector(
      reinterpret_cast<const uint8_t*>(bias.data()),
      sizeof(float) * bias.size())),
  };

  const int32_t input_shape[4] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(input_height),
    static_cast<int32_t>(input_width),
    static_cast<int32_t>(groups * group_input_channels)
  };
  const int32_t output_shape[4] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(output_height),
    static_cast<int32_t>(output_width),
    static_cast<int32_t>(groups * group_output_channels)
  };
  const int32_t filter_shape[4] = {
    static_cast<int32_t>(group_output_channels),
    static_cast<int32_t>(kernel_height),
    static_cast<int32_t>(kernel_width),
    static_cast<int32_t>(groups * group_input_channels)
  };
  const int32_t bias_shape[1] = {
    static_cast<int32_t>(groups * group_output_channels)
  };

  flatbuffers::Offset<tflite::Tensor> tensors[4] = {
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(input_shape, 4),
                         tflite::TensorType_FLOAT32,
                         0 /* buffer id */,
                         builder.CreateString("input")),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(filter_shape, 4),
                         tflite::TensorType_FLOAT32,
                         1 /* buffer id */,
                         builder.CreateString("filter")),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(bias_shape, 1),
                         tflite::TensorType_FLOAT32,
                         2 /* buffer id */,
                         builder.CreateString("bias")),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(output_shape, 4),
                         tflite::TensorType_FLOAT32,
                         0 /* buffer id */,
                         builder.CreateString("output")),
  };

  const int32_t op_inputs[3] = { 0, 1, 2 };
  const int32_t op_outputs[1] = { 3 };
  flatbuffers::Offset<tflite::Operator> op = CreateOperator(
      builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs, 3),
      builder.CreateVector<int32_t>(op_outputs, 1),
      is_depthwise ? tflite::BuiltinOptions_DepthwiseConv2DOptions : tflite::BuiltinOptions_Conv2DOptions,
      is_depthwise ? dwconv2d_options.Union() : conv2d_options.Union(),
      /*custom_options */ 0,
      tflite::CustomOptionsFormat_FLEXBUFFERS);

  const int32_t graph_inputs[1] = { 0 };
  const int32_t graph_outputs[1] = { 3 };
  flatbuffers::Offset<tflite::SubGraph> subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(tensors, 4),
      builder.CreateVector<int32_t>(graph_inputs, 1),
      builder.CreateVector<int32_t>(graph_outputs, 1),
      builder.CreateVector(&op, 1),
      builder.CreateString("Conv2D subgraph"));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("Conv2D model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
      TFLITE_SCHEMA_VERSION,
      builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1),
      description,
      builder.CreateVector(buffers, 3));

  builder.Finish(model_buffer);

  const tflite::Model* model = tflite::GetModel(builder.GetBufferPointer());
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder interpreterBuilder(model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (interpreterBuilder(&interpreter) != kTfLiteOk) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  if (interpreter == nullptr) {
    state.SkipWithError("TFLite interpreter is null");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::generate(
    interpreter->typed_tensor<float>(0),
    interpreter->typed_tensor<float>(0) + batch_size * groups * group_input_channels * input_height * input_width,
    std::ref(f32rng));

  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::WipeCache();
    benchmark::utils::PrefetchToL1(
      interpreter->typed_tensor<float>(0),
      batch_size * groups * group_input_channels * input_height * input_width * sizeof(float));
    state.ResumeTiming();

    if (interpreter->Invoke() != kTfLiteOk) {
      state.SkipWithError("failed to invoke TFLite interpreter");
      return;
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * output_height * output_width *
      groups * group_input_channels * group_output_channels *
      kernel_height * kernel_width,
    benchmark::Counter::kIsRate);

  interpreter.reset();
}
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifdef BENCHMARK_ARM_COMPUTE_LIBRARY
static std::string compare_with_convolution_f32_reference_output(
    const benchmark::State& state, const float* input, size_t input_size,
    const float* kernel, size_t kernel_size, const float* bias, size_t bias_size,
    const float* output, size_t output_size)
{
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding_height = state.range(5);
  const size_t padding_width = state.range(6);
  const size_t subsampling = state.range(7);
  const size_t dilation = state.range(8);
  const size_t groups = state.range(9);
  const size_t group_input_channels = state.range(10);
  const size_t group_output_channels = state.range(11);

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;
  const size_t input_pixel_stride = groups * group_input_channels;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;

  assert(input_size == batch_size * input_height * input_width * groups * group_input_channels);

  assert(kernel_size == group_output_channels * kernel_height * kernel_width * groups * group_input_channels);

  assert(bias_size == groups * group_output_channels);

  assert(output_size == batch_size * output_height * output_width * groups * group_output_channels);

  std::vector<float> output_ref(output_size);
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oy = 0; oy < output_height; oy++) {
      for (size_t ox = 0; ox < output_width; ox++) {
        for (size_t g = 0; g < groups; g++) {
          for (size_t oc = 0; oc < group_output_channels; oc++) {
            output_ref[(((i * output_height + oy) * output_width + ox) * groups + g) * group_output_channels + oc] =
              bias[g * group_output_channels + oc];
          }
        }
      }
    }
  }
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t oy = 0; oy < output_height; oy++) {
      for (size_t ox = 0; ox < output_width; ox++) {
        for (size_t ky = 0; ky < kernel_height; ky++) {
          const size_t iy = oy * subsampling + ky * dilation - padding_top;
          if (iy < input_height) {
            for (size_t kx = 0; kx < kernel_width; kx++) {
              const size_t ix = ox * subsampling + kx * dilation - padding_left;
              if (ix < input_width) {
                for (size_t g = 0; g < groups; g++) {
                  for (size_t oc = 0; oc < group_output_channels; oc++) {
                    for (size_t ic = 0; ic < group_input_channels; ic++) {
                      output_ref[(((i * output_height + oy) * output_width + ox) * groups + g) * group_output_channels + oc] +=
                        input[((i * input_height + iy) * input_width + ix) * input_pixel_stride + g * group_input_channels + ic] *
                        kernel[(((oc * kernel_height + ky) * kernel_width + kx) * groups + g) * group_input_channels + ic];
                    }  // group_input_channels loop
                  }  // group_output_channels loop
                }  // groups loop
              }
            }  // kernel_width loop
          }
        }  // kernel_height loop
      }  // output_width loop
    }  // output_height loop
  }  // batch_size loop

  const float relative_error_tolerance = 1e-4;
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t y = 0; y < output_height; y++) {
      for (size_t x = 0; x < output_width; x++) {
        for (size_t g = 0; g < groups; g++) {
          for (size_t c = 0; c < group_output_channels; c++) {
            const size_t idx = (((i * output_height + y) * output_width + x) * groups + g) * group_output_channels + c;
            const float value_ref = output_ref[idx];
            const float value = output[idx];
            if (std::abs(value - value_ref) > std::max(std::abs(value_ref) * relative_error_tolerance, std::numeric_limits<float>::epsilon())) {
              std::ostringstream error_stream;
              error_stream << "(x, y) = (" << x << ", " << y << "), group = " << g
                       << ", channel = " << c << ", refValue = " << value_ref
                       << ", actualValue = " << value
                       << ", absDiff=" << std::abs(value - value_ref);
              return error_stream.str();
            }
          }
        }
      }
    }
  }
  return "";
}

void armcl_convolution_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding_height = state.range(5);
  const size_t padding_width = state.range(6);
  const size_t subsampling = state.range(7);
  const size_t dilation = state.range(8);
  const size_t groups = state.range(9);
  const size_t group_input_channels = state.range(10);
  const size_t group_output_channels = state.range(11);

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t padding_right = padding_width - padding_left;
  const size_t padding_bottom = padding_height - padding_top;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;

  arm_compute::PadStrideInfo pad_stride_info(
    subsampling /* stride height */,
    subsampling /* stride width */,
    padding_left, padding_right, padding_top, padding_bottom,
    arm_compute::DimensionRoundingType::FLOOR);
  arm_compute::Size2D dilation_info(dilation, dilation);
  // Note: activation is disabled by default.
  arm_compute::ActivationLayerInfo activation_info;

  // Note: no batch size and reverse order of dimensions, i.e. CWHN for NHWC.
  arm_compute::TensorShape input_shape(
    /* C */ groups * group_input_channels,
    /* W */ input_width,
    /* H */ input_height,
    /* N */ batch_size);
  arm_compute::TensorInfo input_info(
    input_shape,
    1 /* number of channels per element (!) */,
    arm_compute::DataType::F32);
  input_info.set_data_layout(arm_compute::DataLayout::NHWC);
  arm_compute::Tensor input_tensor;
  input_tensor.allocator()->init(input_info);
  input_tensor.allocator()->allocate();

  // Note: reverse order of dimensions, i.e. for IWHO for OHWI.
  arm_compute::TensorShape kernel_shape(
    /* I */ groups * group_input_channels,
    /* W */ kernel_width,
    /* H */ kernel_height,
    /* O */ group_output_channels);
  arm_compute::TensorInfo kernel_info(
    kernel_shape,
    1 /* number of channels per element (!) */,
    arm_compute::DataType::F32);
  kernel_info.set_data_layout(arm_compute::DataLayout::NHWC);
  arm_compute::Tensor kernelTensor;
  kernelTensor.allocator()->init(kernel_info);
  kernelTensor.allocator()->allocate();

  arm_compute::TensorShape bias_shape(groups * group_output_channels);
  arm_compute::TensorInfo bias_info(
    bias_shape,
    1 /* number of channels per element (!) */,
    arm_compute::DataType::F32);
  bias_info.set_data_layout(arm_compute::DataLayout::NHWC);
  arm_compute::Tensor bias_tensor;
  bias_tensor.allocator()->init(bias_info);
  bias_tensor.allocator()->allocate();

  // Note: no batch size and reverse order of dimensions, i.e. CWHN for NHWC.
  arm_compute::TensorShape output_shape(
    /* C */ groups * group_output_channels,
    /* W */ output_width,
    /* H */ output_height,
    /* N */ batch_size);
  arm_compute::TensorInfo output_info(
    output_shape,
    1 /* number of channels per element (!) */,
    arm_compute::DataType::F32);
  output_info.set_data_layout(arm_compute::DataLayout::NHWC);
  arm_compute::Tensor output_tensor;
  output_tensor.allocator()->init(output_info);
  output_tensor.allocator()->allocate();

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

  std::generate(
    reinterpret_cast<float*>(input_tensor.buffer()),
    reinterpret_cast<float*>(input_tensor.buffer()) + input_shape.total_size(),
    std::ref(f32rng));
  std::generate(
    reinterpret_cast<float*>(kernelTensor.buffer()),
    reinterpret_cast<float*>(kernelTensor.buffer()) + kernel_shape.total_size(),
    std::ref(f32rng));
  std::generate(
    reinterpret_cast<float*>(bias_tensor.buffer()),
    reinterpret_cast<float*>(bias_tensor.buffer()) + bias_shape.total_size(),
    std::ref(f32rng));
  std::generate(
    reinterpret_cast<float*>(output_tensor.buffer()),
    reinterpret_cast<float*>(output_tensor.buffer()) + output_shape.total_size(),
    std::ref(f32rng));

  bool is_depthwise = false;
  if (groups != 1) {
    // NEConvolutionLayer uses NEGEMMConvolutionLayer by default, which doesn't support grouped convolution.
    // However, depthwise convolution is supported via NEDepthwiseConvolutionLayer.
    if (group_input_channels == 1) {
      is_depthwise = true;
    } else {
      state.SkipWithError("grouped convolution is not supported");
      return;
    }
  }

  std::shared_ptr<arm_compute::IFunction> layer;
  if (is_depthwise) {
    if (dilation != 1) {
      state.SkipWithError("dilated depthwise convolution is not supported");
      return;
    }

    // Avoid NEDepthwiseConvolutionLayer3x3 when stride isn't 2 in order to pass the output verification.
    // TODO(b/130206370) This looks like a bug and needs further investigation.
    if (kernel_height == 3 && kernel_width == 3 && subsampling == 2) {
      auto* depthwise_3x3_convolution_layer = new arm_compute::NEDepthwiseConvolutionLayer3x3();
      layer.reset(depthwise_3x3_convolution_layer);
      depthwise_3x3_convolution_layer->configure(
        &input_tensor, &kernelTensor, &bias_tensor, &output_tensor,
        pad_stride_info, group_output_channels, activation_info);

      if (!depthwise_3x3_convolution_layer->validate(
        &input_info, &kernel_info, &bias_info, &output_info,
        pad_stride_info, group_output_channels, activation_info))
      {
        state.SkipWithError("validation failed");
        return;
      }
    } else {
      auto* depthwise_convolution_layer = new arm_compute::NEDepthwiseConvolutionLayer();
      layer.reset(depthwise_convolution_layer);
      depthwise_convolution_layer->configure(
        &input_tensor, &kernelTensor, &bias_tensor, &output_tensor,
        pad_stride_info, group_output_channels, activation_info);

      if (!depthwise_convolution_layer->validate(
        &input_info, &kernel_info, &bias_info, &output_info,
        pad_stride_info, group_output_channels, activation_info))
      {
        state.SkipWithError("validation failed");
        return;
      }
    }
  } else {
    auto* convolution_layer = new arm_compute::NEConvolutionLayer();
    layer.reset(convolution_layer);
    convolution_layer->configure(
      &input_tensor, &kernelTensor, &bias_tensor, &output_tensor,
      pad_stride_info, arm_compute::WeightsInfo(), dilation_info, activation_info,
      true /* enable fast math */, groups);

    if (!convolution_layer->validate(
      &input_info, &kernel_info, &bias_info, &output_info,
      pad_stride_info, arm_compute::WeightsInfo(), dilation_info, activation_info,
      true /* enable fast math */, groups))
    {
      state.SkipWithError("validation failed");
      return;
    }
  }

  // Dry run to let ACL do one-time initializations.
  arm_compute::CPPScheduler::get().set_num_threads(1);
  layer->run();

  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::WipeCache();
    benchmark::utils::PrefetchToL1(
      input_tensor.buffer(),
      batch_size * groups * group_input_channels * input_height * input_width * sizeof(float));
    state.ResumeTiming();

    layer->run();
  }

  // Validate outputs.
  const std::string error_string = compare_with_convolution_f32_reference_output(
      state, reinterpret_cast<const float*>(input_tensor.buffer()),
      input_shape.total_size(),
      reinterpret_cast<const float*>(kernelTensor.buffer()),
      kernel_shape.total_size(),
      reinterpret_cast<const float*>(bias_tensor.buffer()),
      bias_shape.total_size(),
      reinterpret_cast<const float*>(output_tensor.buffer()),
      output_shape.total_size());

  if (!error_string.empty()) {
    state.SkipWithError(("validation failed: " + error_string).c_str());
    return;
  }

  input_tensor.allocator()->free();
  kernelTensor.allocator()->free();
  bias_tensor.allocator()->free();
  output_tensor.allocator()->free();

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * output_height * output_width *
      groups * group_input_channels * group_output_channels *
      kernel_height * kernel_width,
    benchmark::Counter::kIsRate);
}
#endif  // BENCHMARK_ARM_COMPUTE_LIBRARY

// ShuffleNet v1 with 1 group.
static void ShuffleNetV1G1(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*************************** Conv 1 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   24});
  /******************* Stage 2: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,   36});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  36,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   36,  120});
  /******************* Stage 2: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  144,   36});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1,  36,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   36,  144});
  /******************* Stage 3: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  144,   72});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1,  72,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   72,  144});
  /******************* Stage 3: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  288,   72});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1,  72,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   72,  288});
  /******************* Stage 4: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  288,  144});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 144,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  144,  288});
  /******************* Stage 4: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  576,  144});
  b->Args({1,   7,   7,  3,  3,  2,  2, 2, 1, 144,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  144,  576});
}

// ShuffleNet v1 with 2 groups.
static void ShuffleNetV1G2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*************************** Conv 1 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   24});
  /******************* Stage 2: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,   50});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  50,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   2,   25,   88});
  /******************* Stage 2: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   2,  100,   25});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1,  50,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   2,   25,  100});
  /******************* Stage 3: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   2,  100,   50});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 100,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   2,   50,  100});
  /******************* Stage 3: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   2,  200,   50});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 100,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   2,   50,  200});
  /******************* Stage 4: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   2,  200,  100});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 200,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   2,  100,  200});
  /******************* Stage 4: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   2,  400,  100});
  b->Args({1,   7,   7,  3,  3,  2,  2, 2, 1, 200,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   2,  100,  400});
}

// ShuffleNet v1 with 3 groups.
static void ShuffleNetV1G3(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*************************** Conv 1 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   24});
  /******************* Stage 2: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,   60});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  60,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   3,   20,   72});
  /******************* Stage 2: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   3,   80,   20});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1,  60,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   3,   20,   80});
  /******************* Stage 3: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   3,   80,   40});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 120,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   3,   40,   80});
  /******************* Stage 3: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   3,  160,   40});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 120,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   3,   40,  160});
  /******************* Stage 4: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   3,  160,   80});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 240,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   3,   80,  160});
  /******************* Stage 4: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   3,  320,   80});
  b->Args({1,   7,   7,  3,  3,  2,  2, 2, 1, 240,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   3,   80,  320});
}

// ShuffleNet v1 with 4 groups.
static void ShuffleNetV1G4(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*************************** Conv 1 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   24});
  /******************* Stage 2: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,   68});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  68,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   4,   17,   62});
  /******************* Stage 2: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   4,   68,   17});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1,  68,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   4,   17,   68});
  /******************* Stage 3: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   4,   68,   34});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 136,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   4,   34,   68});
  /******************* Stage 3: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   4,  136,   34});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 136,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   4,   34,  136});
  /******************* Stage 4: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   4,  136,   68});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 272,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   4,   68,  136});
  /******************* Stage 4: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   4,  272,   68});
  b->Args({1,   7,   7,  3,  3,  2,  2, 2, 1, 272,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   4,   68,  272});
}

// ShuffleNet v1 with 8 groups.
static void ShuffleNetV1G8(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*************************** Conv 1 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   24});
  /******************* Stage 2: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,   96});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  96,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   8,   12,   45});
  /******************* Stage 2: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   8,   48,   12});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1,  96,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   8,   12,   48});
  /******************* Stage 3: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   8,   48,   24});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 192,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   8,   24,   48});
  /******************* Stage 3: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   8,   96,   24});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 192,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   8,   24,   96});
  /******************* Stage 4: stride-2 unit ******************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   8,   96,   48});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 384,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   8,   48,   96});
  /******************* Stage 4: stride-1 units *****************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   8,  192,   48});
  b->Args({1,   7,   7,  3,  3,  2,  2, 2, 1, 384,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   8,   48,  192});
}

// ShuffleNet v2 (0.5X scale)
static void ShuffleNetV2X05(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*************************** Conv 1 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   24});
  /************************** Stage 2 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  24,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   24,   24});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,   24});
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1,  24,    1,    1});
  /************************** Stage 3 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1,  48,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   48,   48});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   48,   48});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1,  48,    1,    1});
  /************************** Stage 4 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1,  96,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,   96,   96});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   96,   96});
  b->Args({1,   7,   7,  3,  3,  2,  2, 1, 1,  96,    1,    1});
  /*************************** Conv 5 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  192, 1024});
}

// ShuffleNet v2 (1.0X scale)
static void ShuffleNetV2X10(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*************************** Conv 1 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   24});
  /************************** Stage 2 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  24,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   24,   58});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,   58});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  58,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   58,   58});
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1,  58,    1,    1});
  /************************** Stage 3 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 116,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  116,  116});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  116,  116});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 116,    1,    1});
  /************************** Stage 4 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 232,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  232,  232});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  232,  232});
  b->Args({1,   7,   7,  3,  3,  2,  2, 1, 1, 232,    1,    1});
  /*************************** Conv 5 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  464, 1024});
}

// ShuffleNet v2 (1.5X scale)
static void ShuffleNetV2X15(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*************************** Conv 1 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   24});
  /************************** Stage 2 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  24,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   24,   88});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,   88});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  88,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   88,   88});
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1,  88,    1,    1});
  /************************** Stage 3 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 176,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  176,  176});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  176,  176});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 176,    1,    1});
  /************************** Stage 4 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 352,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  352,  352});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  352,  352});
  b->Args({1,   7,   7,  3,  3,  2,  2, 1, 1, 352,    1,    1});
  /*************************** Conv 5 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  704, 1024});
}

// ShuffleNet v2 (2.0X scale)
static void ShuffleNetV2X20(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*************************** Conv 1 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   24});
  /************************** Stage 2 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  24,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   24,  122});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,  122});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1, 122,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  122,  122});
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1, 122,    1,    1});
  /************************** Stage 3 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 244,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  244,  244});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  244,  244});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 244,    1,    1});
  /************************** Stage 4 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 488,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  488,  488});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  488,  488});
  b->Args({1,   7,   7,  3,  3,  2,  2, 1, 1, 488,    1,    1});
  /*************************** Conv 5 **************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  976, 2048});
}

static void MobileNetV1(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  PH  PW  S  D    G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,    1,    3,   32});
  b->Args({1, 112, 112,  3,  3,  2,  2, 1, 1,   32,    1,    1});
  b->Args({1, 112, 112,  1,  1,  0,  0, 1, 1,    1,   32,   64});
  b->Args({1, 112, 112,  3,  3,  2,  2, 2, 1,   64,    1,    1});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,    1,   64,  128});
  b->Args({1,  56,  56,  3,  3,  2,  2, 1, 1,  128,    1,    1});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,    1,  128,  128});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  128,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,    1,  128,  256});
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1,  256,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,    1,  256,  256});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1,  256,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,    1,  256,  512});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1,  512,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,    1,  512,  512});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1,  512,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,    1,  512, 1024});
  b->Args({1,   7,   7,  3,  3,  2,  2, 1, 1, 1024,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,    1, 1024, 1024});
}

static void MobileNetV2(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   32});

  /************************ Bottleneck 1 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
  b->Args({1, 112, 112,  3,  3,  2,  2, 1, 1,  32,    1,    1});
  b->Args({1, 112, 112,  1,  1,  0,  0, 1, 1,   1,   32,   16});

  /************************ Bottleneck 2 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
  b->Args({1, 112, 112,  1,  1,  0,  0, 1, 1,   1,   16,   96});
  b->Args({1, 112, 112,  3,  3,  2,  2, 2, 1,  96,    1,    1});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   96,   24});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,  144});
  b->Args({1,  56,  56,  3,  3,  2,  2, 1, 1, 144,    1,    1});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,  144,   24});

  /************************ Bottleneck 3 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
//b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,  144});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1, 144,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  144,   32});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   32,  192});
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1, 192,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  192,   32});
//b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   32,  192});
//b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1, 192,    1,    1});
//b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  192,   32});

  /************************ Bottleneck 4 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
//b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   32,  192});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 192,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  192,   64});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   64,  384});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 384,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  384,   64});
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   64,  384});
//b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 384,    1,    1});
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  384,   64});
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   64,  384});
//b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 384,    1,    1});
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  384,   64});

  /************************ Bottleneck 5 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   64,  384});
//b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 384,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  384,   96});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   96,  576});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 576,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  576,   96});
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   96,  576});
//b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 576,    1,    1});
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  576,   96});

  /************************ Bottleneck 6 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   96,  576});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 576,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  576,  160});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  160,  960});
  b->Args({1,   7,   7,  3,  3,  2,  2, 1, 1, 960,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  960,  160});
//b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  160,  960});
//b->Args({1,   7,   7,  3,  3,  2,  2, 1, 1, 960,    1,    1});
//b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  960,  160});

  /************************ Bottleneck 7 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
//b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  160,  960});
//b->Args({1,   7,   7,  3,  3,  2,  2, 1, 1, 960,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  960,  320});

  /******************** Pre-pooling Conv2D *********************/
  /*       N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  320, 1280});
  /******************** Post-pooling Conv2D ********************/
  /*       N   H    W   KH  KW  PH  PW  S  D    G  GCin  GCout */
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1, 1280, 1000});
}

static void MobileNetV3Small(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Initial Stage ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   16});
  /*********************** Bottleneck 1 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 112, 112,  3,  3,  2,  2, 2, 1,  16,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   16,    8});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,    8,   16});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   16,   16});
  /*********************** Bottleneck 2 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   16,   72});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1,  72,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   72,   24});
  /*********************** Bottleneck 3 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   24,   88});
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1,  88,    1,    1});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   88,   24});
  /*********************** Bottleneck 4 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   24,   96});
  b->Args({1,  28,  28,  5,  5,  4,  4, 2, 1,  96,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   96,   24});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   24,   96});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   96,   40});
  /*********************** Bottleneck 5 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   40,  240});
  b->Args({1,  14,  14,  5,  5,  4,  4, 1, 1, 240,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  240,   64});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   64,  240});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  240,   40});
  /*********************** Bottleneck 6 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   40,  240});
//b->Args({1,  14,  14,  5,  5,  4,  4, 1, 1, 240,    1,    1});
//b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  240,   64});
//b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   64,  240});
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  240,   40});
  /*********************** Bottleneck 7 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   40,  120});
  b->Args({1,  14,  14,  5,  5,  4,  4, 1, 1, 120,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  120,   32});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   32,  120});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  120,   48});
  /*********************** Bottleneck 8 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   48,  144});
  b->Args({1,  14,  14,  5,  5,  4,  4, 1, 1, 144,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  144,   40});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   40,  144});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  144,   48});
  /*********************** Bottleneck 9 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   48,  288});
  b->Args({1,  14,  14,  5,  5,  4,  4, 2, 1, 288,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  288,   72});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   72,  288});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  288,   96});
  /*********************** Bottleneck 10 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,   96,  576});
  b->Args({1,   7,   7,  5,  5,  4,  4, 1, 1, 576,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  576,  144});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  144,  576});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  576,   96});
  /*********************** Bottleneck 11 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
//b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,   96,  576});
//b->Args({1,   7,   7,  5,  5,  4,  4, 1, 1, 576,    1,    1});
//b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  576,  144});
//b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  144,  576});
//b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  576,   96});
  /************************ Last Stage  ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
//b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,   96,  576});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  576, 1024});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1, 1024, 1001});
}

static void MobileNetV3Large(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*********************** Initial Stage ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1,   1,    3,   16});
  /*********************** Bottleneck 1 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 112, 112,  3,  3,  2,  2, 1, 1,  16,    1,    1});
  b->Args({1, 112, 112,  1,  1,  0,  0, 1, 1,   1,   16,   16});
  /*********************** Bottleneck 2 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1, 112, 112,  1,  1,  0,  0, 1, 1,   1,   16,   64});
  b->Args({1, 112, 112,  3,  3,  2,  2, 2, 1,  64,    1,    1});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   64,   24});
  /*********************** Bottleneck 3 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,   72});
  b->Args({1,  56,  56,  3,  3,  2,  2, 1, 1,  72,    1,    1});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   72,   24});
  /*********************** Bottleneck 4 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
//b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1,   1,   24,   72});
  b->Args({1,  56,  56,  5,  5,  4,  4, 2, 1,  72,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   72,   24});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   24,   72});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   72,   40});
  /*********************** Bottleneck 5 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   40,  120});
  b->Args({1,  28,  28,  5,  5,  4,  4, 1, 1, 120,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  120,   32});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   32,  120});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  120,   40});
  /*********************** Bottleneck 6 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
//b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   40,  120});
//b->Args({1,  28,  28,  5,  5,  4,  4, 1, 1, 120,    1,    1});
//b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  120,   32});
//b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,   32,  120});
//b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,  120,   40});
  /*********************** Bottleneck 7 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1,   1,   40,  240});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 240,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  240,   80});
  /*********************** Bottleneck 8 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   80,  200});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 200,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  200,   80});
  /*********************** Bottleneck 9 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   80,  184});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 184,    1,    1});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  184,   80});
  /********************** Bottleneck 10 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   80,  184});
//b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 184,    1,    1});
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  184,   80});
  /********************** Bottleneck 11 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,   80,  480});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 480,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  480,  120});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  120,  480});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  480,  112});
  /********************** Bottleneck 12 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  112,  672});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 672,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  672,  168});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  168,  672});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  672,  112});
  /********************** Bottleneck 13 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1,   1,  112,  672});
  b->Args({1,  14,  14,  5,  5,  4,  4, 2, 1, 672,    1,    1});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  672,  160});
  /********************** Bottleneck 14 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  160,  960});
  b->Args({1,   7,   7,  5,  5,  4,  4, 1, 1, 960,    1,    1});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  960,  240});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  240,  960});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  960,  160});
  /********************** Bottleneck 15 ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
//b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  160,  960});
//b->Args({1,   7,   7,  5,  5,  4,  4, 1, 1, 960,    1,    1});
//b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  960,  240});
//b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  240,  960});
//b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  960,  160});
  /************************ Last Stage  ***********************/
  /*       N   H    W   KH  KW  PH  PW  S  D   G   GCin  GCout */
//b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1,   1,  160,  960});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1,  960, 1280});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1,   1, 1280, 1001});
}

// SqueezeNet 1.0
static void SqueezeNetV10(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /************************** Conv 1 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  7,  7,  6,  6, 2, 1, 1,    3,   96});
  /************************** Fire 2 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  55,  55,  1,  1,  0,  0, 1, 1, 1,   96,   16});
  b->Args({1,  55,  55,  1,  1,  0,  0, 1, 1, 1,   16,   64});
  b->Args({1,  55,  55,  3,  3,  2,  2, 1, 1, 1,   16,   64});
  /************************** Fire 3 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  56,  55,  1,  1,  0,  0, 1, 1, 1,  128,   16});
//b->Args({1,  55,  55,  1,  1,  0,  0, 1, 1, 1,   16,   64});
//b->Args({1,  55,  55,  3,  3,  2,  2, 1, 1, 1,   16,   64});
  /************************** Fire 4 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  55,  55,  1,  1,  0,  0, 1, 1, 1,  128,   32});
  b->Args({1,  55,  55,  1,  1,  0,  0, 1, 1, 1,   32,  128});
  b->Args({1,  55,  55,  3,  3,  2,  2, 1, 1, 1,   32,  128});
  /************************** Fire 5 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,  256,   32});
  b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,   32,  128});
  b->Args({1,  27,  27,  3,  3,  2,  2, 1, 1, 1,   32,  128});
  /************************** Fire 6 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,  256,   48});
  b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,   48,  192});
  b->Args({1,  27,  27,  3,  3,  2,  2, 1, 1, 1,   48,  192});
  /************************** Fire 7 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,  384,   48});
//b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,   48,  192});
//b->Args({1,  27,  27,  3,  3,  2,  2, 1, 1, 1,   48,  192});
  /************************** Fire 8 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,  384,   64});
  b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,   64,  256});
  b->Args({1,  27,  27,  3,  3,  2,  2, 1, 1, 1,   64,  256});
  /************************** Fire 9 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,  512,   64});
  b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,   64,  256});
  b->Args({1,  13,  13,  3,  3,  2,  2, 1, 1, 1,   64,  256});
  /************************* Conv 10 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,  512, 1000});
}

// SqueezeNet 1.1
static void SqueezeNetV11(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /************************** Conv 1 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 2, 1, 1,    3,   64});
  /************************** Fire 2 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  55,  55,  1,  1,  0,  0, 1, 1, 1,   64,   16});
  b->Args({1,  55,  55,  1,  1,  0,  0, 1, 1, 1,   16,   64});
  b->Args({1,  55,  55,  3,  3,  2,  2, 1, 1, 1,   16,   64});
  /************************** Fire 3 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  55,  55,  1,  1,  0,  0, 1, 1, 1,  128,   16});
//b->Args({1,  55,  55,  1,  1,  0,  0, 1, 1, 1,   16,   64});
//b->Args({1,  55,  55,  3,  3,  2,  2, 1, 1, 1,   16,   64});
  /************************** Fire 4 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,  128,   32});
  b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,   32,  128});
  b->Args({1,  27,  27,  3,  3,  2,  2, 1, 1, 1,   32,  128});
  /************************** Fire 5 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,  256,   32});
//b->Args({1,  27,  27,  1,  1,  0,  0, 1, 1, 1,   32,  128});
//b->Args({1,  27,  27,  3,  3,  2,  2, 1, 1, 1,   32,  128});
  /************************** Fire 6 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,  256,   48});
  b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,   48,  192});
  b->Args({1,  13,  13,  3,  3,  2,  2, 1, 1, 1,   48,  192});
  /************************** Fire 7 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,  384,   48});
//b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,   48,  192});
//b->Args({1,  13,  13,  3,  3,  2,  2, 1, 1, 1,   48,  192});
  /************************** Fire 8 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,  384,   64});
  b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,   64,  256});
  b->Args({1,  13,  13,  3,  3,  2,  2, 1, 1, 1,   64,  256});
  /************************** Fire 9 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,  512,   64});
//b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,   64,  256});
//b->Args({1,  13,  13,  3,  3,  2,  2, 1, 1, 1,   64,  256});
  /************************* Conv 10 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  13,  13,  1,  1,  0,  0, 1, 1, 1,  512, 1000});
}

static void InceptionV3(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1, 299, 299,  3,  3,  0,  0, 2, 1, 1,    3,   32});
  b->Args({1, 149, 149,  3,  3,  0,  0, 1, 1, 1,   32,   32});
  b->Args({1, 147, 147,  3,  3,  2,  2, 1, 1, 1,   32,   64});
  b->Args({1,  73,  73,  1,  1,  0,  0, 1, 1, 1,   64,   80});
  b->Args({1,  73,  73,  3,  3,  0,  0, 1, 1, 1,   80,  192});
  b->Args({1,  35,  35,  1,  1,  0,  0, 1, 1, 1,  192,   64});
  b->Args({1,  35,  35,  1,  1,  0,  0, 1, 1, 1,  192,   48});
  b->Args({1,  35,  35,  5,  5,  4,  4, 1, 1, 1,   48,   64});
  b->Args({1,  35,  35,  3,  3,  2,  2, 1, 1, 1,   64,   96});
  b->Args({1,  35,  35,  3,  3,  2,  2, 1, 1, 1,   96,   96});
  b->Args({1,  35,  35,  1,  1,  0,  0, 1, 1, 1,  192,   32});
  b->Args({1,  35,  35,  1,  1,  0,  0, 1, 1, 1,  256,   64});
  b->Args({1,  35,  35,  1,  1,  0,  0, 1, 1, 1,  256,   48});
  b->Args({1,  35,  35,  1,  1,  0,  0, 1, 1, 1,  288,   64});
  b->Args({1,  35,  35,  1,  1,  0,  0, 1, 1, 1,  288,   48});
  b->Args({1,  35,  35,  3,  3,  0,  0, 2, 1, 1,  288,  384});
  b->Args({1,  35,  35,  3,  3,  0,  0, 2, 1, 1,   96,   96});
  b->Args({1,  17,  17,  1,  1,  0,  0, 1, 1, 1,  768,  192});
  b->Args({1,  17,  17,  1,  1,  0,  0, 1, 1, 1,  768,  128});
  b->Args({1,  17,  17,  1,  7,  0,  6, 1, 1, 1,  128,  128});
  b->Args({1,  17,  17,  7,  1,  6,  0, 1, 1, 1,  128,  192});
  b->Args({1,  17,  17,  7,  1,  6,  0, 1, 1, 1,  128,  128});
  b->Args({1,  17,  17,  1,  7,  0,  6, 1, 1, 1,  128,  192});
  b->Args({1,  17,  17,  1,  1,  0,  0, 1, 1, 1,  768,  160});
  b->Args({1,  17,  17,  1,  7,  0,  6, 1, 1, 1,  160,  160});
  b->Args({1,  17,  17,  7,  1,  6,  0, 1, 1, 1,  160,  192});
  b->Args({1,  17,  17,  7,  1,  6,  0, 1, 1, 1,  160,  160});
  b->Args({1,  17,  17,  1,  7,  0,  6, 1, 1, 1,  160,  192});
  b->Args({1,  17,  17,  1,  7,  0,  6, 1, 1, 1,  192,  192});
  b->Args({1,  17,  17,  7,  1,  6,  0, 1, 1, 1,  192,  192});
  b->Args({1,  17,  17,  3,  3,  0,  0, 2, 1, 1,  192,  320});
  b->Args({1,  17,  17,  3,  3,  0,  0, 2, 1, 1,  192,  192});
  b->Args({1,   8,   8,  1,  1,  0,  0, 1, 1, 1, 1280,  320});
  b->Args({1,   8,   8,  1,  1,  0,  0, 1, 1, 1, 1280,  384});
  b->Args({1,   8,   8,  1,  3,  0,  2, 1, 1, 1,  384,  384});
  b->Args({1,   8,   8,  3,  1,  2,  0, 1, 1, 1,  384,  384});
  b->Args({1,   8,   8,  1,  1,  0,  0, 1, 1, 1, 1280,  448});
  b->Args({1,   8,   8,  3,  3,  2,  2, 1, 1, 1,  448,  384});
  b->Args({1,   8,   8,  1,  1,  0,  0, 1, 1, 1, 1280,  192});
  b->Args({1,   8,   8,  1,  1,  0,  0, 1, 1, 1, 2048,  320});
  b->Args({1,   8,   8,  1,  1,  0,  0, 1, 1, 1, 2048,  384});
  b->Args({1,   8,   8,  1,  1,  0,  0, 1, 1, 1, 2048,  448});
  b->Args({1,   8,   8,  1,  1,  0,  0, 1, 1, 1, 2048,  192});
  b->Args({1,   1,   1,  1,  1,  0,  0, 1, 1, 1, 2048, 1001});
}

static void ResNet18(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /************************* Conv 1 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1, 224, 224,  7,  7,  6,  6, 2, 1, 1,    3,   64});
  /************************ Conv 2.X ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  56,  56,  3,  3,  2,  2, 1, 1, 1,   64,   64});
  /************************ Conv 3.X ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1, 1,   64,  128});
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1, 1,  128,  128});
  b->Args({1,  56,  56,  1,  1,  0,  0, 2, 1, 1,   64,  128});
  /************************ Conv 4.X ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 1,  128,  256});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 1,  256,  256});
  b->Args({1,  28,  28,  1,  1,  0,  0, 2, 1, 1,  128,  256});
  /************************ Conv 5.X ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 1,  256,  512});
  b->Args({1,   7,   7,  3,  3,  2,  2, 1, 1, 1,  512,  512});
  b->Args({1,  14,  14,  1,  1,  0,  0, 2, 1, 1,  256,  512});
}

static void ResNet50(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /************************* Conv 1 *************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1, 224, 224,  7,  7,  6,  6, 2, 1, 1,    3,   64});
  /************************ Conv 2.1 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1, 1,   64,   64});
  b->Args({1,  56,  56,  3,  3,  2,  2, 1, 1, 1,   64,   64});
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1, 1,   64,  256});
//b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1, 1,   64,  256});
  /************************ Conv 2.X ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1, 1,  256,   64});
//b->Args({1,  56,  56,  3,  3,  2,  2, 1, 1, 1,   64,   64});
//b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1, 1,   64,  256});
  /************************ Conv 3.1 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1, 1,  256,  128});
  b->Args({1,  56,  56,  3,  3,  2,  2, 2, 1, 1,  128,  128});
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1, 1,  128,  512});
  b->Args({1,  56,  56,  1,  1,  0,  0, 2, 1, 1,  256,  512});
  /************************ Conv 3.X ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1, 1,  512,  128});
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1, 1,  128,  128});
//b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1, 1,  128,  512});
  /************************ Conv 4.1 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1, 1,  512,  256});
  b->Args({1,  28,  28,  3,  3,  2,  2, 2, 1, 1,  256,  256});
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1, 1,  256, 1024});
  b->Args({1,  28,  28,  1,  1,  0,  0, 2, 1, 1,  512, 1024});
  /************************ Conv 4.X ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1, 1, 1024,  256});
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 1,  256,  256});
//b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1, 1,  256, 1024});
  /************************ Conv 5.1 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1, 1, 1024,  512});
  b->Args({1,  14,  14,  3,  3,  2,  2, 2, 1, 1,  512,  512});
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1, 1,  512, 2048});
  b->Args({1,  14,  14,  1,  1,  0,  0, 2, 1, 1, 1024, 2048});
  /************************ Conv 5.X ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G GCin  GCout */
  b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1, 1, 2048,  512});
  b->Args({1,   7,   7,  3,  3,  2,  2, 1, 1, 1,  512,  512});
//b->Args({1,   7,   7,  1,  1,  0,  0, 1, 1, 1,  512, 2048});
}

static void VGG(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /************************* Conv 1.1 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 1, 1, 1,    3,   64});
  /************************* Conv 1.2 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1, 224, 224,  3,  3,  2,  2, 1, 1, 1,   64,   64});

  /************************* Conv 2.1 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1, 112, 112,  3,  3,  2,  2, 1, 1, 1,   64,  128});
  /************************* Conv 2.2 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1, 112, 112,  3,  3,  2,  2, 1, 1, 1,  128,  128});

  /************************* Conv 3.1 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  56,  56,  3,  3,  2,  2, 1, 1, 1,  128,  256});
  /************************* Conv 3.2 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  56,  56,  3,  3,  2,  2, 1, 1, 1,  256,  256});
  /************************* Conv 3.3 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  56,  56,  1,  1,  0,  0, 1, 1, 1,  256,  256});

  /************************* Conv 4.1 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1, 1,  256,  512});
  /************************* Conv 4.2 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  28,  28,  3,  3,  2,  2, 1, 1, 1,  512,  512});
  /************************* Conv 4.3 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  28,  28,  1,  1,  0,  0, 1, 1, 1,  512,  512});

  /************************* Conv 5.X ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  14,  14,  3,  3,  2,  2, 1, 1, 1,  512,  512});
  /************************* Conv 5.3 ************************/
  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1,  14,  14,  1,  1,  0,  0, 1, 1, 1,  512,  512});
}

// SRCNN (9-1-5)
static void SRCNN915(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1, 384, 384,  9,  9,  0,  0, 1, 1, 1,    1,   64});
  b->Args({1, 376, 376,  1,  1,  0,  0, 1, 1, 1,   64,   32});
  b->Args({1, 376, 376,  5,  5,  0,  0, 1, 1, 1,   32,    1});
}

// SRCNN (9-3-5)
static void SRCNN935(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1, 384, 384,  9,  9,  0,  0, 1, 1, 1,    1,   64});
  b->Args({1, 376, 376,  3,  3,  0,  0, 1, 1, 1,   64,   32});
  b->Args({1, 374, 374,  5,  5,  0,  0, 1, 1, 1,   32,    1});
}

// SRCNN (9-5-5)
static void SRCNN955(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  PH  PW  S  D  G  GCin  GCout */
  b->Args({1, 384, 384,  9,  9,  0,  0, 1, 1, 1,    1,   64});
  b->Args({1, 376, 376,  5,  5,  0,  0, 1, 1, 1,   64,   32});
  b->Args({1, 372, 372,  5,  5,  0,  0, 1, 1, 1,   32,    1});
}

BENCHMARK_CAPTURE(xnnpack_convolution_f16, mobilenet_v1, "MobileNet v1")->Apply(MobileNetV1)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, mobilenet_v2, "MobileNet v2")->Apply(MobileNetV2)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, mobilenet_v3_small, "MobileNet v3 Small")->Apply(MobileNetV3Small)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, mobilenet_v3_large, "MobileNet v3 Large")->Apply(MobileNetV3Large)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, shufflenet_v2_x05, "ShuffleNet v2 0.5X")->Apply(ShuffleNetV2X05)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, shufflenet_v2_x10, "ShuffleNet v2 1.0X")->Apply(ShuffleNetV2X10)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, shufflenet_v2_x15, "ShuffleNet v2 1.5X")->Apply(ShuffleNetV2X15)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, shufflenet_v2_x20, "ShuffleNet v2 2.0X")->Apply(ShuffleNetV2X20)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, squeezenet_v10, "SqueezeNet 1.0")->Apply(SqueezeNetV10)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, squeezenet_v11, "SqueezeNet 1.1")->Apply(SqueezeNetV11)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, inception_v3, "Inception v3")->Apply(InceptionV3)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, resnet18, "ResNet-18")->Apply(ResNet18)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, resnet50, "ResNet-50")->Apply(ResNet50)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, vgg, "VGG")->Apply(VGG)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, srcnn915, "SRCNN (9-1-5)")->Apply(SRCNN915)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, srcnn935, "SRCNN (9-3-5)")->Apply(SRCNN935)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f16, srcnn955, "SRCNN (9-5-5)")->Apply(SRCNN955)->UseRealTime();

BENCHMARK_CAPTURE(xnnpack_convolution_f32, mobilenet_v1, "MobileNet v1")->Apply(MobileNetV1)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, mobilenet_v2, "MobileNet v2")->Apply(MobileNetV2)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, mobilenet_v3_small, "MobileNet v3 Small")->Apply(MobileNetV3Small)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, mobilenet_v3_large, "MobileNet v3 Large")->Apply(MobileNetV3Large)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, shufflenet_v2_x05, "ShuffleNet v2 0.5X")->Apply(ShuffleNetV2X05)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, shufflenet_v2_x10, "ShuffleNet v2 1.0X")->Apply(ShuffleNetV2X10)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, shufflenet_v2_x15, "ShuffleNet v2 1.5X")->Apply(ShuffleNetV2X15)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, shufflenet_v2_x20, "ShuffleNet v2 2.0X")->Apply(ShuffleNetV2X20)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, squeezenet_v10, "SqueezeNet 1.0")->Apply(SqueezeNetV10)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, squeezenet_v11, "SqueezeNet 1.1")->Apply(SqueezeNetV11)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, inception_v3, "Inception v3")->Apply(InceptionV3)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, resnet18, "ResNet-18")->Apply(ResNet18)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, resnet50, "ResNet-50")->Apply(ResNet50)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, vgg, "VGG")->Apply(VGG)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, srcnn915, "SRCNN (9-1-5)")->Apply(SRCNN915)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, srcnn935, "SRCNN (9-3-5)")->Apply(SRCNN935)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_f32, srcnn955, "SRCNN (9-5-5)")->Apply(SRCNN955)->UseRealTime();

BENCHMARK_CAPTURE(xnnpack_convolution_qu8, mobilenet_v1, "MobileNet v1")->Apply(MobileNetV1)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, mobilenet_v2, "MobileNet v2")->Apply(MobileNetV2)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, mobilenet_v3_small, "MobileNet v3 Small")->Apply(MobileNetV3Small)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, mobilenet_v3_large, "MobileNet v3 Large")->Apply(MobileNetV3Large)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, shufflenet_v2_x05, "ShuffleNet v2 0.5X")->Apply(ShuffleNetV2X05)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, shufflenet_v2_x10, "ShuffleNet v2 1.0X")->Apply(ShuffleNetV2X10)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, shufflenet_v2_x15, "ShuffleNet v2 1.5X")->Apply(ShuffleNetV2X15)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, shufflenet_v2_x20, "ShuffleNet v2 2.0X")->Apply(ShuffleNetV2X20)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, squeezenet_v10, "SqueezeNet 1.0")->Apply(SqueezeNetV10)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, squeezenet_v11, "SqueezeNet 1.1")->Apply(SqueezeNetV11)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, inception_v3, "Inception v3")->Apply(InceptionV3)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, resnet18, "ResNet-18")->Apply(ResNet18)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, resnet50, "ResNet-50")->Apply(ResNet50)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, vgg, "VGG")->Apply(VGG)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, srcnn915, "SRCNN (9-1-5)")->Apply(SRCNN915)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, srcnn935, "SRCNN (9-3-5)")->Apply(SRCNN935)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_convolution_qu8, srcnn955, "SRCNN (9-5-5)")->Apply(SRCNN955)->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
  BENCHMARK_CAPTURE(tflite_convolution_f32, mobilenet_v1, "MobileNet v1")->Apply(MobileNetV1)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, mobilenet_v2, "MobileNet v2")->Apply(MobileNetV2)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, mobilenet_v3_small, "MobileNet v3 Small")->Apply(MobileNetV3Small)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, mobilenet_v3_large, "MobileNet v3 Large")->Apply(MobileNetV3Large)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, shufflenet_v2_x05, "ShuffleNet v2 0.5X")->Apply(ShuffleNetV2X05)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, shufflenet_v2_x10, "ShuffleNet v2 1.0X")->Apply(ShuffleNetV2X10)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, shufflenet_v2_x15, "ShuffleNet v2 1.5X")->Apply(ShuffleNetV2X15)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, shufflenet_v2_x20, "ShuffleNet v2 2.0X")->Apply(ShuffleNetV2X20)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, squeezenet_v10, "SqueezeNet 1.0")->Apply(SqueezeNetV10)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, squeezenet_v11, "SqueezeNet 1.1")->Apply(SqueezeNetV11)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, inception_v3, "Inception v3")->Apply(InceptionV3)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, resnet18, "ResNet-18")->Apply(ResNet18)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, resnet50, "ResNet-50")->Apply(ResNet50)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, vgg, "VGG")->Apply(VGG)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, srcnn915, "SRCNN (9-1-5)")->Apply(SRCNN915)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, srcnn935, "SRCNN (9-3-5)")->Apply(SRCNN935)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_convolution_f32, srcnn955, "SRCNN (9-5-5)")->Apply(SRCNN955)->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifdef BENCHMARK_ARM_COMPUTE_LIBRARY
  BENCHMARK_CAPTURE(armcl_convolution_f32, mobilenet_v1, "MobileNet v1")->Apply(MobileNetV1)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, mobilenet_v2, "MobileNet v2")->Apply(MobileNetV2)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, shufflenet_v1_g1, "ShuffleNet v1 (1 group)")->Apply(ShuffleNetV1G1)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, shufflenet_v1_g2, "ShuffleNet v1 (2 groups)")->Apply(ShuffleNetV1G2)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, shufflenet_v1_g3, "ShuffleNet v1 (3 groups)")->Apply(ShuffleNetV1G3)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, shufflenet_v1_g4, "ShuffleNet v1 (4 groups)")->Apply(ShuffleNetV1G4)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, shufflenet_v1_g8, "ShuffleNet v1 (8 groups)")->Apply(ShuffleNetV1G8)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, shufflenet_v2_x05, "ShuffleNet v2 0.5X")->Apply(ShuffleNetV2X05)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, shufflenet_v2_x10, "ShuffleNet v2 1.0X")->Apply(ShuffleNetV2X10)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, shufflenet_v2_x15, "ShuffleNet v2 1.5X")->Apply(ShuffleNetV2X15)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, shufflenet_v2_x20, "ShuffleNet v2 2.0X")->Apply(ShuffleNetV2X20)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, squeezenet_v10, "SqueezeNet 1.0")->Apply(SqueezeNetV10)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, squeezenet_v11, "SqueezeNet 1.1")->Apply(SqueezeNetV11)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, inception_v3, "Inception v3")->Apply(InceptionV3)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, resnet18, "ResNet-18")->Apply(ResNet18)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, resnet50, "ResNet-50")->Apply(ResNet50)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, vgg, "VGG")->Apply(VGG)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, srcnn915, "SRCNN (9-1-5)")->Apply(SRCNN915)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, srcnn935, "SRCNN (9-3-5)")->Apply(SRCNN935)->UseRealTime();
  BENCHMARK_CAPTURE(armcl_convolution_f32, srcnn955, "SRCNN (9-5-5)")->Apply(SRCNN955)->UseRealTime();
#endif  // BENCHMARK_ARM_COMPUTE_LIBRARY

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
