// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cpuinfo.h>
#include <xnnpack.h>

#include <benchmark/benchmark.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE */
#include "bench/utils.h"


void xnnpack_deconvolution_q8(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding = state.range(5);
  const size_t adjustment = state.range(6);
  const size_t stride = state.range(7);
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
  const size_t padding_left = padding / 2;
  const size_t padding_top = padding / 2;
  const size_t padding_right = padding - padding_left;
  const size_t padding_bottom = padding - padding_top;
  const size_t output_height = std::max(stride * (input_height - 1) + adjustment + effective_kernel_height, padding) - padding;
  const size_t output_width = std::max(stride * (input_width - 1) + adjustment + effective_kernel_width, padding) - padding;

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

  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
    return;
  }
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (kernel.size() + bias.size() + output_elements));
  std::vector<uint8_t> output(output_elements * num_buffers);

  std::vector<xnn_operator_t> deconvolution_operators(num_buffers);
  for (xnn_operator_t& deconvolution_op : deconvolution_operators) {
    status = xnn_create_deconvolution2d_nhwc_q8(
        padding_top, padding_right, padding_bottom, padding_left,
        kernel_height, kernel_width,
        stride, stride,
        dilation, dilation,
        groups, group_input_channels, group_output_channels,
        input_pixel_stride, output_pixel_stride,
        127, 0.5f, 127, 0.5f,
        kernel.data(), bias.data(),
        127, 0.5f, 0, 255,
        0 /* flags */,
        &deconvolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create QINT8 Deconvolution operator");
      return;
    }
  }

  for (size_t i = 0; i < deconvolution_operators.size(); i++) {
    status = xnn_setup_deconvolution2d_nhwc_q8(
        deconvolution_operators[i],
        batch_size, input_height, input_width,
        0 /* height adjustment */, 0 /* width adjustment */,
        input.data(), output.data() + i * output_elements,
        nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup QINT8 Deconvolution operator");
      return;
    }
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(input.data(), input.size() * sizeof(uint8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    status = xnn_run_operator(deconvolution_operators[buffer_index], nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run QINT8 Deconvolution operator");
      return;
    }
  }

  for (xnn_operator_t& deconvolution_op : deconvolution_operators) {
    status = xnn_delete_operator(deconvolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to delete QINT8 Deconvolution operator");
      return;
    }
    deconvolution_op = nullptr;
  }

    state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
    state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * input_width * input_width *
      groups * group_input_channels * group_output_channels *
      kernel_height * kernel_width,
    benchmark::Counter::kIsRate);
}

void xnnpack_deconvolution_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding = state.range(5);
  const size_t adjustment = state.range(6);
  const size_t stride = state.range(7);
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
  const size_t padding_left = padding / 2;
  const size_t padding_top = padding / 2;
  const size_t padding_right = padding - padding_left;
  const size_t padding_bottom = padding - padding_top;
  const size_t output_height = std::max(stride * (input_height - 1) + adjustment + effective_kernel_height, padding) - padding;
  const size_t output_width = std::max(stride * (input_width - 1) + adjustment + effective_kernel_width, padding) - padding;

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

  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
    return;
  }
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (kernel.size() + bias.size() + output_elements));
  std::vector<float> output(output_elements * num_buffers);

  std::vector<xnn_operator_t> deconvolution_operators(num_buffers);
  for (xnn_operator_t& deconvolution_op : deconvolution_operators) {
    status = xnn_create_deconvolution2d_nhwc_f32(
        padding_top, padding_right, padding_bottom, padding_left,
        kernel_height, kernel_width,
        stride, stride,
        dilation, dilation,
        groups, group_input_channels, group_output_channels,
        input_pixel_stride, output_pixel_stride,
        kernel.data(), bias.data(),
        -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
        0 /* flags */,
        &deconvolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create FP32 Deconvolution operator");
      return;
    }
  }

  for (size_t i = 0; i < deconvolution_operators.size(); i++) {
    status = xnn_setup_deconvolution2d_nhwc_f32(
        deconvolution_operators[i],
        batch_size, input_height, input_width,
        0 /* height adjustment */, 0 /* width adjustment */,
        input.data(), output.data() + i * output_elements,
        nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup QINT8 Deconvolution operator");
      return;
    }
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(input.data(), input.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    status = xnn_run_operator(deconvolution_operators[buffer_index], nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run FP32 Deconvolution operator");
      return;
    }
  }

  for (xnn_operator_t& deconvolution_op : deconvolution_operators) {
    status = xnn_delete_operator(deconvolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to delete FP32 Deconvolution operator");
      return;
    }
    deconvolution_op = nullptr;
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * input_width * input_width *
      groups * group_input_channels * group_output_channels *
      kernel_height * kernel_width,
    benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_TENSORFLOW_LITE
void tflite_deconvolution_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding = state.range(5);
  const size_t adjustment = state.range(6);
  const size_t stride = state.range(7);
  const size_t dilation = state.range(8);
  const size_t groups = state.range(9);
  const size_t input_channels = state.range(10);
  const size_t output_channels = state.range(11);

  if (groups != 1) {
    state.SkipWithError("grouped deconvolution is not supported");
    return;
  }
  if (dilation != 1) {
    state.SkipWithError("dilated deconvolution is not supported");
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

  tflite::Padding tf_padding = tflite::Padding_VALID;
  if (padding == (kernel_width - 1) && padding == (kernel_height - 1)) {
    tf_padding = tflite::Padding_SAME;
  } else if (padding == 0) {
    tf_padding = tflite::Padding_VALID;
  } else {
    state.SkipWithError("unsupported padding");
    return;
  }

  const size_t output_height = std::max(stride * (input_height - 1) + adjustment + kernel_height, padding) - padding;
  const size_t output_width = std::max(stride * (input_width - 1) + adjustment + kernel_width, padding) - padding;

  std::vector<float> kernel(output_channels * kernel_height * kernel_width * input_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_TRANSPOSE_CONV, 0);

  flatbuffers::Offset<tflite::TransposeConvOptions> transpose_conv_options = CreateTransposeConvOptions(
      builder,
      tf_padding,
      static_cast<int32_t>(stride), static_cast<int32_t>(stride));

  const int32_t input_shape[4] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(input_height),
    static_cast<int32_t>(input_width),
    static_cast<int32_t>(input_channels)
  };
  const int32_t output_shape[4] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(output_height),
    static_cast<int32_t>(output_width),
    static_cast<int32_t>(output_channels)
  };
  const int32_t filter_shape[4] = {
    static_cast<int32_t>(output_channels),
    static_cast<int32_t>(kernel_height),
    static_cast<int32_t>(kernel_width),
    static_cast<int32_t>(input_channels)
  };
  const int32_t output_shape_shape[1] = { 4 };

  flatbuffers::Offset<tflite::Buffer> buffers[3] = {
    tflite::CreateBuffer(builder, builder.CreateVector({})),
    tflite::CreateBuffer(builder, builder.CreateVector(
      reinterpret_cast<const uint8_t*>(kernel.data()),
      sizeof(float) * kernel.size())),
    tflite::CreateBuffer(builder, builder.CreateVector(
      reinterpret_cast<const uint8_t*>(output_shape),
      sizeof(output_shape))),
  };

  flatbuffers::Offset<tflite::Tensor> tensors[4] = {
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(output_shape_shape, 1),
                         tflite::TensorType_INT32,
                         2 /* buffer id */,
                         builder.CreateString("output_shape")),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(filter_shape, 4),
                         tflite::TensorType_FLOAT32,
                         1 /* buffer id */,
                         builder.CreateString("filter")),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(input_shape, 4),
                         tflite::TensorType_FLOAT32,
                         0 /* buffer id */,
                         builder.CreateString("input")),
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
      tflite::BuiltinOptions_TransposeConvOptions,
      transpose_conv_options.Union());

  const int32_t graph_inputs[1] = { 2 };
  const int32_t graph_outputs[1] = { 3 };
  flatbuffers::Offset<tflite::SubGraph> subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(tensors, 4),
      builder.CreateVector<int32_t>(graph_inputs, 1),
      builder.CreateVector<int32_t>(graph_outputs, 1),
      builder.CreateVector(&op, 1),
      builder.CreateString("TransposeConv subgraph"));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("TransposeConv model");

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
    interpreter->typed_tensor<float>(2),
    interpreter->typed_tensor<float>(2) + batch_size * input_channels * input_height * input_width,
    std::ref(f32rng));

  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::WipeCache();
    benchmark::utils::PrefetchToL1(
      interpreter->typed_tensor<float>(2),
      batch_size * input_channels * input_height * input_width * sizeof(float));
    state.ResumeTiming();

    if (interpreter->Invoke() != kTfLiteOk) {
      state.SkipWithError("failed to invoke TFLite interpreter");
      return;
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * input_width * input_width *
      input_channels * output_channels *
      kernel_height * kernel_width,
    benchmark::Counter::kIsRate);

  interpreter.reset();
}
#endif  // BENCHMARK_TENSORFLOW_LITE

// FCN-32 model (PASCAL VOC version).
// We assume CIF image (352x288) on model input / output.
static void FCN32(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "P", "A", "S", "D", "G", "GCin", "GCout"});

  /*       N  H   W  KH  KW  P  A   S  D  G  GCin  GCout */
  b->Args({1, 9, 11, 64, 64, 0, 0, 32, 1, 1,   21,   21});
}

// FCN-16 model (PASCAL VOC version).
// We assume CIF image (352x288) on model input / output.
static void FCN16(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "P", "A", "S", "D", "G", "GCin", "GCout"});

  /*       N   H   W  KH  KW  P  A   S  D  G  GCin  GCout */
  b->Args({1,  9, 11,  4,  4, 0, 0,  2, 1, 1,   21,   21});
  b->Args({1, 18, 22, 32, 32, 0, 0, 16, 1, 1,   21,   21});
}

// FCN-8 model (PASCAL VOC version).
// We assume CIF image (352x288) on model input / output.
static void FCN8(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "P", "A", "S", "D", "G", "GCin", "GCout"});

  /*       N   H   W  KH  KW  P  A  S  D  G  GCin  GCout */
  b->Args({1,  9, 11,  4,  4, 0, 0, 2, 1, 1,   21,   21});
  b->Args({1, 18, 22,  4,  4, 0, 0, 2, 1, 1,   21,   21});
  b->Args({1, 36, 44, 16, 16, 0, 0, 8, 1, 1,   21,   21});
}

static void ENet(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "P", "A", "S", "D", "G", "GCin", "GCout"});

  /********************* Bottleneck 4.0 ********************/
  /*       N   H    W   KH  KW  P  A  S  D  G  GCin  GCout */
  b->Args({1,  64,  64,  3,  3, 2, 1, 2, 1, 1,   32,   32});
  /********************* Bottleneck 5.0 ********************/
  /*       N   H    W   KH  KW  P  A  S  D  G  GCin  GCout */
  b->Args({1, 128, 128,  3,  3, 2, 1, 2, 1, 1,   16,   16});
  /***************** Final Full Convolution ****************/
  /*       N   H    W   KH  KW  P  A  S  D  G  GCin  GCout */
  b->Args({1, 256, 256,  2,  2, 0, 0, 2, 1, 1,   16,   12});
}

static void ESPNet(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "P", "A", "S", "D", "G", "GCin", "GCout"});

  /*       N   H    W   KH  KW  P  A  S  D  G  GCin  GCout */
  b->Args({1,  64, 128,  2,  2, 0, 0, 2, 1, 1,   20,   20});
  b->Args({1, 128, 256,  2,  2, 0, 0, 2, 1, 1,   20,   20});
  b->Args({1, 256, 512,  2,  2, 0, 0, 2, 1, 1,   20,   20});
}

BENCHMARK_CAPTURE(xnnpack_deconvolution_f32, fcn32, "FCN-32")->Apply(FCN32)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_f32, fcn16, "FCN-16")->Apply(FCN16)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_f32, fcn8, "FCN-8")->Apply(FCN8)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_f32, enet, "ENet")->Apply(ENet)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_f32, espnet, "ESPNet")->Apply(ESPNet)->UseRealTime();

BENCHMARK_CAPTURE(xnnpack_deconvolution_q8, fcn32, "FCN-32")->Apply(FCN32)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_q8, fcn16, "FCN-16")->Apply(FCN16)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_q8, fcn8, "FCN-8")->Apply(FCN8)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_q8, enet, "ENet")->Apply(ENet)->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_q8, espnet, "ESPNet")->Apply(ESPNet)->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
  BENCHMARK_CAPTURE(tflite_deconvolution_f32, fcn32, "FCN-32")->Apply(FCN32)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_deconvolution_f32, fcn16, "FCN-16")->Apply(FCN16)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_deconvolution_f32, fcn8, "FCN-8")->Apply(FCN8)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_deconvolution_f32, enet, "ENet")->Apply(ENet)->UseRealTime();
  BENCHMARK_CAPTURE(tflite_deconvolution_f32, espnet, "ESPNet")->Apply(ESPNet)->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
