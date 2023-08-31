// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

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

void xnnpack_deconvolution_qu8(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding_height = state.range(5);
  const size_t padding_width = state.range(6);
  const size_t adjustment = state.range(7);
  const size_t stride_height = state.range(8);
  const size_t stride_width = state.range(9);
  const size_t dilation = state.range(10);
  const size_t input_channels = state.range(11);
  const size_t output_channels = state.range(12);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto u8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
    std::ref(rng));

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t padding_right = padding_width - padding_left;
  const size_t padding_bottom = padding_height - padding_top;
  const size_t output_height = std::max(stride_height * (input_height - 1) + adjustment + effective_kernel_height, padding_height) - padding_height;
  const size_t output_width = std::max(stride_width * (input_width - 1) + adjustment + effective_kernel_width, padding_width) - padding_width;

  std::vector<uint8_t> input(batch_size * input_height * input_width * input_channels);
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> kernel(output_channels * kernel_height * kernel_width * input_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
  std::vector<int32_t> bias(output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(i32rng));
  const size_t output_elements = batch_size * output_height * output_width * output_channels;

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (kernel.size() + bias.size() + output_elements));
  std::vector<uint8_t> output(output_elements * num_buffers);

  std::vector<xnn_operator_t> deconvolution_operators(num_buffers);
  for (xnn_operator_t& deconvolution_op : deconvolution_operators) {
    status = xnn_create_deconvolution2d_nhwc_qu8(
        padding_top, padding_right, padding_bottom, padding_left,
        kernel_height, kernel_width,
        stride_height, stride_width,
        dilation, dilation,
        /*groups=*/1, input_channels, output_channels,
        /*input_pixel_stride=*/input_channels, /*output_pixel_stride=*/output_channels,
        127, 0.5f, 127, 0.5f,
        kernel.data(), bias.data(),
        127, 0.5f, 0, 255,
        0 /* flags */,
        nullptr, nullptr,
        &deconvolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create QINT8 Deconvolution operator");
      return;
    }
  }

  for (size_t i = 0; i < deconvolution_operators.size(); i++) {
    status = xnn_reshape_deconvolution2d_nhwc_qu8(
        deconvolution_operators[i],
        batch_size, input_height, input_width,
        0 /* height adjustment */, 0 /* width adjustment */,
	/*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
        /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup QINT8 Deconvolution operator");
      return;
    }
  }

  for (size_t i = 0; i < deconvolution_operators.size(); i++) {
    status = xnn_setup_deconvolution2d_nhwc_qu8(
        deconvolution_operators[i],
        input.data(), output.data() + i * output_elements);
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

    status = xnn_run_operator(deconvolution_operators[buffer_index], /*threadpool=*/nullptr);
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

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] = benchmark::Counter(
  uint64_t(state.iterations()) * 2 *
    batch_size * input_width * input_width *
    input_channels * output_channels *
    kernel_height * kernel_width,
  benchmark::Counter::kIsRate);
}

void xnnpack_deconvolution_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t input_height = state.range(1);
  const size_t input_width = state.range(2);
  const size_t kernel_height = state.range(3);
  const size_t kernel_width = state.range(4);
  const size_t padding_height = state.range(5);
  const size_t padding_width = state.range(6);
  const size_t adjustment = state.range(7);
  const size_t stride_height = state.range(8);
  const size_t stride_width = state.range(9);
  const size_t dilation = state.range(10);
  const size_t input_channels = state.range(11);
  const size_t output_channels = state.range(12);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t padding_right = padding_width - padding_left;
  const size_t padding_bottom = padding_height - padding_top;
  const size_t output_height = std::max(stride_height * (input_height - 1) + adjustment + effective_kernel_height, padding_height) - padding_height;
  const size_t output_width = std::max(stride_width * (input_width - 1) + adjustment + effective_kernel_width, padding_width) - padding_width;

  std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
    batch_size * input_height * input_width * input_channels);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> kernel(output_channels * kernel_height * kernel_width * input_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
  std::vector<float> bias(output_channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));
  const size_t output_elements = batch_size * output_height * output_width * output_channels;

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
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
        stride_height, stride_width,
        dilation, dilation,
        /*groups=*/1, input_channels, output_channels,
        /*input_pixel_stride=*/input_channels, /*output_pixel_stride=*/output_channels,
        kernel.data(), bias.data(),
        -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity(),
        0 /* flags */,
        nullptr,
        nullptr,
        &deconvolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create FP32 Deconvolution operator");
      return;
    }
  }

  for (size_t i = 0; i < deconvolution_operators.size(); i++) {
    status = xnn_reshape_deconvolution2d_nhwc_f32(
        deconvolution_operators[i],
        batch_size, input_height, input_width,
        0 /* height adjustment */, 0 /* width adjustment */,
	/*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
        /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to setup QINT8 Deconvolution operator");
      return;
    }
  }

  for (size_t i = 0; i < deconvolution_operators.size(); i++) {
    status = xnn_setup_deconvolution2d_nhwc_f32(
        deconvolution_operators[i],
        input.data(), output.data() + i * output_elements);
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

    status = xnn_run_operator(deconvolution_operators[buffer_index], /*threadpool=*/nullptr);
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

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      batch_size * input_width * input_width *
      input_channels * output_channels *
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
  const size_t padding_height = state.range(5);
  const size_t padding_width = state.range(6);
  const size_t adjustment = state.range(7);
  const size_t stride_height = state.range(8);
  const size_t stride_width = state.range(9);
  const size_t dilation = state.range(10);
  const size_t input_channels = state.range(11);
  const size_t output_channels = state.range(12);

  if (dilation != 1) {
    state.SkipWithError("dilated deconvolution is not supported");
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

  tflite::Padding tf_padding = tflite::Padding_VALID;
  if (padding_width == kernel_width - stride_width && padding_height == kernel_height - stride_height) {
    tf_padding = tflite::Padding_SAME;
  } else if (padding_width == 0 && padding_height == 0) {
    tf_padding = tflite::Padding_VALID;
  } else {
    state.SkipWithError("unsupported padding");
    return;
  }

  const size_t output_height = std::max(stride_height * (input_height - 1) + adjustment + kernel_height, padding_height) - padding_height;
  const size_t output_width = std::max(stride_width * (input_width - 1) + adjustment + kernel_width, padding_width) - padding_width;

  std::vector<float> kernel(output_channels * kernel_height * kernel_width * input_channels);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_TRANSPOSE_CONV, 0);

  flatbuffers::Offset<tflite::TransposeConvOptions> transpose_conv_options = CreateTransposeConvOptions(
      builder,
      tf_padding,
      static_cast<int32_t>(stride_width), static_cast<int32_t>(stride_height));

  const std::array<int32_t, 4> input_shape{{
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(input_height),
    static_cast<int32_t>(input_width),
    static_cast<int32_t>(input_channels)
  }};
  const std::array<int32_t, 4> output_shape{{
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(output_height),
    static_cast<int32_t>(output_width),
    static_cast<int32_t>(output_channels)
  }};
  const std::array<int32_t, 4> filter_shape{{
    static_cast<int32_t>(output_channels),
    static_cast<int32_t>(kernel_height),
    static_cast<int32_t>(kernel_width),
    static_cast<int32_t>(input_channels)
  }};
  const std::array<int32_t, 1> output_shape_shape{{ 4 }};

  const std::array<flatbuffers::Offset<tflite::Buffer>, 3> buffers{{
    tflite::CreateBuffer(builder, builder.CreateVector({})),
    tflite::CreateBuffer(builder, builder.CreateVector(
      reinterpret_cast<const uint8_t*>(kernel.data()),
      sizeof(float) * kernel.size())),
    tflite::CreateBuffer(builder, builder.CreateVector(
      reinterpret_cast<const uint8_t*>(output_shape.data()),
      sizeof(int32_t) * output_shape.size())),
  }};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 4> tensors{{
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(output_shape_shape.data(), output_shape_shape.size()),
                         tflite::TensorType_INT32,
                         2 /* buffer id */),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(filter_shape.data(), filter_shape.size()),
                         tflite::TensorType_FLOAT32,
                         1 /* buffer id */),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
                         tflite::TensorType_FLOAT32),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(output_shape.data(), output_shape.size()),
                         tflite::TensorType_FLOAT32),
  }};

  const std::array<int32_t, 3> op_inputs{{ 0, 1, 2 }};
  const std::array<int32_t, 1> op_outputs{{ 3 }};
  flatbuffers::Offset<tflite::Operator> op = CreateOperator(
      builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      tflite::BuiltinOptions_TransposeConvOptions,
      transpose_conv_options.Union());

  const std::array<int32_t, 1> graph_inputs{{ 2 }};
  const std::array<int32_t, 1> graph_outputs{{ 3 }};
  flatbuffers::Offset<tflite::SubGraph> subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1),
      builder.CreateString("TransposeConv subgraph"));

  const flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("TransposeConv model");

  const flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
      TFLITE_SCHEMA_VERSION,
      builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1),
      description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  const tflite::Model* model = tflite::GetModel(builder.GetBufferPointer());
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
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

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

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
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "A", "SH", "SW", "D", "Cin", "Cout"});

  /*       N  H   W  KH  KW  PH  PW  A  SH  SW  D  Cin  Cout */
  b->Args({1, 9, 11, 64, 64,  0,  0, 0, 32, 32, 1,  21,  21});
}

// FCN-16 model (PASCAL VOC version).
// We assume CIF image (352x288) on model input / output.
static void FCN16(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "A", "SH", "SW", "D", "Cin", "Cout"});

  /*       N   H   W  KH  KW  PH  PW  A  SH  SW  D  Cin  Cout */
  b->Args({1,  9, 11,  4,  4,  0,  0, 0,  2,  2, 1,  21,  21});
  b->Args({1, 18, 22, 32, 32,  0,  0, 0, 16, 16, 1,  21,  21});
}

// FCN-8 model (PASCAL VOC version).
// We assume CIF image (352x288) on model input / output.
static void FCN8(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "A", "SH", "SW", "D", "Cin", "Cout"});

  /*       N   H   W  KH  KW  PH  PW  A  SH  SW  D  Cin  Cout */
  b->Args({1,  9, 11,  4,  4,  0,  0, 0,  2,  2, 1,  21,  21});
  b->Args({1, 18, 22,  4,  4,  0,  0, 0,  2,  2, 1,  21,  21});
  b->Args({1, 36, 44, 16, 16,  0,  0, 0,  8,  8, 1,  21,  21});
}

static void ENet(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "A", "SH", "SW", "D", "Cin", "Cout"});

  /*********************** Bottleneck 4.0 ***********************/
  /*       N   H    W   KH  KW  PH  PW  A  SH  SW  D  Cin  Cout */
  b->Args({1,  64,  64,  3,  3,  2,  2, 1,  2,  2, 1,  32,  32});
  /*********************** Bottleneck 5.0 ***********************/
  /*       N   H    W   KH  KW  PH  PW  A  SH  SW  D  Cin  Cout */
  b->Args({1, 128, 128,  3,  3,  2,  2, 1,  2,  2, 1,  16,  16});
  /******************* Final Full Convolution *******************/
  /*       N   H    W   KH  KW  PH  PW  A  SH  SH  D  Cin  Cout */
  b->Args({1, 256, 256,  2,  2,  0,  0, 0,  2,  2, 1,  16,  12});
}

static void ESPNet(benchmark::internal::Benchmark* b) {
  b->ArgNames({"N", "H", "W", "KH", "KW", "PH", "PW", "A", "SH", "SW", "D", "Cin", "Cout"});

  /*       N   H    W   KH  KW  PH  PW  A  SH  SW  D  Cin  Cout */
  b->Args({1,  64, 128,  2,  2,  0,  0, 0,  2,  2, 1,  20,  20});
  b->Args({1, 128, 256,  2,  2,  0,  0, 0,  2,  2, 1,  20,  20});
  b->Args({1, 256, 512,  2,  2,  0,  0, 0,  2,  2, 1,  20,  20});
}

BENCHMARK_CAPTURE(xnnpack_deconvolution_f32, fcn32, "FCN-32")
  ->Apply(FCN32)
  ->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_f32, fcn16, "FCN-16")
  ->Apply(FCN16)
  ->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_f32, fcn8, "FCN-8")
  ->Apply(FCN8)
  ->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_f32, enet, "ENet")
  ->Apply(ENet)
  ->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_f32, espnet, "ESPNet")
  ->Apply(ESPNet)
  ->UseRealTime();

BENCHMARK_CAPTURE(xnnpack_deconvolution_qu8, fcn32, "FCN-32")
  ->Apply(FCN32)
  ->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_qu8, fcn16, "FCN-16")
  ->Apply(FCN16)
  ->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_qu8, fcn8, "FCN-8")
  ->Apply(FCN8)
  ->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_qu8, enet, "ENet")
  ->Apply(ENet)
  ->UseRealTime();
BENCHMARK_CAPTURE(xnnpack_deconvolution_qu8, espnet, "ESPNet")
  ->Apply(ESPNet)
  ->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
  BENCHMARK_CAPTURE(tflite_deconvolution_f32, fcn32, "FCN-32")
    ->Apply(FCN32)
    ->UseRealTime();
  BENCHMARK_CAPTURE(tflite_deconvolution_f32, fcn16, "FCN-16")
    ->Apply(FCN16)
    ->UseRealTime();
  BENCHMARK_CAPTURE(tflite_deconvolution_f32, fcn8, "FCN-8")
    ->Apply(FCN8)
    ->UseRealTime();
  BENCHMARK_CAPTURE(tflite_deconvolution_f32, enet, "ENet")
    ->Apply(ENet)
    ->UseRealTime();
  BENCHMARK_CAPTURE(tflite_deconvolution_f32, espnet, "ESPNet")
    ->Apply(ESPNet)
    ->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
