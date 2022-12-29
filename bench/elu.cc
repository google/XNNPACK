// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <fp16.h>

#include <xnnpack.h>

#include <benchmark/benchmark.h>
#include "bench/utils.h"
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE


#ifndef XNN_NO_F16_OPERATORS
static void xnnpack_elu_f16(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-20.0f, 20.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t> input(batch_size + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> output(batch_size);
  std::generate(input.begin(), input.end(), std::ref(f16rng));
  std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t elu_op = nullptr;
  status = xnn_create_elu_nc_f16(
    1 /* channels */, 1 /* input stride */, 1 /* output stride */,
    1.0f /* alpha */, 0 /* flags */, &elu_op);
  if (status != xnn_status_success || elu_op == nullptr) {
    state.SkipWithError("failed to create ELU operator");
    return;
  }

  status = xnn_setup_elu_nc_f16(
    elu_op, batch_size,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup ELU operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(elu_op, nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run ELU operator");
      return;
    }
  }

  status = xnn_delete_operator(elu_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete ELU operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(uint16_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}
#endif  // XNN_NO_F16_OPERATORS

static void xnnpack_elu_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-20.0f, 20.0f), std::ref(rng));

  std::vector<float> input(batch_size + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float> output(batch_size);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t elu_op = nullptr;
  status = xnn_create_elu_nc_f32(
    1 /* channels */, 1 /* input stride */, 1 /* output stride */,
    1.0f /* alpha */, 0 /* flags */, &elu_op);
  if (status != xnn_status_success || elu_op == nullptr) {
    state.SkipWithError("failed to create ELU operator");
    return;
  }

  status = xnn_setup_elu_nc_f32(
    elu_op, batch_size,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup ELU operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(elu_op, nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run ELU operator");
      return;
    }
  }

  status = xnn_delete_operator(elu_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete ELU operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#ifndef XNN_NO_QS8_OPERATORS
static void xnnpack_elu_qs8(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> input(batch_size + XNN_EXTRA_BYTES / sizeof(int8_t));
  XNN_PRAGMA("GCC diagnostic push")
  XNN_PRAGMA("GCC diagnostic ignored \"-Walloc-size-larger-than=\"")
  std::vector<int8_t> output(batch_size);
  XNN_PRAGMA("GCC diagnostic pop")
  std::generate(input.begin(), input.end(), std::ref(i8rng));
  std::fill(output.begin(), output.end(), INT8_C(0xA5));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t elu_op = nullptr;
  status = xnn_create_elu_nc_qs8(
    1 /* channels */, 1 /* input stride */, 1 /* output stride */,
    1.0f /* alpha */,
    0 /* input zero point */, 1.0f /* input scale */,
    0 /* output zero point */, 1.0f /* output scale */,
    std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
    0 /* flags */, &elu_op);
  if (status != xnn_status_success || elu_op == nullptr) {
    state.SkipWithError("failed to create ELU operator");
    return;
  }

  status = xnn_setup_elu_nc_qs8(
    elu_op, batch_size,
    input.data(), output.data(),
    nullptr /* thread pool */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup ELU operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(elu_op, nullptr /* thread pool */);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run ELU operator");
      return;
    }
  }

  status = xnn_delete_operator(elu_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete ELU operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(int8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}
#endif  // XNN_NO_QS8_OPERATORS

#ifdef BENCHMARK_TENSORFLOW_LITE
static void tflite_elu_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-20.0f, 20.0f), std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  const flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_ELU);

  const std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
    tflite::CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 1> shape{{
    static_cast<int32_t>(batch_size)
  }};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 2> tensors{{
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_FLOAT32),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_FLOAT32),
  }};

  const std::array<int32_t, 1> op_inputs{{ 0 }};
  const std::array<int32_t, 1> op_outputs{{ 1 }};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(
      builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{ 0 }};
  const std::array<int32_t, 1> graph_outputs{{ 1 }};
  const flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
      TFLITE_SCHEMA_VERSION,
      builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("ELU model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  const tflite::Model* model = tflite::GetModel(builder.GetBufferPointer());
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterBuilder interpreterBuilder(model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (interpreterBuilder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::generate(
    interpreter->typed_tensor<float>(0),
    interpreter->typed_tensor<float>(0) + batch_size,
    std::ref(f32rng));

  for (auto _ : state) {
    if (interpreter->Invoke() != kTfLiteOk) {
      state.SkipWithError("failed to invoke TFLite interpreter");
      return;
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);

  interpreter.reset();
}

static void tflite_elu_qs8(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  const flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_ELU);

  const std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
    tflite::CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 1> shape{{
    static_cast<int32_t>(batch_size)
  }};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 2> tensors{{
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_INT8, 0 /* buffer */, 0 /* name */,
                         tflite::CreateQuantizationParameters(builder,
                           0 /*min*/, 0 /*max*/,
                           builder.CreateVector<float>({1.0f /* scale */}),
                           builder.CreateVector<int64_t>({1 /* zero point */}))),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_INT8, 0 /* buffer */, 0 /* name */,
                         tflite::CreateQuantizationParameters(builder,
                           0 /*min*/, 0 /*max*/,
                           builder.CreateVector<float>({1.0f /* scale */}),
                           builder.CreateVector<int64_t>({1 /* zero point */}))),
  }};

  const std::array<int32_t, 1> op_inputs{{ 0 }};
  const std::array<int32_t, 1> op_outputs{{ 1 }};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(
      builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{ 0 }};
  const std::array<int32_t, 1> graph_outputs{{ 1 }};
  const flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
      TFLITE_SCHEMA_VERSION,
      builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1),
      builder.CreateString("ELU model"),
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  const tflite::Model* model = tflite::GetModel(builder.GetBufferPointer());
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  tflite::InterpreterBuilder interpreterBuilder(model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (interpreterBuilder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::generate(
    interpreter->typed_tensor<int8_t>(0),
    interpreter->typed_tensor<int8_t>(0) + batch_size,
    std::ref(i8rng));

  for (auto _ : state) {
    if (interpreter->Invoke() != kTfLiteOk) {
      state.SkipWithError("failed to invoke TFLite interpreter");
      return;
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(int8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);

  interpreter.reset();
}
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNN_NO_F16_OPERATORS
  BENCHMARK(xnnpack_elu_f16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, uint16_t>)
    ->UseRealTime();
#endif  // XNN_NO_F16_OPERATORS
BENCHMARK(xnnpack_elu_f32)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
#ifndef XNN_NO_QS8_OPERATORS
  BENCHMARK(xnnpack_elu_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_NO_QS8_OPERATORS

#ifdef BENCHMARK_TENSORFLOW_LITE
  BENCHMARK(tflite_elu_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK(tflite_elu_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
