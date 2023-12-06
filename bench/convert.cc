// Copyright 2021 Google LLC
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
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>
#include "bench/utils.h"
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE


void xnnpack_convert_f16_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t> input(batch_size + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::generate(input.begin(), input.end(), std::ref(f16rng));
  std::vector<float> output(batch_size);
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t convert_op = nullptr;
  status = xnn_create_convert_nc_f16_f32(
    0 /* flags */, &convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create F16->F32 Convert operator");
    return;
  }

  status = xnn_reshape_convert_nc_f16_f32(convert_op, batch_size,
    /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape F16->F32 Convert operator");
    return;
  }

  status = xnn_setup_convert_nc_f16_f32(convert_op, input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup F16->F32 Convert operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(convert_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run F16->F32 Convert operator");
      return;
    }
  }

  status = xnn_delete_operator(convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete F16->F32 Convert operator");
    return;
  }
  convert_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = batch_size * (sizeof(uint16_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

void xnnpack_convert_f32_f16(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  std::vector<float> input(batch_size + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<uint16_t> output(batch_size);
  std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t convert_op = nullptr;
  status = xnn_create_convert_nc_f32_f16(
    0 /* flags */, &convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create F32->F16 Convert operator");
    return;
  }

  status = xnn_reshape_convert_nc_f32_f16(convert_op, batch_size,
    /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape F32->F16 Convert operator");
    return;
  }

  status = xnn_setup_convert_nc_f32_f16(convert_op, input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup F32->F16 Convert operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(convert_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run F32->F16 Convert operator");
      return;
    }
  }

  status = xnn_delete_operator(convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete F32->F16 Convert operator");
    return;
  }
  convert_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = batch_size * (sizeof(float) + sizeof(uint16_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

void xnnpack_convert_f32_qs8(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  std::vector<float> input(batch_size + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<int8_t> output(batch_size);
  std::fill(output.begin(), output.end(), 0);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t convert_op = nullptr;
  status = xnn_create_convert_nc_f32_qs8(
    1.0f / 128.0f /* scale */, 1 /* zero point */,
    std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
    0 /* flags */, &convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create F32->QS8 Convert operator");
    return;
  }

  status = xnn_reshape_convert_nc_f32_qs8(convert_op, batch_size,
    /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape F32->QS8 Convert operator");
    return;
  }

  status = xnn_setup_convert_nc_f32_qs8(convert_op, input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup F32->QS8 Convert operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(convert_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run F32->QS8 Convert operator");
      return;
    }
  }

  status = xnn_delete_operator(convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete F32->QS8 Convert operator");
    return;
  }
  convert_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = batch_size * (sizeof(float) + sizeof(int8_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

void xnnpack_convert_f32_qu8(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  std::vector<float> input(batch_size + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<uint8_t> output(batch_size);
  std::fill(output.begin(), output.end(), 0);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t convert_op = nullptr;
  status = xnn_create_convert_nc_f32_qu8(
    1.0f / 128.0f /* scale */, 127 /* zero point */,
    std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max(),
    0 /* flags */, &convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create F32->QU8 Convert operator");
    return;
  }

  status = xnn_reshape_convert_nc_f32_qu8(convert_op, batch_size,
    /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape F32->QU8 Convert operator");
    return;
  }

  status = xnn_setup_convert_nc_f32_qu8(convert_op, input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup F32->QU8 Convert operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(convert_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run F32->QU8 Convert operator");
      return;
    }
  }

  status = xnn_delete_operator(convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete F32->QU8 Convert operator");
    return;
  }
  convert_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = batch_size * (sizeof(float) + sizeof(uint8_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

void xnnpack_convert_qs8(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> input(batch_size + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(input.begin(), input.end(), std::ref(i8rng));
  std::vector<int8_t> output(batch_size);
  std::fill(output.begin(), output.end(), INT8_C(0xAA));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t convert_op = nullptr;
  status = xnn_create_convert_nc_qs8(
    0.75f /* input scale */, -1 /* input zero point */,
    0.5f /* output scale */, 1 /* output zero point */,
    0 /* flags */, &convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create QS8 Convert operator");
    return;
  }

  status = xnn_reshape_convert_nc_qs8(convert_op, batch_size,
    /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape QS8 Convert operator");
    return;
  }

  status = xnn_setup_convert_nc_qs8(convert_op, input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup QS8 Convert operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(convert_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run QS8 Convert operator");
      return;
    }
  }

  status = xnn_delete_operator(convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete QS8 Convert operator");
    return;
  }
  convert_op = nullptr;

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

void xnnpack_convert_qs8_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t> input(batch_size + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(input.begin(), input.end(), std::ref(i8rng));
  std::vector<float> output(batch_size);
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t convert_op = nullptr;
  status = xnn_create_convert_nc_qs8_f32(
    1.0f / 255.0f /* scale */, -128 /* zero point */,
    0 /* flags */, &convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create QS8->F32 Convert operator");
    return;
  }

  status = xnn_reshape_convert_nc_qs8_f32(convert_op, batch_size,
    /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape QS8->F32 Convert operator");
    return;
  }

  status = xnn_setup_convert_nc_qs8_f32(convert_op, input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup QS8->F32 Convert operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(convert_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run QS8->F32 Convert operator");
      return;
    }
  }

  status = xnn_delete_operator(convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete QS8->F32 Convert operator");
    return;
  }
  convert_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = batch_size * (sizeof(int8_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

void xnnpack_convert_qu8(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
    std::ref(rng));

  std::vector<uint8_t> input(batch_size + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<uint8_t> output(batch_size);
  std::fill(output.begin(), output.end(), UINT8_C(0xAA));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t convert_op = nullptr;
  status = xnn_create_convert_nc_qu8(
    0.75f /* scale */, 125 /* zero point */,
    0.5f /* scale */, 130 /* zero point */,
    0 /* flags */, &convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create QU8 Convert operator");
    return;
  }

  status = xnn_reshape_convert_nc_qu8(convert_op, batch_size,
    /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape QU8 Convert operator");
    return;
  }

  status = xnn_setup_convert_nc_qu8(convert_op, input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup QU8 Convert operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(convert_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run QU8 Convert operator");
      return;
    }
  }

  status = xnn_delete_operator(convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete QU8 Convert operator");
    return;
  }
  convert_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(uint8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

void xnnpack_convert_qu8_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
    std::ref(rng));

  std::vector<uint8_t> input(batch_size + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::vector<float> output(batch_size);
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t convert_op = nullptr;
  status = xnn_create_convert_nc_qu8_f32(
    1.0f / 128.0f /* scale */, 128 /* zero point */,
    0 /* flags */, &convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create QU8->F32 Convert operator");
    return;
  }

  status = xnn_reshape_convert_nc_qu8_f32(convert_op, batch_size,
    /*channels=*/1, /*input_stride=*/1, /*output_stride=*/1, /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape QU8->F32 Convert operator");
    return;
  }

  status = xnn_setup_convert_nc_qu8_f32(convert_op, input.data(), output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup QU8->F32 Convert operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(convert_op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run QU8->F32 Convert operator");
      return;
    }
  }

  status = xnn_delete_operator(convert_op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete QU8->F32 Convert operator");
    return;
  }
  convert_op = nullptr;

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = batch_size * (sizeof(uint8_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_TENSORFLOW_LITE
void tflite_convert_f16_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_DEQUANTIZE);

  std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
    tflite::CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 1> shape{{
    static_cast<int32_t>(batch_size)
  }};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 2> tensors{{
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_FLOAT16),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_FLOAT32)
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{0}};
  const std::array<int32_t, 1> graph_outputs{{1}};
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("Dequantize model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
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
  if (interpreterBuilder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  uint16_t* input_data = reinterpret_cast<uint16_t*>(interpreter->tensor(0)->data.data);
  std::generate_n(input_data, batch_size, std::ref(f16rng));

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

  const size_t bytes_per_iteration = batch_size * (sizeof(uint16_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);

  interpreter.reset();
}

void tflite_convert_f32_qs8(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_QUANTIZE);

  std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
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
                         tflite::TensorType_INT8, 0 /* buffer */, 0 /* name */,
                         tflite::CreateQuantizationParameters(builder,
                           0 /*min*/, 0 /*max*/,
                           builder.CreateVector<float>({1.0f / 128.0f /* scale */}),
                           builder.CreateVector<int64_t>({1 /* zero point */})))
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{0}};
  const std::array<int32_t, 1> graph_outputs{{1}};
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("Quantize model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
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
  if (interpreterBuilder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::generate_n(interpreter->typed_tensor<float>(0), batch_size, std::ref(f32rng));

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

  const size_t bytes_per_iteration = batch_size * (sizeof(float) + sizeof(int8_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);

  interpreter.reset();
}

void tflite_convert_f32_qu8(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_QUANTIZE);

  std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
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
                         tflite::TensorType_UINT8, 0 /* buffer */, 0 /* name */,
                         tflite::CreateQuantizationParameters(builder,
                           0 /*min*/, 0 /*max*/,
                           builder.CreateVector<float>({1.0f / 128.0f /* scale */}),
                           builder.CreateVector<int64_t>({127 /* zero point */})))
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{0}};
  const std::array<int32_t, 1> graph_outputs{{1}};
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("Quantize model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
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
  if (interpreterBuilder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::generate_n(interpreter->typed_tensor<float>(0), batch_size, std::ref(f32rng));

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

  const size_t bytes_per_iteration = batch_size * (sizeof(float) + sizeof(uint8_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);

  interpreter.reset();
}

void tflite_convert_qs8(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_QUANTIZE);

  std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
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
                           builder.CreateVector<float>({0.75f /* scale */}),
                           builder.CreateVector<int64_t>({-1 /* zero point */}))),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_INT8, 0 /* buffer */, 0 /* name */,
                         tflite::CreateQuantizationParameters(builder,
                           0 /*min*/, 0 /*max*/,
                           builder.CreateVector<float>({0.5f /* scale */}),
                           builder.CreateVector<int64_t>({1 /* zero point */}))),
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{0}};
  const std::array<int32_t, 1> graph_outputs{{1}};
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("Quantize model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
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
  if (interpreterBuilder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::generate_n(interpreter->typed_tensor<int8_t>(0), batch_size, std::ref(i8rng));

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

void tflite_convert_qs8_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_DEQUANTIZE);

  std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
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
                           builder.CreateVector<float>({1.0f / 255.0f /* scale */}),
                           builder.CreateVector<int64_t>({-128 /* zero point */}))),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_FLOAT32)
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{0}};
  const std::array<int32_t, 1> graph_outputs{{1}};
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("Dequantize model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
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
  if (interpreterBuilder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::generate_n(interpreter->typed_tensor<int8_t>(0), batch_size, std::ref(i8rng));

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

  const size_t bytes_per_iteration = batch_size * (sizeof(int8_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);

  interpreter.reset();
}

void tflite_convert_qu8(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
    std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_QUANTIZE);

  std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
    tflite::CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 1> shape{{
    static_cast<int32_t>(batch_size)
  }};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 2> tensors{{
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_UINT8, 0 /* buffer */, 0 /* name */,
                         tflite::CreateQuantizationParameters(builder,
                           0 /*min*/, 0 /*max*/,
                           builder.CreateVector<float>({0.75f /* scale */}),
                           builder.CreateVector<int64_t>({125 /* zero point */}))),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_UINT8, 0 /* buffer */, 0 /* name */,
                         tflite::CreateQuantizationParameters(builder,
                           0 /*min*/, 0 /*max*/,
                           builder.CreateVector<float>({0.5f /* scale */}),
                           builder.CreateVector<int64_t>({130 /* zero point */})))
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{0}};
  const std::array<int32_t, 1> graph_outputs{{1}};
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("Quantize model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
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
  if (interpreterBuilder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::generate_n(interpreter->typed_tensor<uint8_t>(0), batch_size, std::ref(u8rng));

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

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(uint8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);

  interpreter.reset();
}

void tflite_convert_qu8_f32(benchmark::State& state) {
  const size_t batch_size = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
    std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, tflite::BuiltinOperator_DEQUANTIZE);

  std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
    tflite::CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 1> shape{{
    static_cast<int32_t>(batch_size)
  }};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 2> tensors{{
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_UINT8, 0 /* buffer */, 0 /* name */,
                         tflite::CreateQuantizationParameters(builder,
                           0 /*min*/, 0 /*max*/,
                           builder.CreateVector<float>({1.0f / 128.0f /* scale */}),
                           builder.CreateVector<int64_t>({128 /* zero point */}))),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(shape.data(), shape.size()),
                         tflite::TensorType_FLOAT32)
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{0}};
  const std::array<int32_t, 1> graph_outputs{{1}};
  flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder,
      builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("Dequantize model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
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
  if (interpreterBuilder(&interpreter) != kTfLiteOk || interpreter == nullptr) {
    state.SkipWithError("failed to create TFLite interpreter");
    return;
  }
  interpreter->SetNumThreads(1);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    state.SkipWithError("failed to allocate tensors");
    return;
  }

  std::generate_n(interpreter->typed_tensor<uint8_t>(0), batch_size, std::ref(u8rng));

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

  const size_t bytes_per_iteration = batch_size * (sizeof(uint8_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);

  interpreter.reset();
}
#endif  // BENCHMARK_TENSORFLOW_LITE

BENCHMARK(xnnpack_convert_f16_f32)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_f32_f16)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint16_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_f32_qs8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, int8_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_f32_qu8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_qs8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_qs8_f32)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_qu8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
  ->UseRealTime();
BENCHMARK(xnnpack_convert_qu8_f32)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, float>)
  ->UseRealTime();

#ifdef BENCHMARK_TENSORFLOW_LITE
  BENCHMARK(tflite_convert_f16_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_f32_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, int8_t>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_f32_qu8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_qs8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_qs8_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, float>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_qu8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK(tflite_convert_qu8_f32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, float>)
    ->UseRealTime();
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
