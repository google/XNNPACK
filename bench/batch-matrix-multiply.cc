// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>
#include "bench/utils.h"
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE

void xnnpack_batch_matrix_multiply_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t m = state.range(1);
  const size_t k = state.range(1);
  const size_t n = state.range(1);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

  std::vector<float> input1(batch_size * m * k);
  std::generate(input1.begin(), input1.end(), std::ref(f32rng));
  std::vector<float> input2(batch_size * k * n);
  std::generate(input2.begin(), input2.end(), std::ref(f32rng));
  const size_t output_elements = batch_size * m * n;

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  const size_t num_buffers =
    1 + benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), sizeof(float) * (output_elements));
  std::vector<float> output(output_elements * num_buffers);

  std::vector<xnn_operator_t> ops(num_buffers);

  for (xnn_operator_t& op : ops) {
    status = xnn_create_batch_matrix_multiply_nc_f32(/*flags=*/0, &op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to create FP32 Convolution operator");
      return;
    }
  }

  std::vector<std::unique_ptr<std::vector<char>>> workspaces;

  for (xnn_operator_t& op : ops) {
    size_t workspace_size = 0;
    size_t workspace_alignment = 0;
    status =
      xnn_reshape_batch_matrix_multiply_nc_f32(op, batch_size, m, k, n, &workspace_size, &workspace_alignment, nullptr);

    auto workspace = std::make_unique<std::vector<char>>(workspace_size);
    char* workspace_ptr = workspace->data();

    workspaces.push_back(std::move(workspace));

    if (status != xnn_status_success) {
      state.SkipWithError("failed to create FP32 Convolution operator");
      return;
    }

    status = xnn_setup_batch_matrix_multiply_nc_f32(op, workspace_ptr, input1.data(), input2.data(), output.data());
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    status = xnn_run_operator(ops[buffer_index], /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run FP32 Convolution operator");
      return;
    }
  }

  for (xnn_operator_t& convolution_op : ops) {
    status = xnn_delete_operator(convolution_op);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to delete FP32 Convolution operator");
      return;
    }
    convolution_op = nullptr;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * batch_size * m * k * n,
    benchmark::Counter::kIsRate);
}

#ifdef BENCHMARK_TENSORFLOW_LITE
void tflite_batch_matrix_multiply_f32(benchmark::State& state, const char* net) {
  const size_t batch_size = state.range(0);
  const size_t m = state.range(1);
  const size_t k = state.range(1);
  const size_t n = state.range(1);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

  std::vector<float> input1(batch_size * m * k);
  std::generate(input1.begin(), input1.end(), std::ref(f32rng));
  std::vector<float> input2(batch_size * k * n);
  std::generate(input2.begin(), input2.end(), std::ref(f32rng));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<tflite::OperatorCode> operator_code =
    CreateOperatorCode(builder, tflite::BuiltinOperator_BATCH_MATMUL, 0);

  flatbuffers::Offset<tflite::BatchMatMulOptions> batch_mat_mul_options =
    tflite::CreateBatchMatMulOptions(builder, false, false, false);

  flatbuffers::Offset<tflite::Buffer> buffers[1] = {
    tflite::CreateBuffer(builder, builder.CreateVector({})),
  };

  const int32_t input1_shape[3] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(m),
    static_cast<int32_t>(k),
  };
  const int32_t input2_shape[3] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(k),
    static_cast<int32_t>(n),
  };
  const int32_t output_shape[3] = {
    static_cast<int32_t>(batch_size),
    static_cast<int32_t>(m),
    static_cast<int32_t>(n),
  };

  flatbuffers::Offset<tflite::Tensor> tensors[4] = {
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(input1_shape, 3),
                         tflite::TensorType_FLOAT32,
                         0 /* buffer id */,
                         builder.CreateString("input1")),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(input2_shape, 3),
                         tflite::TensorType_FLOAT32,
                         0 /* buffer id */,
                         builder.CreateString("input2")),
    tflite::CreateTensor(builder,
                         builder.CreateVector<int32_t>(output_shape, 2),
                         tflite::TensorType_FLOAT32,
                         0 /* buffer id */,
                         builder.CreateString("output")),
  };

  const int32_t op_inputs[2] = { 0, 1 };
  const int32_t op_outputs[1] = { 2 };
  flatbuffers::Offset<tflite::Operator> op = CreateOperator(
      builder,
      0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs, 2),
      builder.CreateVector<int32_t>(op_outputs, 1),
      tflite::BuiltinOptions_BatchMatMulOptions,
      batch_mat_mul_options.Union());

  const int32_t graph_inputs[2] = { 0, 1 };
  const int32_t graph_outputs[1] = { 2 };
  flatbuffers::Offset<tflite::SubGraph> subgraph = CreateSubGraph(
      builder,
      builder.CreateVector(tensors, 3),
      builder.CreateVector<int32_t>(graph_inputs, 2),
      builder.CreateVector<int32_t>(graph_outputs, 1),
      builder.CreateVector(&op, 1),
      builder.CreateString("BatchMatMul subgraph"));

  flatbuffers::Offset<flatbuffers::String> description = builder.CreateString("BatchMatMul model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(builder,
      TFLITE_SCHEMA_VERSION,
      builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1),
      description,
      builder.CreateVector(buffers, 1));

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
    interpreter->typed_tensor<float>(0),
    interpreter->typed_tensor<float>(0) + batch_size * m * k,
    std::ref(f32rng));

  std::generate(
    interpreter->typed_tensor<float>(1),
    interpreter->typed_tensor<float>(1) + batch_size * k * n,
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

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * batch_size * m * k * n,
    benchmark::Counter::kIsRate);

  interpreter.reset();
}
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
