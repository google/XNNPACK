// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>

#include "utils.h"
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include <benchmark/benchmark.h>
#include "pthreadpool.h"

#if XNN_ENABLE_CPUINFO
#include <cpuinfo.h>
#endif  // XNN_ENABLE_CPUINFO

#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/buffer.h"
#include "flatbuffers/include/flatbuffers/flatbuffer_builder.h"
#include "flatbuffers/include/flatbuffers/string.h"
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE

// Pthreadpool-compatible function to wipe the cache in each thread.
void PthreadpoolClearL2Cache(void* context, size_t id) {
#if XNN_ENABLE_CPUINFO
  static const size_t wipe_buffer_size = []() {
    const auto* l2_cache = cpuinfo_get_l2_cache(0);
    return l2_cache == nullptr ? 0 : l2_cache->size;
  }();
  static const char* wipe_buffer = wipe_buffer_size ? [&]() -> char* {
    char* const buff = (char*)malloc(wipe_buffer_size);
    memset(buff, 0xA5, wipe_buffer_size);
    return buff;
  }()
      : nullptr;
  if (wipe_buffer_size) {
    benchmark::utils::PrefetchToL1(wipe_buffer, wipe_buffer_size);
  } else {
    benchmark::utils::WipeCache();
  }
#else
  benchmark::utils::WipeCache();
#endif  // XNN_ENABLE_CPUINFO
}

void xnnpack_batch_matrix_multiply_f32(benchmark::State& state,
                                       const char* net) {
  const size_t batch_size = state.range(0);
  const size_t m = state.range(1);
  const size_t k = state.range(2);
  const size_t n = state.range(3);
  const bool transpose_b = state.range(4);
  const size_t num_threads = state.range(5);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f),
                          std::ref(rng));

  const size_t input1_elements = batch_size * m * k;
  const size_t input2_elements = batch_size * k * n;
  const size_t output_elements = batch_size * m * n;

  xnnpack::Buffer<float> input1(input1_elements +
                                XNN_EXTRA_BYTES / sizeof(float));
  std::generate(input1.begin(), input1.end(), std::ref(f32rng));
  xnnpack::Buffer<float> input2(input2_elements);
  std::generate(input2.begin(), input2.end(), std::ref(f32rng));
  xnnpack::Buffer<float> output(output_elements);

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t op;

  uint32_t flags = transpose_b ? XNN_FLAG_TRANSPOSE_B : 0;
  status = xnn_create_batch_matrix_multiply_nc_f32_const_weights(
      batch_size, k, n, input2.data(), flags, &op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to create FP32 BatchMatrixMultiply operator");
    return;
  }

  pthreadpool_t threadpool = pthreadpool_create(num_threads);

  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  status = xnn_reshape_batch_matrix_multiply_nc_f32(
      op, /*num_batch_dims=*/1, /*batch_dims_a=*/&batch_size,
      /*batch_dims_b=*/&batch_size, m, k, n, &workspace_size,
      &workspace_alignment, threadpool);

  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape FP32 BatchMatrixMultiply operator");
    return;
  }

  status = xnn_setup_batch_matrix_multiply_nc_f32(
      op, /*workspace=*/nullptr, input1.data(),
      /*input_b=*/nullptr, output.data());

  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup FP32 BatchMatrixMultiply operator");
    return;
  }

  for (auto _ : state) {
    state.PauseTiming();
    pthreadpool_parallelize_1d(threadpool, PthreadpoolClearL2Cache, nullptr,
                               num_threads, 0);
    state.ResumeTiming();

    status = xnn_run_operator(op, threadpool);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run FP32 BatchMatrixMultiply operator");
      return;
    }
  }

  status = xnn_delete_operator(op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete FP32 BatchMatrixMultiply operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * batch_size * m * k * n,
      benchmark::Counter::kIsRate);

  pthreadpool_destroy(threadpool);
}

void xnnpack_batch_matrix_multiply_qd8_f32_qc8w(benchmark::State& state,
                                                const char* net) {
  const size_t batch_size = state.range(0);
  const size_t m = state.range(1);
  const size_t k = state.range(2);
  const size_t n = state.range(3);
  const bool transpose_b = state.range(4);
  const size_t num_threads = state.range(5);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f),
                          std::ref(rng));
  auto q8rng = std::bind(
      std::uniform_int_distribution<int8_t>(std::numeric_limits<int8_t>::min(),
                                            std::numeric_limits<int8_t>::max()),
      std::ref(rng));

  const size_t input1_elements = batch_size * m * k;
  const size_t input2_elements = batch_size * k * n;
  const size_t output_elements = batch_size * m * n;

  xnnpack::Buffer<int8_t> input1(input1_elements +
                                 XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(input1.begin(), input1.end(), std::ref(q8rng));
  xnnpack::Buffer<int8_t> input2(input2_elements);
  std::generate(input2.begin(), input2.end(), std::ref(q8rng));
  xnnpack::Buffer<float> output(output_elements);

  // Allocate and fill the quantization parameters.
  xnnpack::Buffer<float> channelwise_scales(batch_size * n +
                                            XNN_EXTRA_BYTES / sizeof(float));
  std::generate(channelwise_scales.begin(), channelwise_scales.end(),
                std::ref(f32rng));
  xnnpack::Buffer<xnn_quantization_params> quantization_params(
      batch_size * m + XNN_EXTRA_QUANTIZATION_PARAMS);
  for (int i = 0; i < batch_size * m; i++) {
    quantization_params[i] = {.zero_point = q8rng(), .scale = f32rng()};
  }

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t op;

  uint32_t flags = transpose_b ? XNN_FLAG_TRANSPOSE_B : 0;
  status = xnn_create_batch_matrix_multiply_nc_qd8_f32_qc8w(
      batch_size, k, n, input2.data(), channelwise_scales.data(), flags, &op);
  if (status != xnn_status_success) {
    state.SkipWithError(
        "failed to create QD8_F32_QC8W BatchMatrixMultiply operator");
    return;
  }

  pthreadpool_t threadpool = pthreadpool_create(num_threads);

  status = xnn_reshape_batch_matrix_multiply_nc_qd8_f32_qc8w(
      op, /*num_batch_dims=*/1, /*batch_dims_a=*/&batch_size,
      /*batch_dims_b=*/&batch_size, m, k, n, threadpool);

  if (status != xnn_status_success) {
    state.SkipWithError(
        "failed to reshape QD8_F32_QC8W BatchMatrixMultiply operator");
    return;
  }

  status = xnn_setup_batch_matrix_multiply_nc_qd8_f32_qc8w(
      op, input1.data(), quantization_params.data(), output.data());

  if (status != xnn_status_success) {
    state.SkipWithError(
        "failed to setup QD8_F32_QC8W BatchMatrixMultiply operator");
    return;
  }

  for (auto _ : state) {
    state.PauseTiming();
    pthreadpool_parallelize_1d(threadpool, PthreadpoolClearL2Cache, nullptr,
                               num_threads, 0);
    state.ResumeTiming();

    status = xnn_run_operator(op, threadpool);
    if (status != xnn_status_success) {
      state.SkipWithError(
          "failed to run QD8_F32_QC8W BatchMatrixMultiply operator");
      return;
    }
  }

  status = xnn_delete_operator(op);
  if (status != xnn_status_success) {
    state.SkipWithError(
        "failed to delete QD8_F32_QC8W BatchMatrixMultiply operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * batch_size * m * k * n,
      benchmark::Counter::kIsRate);

  pthreadpool_destroy(threadpool);
}

#ifdef BENCHMARK_TENSORFLOW_LITE
void tflite_batch_matrix_multiply_f32(benchmark::State& state,
                                      const char* net) {
  const size_t batch_size = state.range(0);
  const size_t m = state.range(1);
  const size_t k = state.range(1);
  const size_t n = state.range(1);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f),
                          std::ref(rng));

  xnnpack::Buffer<float> input1(batch_size * m * k);
  std::generate(input1.begin(), input1.end(), std::ref(f32rng));
  xnnpack::Buffer<float> input2(batch_size * k * n);
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
                           tflite::TensorType_FLOAT32, 0 /* buffer id */,
                           builder.CreateString("input1")),
      tflite::CreateTensor(builder,
                           builder.CreateVector<int32_t>(input2_shape, 3),
                           tflite::TensorType_FLOAT32, 0 /* buffer id */,
                           builder.CreateString("input2")),
      tflite::CreateTensor(builder,
                           builder.CreateVector<int32_t>(output_shape, 2),
                           tflite::TensorType_FLOAT32, 0 /* buffer id */,
                           builder.CreateString("output")),
  };

  const int32_t op_inputs[2] = {0, 1};
  const int32_t op_outputs[1] = {2};
  flatbuffers::Offset<tflite::Operator> op = CreateOperator(
      builder, 0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs, 2),
      builder.CreateVector<int32_t>(op_outputs, 1),
      tflite::BuiltinOptions_BatchMatMulOptions, batch_mat_mul_options.Union());

  const int32_t graph_inputs[2] = {0, 1};
  const int32_t graph_outputs[1] = {2};
  flatbuffers::Offset<tflite::SubGraph> subgraph =
      CreateSubGraph(builder, builder.CreateVector(tensors, 3),
                     builder.CreateVector<int32_t>(graph_inputs, 2),
                     builder.CreateVector<int32_t>(graph_outputs, 1),
                     builder.CreateVector(&op, 1),
                     builder.CreateString("BatchMatMul subgraph"));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("BatchMatMul model");

  flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), description,
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

  std::generate(interpreter->typed_tensor<float>(0),
                interpreter->typed_tensor<float>(0) + batch_size * m * k,
                std::ref(f32rng));

  std::generate(interpreter->typed_tensor<float>(1),
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
      static_cast<uint64_t>(state.iterations()) * batch_size * m * k * n,
      benchmark::Counter::kIsRate);

  interpreter.reset();
}
#endif  // BENCHMARK_TENSORFLOW_LITE

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
