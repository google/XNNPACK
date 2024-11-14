// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "utils.h"
#include "xnnpack.h"
#include "xnnpack/datatype.h"
#include "xnnpack/buffer.h"
#include "xnnpack/math.h"
#include <benchmark/benchmark.h>
#ifdef BENCHMARK_TENSORFLOW_LITE
#include "flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#endif  // BENCHMARK_TENSORFLOW_LITE

void init_params(xnn_binary_operator op_type, xnn_datatype datatype,
                 xnn_binary_params& params,
                 xnn_quantization_params& input_quantization,
                 xnn_quantization_params& output_quantization) {
  switch (op_type) {
    case xnn_datatype_qint8:
      input_quantization = {0, 1.0f / 128.0f};
      output_quantization = {128, 1.0f / 128.0f};
      break;
    case xnn_datatype_quint8:
      input_quantization = {128, 1.0f / 128.0f};
      output_quantization = {0, 1.0f / 256.0f};
      break;
    default:
      input_quantization = {0, 1.0f};
      output_quantization = {0, 1.0f};
      break;
  }
}

template <typename T>
static void benchmark_binary_operator(benchmark::State& state,
                                      xnn_binary_operator op_type) {
  const size_t batch_size = state.range(0);

  state.SetItemsProcessed(batch_size);
  state.SetBytesProcessed(batch_size * (sizeof(T) * 3));

  xnn_binary_params params;
  xnn_quantization_params input_quantization;
  xnn_quantization_params output_quantization;
  init_params(op_type, xnn_datatype_of<T>(), params, input_quantization,
              output_quantization);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32dist = std::uniform_real_distribution<float>(
      std::max<float>(std::numeric_limits<T>::lowest(), -128.0f),
      std::min<float>(std::numeric_limits<T>::max(), 127.0f));

  xnnpack::Buffer<T> input1(batch_size + XNN_EXTRA_BYTES / sizeof(T));
  xnnpack::Buffer<T> input2(batch_size + XNN_EXTRA_BYTES / sizeof(T));
  xnnpack::Buffer<T> output(batch_size);
  std::generate(input1.begin(), input1.end(), [&]() { return f32dist(rng); });
  std::generate(input2.begin(), input2.end(), [&]() { return f32dist(rng); });

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t op = nullptr;
  status = xnn_create_binary_elementwise_nd(
      op_type, xnn_datatype_of<T>(), &input_quantization, &input_quantization,
      &output_quantization, /*flags*/ 0, &op);
  if (status != xnn_status_success || op == nullptr) {
    state.SkipWithError("failed to create operator");
    return;
  }

  const size_t input_shape[] = {batch_size};
  status =
      xnn_reshape_binary_elementwise_nd(op, /*num_input1_dims*/ 1, input_shape,
                                        /*num_input2_dims*/ 1, input_shape,
                                        /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape operator");
    return;
  }

  status = xnn_setup_binary_elementwise_nd(op, input1.data(), input2.data(),
                                           output.data());
  if (status != xnn_status_success) {
    state.SkipWithError("failed to setup operator");
    return;
  }

  for (auto _ : state) {
    status = xnn_run_operator(op, /*threadpool=*/nullptr);
    if (status != xnn_status_success) {
      state.SkipWithError("failed to run operator");
      return;
    }
  }

  status = xnn_delete_operator(op);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to delete operator");
    return;
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

#ifdef BENCHMARK_TENSORFLOW_LITE
tflite::BuiltinOperator xnn_binary_operator_to_tflite(xnn_binary_operator op) {
  switch (op) {
    case xnn_binary_add:
      return tflite::BuiltinOperator_STABLEHLO_ADD;
    case xnn_binary_subtract:
      return tflite::BuiltinOperator_STABLEHLO_SUBTRACT;
    case xnn_binary_multiply:
      return tflite::BuiltinOperator_STABLEHLO_MULTIPLY;
    case xnn_binary_divide:
      return tflite::BuiltinOperator_STABLEHLO_DIVIDE;
    case xnn_binary_maximum:
      return tflite::BuiltinOperator_STABLEHLO_MAXIMUM;
    case xnn_binary_minimum:
      return tflite::BuiltinOperator_STABLEHLO_MINIMUM;
    case xnn_binary_copysign:
      return tflite::BuiltinOperator_CUSTOM;  // No corresponding TFlite op
    case xnn_binary_squared_difference:
      return tflite::BuiltinOperator_SQUARED_DIFFERENCE;
    case xnn_binary_prelu:
      return tflite::BuiltinOperator_PRELU;
    case xnn_binary_modulus:
      return tflite::BuiltinOperator_CUSTOM;  // No corresponding TFlite op
    case xnn_binary_atan2:
      return tflite::BuiltinOperator_ATAN2;
    case xnn_binary_pow:
      return tflite::BuiltinOperator_STABLEHLO_POWER;
    case xnn_binary_bitwise_and:
      return tflite::BuiltinOperator_STABLEHLO_AND;
    case xnn_binary_bitwise_or:
      return tflite::BuiltinOperator_STABLEHLO_OR;
    case xnn_binary_bitwise_xor:
      return tflite::BuiltinOperator_BITWISE_XOR;
    case xnn_binary_shift_left:
      return tflite::BuiltinOperator_STABLEHLO_SHIFT_LEFT;
    case xnn_binary_shift_right_logical:
      // TODO: BuiltinOperator_RIGHT_SHIFT is logical only for unsigned types
      return tflite::BuiltinOperator_RIGHT_SHIFT;
    case xnn_binary_shift_right_arithmetic:
      // TODO: BuiltinOperator_RIGHT_SHIFT is arithmetic only for unsigned types
      return tflite::BuiltinOperator_RIGHT_SHIFT;
    default:
      XNN_UNREACHABLE;
      return tflite::BuiltinOperator_CUSTOM;
  }
}

template <typename T>
struct TypeToTfliteType {
  using type = T;
  static constexpr auto tensor_type =
      tflite::TensorTypeFor<typename xnnpack::unwrap_quantized<T>::type>::value;
};
template <>
struct TypeToTfliteType<xnn_float16> {
  using type = TfLiteFloat16;
  static constexpr auto tensor_type = tflite::TensorType_FLOAT16;
};
template <>
struct TypeToTfliteType<xnn_bfloat16> {
  using type = TfLiteBFloat16;
  static constexpr auto tensor_type = tflite::TensorType_BFLOAT16;
};

template <typename T, class BuildInQuantization, class BuildOutQuantization>
static void benchmark_tflite_binary_operator(
    benchmark::State& state, BuildInQuantization in_quantization,
    BuildOutQuantization out_quantization, tflite::BuiltinOperator op_code) {
  const size_t batch_size = state.range(0);

  flatbuffers::FlatBufferBuilder builder;
  const flatbuffers::Offset<tflite::OperatorCode> operator_code =
      CreateOperatorCode(builder, op_code);

  const std::array<flatbuffers::Offset<tflite::Buffer>, 1> buffers{{
      tflite::CreateBuffer(builder, builder.CreateVector({})),
  }};

  const std::array<int32_t, 1> shape{{static_cast<int32_t>(batch_size)}};

  const std::array<flatbuffers::Offset<tflite::Tensor>, 3> tensors{{
      tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>(shape.data(), shape.size()),
          TypeToTfliteType<T>::tensor_type,
          /*buffer=*/0, /*name=*/0, in_quantization(builder)),
      tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>(shape.data(), shape.size()),
          TypeToTfliteType<T>::tensor_type,
          /*buffer=*/0, /*name=*/0, in_quantization(builder)),
      tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>(shape.data(), shape.size()),
          TypeToTfliteType<T>::tensor_type,
          /*buffer=*/0, /*name=*/0, out_quantization(builder)),
  }};

  const std::array<int32_t, 2> op_inputs{{0, 1}};
  const std::array<int32_t, 1> op_outputs{{2}};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(
      builder, 0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 2> graph_inputs{{0, 1}};
  const std::array<int32_t, 1> graph_outputs{{2}};
  const flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("Binary model"),
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

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32dist = std::uniform_real_distribution<float>(
      std::max<float>(std::numeric_limits<T>::lowest(), -128.0f),
      std::min<float>(std::numeric_limits<T>::max(), 127.0f));
  T* input_ptr = reinterpret_cast<T*>(interpreter->tensor(0)->data.raw);
  std::generate(input_ptr, input_ptr + batch_size,
                [&]() { return f32dist(rng); });

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

  state.counters["elements"] = benchmark::Counter(
      uint64_t(state.iterations()) * batch_size, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * batch_size * sizeof(float);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);

  interpreter.reset();
}

static flatbuffers::Offset<tflite::QuantizationParameters> no_quantization(
    flatbuffers::FlatBufferBuilder& builder) {
  return 0;
}

static auto CreateTfLiteQuantizationParameters(
    flatbuffers::FlatBufferBuilder& builder,
    const xnn_quantization_params& params) {
  return tflite::CreateQuantizationParameters(
      builder, 0 /*min*/, 0 /*max*/,
      builder.CreateVector<float>({params.scale}),
      builder.CreateVector<int64_t>({params.zero_point}));
}

template <typename T>
static void benchmark_tflite_binary_operator(benchmark::State& state,
                                             xnn_binary_operator op) {
  xnn_binary_params params;
  xnn_quantization_params input_quantization;
  xnn_quantization_params output_quantization;
  init_params(op, xnn_datatype_of<T>(), params, input_quantization,
              output_quantization);
  auto in_quantization = [=](flatbuffers::FlatBufferBuilder& builder) {
    return CreateTfLiteQuantizationParameters(builder, input_quantization);
  };
  auto out_quantization = [=](flatbuffers::FlatBufferBuilder& builder) {
    return CreateTfLiteQuantizationParameters(builder, output_quantization);
  };

  tflite::BuiltinOperator op_code = xnn_binary_operator_to_tflite(op);
  if (op_code == tflite::BuiltinOperator_CUSTOM) {
    state.SkipWithMessage("no corresponding TFLite operator");
    return;
  }
  if (!xnnpack::is_quantized<T>::value) {
    return benchmark_tflite_binary_operator<T>(state, no_quantization,
                                               no_quantization, op_code);
  } else {
    return benchmark_tflite_binary_operator<T>(state, in_quantization,
                                               out_quantization, op_code);
  }
}

#define BENCHMARK_OP_TYPE(op, type_name, type)                           \
  void xnnpack_##op##_##type_name(benchmark::State& state) {             \
    benchmark_binary_operator<type>(state, xnn_binary_##op);             \
  }                                                                      \
  void tflite_##op##_##type_name(benchmark::State& state) {              \
    benchmark_tflite_binary_operator<type>(state, xnn_binary_##op);      \
  }                                                                      \
  BENCHMARK(xnnpack_##op##_##type_name)                                  \
      ->Apply(benchmark::utils::BinaryElementwiseParameters<type, type>) \
      ->UseRealTime();                                                   \
  BENCHMARK(tflite_##op##_##type_name)                                   \
      ->Apply(benchmark::utils::BinaryElementwiseParameters<type, type>) \
      ->UseRealTime();

#else  // BENCHMARK_TENSORFLOW_LITE

#define BENCHMARK_OP_TYPE(op, type_name, type)                           \
  void xnnpack_##op##_##type_name(benchmark::State& state) {             \
    benchmark_binary_operator<type>(state, xnn_binary_##op);             \
  }                                                                      \
  BENCHMARK(xnnpack_##op##_##type_name)                                  \
      ->Apply(benchmark::utils::BinaryElementwiseParameters<type, type>) \
      ->UseRealTime();

#endif  // BENCHMARK_TENSORFLOW_LITE

#define BENCHMARK_OP_REAL(op)                            \
  BENCHMARK_OP_TYPE(op, f32, float)                      \
  BENCHMARK_OP_TYPE(op, f16, xnn_float16)                \
  BENCHMARK_OP_TYPE(op, bf16, xnn_bfloat16)              \
  BENCHMARK_OP_TYPE(op, qs8, xnnpack::quantized<int8_t>) \
  BENCHMARK_OP_TYPE(op, qu8, xnnpack::quantized<uint8_t>)

#define BENCHMARK_OP_INTEGRAL(op) BENCHMARK_OP_TYPE(op, s32, int32_t)

#define BENCHMARK_OP_ALL(op) \
  BENCHMARK_OP_REAL(op)      \
  BENCHMARK_OP_INTEGRAL(op)

BENCHMARK_OP_ALL(add);
BENCHMARK_OP_ALL(subtract);
BENCHMARK_OP_ALL(multiply);
BENCHMARK_OP_ALL(divide);
BENCHMARK_OP_ALL(maximum);
BENCHMARK_OP_ALL(minimum);
// BENCHMARK_OP_ALL(copysign);  // Missing in TFLite
BENCHMARK_OP_REAL(squared_difference);
BENCHMARK_OP_REAL(prelu);
// BENCHMARK_OP_ALL(modulus);  // Missing in TFLite
BENCHMARK_OP_REAL(atan2);
BENCHMARK_OP_ALL(pow);
BENCHMARK_OP_INTEGRAL(bitwise_and);
BENCHMARK_OP_INTEGRAL(bitwise_or);
BENCHMARK_OP_INTEGRAL(bitwise_xor);
BENCHMARK_OP_INTEGRAL(shift_left);
BENCHMARK_OP_INTEGRAL(shift_right_logical);
BENCHMARK_OP_INTEGRAL(shift_right_arithmetic);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
