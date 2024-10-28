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

void init_params(xnn_unary_operator op, xnn_datatype in_type,
                 xnn_datatype out_type, xnn_unary_params& params,
                 xnn_quantization_params& input_quantization,
                 xnn_quantization_params& output_quantization) {
  switch (op) {
    case xnn_unary_clamp:
      params.clamp.min = -10.0f;
      params.clamp.max = 10.0f;
      break;
    case xnn_unary_elu:
      params.elu.alpha = 0.5f;
      break;
    case xnn_unary_leaky_relu:
      params.leaky_relu.negative_slope = 0.5f;
      break;
    case xnn_unary_tanh:
      switch (out_type) {
        case xnn_datatype_qint8:
          output_quantization = {0, 1.0f / 128.0f};
          break;
        case xnn_datatype_quint8:
          output_quantization = {128, 1.0f / 128.0f};
          break;
        default:
          break;
      }
      break;
    case xnn_unary_sigmoid:
      switch (out_type) {
        case xnn_datatype_qint8:
          output_quantization = {-128, 1.0f / 256.0f};
          break;
        case xnn_datatype_quint8:
          output_quantization = {0, 1.0f / 256.0f};
          break;
        default:
          break;
      }
      break;
    default:
      break;
  }
  switch (in_type) {
    case xnn_datatype_qint8:
      input_quantization = {0, 1.0f / 128.0f};
      break;
    case xnn_datatype_quint8:
      input_quantization = {128, 1.0f / 128.0f};
      break;
    default:
      break;
  }
  switch (out_type) {
    case xnn_datatype_qint8:
      output_quantization = {128, 1.0f / 128.0f};
      break;
    case xnn_datatype_quint8:
      output_quantization = {0, 1.0f / 256.0f};
      break;
    default:
      break;
  }
}

template <typename In, typename Out>
static void benchmark_unary_operator(benchmark::State& state,
                                     xnn_unary_operator op_type) {
  const size_t batch_size = state.range(0);

  state.SetItemsProcessed(batch_size);
  state.SetBytesProcessed(batch_size * (sizeof(In) + sizeof(Out)));

  xnn_unary_params params;
  xnn_quantization_params input_quantization = {0, 1.0f};
  xnn_quantization_params output_quantization = {0, 1.0f};
  init_params(op_type, xnn_datatype_of<In>(), xnn_datatype_of<Out>(),
              params, input_quantization, output_quantization);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32dist = std::uniform_real_distribution<float>(
      std::max<float>(std::numeric_limits<In>::lowest(), -128.0f),
      std::min<float>(std::numeric_limits<In>::max(), 127.0f));

  xnnpack::Buffer<In> input(batch_size + XNN_EXTRA_BYTES / sizeof(In));
  xnnpack::Buffer<Out> output(batch_size);
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

  xnn_status status = xnn_initialize(nullptr /* allocator */);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  xnn_operator_t op = nullptr;
  status = xnn_create_unary_elementwise_nc(
      op_type, xnn_datatype_of<In>(), xnn_datatype_of<Out>(), &params,
      &input_quantization, &output_quantization, 0 /* flags */, &op);
  if (status != xnn_status_success || op == nullptr) {
    state.SkipWithError("failed to create operator");
    return;
  }

  status = xnn_reshape_unary_elementwise_nc(op, batch_size,
                                            /*channels=*/1, /*input_stride=*/1,
                                            /*output_stride=*/1,
                                            /*threadpool=*/nullptr);
  if (status != xnn_status_success) {
    state.SkipWithError("failed to reshape operator");
    return;
  }

  status = xnn_setup_unary_elementwise_nc(op, input.data(), output.data());
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

template <typename In, typename Out>
static void benchmark_convert(benchmark::State& state) {
  benchmark_unary_operator<In, Out>(state, xnn_unary_convert);
}

#ifdef BENCHMARK_TENSORFLOW_LITE

tflite::BuiltinOperator xnn_unary_operator_to_tflite(xnn_unary_operator op) {
  switch (op) {
    case xnn_unary_convert:
      return tflite::BuiltinOperator_CAST;
    case xnn_unary_clamp:
      return tflite::BuiltinOperator_STABLEHLO_CLAMP;
    case xnn_unary_abs:
      return tflite::BuiltinOperator_ABS;
    case xnn_unary_bankers_rounding:
      return tflite::BuiltinOperator_ROUND;
    case xnn_unary_ceiling:
      return tflite::BuiltinOperator_CEIL;
    case xnn_unary_elu:
      return tflite::BuiltinOperator_ELU;
    case xnn_unary_exp:
      return tflite::BuiltinOperator_EXP;
    case xnn_unary_floor:
      return tflite::BuiltinOperator_FLOOR;
    case xnn_unary_gelu:
      return tflite::BuiltinOperator_GELU;
    case xnn_unary_hardswish:
      return tflite::BuiltinOperator_HARD_SWISH;
    case xnn_unary_leaky_relu:
      return tflite::BuiltinOperator_LEAKY_RELU;
    case xnn_unary_log:
      return tflite::BuiltinOperator_LOG;
    case xnn_unary_negate:
      return tflite::BuiltinOperator_NEG;
    case xnn_unary_sigmoid:
      return tflite::BuiltinOperator_LOGISTIC;
    case xnn_unary_square:
      return tflite::BuiltinOperator_SQUARE;
    case xnn_unary_square_root:
      return tflite::BuiltinOperator_SQRT;
    case xnn_unary_reciprocal_square_root:
      return tflite::BuiltinOperator_RSQRT;
    case xnn_unary_tanh:
      return tflite::BuiltinOperator_TANH;
    case xnn_unary_invalid:
      break;
  }
  XNN_UNREACHABLE;
  return tflite::BuiltinOperator_CUSTOM;
}

template <typename T>
struct TypeToTfliteType {
  using type = T;
};
template <>
struct TypeToTfliteType<xnn_float16> {
  using type = TfLiteFloat16;
};
template <>
struct TypeToTfliteType<xnn_bfloat16> {
  using type = TfLiteBFloat16;
};

template <typename In, typename Out, class BuildInQuantization,
          class BuildOutQuantization>
static void benchmark_tflite_unary_operator(
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

  const std::array<flatbuffers::Offset<tflite::Tensor>, 2> tensors{{
      tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>(shape.data(), shape.size()),
          tflite::TensorTypeFor<typename TypeToTfliteType<In>::type>::value,
          /*buffer=*/0, /*name=*/0, in_quantization(builder)),
      tflite::CreateTensor(
          builder, builder.CreateVector<int32_t>(shape.data(), shape.size()),
          tflite::TensorTypeFor<typename TypeToTfliteType<Out>::type>::value,
          /*buffer=*/0, /*name=*/0, out_quantization(builder)),
  }};

  const std::array<int32_t, 1> op_inputs{{0}};
  const std::array<int32_t, 1> op_outputs{{1}};
  flatbuffers::Offset<tflite::Operator> op = tflite::CreateOperator(
      builder, 0 /* opcode_index */,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()));

  const std::array<int32_t, 1> graph_inputs{{0}};
  const std::array<int32_t, 1> graph_outputs{{1}};
  const flatbuffers::Offset<tflite::SubGraph> subgraph = tflite::CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(graph_inputs.data(), graph_inputs.size()),
      builder.CreateVector<int32_t>(graph_outputs.data(), graph_outputs.size()),
      builder.CreateVector(&op, 1));

  const flatbuffers::Offset<tflite::Model> model_buffer = tflite::CreateModel(
      builder, TFLITE_SCHEMA_VERSION, builder.CreateVector(&operator_code, 1),
      builder.CreateVector(&subgraph, 1), builder.CreateString("Abs model"),
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
      std::max<float>(std::numeric_limits<In>::lowest(), -128.0f),
      std::min<float>(std::numeric_limits<In>::max(), 127.0f));
  In* input_ptr = reinterpret_cast<In*>(
      interpreter->typed_tensor<typename TypeToTfliteType<In>::type>(0));
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

bool is_quantized(int8_t) { return true; }
bool is_quantized(uint8_t) { return true; }
bool is_quantized(xnn_float16) { return false; }
bool is_quantized(float) { return false; }

template <typename In, typename Out>
static void benchmark_tflite_unary_operator(benchmark::State& state,
                                            xnn_unary_operator op) {
  xnn_unary_params params;
  xnn_quantization_params input_quantization = {0, 1.0f};
  xnn_quantization_params output_quantization = {0, 1.0f};
  init_params(op, xnn_datatype_of<In>(), xnn_datatype_of<Out>(),
              params, input_quantization, output_quantization);
  auto in_quantization = [=](flatbuffers::FlatBufferBuilder& builder) {
    return CreateTfLiteQuantizationParameters(builder, input_quantization);
  };
  auto out_quantization = [=](flatbuffers::FlatBufferBuilder& builder) {
    return CreateTfLiteQuantizationParameters(builder, output_quantization);
  };

  if (!is_quantized(In()) && !is_quantized(Out())) {
    tflite::BuiltinOperator op_code = xnn_unary_operator_to_tflite(op);
    return benchmark_tflite_unary_operator<In, Out>(state, no_quantization,
                                                    no_quantization, op_code);
  } else if (is_quantized(In()) && !is_quantized(Out())) {
    assert(op == xnn_unary_convert);
    return benchmark_tflite_unary_operator<In, Out>(
        state, in_quantization, no_quantization,
        tflite::BuiltinOperator_DEQUANTIZE);
  } else if (!is_quantized(In()) && is_quantized(Out())) {
    assert(op == xnn_unary_convert);
    return benchmark_tflite_unary_operator<In, Out>(
        state, no_quantization, out_quantization,
        tflite::BuiltinOperator_QUANTIZE);
  } else if (is_quantized(In()) && is_quantized(Out())) {
    tflite::BuiltinOperator op_code;
    if (op == xnn_unary_convert) {
      op_code = tflite::BuiltinOperator_QUANTIZE;
    } else {
      op_code = xnn_unary_operator_to_tflite(op);
    }
    return benchmark_tflite_unary_operator<In, Out>(state, in_quantization,
                                                    out_quantization, op_code);
  }
}

template <typename In, typename Out>
static void benchmark_tflite_convert(benchmark::State& state) {
  benchmark_tflite_unary_operator<In, Out>(state, xnn_unary_convert);
}

#define BENCHMARK_OP_TYPE(op, type_name, type)                          \
  void xnnpack_##op##_##type_name(benchmark::State& state) {            \
    benchmark_unary_operator<type, type>(state, xnn_unary_##op);        \
  }                                                                     \
  void tflite_##op##_##type_name(benchmark::State& state) {             \
    benchmark_tflite_unary_operator<type, type>(state, xnn_unary_##op); \
  }                                                                     \
  BENCHMARK(xnnpack_##op##_##type_name)                                 \
      ->Apply(benchmark::utils::UnaryElementwiseParameters<type, type>) \
      ->UseRealTime();                                                  \
  BENCHMARK(tflite_##op##_##type_name)                                  \
      ->Apply(benchmark::utils::UnaryElementwiseParameters<type, type>) \
      ->UseRealTime();

#define BENCHMARK_CONVERT(name, in, out)                             \
  BENCHMARK_TEMPLATE(benchmark_convert, in, out)                     \
      ->Apply(benchmark::utils::UnaryElementwiseParameters<in, out>) \
      ->Name("xnnpack_convert_" #name)                               \
      ->UseRealTime();                                               \
  BENCHMARK_TEMPLATE(benchmark_tflite_convert, in, out)              \
      ->Apply(benchmark::utils::UnaryElementwiseParameters<in, out>) \
      ->Name("tflite_convert_" #name)                                \
      ->UseRealTime()

#else  // BENCHMARK_TENSORFLOW_LITE

#define BENCHMARK_OP_TYPE(op, type_name, type)                          \
  void xnnpack_##op##_##type_name(benchmark::State& state) {            \
    benchmark_unary_operator<type, type>(state, xnn_unary_##op);        \
  }                                                                     \
  BENCHMARK(xnnpack_##op##_##type_name)                                 \
      ->Apply(benchmark::utils::UnaryElementwiseParameters<type, type>) \
      ->UseRealTime();

#define BENCHMARK_CONVERT(name, in, out)                             \
  BENCHMARK_TEMPLATE(benchmark_convert, in, out)                     \
      ->Apply(benchmark::utils::UnaryElementwiseParameters<in, out>) \
      ->Name("xnnpack_" #name)                                       \
      ->UseRealTime()

#endif  // BENCHMARK_TENSORFLOW_LITE

#define BENCHMARK_OP(op)                  \
  BENCHMARK_OP_TYPE(op, f32, float)       \
  BENCHMARK_OP_TYPE(op, f16, xnn_float16) \
  BENCHMARK_OP_TYPE(op, qs8, int8_t)      \
  BENCHMARK_OP_TYPE(op, qu8, uint8_t)

BENCHMARK_OP(clamp);
BENCHMARK_OP(abs);
BENCHMARK_OP(bankers_rounding);
BENCHMARK_OP(ceiling);
BENCHMARK_OP(elu);
BENCHMARK_OP(exp);
BENCHMARK_OP(floor);
BENCHMARK_OP(gelu);
BENCHMARK_OP(hardswish);
BENCHMARK_OP(leaky_relu);
BENCHMARK_OP(log);
BENCHMARK_OP(negate);
BENCHMARK_OP(sigmoid);
BENCHMARK_OP(square);
BENCHMARK_OP(square_root);
BENCHMARK_OP(reciprocal_square_root);
BENCHMARK_OP(tanh);

BENCHMARK_CONVERT(qs8_qs8, int8_t, int8_t);
BENCHMARK_CONVERT(qs8_qu8, int8_t, uint8_t);
BENCHMARK_CONVERT(qs8_f16, int8_t, xnn_float16);
BENCHMARK_CONVERT(qs8_f32, int8_t, float);

BENCHMARK_CONVERT(qu8_qs8, uint8_t, int8_t);
BENCHMARK_CONVERT(qu8_qu8, uint8_t, uint8_t);
BENCHMARK_CONVERT(qu8_f16, uint8_t, xnn_float16);
BENCHMARK_CONVERT(qu8_f32, uint8_t, float);

BENCHMARK_CONVERT(f16_qs8, xnn_float16, int8_t);
BENCHMARK_CONVERT(f16_qu8, xnn_float16, uint8_t);
// BENCHMARK_CONVERT(f16_f16, xnn_float16, xnn_float16);
BENCHMARK_CONVERT(f16_f32, xnn_float16, float);

BENCHMARK_CONVERT(f32_qs8, float, int8_t);
BENCHMARK_CONVERT(f32_qu8, float, uint8_t);
BENCHMARK_CONVERT(f32_f16, float, xnn_float16);
// BENCHMARK_CONVERT(f32_f32, float, float);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
