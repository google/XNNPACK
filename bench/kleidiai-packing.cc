// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/pack.h"

#if XNN_ENABLE_KLEIDIAI && XNN_ARCH_ARM64
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"

namespace {

enum class PackingFormat {
  kF16Conv,
  kQs8Conv,
  kF32Conv,
  kQs8Gemm,
  kQs8GemmTransposed,
};

template <PackingFormat format>
void KleidiAIPacking(benchmark::State& state) {
  if (!benchmark::utils::CheckArchFlags(state, xnn_arch_arm_sme)) {
    return;
  }
  if (xnn_initialize(nullptr) != xnn_status_success) {
    state.SkipWithError("XNNPACK initialization failed");
    return;
  }

  constexpr bool quantized = format == PackingFormat::kQs8Conv ||
                             format == PackingFormat::kQs8Gemm ||
                             format == PackingFormat::kQs8GemmTransposed;
  using Weight = std::conditional_t<
      quantized, int8_t,
      std::conditional_t<format == PackingFormat::kF16Conv, uint16_t, float>>;
  using Bias = std::conditional_t<quantized, int32_t, Weight>;
  const size_t groups = state.range(0);
  const size_t channels = state.range(1);
  const size_t kernel_size = state.range(2);
  const size_t depth = state.range(3);
  const bool has_bias = state.range(4) != 0;
  xnnpack::Buffer<Weight> weights(groups * channels * kernel_size * depth);
  xnnpack::Buffer<Bias> bias(groups * channels);
  xnnpack::Buffer<float> scales(groups * channels);
  for (size_t index = 0; index < weights.size(); ++index) {
    size_t native_index = index;
    if constexpr (format == PackingFormat::kQs8GemmTransposed) {
      native_index = (index % channels) * depth + index / channels;
    }
    if constexpr (format == PackingFormat::kF16Conv) {
      weights[index] = 0x3000 + native_index % 1024;
    } else {
      weights[index] = static_cast<int32_t>(native_index % 17) - 8;
    }
  }
  for (size_t index = 0; index < bias.size(); ++index) {
    bias[index] = format == PackingFormat::kF16Conv ? 0x3800 : index % 7;
    scales[index] = 0.5f + (index % 3) * 0.125f;
  }

  size_t packed_group_size;
  if constexpr (format == PackingFormat::kF16Conv) {
    packed_group_size =
        kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme(
            channels, kernel_size, depth);
  } else if constexpr (format == PackingFormat::kQs8Conv) {
    packed_group_size =
        kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
            channels, kernel_size, depth);
  } else if constexpr (format == PackingFormat::kF32Conv) {
    packed_group_size =
        kai_get_rhs_packed_size_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme(
            channels, kernel_size, depth);
  } else {
    packed_group_size =
        kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
            channels, depth);
  }
  xnnpack::Buffer<uint8_t> packed(groups * packed_group_size, uint8_t{0xA5});
  xnn_gemm_config config = {};
  if constexpr (format == PackingFormat::kQs8Gemm ||
                format == PackingFormat::kQs8GemmTransposed) {
    config.nr = kai_get_n_step_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme();
    config.log2_kr = 2;
  }
  const xnn_qs8_packing_params params = {-7};
  const Bias* bias_data = has_bias ? bias.data() : nullptr;
  const auto pack = [&]() {
    if constexpr (format == PackingFormat::kF16Conv) {
      xnn_pack_kai_f16_conv_goki_w_sme(
          groups, channels, kernel_size, depth, 16, 1, 1, weights.data(),
          bias_data, scales.data(), packed.data(), 0, nullptr);
    } else if constexpr (format == PackingFormat::kQs8Conv) {
      xnn_pack_kai_qs8_conv_goki_w_sme(
          groups, channels, kernel_size, depth, 16, 4, 1, weights.data(),
          bias_data, scales.data(), packed.data(), 0, &params);
    } else if constexpr (format == PackingFormat::kF32Conv) {
      xnn_pack_kai_pf32_conv_goki_w_sme(
          groups, channels, kernel_size, depth, 16, 1, 1, weights.data(),
          bias_data, scales.data(), reinterpret_cast<float*>(packed.data()), 0,
          nullptr);
    } else {
      constexpr bool transposed = format == PackingFormat::kQs8GemmTransposed;
      xnn_pack_kai_qs8_qc8w_weights_and_biases_sme(
          transposed ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0, &config, depth, channels,
          groups, 0, transposed ? channels : depth, bias_data, weights.data(),
          nullptr, scales.data(), sizeof(float), nullptr, nullptr, 0,
          packed.data(), &params);
    }
  };

  pack();
  for (auto _ : state) {
    pack();
    benchmark::ClobberMemory();
  }
  uint32_t checksum = UINT32_C(2166136261);
  for (size_t index = 0; index < packed.size(); ++index) {
    checksum = (checksum ^ packed[index]) * UINT32_C(16777619);
  }
  state.counters["checksum"] = checksum;
  state.counters["input_bytes"] = weights.size() * sizeof(Weight);
  state.counters["packed_bytes"] = packed.size();
  state.SetItemsProcessed(state.iterations() * weights.size());
}

template <PackingFormat format>
void KleidiAIOperatorSetup(benchmark::State& state) {
  if (!benchmark::utils::CheckArchFlags(state, xnn_arch_arm_sme2)) {
    return;
  }
  const auto check_status = [&](xnn_status status) {
    if (status == xnn_status_success) {
      return true;
    }
    state.SkipWithError("operator lifecycle failed");
    return false;
  };
  if (!check_status(xnn_initialize(nullptr))) {
    return;
  }
  constexpr bool gemm = format == PackingFormat::kQs8Gemm ||
                        format == PackingFormat::kQs8GemmTransposed;
  constexpr bool quantized = gemm || format == PackingFormat::kQs8Conv;
  constexpr bool transposed = format == PackingFormat::kQs8GemmTransposed;
  using Value =
      std::conditional_t<quantized, int8_t,
                         std::conditional_t<format == PackingFormat::kF16Conv,
                                            xnn_float16, float>>;
  using Bias = std::conditional_t<quantized, int32_t, Value>;
  const auto convert = [](float value) -> Value {
    if constexpr (format == PackingFormat::kF16Conv) {
      return xnn_float16_from_float(value);
    } else {
      return static_cast<Value>(value);
    }
  };
  const size_t groups = state.range(0);
  const size_t channels = state.range(1);
  const size_t kernel_size = state.range(2);
  const size_t depth = state.range(3);
  const bool has_bias = state.range(4) != 0;
  const size_t kernel_height = kernel_size == 9 ? 3 : 1;
  const size_t kernel_width = kernel_size == 1 ? 1 : 3;
  const size_t padding_height = kernel_height / 2;
  const size_t padding_width = kernel_width / 2;
  const size_t pixels = gemm ? 8 : 16;
  xnnpack::Buffer<Value> weights(groups * channels * kernel_size * depth);
  xnnpack::Buffer<Bias> bias(groups * channels);
  xnnpack::Buffer<float> scales(groups * channels);
  xnnpack::Buffer<Value> input(pixels * groups * depth +
                               XNN_EXTRA_BYTES / sizeof(Value));
  xnnpack::Buffer<Value> output(pixels * groups * channels);
  for (size_t index = 0; index < weights.size(); ++index) {
    const size_t native_index =
        transposed ? (index % channels) * depth + index / channels : index;
    if constexpr (quantized) {
      weights[index] = static_cast<int32_t>(native_index % 17) - 8;
    } else {
      const int sign =
          static_cast<int>((index / (kernel_size * depth)) % 3) - 1;
      weights[index] = convert(sign * 0.25f);
    }
  }
  for (size_t index = 0; index < input.size(); ++index) {
    if constexpr (quantized) {
      input[index] = static_cast<int32_t>(index % 31) - 15;
    } else {
      input[index] = convert(0.5f);
    }
  }
  for (size_t index = 0; index < bias.size(); ++index) {
    if constexpr (quantized) {
      bias[index] = static_cast<int32_t>(index % 7) - 3;
    } else {
      bias[index] = convert(0.5f);
    }
    scales[index] = 0.005f + 0.0025f * (index % 3);
  }
  const Bias* bias_data = has_bias ? bias.data() : nullptr;
  const uint32_t flags = XNN_FLAG_INLINE_LHS_PACKING |
                         (transposed ? XNN_FLAG_TRANSPOSE_WEIGHTS : 0);
  bool validated = false;
  uint32_t checksum = UINT32_C(2166136261);
  size_t workspace_bytes = 0;
  xnn_microkernel_type ukernel_type = xnn_microkernel_type_default;
  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    xnn_operator_t raw_operation = nullptr;
    xnn_status status;
    if constexpr (format == PackingFormat::kF16Conv) {
      status = xnn_create_convolution2d_nhwc_pf16(
          padding_height, padding_width, padding_height, padding_width,
          kernel_height, kernel_width, 1, 1, 1, 1, groups, depth, channels,
          groups * depth, groups * channels, weights.data(), bias_data,
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), flags, nullptr,
          &raw_operation);
    } else if constexpr (format == PackingFormat::kF32Conv) {
      status = xnn_create_convolution2d_nhwc_pf32(
          padding_height, padding_width, padding_height, padding_width,
          kernel_height, kernel_width, 1, 1, 1, 1, groups, depth, channels,
          groups * depth, groups * channels, weights.data(), bias_data,
          -std::numeric_limits<float>::infinity(),
          std::numeric_limits<float>::infinity(), flags, nullptr,
          &raw_operation);
    } else if constexpr (format == PackingFormat::kQs8Conv) {
      status = xnn_create_convolution2d_nhwc_pqs8_qs8_qc8w(
          padding_height, padding_width, padding_height, padding_width,
          kernel_height, kernel_width, 1, 1, 1, 1, groups, depth, channels,
          groups * depth, groups * channels, -7, 1.0f, scales.data(),
          weights.data(), bias_data, 0, 1.0f, -128, 127, flags, nullptr,
          &raw_operation);
    } else {
      status = xnn_create_fully_connected_nc_pqs8_qc8w(
          depth, channels, depth, channels, -7, 1.0f, scales.data(),
          weights.data(), bias_data, 0, 1.0f, -128, 127, flags, nullptr,
          &raw_operation);
    }
    std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> operation(
        raw_operation, xnn_delete_operator);
    if (!check_status(status)) {
      return;
    }
    size_t output_height = 4;
    size_t output_width = 4;
    if constexpr (gemm) {
      status = xnn_reshape_fully_connected_nc_pqs8_qc8w(
          operation.get(), pixels, &workspace_bytes, nullptr);
    } else {
      const auto reshape =
          format == PackingFormat::kF16Conv
              ? xnn_reshape_convolution2d_nhwc_pf16
              : (format == PackingFormat::kF32Conv
                     ? xnn_reshape_convolution2d_nhwc_pf32
                     : xnn_reshape_convolution2d_nhwc_pqs8_qs8_qc8w);
      status = reshape(operation.get(), 1, 4, 4, &workspace_bytes,
                       &output_height, &output_width, nullptr);
    }
    if (!check_status(status)) {
      return;
    }
    if (output_height != 4 || output_width != 4) {
      state.SkipWithError("unexpected convolution output shape");
      return;
    }
    xnnpack::Buffer<uint8_t> workspace(workspace_bytes);
    if constexpr (format == PackingFormat::kF16Conv) {
      status = xnn_setup_convolution2d_nhwc_pf16(
          operation.get(), workspace.data(), input.data(), output.data());
    } else if constexpr (format == PackingFormat::kF32Conv) {
      status = xnn_setup_convolution2d_nhwc_pf32(
          operation.get(), workspace.data(), input.data(), output.data());
    } else if constexpr (format == PackingFormat::kQs8Conv) {
      status = xnn_setup_convolution2d_nhwc_pqs8_qs8_qc8w(
          operation.get(), workspace.data(), input.data(), output.data());
    } else {
      status = xnn_setup_fully_connected_nc_pqs8_qc8w(
          operation.get(), input.data(), output.data(), workspace.data());
    }
    const auto end = std::chrono::steady_clock::now();
    if (!check_status(status)) {
      return;
    }
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
    ukernel_type = operation->ukernel.type;
    if (!validated) {
      if (!check_status(xnn_run_operator(operation.get(), nullptr))) {
        return;
      }
      if constexpr (!quantized) {
        for (size_t pixel = 0; pixel < pixels; ++pixel) {
          const size_t row = pixel / 4;
          const size_t column = pixel % 4;
          const size_t valid_rows =
              kernel_height - ((row == 0 || row == 3) ? padding_height : 0);
          const size_t valid_columns =
              kernel_width - ((column == 0 || column == 3) ? padding_width : 0);
          for (size_t channel = 0; channel < groups * channels; ++channel) {
            const int sign = static_cast<int>(channel % 3) - 1;
            const float expected =
                sign * 0.125f * (valid_rows * valid_columns * depth) +
                (has_bias ? 0.5f : 0.0f);
            float actual;
            if constexpr (format == PackingFormat::kF16Conv) {
              actual = xnn_float16_to_float(
                  output[pixel * groups * channels + channel]);
            } else {
              actual = output[pixel * groups * channels + channel];
            }
            if (actual != expected) {
              state.SkipWithError("float output differs from exact reference");
              return;
            }
          }
        }
      }
      const auto* bytes = reinterpret_cast<const uint8_t*>(output.data());
      for (size_t index = 0; index < output.size() * sizeof(Value); ++index) {
        checksum = (checksum ^ bytes[index]) * UINT32_C(16777619);
      }
      validated = true;
    }
    benchmark::ClobberMemory();
  }
  state.counters["checksum"] = checksum;
  state.counters["input_bytes"] = weights.size() * sizeof(Value);
  state.counters["output_bytes"] = output.size() * sizeof(Value);
  state.counters["workspace_bytes"] = workspace_bytes;
  state.counters["ukernel_type"] = static_cast<int>(ukernel_type);
}

void ConvArguments(benchmark::Benchmark* benchmark) {
  for (int64_t has_bias : {0, 1}) {
    benchmark->Args({1, 1, 1, 1, has_bias});
    benchmark->Args({1, 3, 1, 15, has_bias});
    benchmark->Args({1, 33, 3, 17, has_bias});
    benchmark->Args({1, 64, 9, 64, has_bias});
    benchmark->Args({2, 64, 9, 64, has_bias});
    benchmark->Args({1, 256, 9, 256, has_bias});
  }
}

void GemmArguments(benchmark::Benchmark* benchmark) {
  for (int64_t has_bias : {0, 1}) {
    benchmark->Args({1, 1, 1, 1, has_bias});
    benchmark->Args({1, 3, 1, 15, has_bias});
    benchmark->Args({1, 33, 1, 17, has_bias});
    benchmark->Args({1, 256, 1, 256, has_bias});
    benchmark->Args({1, 1024, 1, 1024, has_bias});
  }
}

BENCHMARK_TEMPLATE(KleidiAIPacking, PackingFormat::kF16Conv)
    ->Name("kai_f16_conv_goki")
    ->Apply(ConvArguments)
    ->ArgNames({"g", "nc", "ks", "kc", "bias"})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(KleidiAIPacking, PackingFormat::kQs8Conv)
    ->Name("kai_qs8_conv_goki")
    ->Apply(ConvArguments)
    ->ArgNames({"g", "nc", "ks", "kc", "bias"})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(KleidiAIPacking, PackingFormat::kF32Conv)
    ->Name("kai_f32_conv_goki")
    ->Apply(ConvArguments)
    ->ArgNames({"g", "nc", "ks", "kc", "bias"})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(KleidiAIPacking, PackingFormat::kQs8Gemm)
    ->Name("kai_qs8_qc8w_gemm")
    ->Apply(GemmArguments)
    ->ArgNames({"g", "nc", "ks", "kc", "bias"})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();
BENCHMARK_TEMPLATE(KleidiAIPacking, PackingFormat::kQs8GemmTransposed)
    ->Name("kai_qs8_qc8w_gemm_transposed")
    ->Apply(GemmArguments)
    ->ArgNames({"g", "nc", "ks", "kc", "bias"})
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

BENCHMARK_TEMPLATE(KleidiAIOperatorSetup, PackingFormat::kF16Conv)
    ->Name("kai_setup_f16_conv_goki")
    ->Apply(ConvArguments)
    ->ArgNames({"g", "nc", "ks", "kc", "bias"})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
BENCHMARK_TEMPLATE(KleidiAIOperatorSetup, PackingFormat::kQs8Conv)
    ->Name("kai_setup_qs8_conv_goki")
    ->Apply(ConvArguments)
    ->ArgNames({"g", "nc", "ks", "kc", "bias"})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
BENCHMARK_TEMPLATE(KleidiAIOperatorSetup, PackingFormat::kF32Conv)
    ->Name("kai_setup_f32_conv_goki")
    ->Apply(ConvArguments)
    ->ArgNames({"g", "nc", "ks", "kc", "bias"})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
BENCHMARK_TEMPLATE(KleidiAIOperatorSetup, PackingFormat::kQs8Gemm)
    ->Name("kai_setup_qs8_qc8w_gemm")
    ->Apply(GemmArguments)
    ->ArgNames({"g", "nc", "ks", "kc", "bias"})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
BENCHMARK_TEMPLATE(KleidiAIOperatorSetup, PackingFormat::kQs8GemmTransposed)
    ->Name("kai_setup_qs8_qc8w_gemm_transposed")
    ->Apply(GemmArguments)
    ->ArgNames({"g", "nc", "ks", "kc", "bias"})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

}  // namespace
#endif

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
