// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/dwconv.h"
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/indirection.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"


static void f32_dwconv2d_chw(benchmark::State& state,
  xnn_f32_dwconv2d_chw_ukernel_fn dwconv,
  xnn_init_f32_chw_params_fn init_params,
  uint32_t kernel_height, uint32_t kernel_width, uint32_t padding_width, uint32_t stride,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t input_height = state.range(0);
  const size_t input_width = state.range(1);
  const size_t kernel_height_arg = state.range(2);
  const size_t kernel_width_arg = state.range(3);
  const size_t padding_height = state.range(4);
  const size_t padding_width_left_plus_right = state.range(5);
  const size_t subsampling = state.range(6);
  const size_t dilation = state.range(7);
  const size_t channels = state.range(8);

  if (kernel_height != kernel_height_arg) {
    state.SkipWithError("kernel height mismatch");
    return;
  }

  if (kernel_width != kernel_width_arg) {
    state.SkipWithError("kernel width mismatch");
    return;
  }

  if (subsampling != stride) {
    state.SkipWithError("subsampling mismatch");
    return;
  }

  if (padding_width_left_plus_right % 2 != 0 || padding_width_left_plus_right / 2 != padding_width) {
    state.SkipWithError("padding width mismatch");
    return;
  }

  if (dilation != 1) {
    state.SkipWithError("unsupported dilation");
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width_left_plus_right - effective_kernel_width) / subsampling + 1;

  const size_t inputSize = (input_height + padding_height) * input_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_size = output_height * output_width;

  std::vector<float> input(inputSize * channels + 2 * XNN_EXTRA_BYTES);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> bias(channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));
  std::vector<float> kernel(channels * kernel_size);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
  std::vector<float> zero(input_width + padding_width_left_plus_right);

  const size_t w_elements = (kernel_size + 1) * channels;
  const size_t o_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + o_elements));

  std::vector<float, AlignedAllocator<float, 64>> packed_weights(w_elements * num_buffers);
  std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);
  for (size_t c = 0; c < channels; c++) {
    packed_weights[c * kernel_size + c] = bias[c];
    for (size_t i = 0; i < kernel_size; i++) {
      packed_weights[c * kernel_size + c + 1 + i] = kernel[c * kernel_size + i];
    }
  }
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(packed_weights.cbegin(), packed_weights.cbegin() + w_elements, packed_weights.begin() + n * w_elements);
  }

  std::vector<float> output(o_elements * num_buffers);
  std::fill(output.begin(), output.end(), std::nanf(""));

  xnn_f32_chw_params chw_params;
  init_params(&chw_params,
    input_width, -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(input.data(), input.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t channel = 0; channel < channels; channel++) {
      dwconv(
        input_height, input_width * sizeof(float),
        input.data() + channel * inputSize,
        packed_weights.data() + channel * (kernel_size + 1) + buffer_index * w_elements,
        zero.data(),
        output.data() + channel * output_size + buffer_index * o_elements,
        padding_height / 2,  // padding_top
        &chw_params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * output_size * channels * kernel_size,
    benchmark::Counter::kIsRate);

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) * (output_size + inputSize + kernel_size + 1 /* bias */) * channels * sizeof(float),
    benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM
  static void dwconv2d_chw_3x3p1__neon_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_1x4,
      xnn_init_f32_chw_neon_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3p1__neon_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_2x4,
      xnn_init_f32_chw_neon_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3p1__neon_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_3x4,
      xnn_init_f32_chw_neon_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3p1__neon_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_4x4,
      xnn_init_f32_chw_neon_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3p1__neon_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_5x4,
      xnn_init_f32_chw_neon_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3p1__neon_6x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_6x4,
      xnn_init_f32_chw_neon_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3p1__neon_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_1x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3p1__neon_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_1x4_acc3,
      xnn_init_f32_chw_neon_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3p1__neon_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_1x4_acc4,
      xnn_init_f32_chw_neon_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3p1__neon_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_2x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }

  static void dwconv2d_chw_3x3s2p1__neon_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_1x4,
      xnn_init_f32_chw_neon_stride2_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3s2p1__neon_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_2x4,
      xnn_init_f32_chw_neon_stride2_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3s2p1__neon_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_3x4,
      xnn_init_f32_chw_neon_stride2_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3s2p1__neon_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_4x4,
      xnn_init_f32_chw_neon_stride2_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3s2p1__neon_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_1x4_acc2,
      xnn_init_f32_chw_neon_stride2_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3s2p1__neon_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_1x4_acc3,
      xnn_init_f32_chw_neon_stride2_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3s2p1__neon_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_1x4_acc4,
      xnn_init_f32_chw_neon_stride2_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_3x3s2p1__neon_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_2x4_acc2,
      xnn_init_f32_chw_neon_stride2_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }

  static void dwconv2d_chw_5x5p2__neon_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_2x4,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_3x4,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_4x4,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_5x4,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4_acc3,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4_acc4,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4_acc5,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_2x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_2x4_acc3,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_3x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5p2__neon_4x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_4x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckNEON);
  }

  static void dwconv2d_chw_5x5s2p2__neon_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4,
      xnn_init_f32_chw_neon_stride2_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5s2p2__neon_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_2x4,
      xnn_init_f32_chw_neon_stride2_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5s2p2__neon_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_3x4,
      xnn_init_f32_chw_neon_stride2_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5s2p2__neon_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4_acc2,
      xnn_init_f32_chw_neon_stride2_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5s2p2__neon_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4_acc3,
      xnn_init_f32_chw_neon_stride2_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5s2p2__neon_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4_acc4,
      xnn_init_f32_chw_neon_stride2_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5s2p2__neon_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4_acc5,
      xnn_init_f32_chw_neon_stride2_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5s2p2__neon_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_2x4_acc2,
      xnn_init_f32_chw_neon_stride2_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5s2p2__neon_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_2x4_acc3,
      xnn_init_f32_chw_neon_stride2_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }
  static void dwconv2d_chw_5x5s2p2__neon_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4_acc5,
      xnn_init_f32_chw_neon_stride2_params,
      5 /* kernel height */, 5 /* kernel width */, 2 /* width padding */, 2 /* stride */,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__neon_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__neon_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__neon_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__neon_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__neon_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__neon_6x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__neon_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__neon_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__neon_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__neon_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__neon_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__neon_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__neon_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__neon_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__neon_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__neon_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__neon_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__neon_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_3x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__neon_4x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__neon_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__neon_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__neon_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__neon_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__neon_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__neon_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__neon_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__neon_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__neon_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__neon_3x4_acc2)
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_ARM64
  static void dwconv2d_chw_3x3p1__aarch64_neonfma_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_1x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__aarch64_neonfma_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_2x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__aarch64_neonfma_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_3x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__aarch64_neonfma_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_4x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__aarch64_neonfma_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_5x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__aarch64_neonfma_6x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_6x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__aarch64_neonfma_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_1x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__aarch64_neonfma_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_1x4_acc3,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__aarch64_neonfma_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_1x4_acc4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__aarch64_neonfma_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__aarch64_neonfma_2x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }

  static void dwconv2d_chw_3x3s2p1__aarch64_neonfma_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__aarch64_neonfma_1x4,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__aarch64_neonfma_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__aarch64_neonfma_2x4,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__aarch64_neonfma_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__aarch64_neonfma_3x4,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__aarch64_neonfma_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__aarch64_neonfma_4x4,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__aarch64_neonfma_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__aarch64_neonfma_1x4_acc2,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__aarch64_neonfma_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__aarch64_neonfma_1x4_acc3,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__aarch64_neonfma_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__aarch64_neonfma_1x4_acc4,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__aarch64_neonfma_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__aarch64_neonfma_2x4_acc2,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }

  static void dwconv2d_chw_5x5p2__aarch64_neonfma_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_1x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_2x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_3x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_4x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_5x4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_1x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_1x4_acc3,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_1x4_acc4,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_1x4_acc5,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_2x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_2x4_acc3,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_3x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__aarch64_neonfma_4x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__aarch64_neonfma_4x4_acc2,
      xnn_init_f32_chw_neon_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }

  static void dwconv2d_chw_5x5s2p2__aarch64_neonfma_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_1x4,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__aarch64_neonfma_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_2x4,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__aarch64_neonfma_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_3x4,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__aarch64_neonfma_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_1x4_acc2,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__aarch64_neonfma_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_1x4_acc3,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__aarch64_neonfma_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_1x4_acc4,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__aarch64_neonfma_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_1x4_acc5,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__aarch64_neonfma_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_2x4_acc2,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__aarch64_neonfma_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_2x4_acc3,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__aarch64_neonfma_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__aarch64_neonfma_1x4_acc5,
      xnn_init_f32_chw_neon_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }

  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__aarch64_neonfma_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__aarch64_neonfma_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__aarch64_neonfma_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__aarch64_neonfma_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__aarch64_neonfma_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__aarch64_neonfma_6x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__aarch64_neonfma_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__aarch64_neonfma_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__aarch64_neonfma_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__aarch64_neonfma_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__aarch64_neonfma_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__aarch64_neonfma_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__aarch64_neonfma_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__aarch64_neonfma_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__aarch64_neonfma_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__aarch64_neonfma_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__aarch64_neonfma_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__aarch64_neonfma_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_3x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__aarch64_neonfma_4x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__aarch64_neonfma_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__aarch64_neonfma_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__aarch64_neonfma_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__aarch64_neonfma_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__aarch64_neonfma_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__aarch64_neonfma_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__aarch64_neonfma_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__aarch64_neonfma_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__aarch64_neonfma_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__aarch64_neonfma_3x4_acc2)
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void dwconv2d_chw_3x3p1__ssse3_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_1x4,
      xnn_init_f32_chw_sse_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckSSSE3);
  }
  static void dwconv2d_chw_3x3p1__ssse3_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_2x4,
      xnn_init_f32_chw_sse_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckSSSE3);
  }
  static void dwconv2d_chw_3x3p1__ssse3_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_3x4,
      xnn_init_f32_chw_sse_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckSSSE3);
  }
  static void dwconv2d_chw_3x3p1__ssse3_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_4x4,
      xnn_init_f32_chw_sse_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckSSSE3);
  }
  static void dwconv2d_chw_3x3p1__ssse3_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_5x4,
      xnn_init_f32_chw_sse_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckSSSE3);
  }
  static void dwconv2d_chw_3x3p1__ssse3_6x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_6x4,
      xnn_init_f32_chw_sse_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckSSSE3);
  }
  static void dwconv2d_chw_3x3p1__ssse3_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_1x4_acc2,
      xnn_init_f32_chw_sse_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckSSSE3);
  }
  static void dwconv2d_chw_3x3p1__ssse3_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_1x4_acc3,
      xnn_init_f32_chw_sse_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckSSSE3);
  }
  static void dwconv2d_chw_3x3p1__ssse3_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_1x4_acc4,
      xnn_init_f32_chw_sse_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckSSSE3);
  }
  static void dwconv2d_chw_3x3p1__ssse3_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_2x4_acc2,
      xnn_init_f32_chw_sse_stride1_params,
      3 /* kernel height */, 3 /* kernel width */, 1 /* width padding */, 1 /* stride */,
      benchmark::utils::CheckSSSE3);
  }

  static void dwconv2d_chw_3x3p1__sse_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_1x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__sse_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_2x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__sse_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_3x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__sse_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_4x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__sse_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_5x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__sse_6x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_6x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__sse_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_1x4_acc2,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__sse_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_1x4_acc3,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__sse_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_1x4_acc4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__sse_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_2x4_acc2,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }

  static void dwconv2d_chw_3x3s2p1__sse_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__sse_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_2x4,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__sse_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_3x4,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__sse_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_4x4,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__sse_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4_acc2,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__sse_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4_acc3,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__sse_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4_acc4,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__sse_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_2x4_acc2,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }

  static void dwconv2d_chw_5x5p2__sse_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_1x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_2x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_3x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_4x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_5x4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_1x4_acc2,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_1x4_acc3,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_1x4_acc4,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_1x4_acc5,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_2x4_acc2,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_2x4_acc3,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_3x4_acc2,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__sse_4x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_4x4_acc2,
      xnn_init_f32_chw_sse_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }

  static void dwconv2d_chw_5x5s2p2__sse_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__sse_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_2x4,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__sse_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_3x4,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__sse_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4_acc2,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__sse_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4_acc3,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__sse_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4_acc4,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__sse_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4_acc5,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__sse_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_2x4_acc2,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__sse_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_2x4_acc3,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__sse_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4_acc5,
      xnn_init_f32_chw_sse_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }

  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__ssse3_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__ssse3_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__ssse3_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__ssse3_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__ssse3_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__ssse3_6x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__ssse3_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__ssse3_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__ssse3_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__ssse3_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__sse_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__sse_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__sse_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__sse_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__sse_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__sse_6x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__sse_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__sse_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__sse_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__sse_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__sse_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__sse_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__sse_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__sse_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__sse_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__sse_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__sse_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__sse_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_3x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__sse_4x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__sse_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__sse_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__sse_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__sse_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__sse_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__sse_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__sse_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__sse_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__sse_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__sse_3x4_acc2)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_1x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_2x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_3x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_4x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_5x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_6x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_6x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_1x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_2x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_3x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_4x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_5x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_6x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_6x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_splat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_1x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_splat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_2x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_splat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_3x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_splat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_4x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_splat_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_5x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_splat_6x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_6x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_splat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_splat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_splat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_arm_splat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_splat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_1x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_splat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_2x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_splat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_3x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_splat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_4x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_splat_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_5x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_splat_6x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_6x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_splat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_splat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_splat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }
  static void dwconv2d_chw_3x3p1__wasmsimd_x86_splat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
  }

  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_1x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_2x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_3x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_4x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_1x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_2x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_3x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_4x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_1x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_2x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_3x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_4x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_1x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_2x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_3x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_4x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }
  static void dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
  }

  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_1x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_2x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_3x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_4x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_5x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_2x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_3x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_4x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_4x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_2x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_3x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_4x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_5x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_2x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_3x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_4x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_4x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_1x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_2x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_3x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_4x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_5x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_2x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_3x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_arm_splat_4x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_4x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_1x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_2x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_3x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_4x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_4x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_5x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_5x4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_2x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_3x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }
  static void dwconv2d_chw_5x5p2__wasmsimd_x86_splat_4x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_4x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride1_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
  }

  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_2x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_3x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_2x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_2x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_3x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_2x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_2x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_3x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_2x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_1x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_2x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_2x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_3x4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_3x4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_1x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_1x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_1x4_acc4(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc4,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_1x4_acc5(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_2x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_2x4_acc2,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_2x4_acc3(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_2x4_acc3,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }
  static void dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_3x4_acc2(benchmark::State& state, const char* net) {
    f32_dwconv2d_chw(state,
      xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc5,
      xnn_init_f32_chw_wasmsimd_stride2_params,
      /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
  }

  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_6x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_loadsplat_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_6x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_loadsplat_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_splat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_splat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_splat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_splat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_splat_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_splat_6x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_splat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_splat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_splat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_arm_splat_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_splat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_splat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_splat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_splat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_splat_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_splat_6x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_splat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_splat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_splat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__wasmsimd_x86_splat_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_loadsplat_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_loadsplat_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_arm_splat_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__wasmsimd_x86_splat_2x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_3x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_loadsplat_4x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_3x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_loadsplat_4x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_3x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_arm_splat_4x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_4x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_5x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_3x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__wasmsimd_x86_splat_4x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_loadsplat_3x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_loadsplat_3x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_arm_splat_3x4_acc2)

  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_1x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_2x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_3x4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_1x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_1x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_1x4_acc4)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_1x4_acc5)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_2x4_acc2)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_2x4_acc3)
  BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__wasmsimd_x86_splat_3x4_acc2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

static void dwconv2d_chw_3x3p1__scalar_1x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
}
static void dwconv2d_chw_3x3p1__scalar_2x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
}
static void dwconv2d_chw_3x3p1__scalar_3x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_3x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
}
static void dwconv2d_chw_3x3p1__scalar_4x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_4x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
}
static void dwconv2d_chw_3x3p1__scalar_5x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_5x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
}
static void dwconv2d_chw_3x3p1__scalar_6x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_6x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
}
static void dwconv2d_chw_3x3p1__scalar_1x1_acc2(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1_acc2,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
}
static void dwconv2d_chw_3x3p1__scalar_1x1_acc3(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1_acc3,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
}
static void dwconv2d_chw_3x3p1__scalar_1x1_acc4(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1_acc4,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
}
static void dwconv2d_chw_3x3p1__scalar_2x1_acc2(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1_acc2,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/1);
}

static void dwconv2d_chw_3x3s2p1__scalar_1x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
}
static void dwconv2d_chw_3x3s2p1__scalar_2x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_2x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
}
static void dwconv2d_chw_3x3s2p1__scalar_3x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_3x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
}
static void dwconv2d_chw_3x3s2p1__scalar_4x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_4x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
}
static void dwconv2d_chw_3x3s2p1__scalar_1x1_acc2(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc2,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
}
static void dwconv2d_chw_3x3s2p1__scalar_1x1_acc3(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc3,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
}
static void dwconv2d_chw_3x3s2p1__scalar_1x1_acc4(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc4,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
}
static void dwconv2d_chw_3x3s2p1__scalar_2x1_acc2(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_2x1_acc2,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/3, /*kernel_width=*/3, /*padding_width=*/1, /*stride=*/2);
}

static void dwconv2d_chw_5x5p2__scalar_1x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
}
static void dwconv2d_chw_5x5p2__scalar_2x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
}
static void dwconv2d_chw_5x5p2__scalar_3x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_3x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
}
static void dwconv2d_chw_5x5p2__scalar_1x1_acc2(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc2,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
}
static void dwconv2d_chw_5x5p2__scalar_1x1_acc3(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc3,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
}
static void dwconv2d_chw_5x5p2__scalar_1x1_acc4(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc4,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
}
static void dwconv2d_chw_5x5p2__scalar_1x1_acc5(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc5,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
}
static void dwconv2d_chw_5x5p2__scalar_2x1_acc2(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1_acc2,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
}
static void dwconv2d_chw_5x5p2__scalar_2x1_acc3(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1_acc3,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
}
static void dwconv2d_chw_5x5p2__scalar_3x1_acc2(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_3x1_acc2,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/1);
}

static void dwconv2d_chw_5x5s2p2__scalar_1x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
}
static void dwconv2d_chw_5x5s2p2__scalar_2x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
}
static void dwconv2d_chw_5x5s2p2__scalar_3x1(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_3x1,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
}
static void dwconv2d_chw_5x5s2p2__scalar_1x1_acc2(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc2,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
}
static void dwconv2d_chw_5x5s2p2__scalar_1x1_acc3(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc3,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
}
static void dwconv2d_chw_5x5s2p2__scalar_1x1_acc4(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc4,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
}
static void dwconv2d_chw_5x5s2p2__scalar_1x1_acc5(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc5,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
}
static void dwconv2d_chw_5x5s2p2__scalar_2x1_acc2(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1_acc2,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
}
static void dwconv2d_chw_5x5s2p2__scalar_2x1_acc3(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1_acc3,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
}
static void dwconv2d_chw_5x5s2p2__scalar_3x1_acc2(benchmark::State& state, const char* net) {
  f32_dwconv2d_chw(state,
    xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc5,
    xnn_init_f32_chw_scalar_params,
    /*kernel_height=*/5, /*kernel_width=*/5, /*padding_width=*/2, /*stride=*/2);
}

BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__scalar_1x1)
BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__scalar_2x1)
BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__scalar_3x1)
BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__scalar_4x1)
BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__scalar_5x1)
BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__scalar_6x1)
BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__scalar_1x1_acc2)
BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__scalar_1x1_acc3)
BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__scalar_1x1_acc4)
BENCHMARK_DWCONV(dwconv2d_chw_3x3p1__scalar_2x1_acc2)

BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__scalar_1x1)
BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__scalar_2x1)
BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__scalar_3x1)
BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__scalar_4x1)
BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__scalar_1x1_acc2)
BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__scalar_1x1_acc3)
BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__scalar_1x1_acc4)
BENCHMARK_DWCONV(dwconv2d_chw_3x3s2p1__scalar_2x1_acc2)

BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__scalar_1x1)
BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__scalar_2x1)
BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__scalar_3x1)
BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__scalar_1x1_acc2)
BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__scalar_1x1_acc3)
BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__scalar_1x1_acc4)
BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__scalar_1x1_acc5)
BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__scalar_2x1_acc2)
BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__scalar_2x1_acc3)
BENCHMARK_DWCONV(dwconv2d_chw_5x5p2__scalar_3x1_acc2)

BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__scalar_1x1)
BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__scalar_2x1)
BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__scalar_3x1)
BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__scalar_1x1_acc2)
BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__scalar_1x1_acc3)
BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__scalar_1x1_acc4)
BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__scalar_1x1_acc5)
BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__scalar_2x1_acc2)
BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__scalar_2x1_acc3)
BENCHMARK_DWCONV(dwconv2d_chw_5x5s2p2__scalar_3x1_acc2)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
