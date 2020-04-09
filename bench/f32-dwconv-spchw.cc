// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <cpuinfo.h>

#include <benchmark/benchmark.h>
#include "bench/dwconv.h"
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/indirection.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static void DWConvCHWBenchmark(benchmark::State& state,
  xnn_f32_dwconv_spchw_ukernel_function dwconv,
  uint32_t it, uint32_t ot, uint32_t kh, uint32_t kw, uint32_t pw, uint32_t s)
{
  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
    return;
  }

  const size_t input_height = state.range(0);
  const size_t input_width = state.range(1);
  const size_t kernel_height = state.range(2);
  const size_t kernel_width = state.range(3);
  const size_t padding_height = state.range(4);
  const size_t padding_width = state.range(5);
  const size_t subsampling = state.range(6);
  const size_t dilation = state.range(7);
  const size_t channels = state.range(8);

  if (kernel_height != kh) {
    state.SkipWithError("kernel height mismatch");
    return;
  }

  if (kernel_width != kw) {
    state.SkipWithError("kernel width mismatch");
    return;
  }

  if (subsampling != s) {
    state.SkipWithError("subsampling mismatch");
    return;
  }

  if (padding_width % 2 != 0 || padding_width / 2 != pw) {
    state.SkipWithError("padding width mismatch");
    return;
  }

  if (dilation != 1) {
    state.SkipWithError("unsupported dilation");
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;

  const size_t inputSize = (input_height + padding_height) * input_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_size = output_height * output_width;

  std::vector<float> input(inputSize * channels + 2 * it);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> bias(channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));
  std::vector<float> kernel(channels * kernel_size);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));

  const size_t w_elements = (kernel_size + 1) * channels;
  const size_t o_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + o_elements));

  std::vector<float, AlignedAllocator<float, 32>> packed_weights(w_elements * num_buffers);
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

  xnn_f32_spchw_params spchw_params =
    xnn_init_f32_spchw_params(input_width, -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(input.data(), input.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t channel = 0; channel < channels; channel++) {
      dwconv(
        output_height, input_width,
        input.data() + channel * inputSize,
        packed_weights.data() + channel * (kernel_size + 1) + buffer_index * w_elements,
        output.data() + channel * output_size + buffer_index * o_elements,
        it * sizeof(float), ot * sizeof(float),
        input_width * sizeof(float), output_width * sizeof(float),
        &spchw_params);
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * output_size * channels * kernel_size,
    benchmark::Counter::kIsRate);

  state.counters["BYTES"] = benchmark::Counter(
    uint64_t(state.iterations()) * (output_size + inputSize + kernel_size + 1 /* bias */) * channels * sizeof(float),
    benchmark::Counter::kIsRate);
}

static void DWConvHWoTCTBenchmark(benchmark::State& state,
  xnn_f32_dwconv_spchw_ukernel_function dwconv,
  uint32_t it, uint32_t ot, uint32_t kh, uint32_t kw, uint32_t pw, uint32_t s)
{
  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
    return;
  }

  const size_t input_height = state.range(0);
  const size_t input_width = state.range(1);
  const size_t kernel_height = state.range(2);
  const size_t kernel_width = state.range(3);
  const size_t padding_height = state.range(4);
  const size_t padding_width = state.range(5);
  const size_t subsampling = state.range(6);
  const size_t dilation = state.range(7);
  const size_t channels = state.range(8);

  if (kernel_height != kh) {
    state.SkipWithError("kernel height mismatch");
    return;
  }

  if (kernel_width != kw) {
    state.SkipWithError("kernel width mismatch");
    return;
  }

  if (subsampling != s) {
    state.SkipWithError("subsampling mismatch");
    return;
  }

  if (padding_width % 2 != 0 || padding_width / 2 != pw) {
    state.SkipWithError("padding width mismatch");
    return;
  }

  if (dilation != 1) {
    state.SkipWithError("unsupported dilation");
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;

  const size_t inputSize = (input_height + padding_height) * input_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_size = output_height * output_width;

  std::vector<float> input(input_height * benchmark::utils::RoundUp<size_t>(input_width, it) * channels);
  std::generate(input.begin(), input.end(), std::ref(f32rng));
  std::vector<float> bias(channels);
  std::generate(bias.begin(), bias.end(), std::ref(f32rng));
  std::vector<float> kernel(channels * kernel_size);
  std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));

  const size_t w_elements = (kernel_size + 1) * channels;
  const size_t o_elements = output_height * benchmark::utils::RoundUp<size_t>(output_width, ot) * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + o_elements));

  std::vector<float, AlignedAllocator<float, 32>> packed_weights(w_elements * num_buffers);
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

  xnn_f32_spchw_params spchw_params =
    xnn_init_f32_spchw_params(input_width, -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(input.data(), input.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t channel = 0; channel < channels; channel++) {
      dwconv(
        output_height, input_width,
        input.data() + channel * it,
        packed_weights.data() + channel * (kernel_size + 1) + buffer_index * w_elements,
        output.data() + channel * ot + buffer_index * o_elements,
        it * channels * sizeof(float), ot * channels * sizeof(float),
        benchmark::utils::RoundUp<size_t>(input_width, it) * channels * sizeof(float),
        benchmark::utils::RoundUp<size_t>(output_width, ot) * channels * sizeof(float),
        &spchw_params);
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * output_size * channels * kernel_size,
    benchmark::Counter::kIsRate);

  state.counters["BYTES"] = benchmark::Counter(
    uint64_t(state.iterations()) * (output_size + inputSize + kernel_size + 1 /* bias */) * channels * sizeof(float),
    benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64
  static void CHW_3x3p1__neonfma(benchmark::State& state, const char* net) {
    DWConvCHWBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3p1__neonfma, 4, 4, 3, 3, 1, 1);
  }

  static void CHW_5x5p2__neonfma(benchmark::State& state, const char* net) {
    DWConvCHWBenchmark(state, xnn_f32_dwconv_spchw_ukernel_5x5p2__neonfma, 4, 4, 5, 5, 2, 1);
  }

  static void CHW_3x3s2p1__neonfma(benchmark::State& state, const char* net) {
    DWConvCHWBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3s2p1__neonfma, 4, 4, 3, 3, 1, 2);
  }

  static void CHW_5x5s2p2__neonfma(benchmark::State& state, const char* net) {
    DWConvCHWBenchmark(state, xnn_f32_dwconv_spchw_ukernel_5x5s2p2__neonfma, 4, 4, 5, 5, 2, 2);
  }

  static void HWo4C4_3x3p1__neonfma(benchmark::State& state, const char* net) {
    DWConvHWoTCTBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3p1__neonfma, 4, 4, 3, 3, 1, 1);
  }

  static void HWo4C4_5x5p2__neonfma(benchmark::State& state, const char* net) {
    DWConvHWoTCTBenchmark(state, xnn_f32_dwconv_spchw_ukernel_5x5p2__neonfma, 4, 4, 5, 5, 2, 1);
  }

  static void HWo4C4_3x3s2p1__neonfma(benchmark::State& state, const char* net) {
    DWConvHWoTCTBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3s2p1__neonfma, 4, 4, 3, 3, 1, 2);
  }

  static void HWo4C4_5x5s2p2__neonfma(benchmark::State& state, const char* net) {
    DWConvHWoTCTBenchmark(state, xnn_f32_dwconv_spchw_ukernel_5x5s2p2__neonfma, 4, 4, 5, 5, 2, 2);
  }

  BENCHMARK_DWCONV(CHW_3x3p1__neonfma)
  BENCHMARK_DWCONV(CHW_5x5p2__neonfma)
  BENCHMARK_DWCONV(CHW_3x3s2p1__neonfma)
  BENCHMARK_DWCONV(CHW_5x5s2p2__neonfma)
  BENCHMARK_DWCONV(HWo4C4_3x3p1__neonfma)
  BENCHMARK_DWCONV(HWo4C4_5x5p2__neonfma)
  BENCHMARK_DWCONV(HWo4C4_3x3s2p1__neonfma)
  BENCHMARK_DWCONV(HWo4C4_5x5s2p2__neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void CHW_3x3p1__sse(benchmark::State& state, const char* net) {
    DWConvCHWBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3p1__sse, 4, 4, 3, 3, 1, 1);
  }

  static void CHW_3x3s2p1__sse(benchmark::State& state, const char* net) {
    DWConvCHWBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3s2p1__sse, 4, 4, 3, 3, 1, 2);
  }

  static void HWo4C4_3x3p1__sse(benchmark::State& state, const char* net) {
    DWConvHWoTCTBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3p1__sse, 4, 4, 3, 3, 1, 1);
  }

  static void HWo4C4_3x3s2p1__sse(benchmark::State& state, const char* net) {
    DWConvHWoTCTBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3s2p1__sse, 4, 4, 3, 3, 1, 2);
  }

  BENCHMARK_DWCONV(CHW_3x3p1__sse)
  BENCHMARK_DWCONV(CHW_3x3s2p1__sse)
  BENCHMARK_DWCONV(HWo4C4_3x3p1__sse)
  BENCHMARK_DWCONV(HWo4C4_3x3s2p1__sse)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

  static void CHW_3x3p1__scalar(benchmark::State& state, const char* net) {
    DWConvCHWBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3p1__scalar, 1, 1, 3, 3, 1, 1);
  }

  static void CHW_5x5p2__scalar(benchmark::State& state, const char* net) {
    DWConvCHWBenchmark(state, xnn_f32_dwconv_spchw_ukernel_5x5p2__scalar, 1, 1, 5, 5, 2, 1);
  }

  static void CHW_3x3s2p1__scalar(benchmark::State& state, const char* net) {
    DWConvCHWBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3s2p1__scalar, 1, 1, 3, 3, 1, 2);
  }

  static void CHW_5x5s2p2__scalar(benchmark::State& state, const char* net) {
    DWConvCHWBenchmark(state, xnn_f32_dwconv_spchw_ukernel_5x5s2p2__scalar, 1, 1, 5, 5, 2, 2);
  }

  static void HWC_3x3p1__scalar(benchmark::State& state, const char* net) {
    DWConvHWoTCTBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3p1__scalar, 1, 1, 3, 3, 1, 1);
  }

  static void HWC_5x5p2__scalar(benchmark::State& state, const char* net) {
    DWConvHWoTCTBenchmark(state, xnn_f32_dwconv_spchw_ukernel_5x5p2__scalar, 1, 1, 5, 5, 2, 1);
  }

  static void HWC_3x3s2p1__scalar(benchmark::State& state, const char* net) {
    DWConvHWoTCTBenchmark(state, xnn_f32_dwconv_spchw_ukernel_3x3s2p1__scalar, 1, 1, 3, 3, 1, 2);
  }

  static void HWC_5x5s2p2__scalar(benchmark::State& state, const char* net) {
    DWConvHWoTCTBenchmark(state, xnn_f32_dwconv_spchw_ukernel_5x5s2p2__scalar, 1, 1, 5, 5, 2, 2);
  }


  BENCHMARK_DWCONV(CHW_3x3p1__scalar)
  BENCHMARK_DWCONV(CHW_5x5p2__scalar)
  BENCHMARK_DWCONV(CHW_3x3s2p1__scalar)
  BENCHMARK_DWCONV(CHW_5x5s2p2__scalar)
  BENCHMARK_DWCONV(HWC_3x3p1__scalar)
  BENCHMARK_DWCONV(HWC_5x5p2__scalar)
  BENCHMARK_DWCONV(HWC_3x3s2p1__scalar)
  BENCHMARK_DWCONV(HWC_5x5s2p2__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
