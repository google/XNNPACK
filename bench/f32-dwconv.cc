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

#include "bench/dwconv.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/indirection.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microkernel-utils.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"


static void f32_dwconv(benchmark::State& state,
  xnn_f32_dwconv_minmax_unipass_ukernel_fn dwconv,
  xnn_init_f32_minmax_params_fn init_params,
  uint32_t channel_tile, uint32_t primary_tile,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
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

  const size_t kernel_size = kernel_height * kernel_width;
  if (kernel_size > primary_tile) {
    state.SkipWithError("kernel size mismatch");
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;
  const size_t output_size = output_height * output_width;
  const size_t step_width = dilation == 1 ? std::min(subsampling, kernel_width) : kernel_width;
  const size_t step_height = kernel_size + (output_width - 1) * step_width * kernel_height;

  const size_t c_stride = benchmark::utils::RoundUp<size_t>(channels, channel_tile);

  std::vector<float> a(channels * input_height * input_width + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(channels * kernel_height * kernel_width);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(channels);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  std::vector<float> z(channels + XNN_EXTRA_BYTES / sizeof(float));

  const size_t w_elements = (kernel_size + 1) * c_stride;
  // Can read (primary_tile - kernel_size) elements after end of indirection buffer.
  const size_t i_elements = (primary_tile - kernel_size) + output_height * step_height;
  const size_t c_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_dwconv_ghw_w(primary_tile, 0, 0, kernel_height, kernel_width, channels,
                            channel_tile, channel_tile, /*channel_round=*/1,
                            k.data(), b.data(), /*scale=*/nullptr, w.data(),
                            /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const float*> i(i_elements * num_buffers);
  xnn_indirection_init_dwconv2d(
    /*output_y_start=*/0, /*output_y_end=*/output_height,
    reinterpret_cast<const void**>(i.data()),
    a.data(),
    channels << XNN_LOG2_SIZEOF_FLOAT,
    z.data(),
    input_height, input_width,
    output_height, output_width,
    kernel_height, kernel_width,
    subsampling, subsampling,
    dilation, dilation,
    padding_top, padding_left,
    step_height, step_width, primary_tile);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params, -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (size_t y = 0; y < output_height; y++) {
      dwconv(channels, output_width,
        i.data() + buffer_index * i_elements + step_height * y,
        w.data() + buffer_index * w_elements,
        c.data() + buffer_index * c_elements + y * output_width * channels,
        kernel_height * step_width * sizeof(void*), 0,
        0, z.data(), &params);
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
    uint64_t(state.iterations()) * (output_size + input_height * input_width + kernel_size + 1 /* bias */) * channels * sizeof(float),
    benchmark::Counter::kIsRate);
}

static void f32_dwconv(
  benchmark::State& state,
  xnn_f32_dwconv_minmax_multipass_ukernel_fn dwconv,
  xnn_init_f32_minmax_params_fn init_params,
  uint32_t first_pass_tile,
  uint32_t middle_pass_tile,
  uint32_t last_pass_tile,
  uint32_t channel_tile,
  uint32_t channel_subtile,
  uint32_t channel_round,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
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

  const size_t kernel_size = kernel_height * kernel_width;

  if (kernel_size <= first_pass_tile) {
    state.SkipWithError("kernel size mismatch");
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;
  const size_t output_size = output_height * output_width;
  const size_t step_width = dilation == 1 ? std::min(subsampling, kernel_width) : kernel_width;
  const size_t step_height = kernel_size + (output_width - 1) * step_width * kernel_height;

  std::vector<float> a(channels * input_height * input_width + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(channels * kernel_size);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(channels);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  std::vector<float> z(channels + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<float, AlignedAllocator<float, 64>> buffer(channels + XNN_MULTIPASS_EXTRA_BYTES / sizeof(float));

  const size_t tile_size = xnn_dwconv_multipass_tile_size(
    kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile);
  const size_t w_elements =
    xnn_dwconv_multipass_weights_size(
      tile_size, channels, channel_tile, channel_subtile, channel_round,
      /*bias_element_size=*/sizeof(float), /*log2_filter_element_size=*/2, /*extra_weights_byte=*/0) /
    sizeof(float);
  // Can read (tile_size - kernel_size) elements after end of indirection buffer.
  const size_t i_elements = tile_size - kernel_size + output_height * step_height;
  const size_t c_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_dwconv_ghw_w(first_pass_tile, middle_pass_tile, last_pass_tile,
                            kernel_height, kernel_width,
                            channels, channel_tile, channel_subtile, channel_round,
                            k.data(), b.data(), /*scale=*/nullptr, w.data(),
                            /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, /*params=*/nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const float*> i(i_elements * num_buffers);
  xnn_indirection_init_dwconv2d(
    /*output_y_start=*/0, /*output_y_end=*/output_height,
    reinterpret_cast<const void**>(i.data()),
    a.data(),
    channels << XNN_LOG2_SIZEOF_FLOAT,
    z.data(),
    input_height, input_width,
    output_height, output_width,
    kernel_height, kernel_width,
    subsampling, subsampling,
    dilation, dilation,
    padding_top, padding_left,
    step_height, step_width, tile_size);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params, -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  const int input_advanced = tile_size - last_pass_tile;
  const int input_stride_elements = kernel_height * step_width - input_advanced;
  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (size_t y = 0; y < output_height; y++) {
      dwconv(channels, output_width,
        i.data() + buffer_index * i_elements + step_height * y,
        w.data() + buffer_index * w_elements,
        c.data() + buffer_index * c_elements + y * output_width * channels,
        input_stride_elements * sizeof(void*), 0,
        0, z.data(), kernel_size, buffer.data(), &params);
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
    uint64_t(state.iterations()) * (output_size + input_height * input_width + kernel_size + 1 /* bias */) * channels * sizeof(float),
    benchmark::Counter::kIsRate);
}


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_dwconv_9p4c__asm_aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p4c__neon,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }
  static void f32_dwconv_9p4c__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p4c__neonfma,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  BENCHMARK_DWCONV(f32_dwconv_9p4c__asm_aarch64_neonfma)
  BENCHMARK_DWCONV(f32_dwconv_9p4c__asm_aarch64_neonfma_cortex_a55)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_dwconv_4p4c__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p4c__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_4p4c__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p4c__neon,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_4p4c__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p4c__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_4p4c__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p4c__neonfma,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_9p4c__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p4c__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_9p4c__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p4c__neon,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_9p4c__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p4c__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_9p4c__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p4c__neonfma,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_25p4c__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_25p4c__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__neon,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_25p4c__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_25p4c__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__neonfma,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_4p8c__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p8c__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_4p8c__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p8c__neon,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_4p8c__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p8c__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_4p8c__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p8c__neonfma,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_9p8c__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p8c__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_9p8c__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p8c__neon,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_9p8c__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p8c__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_9p8c__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p8c__neonfma,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_25p8c__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_25p8c__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__neon,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_25p8c__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_25p8c__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__neonfma,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_4p16c__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p16c__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_4p16c__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p16c__neon,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_4p16c__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p16c__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_4p16c__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_4p16c__neonfma,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 4 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_9p16c__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p16c__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_9p16c__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p16c__neon,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_9p16c__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p16c__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_9p16c__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p16c__neonfma,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_25p16c__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p16c__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_25p16c__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p16c__neon,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_25p16c__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p16c__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_dwconv_25p16c__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p16c__neonfma,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 25 /* primary tile */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_dwconv_5f5m5l4c4s4r__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma,
      xnn_init_f32_minmax_scalar_params,
      5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_5f5m5l4c4s4r__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_5f5m5l8c4s4r__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma,
      xnn_init_f32_minmax_scalar_params,
      5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_5f5m5l8c4s4r__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_6f6m7l4c4s4r__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma,
      xnn_init_f32_minmax_scalar_params,
      6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_6f6m7l4c4s4r__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_6f6m7l8c4s4r__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma,
      xnn_init_f32_minmax_scalar_params,
      6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_6f6m7l8c4s4r__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_8f8m9l4c4s4r__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma,
      xnn_init_f32_minmax_scalar_params,
      8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_8f8m9l4c4s4r__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* first pass tile */, 8 /* middle pass tile */, 8 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_8f8m9l8c4s4r__neonfma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma,
      xnn_init_f32_minmax_scalar_params,
      8 /* first pass tile */, 8 /* middle pass tile */, 8 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_8f8m9l8c4s4r__neonfma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* first pass tile */, 8 /* middle pass tile */, 8 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_5f5m5l4c4s4r__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon,
      xnn_init_f32_minmax_scalar_params,
      5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_5f5m5l4c4s4r__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_5f5m5l8c4s4r__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon,
      xnn_init_f32_minmax_scalar_params,
      5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_5f5m5l8c4s4r__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_6f6m7l4c4s4r__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon,
      xnn_init_f32_minmax_scalar_params,
      6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_6f6m7l4c4s4r__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_6f6m7l8c4s4r__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon,
      xnn_init_f32_minmax_scalar_params,
      6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_6f6m7l8c4s4r__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_8f8m9l4c4s4r__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon,
      xnn_init_f32_minmax_scalar_params,
      8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_8f8m9l4c4s4r__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
      4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_8f8m9l8c4s4r__neon(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon,
      xnn_init_f32_minmax_scalar_params,
      8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }
  static void f32_dwconv_8f8m9l8c4s4r__neon_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
      8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_DWCONV(f32_dwconv_4p4c__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_4p4c__neonfma_acc2)
  BENCHMARK_DWCONV(f32_dwconv_4p8c__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_4p8c__neonfma_acc2)
  BENCHMARK_DWCONV(f32_dwconv_4p16c__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_4p16c__neonfma_acc2)

  BENCHMARK_DWCONV(f32_dwconv_9p4c__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_9p4c__neonfma_acc2)
  BENCHMARK_DWCONV(f32_dwconv_9p8c__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_9p8c__neonfma_acc2)
  BENCHMARK_DWCONV(f32_dwconv_9p16c__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_9p16c__neonfma_acc2)

  BENCHMARK_DWCONV(f32_dwconv_25p4c__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_25p4c__neonfma_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__neonfma_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p16c__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_25p16c__neonfma_acc2)

  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__neonfma_acc2)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l8c4s4r__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l8c4s4r__neonfma_acc2)

  BENCHMARK_DWCONV(f32_dwconv_6f6m7l4c4s4r__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l4c4s4r__neonfma_acc2)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l8c4s4r__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l8c4s4r__neonfma_acc2)

  BENCHMARK_DWCONV(f32_dwconv_8f8m9l4c4s4r__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l4c4s4r__neonfma_acc2)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l8c4s4r__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l8c4s4r__neonfma_acc2)

  BENCHMARK_DWCONV(f32_dwconv_4p4c__neon)
  BENCHMARK_DWCONV(f32_dwconv_4p4c__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_4p8c__neon)
  BENCHMARK_DWCONV(f32_dwconv_4p8c__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_4p16c__neon)
  BENCHMARK_DWCONV(f32_dwconv_4p16c__neon_acc2)

  BENCHMARK_DWCONV(f32_dwconv_9p4c__neon)
  BENCHMARK_DWCONV(f32_dwconv_9p4c__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_9p8c__neon)
  BENCHMARK_DWCONV(f32_dwconv_9p8c__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_9p16c__neon)
  BENCHMARK_DWCONV(f32_dwconv_9p16c__neon_acc2)

  BENCHMARK_DWCONV(f32_dwconv_25p4c__neon)
  BENCHMARK_DWCONV(f32_dwconv_25p4c__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__neon)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p16c__neon)
  BENCHMARK_DWCONV(f32_dwconv_25p16c__neon_acc2)

  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__neon)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l8c4s4r__neon)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l8c4s4r__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l4c4s4r__neon)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l4c4s4r__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l8c4s4r__neon)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l8c4s4r__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l4c4s4r__neon)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l4c4s4r__neon_acc2)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l8c4s4r__neon)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l8c4s4r__neon_acc2)

#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_dwconv_4p4c__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_4p4c__sse,
               xnn_init_f32_minmax_sse_params,
               4 /* channel tile */, 4 /* primary tile */);
  }
  static void f32_dwconv_9p4c__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_9p4c__sse,
               xnn_init_f32_minmax_sse_params,
               4 /* channel tile */, 9 /* primary tile */);
  }
  static void f32_dwconv_25p4c__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p4c__sse,
               xnn_init_f32_minmax_sse_params,
               4 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p8c__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p8c__sse,
               xnn_init_f32_minmax_sse_params,
               8 /* channel tile */, 25 /* primary tile */);
  }

  static void f32_dwconv_5f5m5l4c4s4r__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse, xnn_init_f32_minmax_sse_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_5f5m5l4c4s4r__sse_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_5f5m5l8c4s4r__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse, xnn_init_f32_minmax_sse_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_5f5m5l8c4s4r__sse_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_5f5m5l16c4s4r__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse, xnn_init_f32_minmax_sse_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               16 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_5f5m5l16c4s4r__sse_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               16 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }

  static void f32_dwconv_6f6m7l4c4s4r__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse, xnn_init_f32_minmax_sse_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_6f6m7l4c4s4r__sse_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_6f6m7l8c4s4r__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse, xnn_init_f32_minmax_sse_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_6f6m7l8c4s4r__sse_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_6f6m7l16c4s4r__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse, xnn_init_f32_minmax_sse_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               16 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_6f6m7l16c4s4r__sse_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               16 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }

  static void f32_dwconv_8f8m9l4c4s4r__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse, xnn_init_f32_minmax_sse_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_8f8m9l4c4s4r__sse_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l4c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_8f8m9l8c4s4r__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse, xnn_init_f32_minmax_sse_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_8f8m9l8c4s4r__sse_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l8c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               8 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_8f8m9l16c4s4r__sse(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse, xnn_init_f32_minmax_sse_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               16 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_8f8m9l16c4s4r__sse_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l16c4s4r__sse_acc2, xnn_init_f32_minmax_sse_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               16 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }

  static void f32_dwconv_25p8c__avx(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p8c__avx,
               xnn_init_f32_minmax_avx_params,
               8 /* channel tile */, 25 /* primary tile */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_25p8c__avx_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p8c__avx_acc2,
               xnn_init_f32_minmax_avx_params,
               8 /* channel tile */, 25 /* primary tile */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_25p16c__avx(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p16c__avx,
               xnn_init_f32_minmax_avx_params,
               16 /* channel tile */, 25 /* primary tile */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_25p16c__avx_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p16c__avx_acc2,
               xnn_init_f32_minmax_avx_params,
               16 /* channel tile */, 25 /* primary tile */,
               benchmark::utils::CheckAVX);
  }

  static void f32_dwconv_5f5m5l8c8s4r__avx(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx, xnn_init_f32_minmax_avx_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               8 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_5f5m5l8c8s4r__avx_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               8 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_5f5m5l16c8s4r__avx(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx, xnn_init_f32_minmax_avx_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               16 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_5f5m5l16c8s4r__avx_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               16 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }

  static void f32_dwconv_6f6m7l8c8s4r__avx(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx, xnn_init_f32_minmax_avx_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               8 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_6f6m7l8c8s4r__avx_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               8 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_6f6m7l16c8s4r__avx(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx, xnn_init_f32_minmax_avx_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               16 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_6f6m7l16c8s4r__avx_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               16 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }

  static void f32_dwconv_8f8m9l8c8s4r__avx(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx, xnn_init_f32_minmax_avx_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               8 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_8f8m9l8c8s4r__avx_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l8c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               8 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_8f8m9l16c8s4r__avx(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx, xnn_init_f32_minmax_avx_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               16 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_8f8m9l16c8s4r__avx_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l16c8s4r__avx_acc2, xnn_init_f32_minmax_avx_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               16 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckAVX);
  }

  static void f32_dwconv_25p8c__fma3(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p8c__fma3,
               xnn_init_f32_minmax_avx_params,
               8 /* channel tile */, 25 /* primary tile */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_25p8c__fma3_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p8c__fma3_acc2,
               xnn_init_f32_minmax_avx_params,
               8 /* channel tile */, 25 /* primary tile */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_25p16c__fma3(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p16c__fma3,
               xnn_init_f32_minmax_avx_params,
               16 /* channel tile */, 25 /* primary tile */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_25p16c__fma3_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p16c__fma3_acc2,
               xnn_init_f32_minmax_avx_params,
               16 /* channel tile */, 25 /* primary tile */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_5f5m5l8c8s4r__fma3(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f32_minmax_avx_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               8 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_5f5m5l8c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3_acc2, xnn_init_f32_minmax_avx_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               8 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_5f5m5l16c8s4r__fma3(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f32_minmax_avx_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               16 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_5f5m5l16c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3_acc2, xnn_init_f32_minmax_avx_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               16 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_5f5m5l32c8s4r__fma3(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f32_minmax_avx_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               32 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_5f5m5l32c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3_acc2, xnn_init_f32_minmax_avx_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               32 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_7f6m6l8c8s4r__fma3(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3, xnn_init_f32_minmax_avx_params,
               7 /* first pass tile */, 6 /* middle pass tile */, 6 /* last pass tile */,
               8 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_7f6m6l8c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_7f6m6l8c8s4r__fma3_acc2, xnn_init_f32_minmax_avx_params,
               7 /* first pass tile */, 6 /* middle pass tile */, 6 /* last pass tile */,
               8 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_7f6m6l16c8s4r__fma3(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3, xnn_init_f32_minmax_avx_params,
               7 /* first pass tile */, 6 /* middle pass tile */, 6 /* last pass tile */,
               16 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_7f6m6l16c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_7f6m6l16c8s4r__fma3_acc2, xnn_init_f32_minmax_avx_params,
               7 /* first pass tile */, 6 /* middle pass tile */, 6 /* last pass tile */,
               16 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_7f6m6l32c8s4r__fma3(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3, xnn_init_f32_minmax_avx_params,
               7 /* first pass tile */, 6 /* middle pass tile */, 6 /* last pass tile */,
               32 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_7f6m6l32c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_7f6m6l32c8s4r__fma3_acc2, xnn_init_f32_minmax_avx_params,
               7 /* first pass tile */, 6 /* middle pass tile */, 6 /* last pass tile */,
               32 /* channel tile */, 8 /* channel subtile */, 4 /* channel round */,
               benchmark::utils::CheckFMA3);
  }

  static void f32_dwconv_25p16c__avx512f(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p16c__avx512f,
               xnn_init_f32_minmax_scalar_params,
               16 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_25p16c__avx512f_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p16c__avx512f_acc2,
               xnn_init_f32_minmax_scalar_params,
               16 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_25p32c__avx512f(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p32c__avx512f,
               xnn_init_f32_minmax_scalar_params,
               32 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_25p32c__avx512f_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_25p32c__avx512f_acc2,
               xnn_init_f32_minmax_scalar_params,
               32 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckAVX512F);
  }

  static void f32_dwconv_5f5m5l16c16s1r__avx512f(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f, xnn_init_f32_minmax_scalar_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               16 /* channel tile */, 16 /* channel subtile */, 1 /* channel round */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_5f5m5l16c16s1r__avx512f_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l16c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               16 /* channel tile */, 16 /* channel subtile */, 1 /* channel round */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_5f5m5l32c16s1r__avx512f(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f, xnn_init_f32_minmax_scalar_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               32 /* channel tile */, 16 /* channel subtile */, 1 /* channel round */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_5f5m5l32c16s1r__avx512f_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l32c16s1r__avx512f_acc2, xnn_init_f32_minmax_scalar_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               32 /* channel tile */, 16 /* channel subtile */, 1 /* channel round */, benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_DWCONV(f32_dwconv_4p4c__sse)
  BENCHMARK_DWCONV(f32_dwconv_9p4c__sse)
  BENCHMARK_DWCONV(f32_dwconv_25p4c__sse)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__sse)

  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__sse)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__sse_acc2)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l8c4s4r__sse)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l8c4s4r__sse_acc2)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l16c4s4r__sse)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l16c4s4r__sse_acc2)

  BENCHMARK_DWCONV(f32_dwconv_6f6m7l4c4s4r__sse)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l4c4s4r__sse_acc2)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l8c4s4r__sse)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l8c4s4r__sse_acc2)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l16c4s4r__sse)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l16c4s4r__sse_acc2)

  BENCHMARK_DWCONV(f32_dwconv_8f8m9l4c4s4r__sse)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l4c4s4r__sse_acc2)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l8c4s4r__sse)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l8c4s4r__sse_acc2)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l16c4s4r__sse)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l16c4s4r__sse_acc2)

  BENCHMARK_DWCONV(f32_dwconv_25p8c__avx)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__avx_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p16c__avx)
  BENCHMARK_DWCONV(f32_dwconv_25p16c__avx_acc2)

  BENCHMARK_DWCONV(f32_dwconv_5f5m5l8c8s4r__avx)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l8c8s4r__avx_acc2)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l16c8s4r__avx)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l16c8s4r__avx_acc2)

  BENCHMARK_DWCONV(f32_dwconv_6f6m7l8c8s4r__avx)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l8c8s4r__avx_acc2)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l16c8s4r__avx)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l16c8s4r__avx_acc2)

  BENCHMARK_DWCONV(f32_dwconv_8f8m9l8c8s4r__avx)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l8c8s4r__avx_acc2)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l16c8s4r__avx)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l16c8s4r__avx_acc2)

  BENCHMARK_DWCONV(f32_dwconv_25p8c__fma3)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__fma3_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p16c__fma3)
  BENCHMARK_DWCONV(f32_dwconv_25p16c__fma3_acc2)

  BENCHMARK_DWCONV(f32_dwconv_5f5m5l8c8s4r__fma3)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l8c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l16c8s4r__fma3)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l16c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l32c8s4r__fma3)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l32c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f32_dwconv_7f6m6l8c8s4r__fma3)
  BENCHMARK_DWCONV(f32_dwconv_7f6m6l8c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f32_dwconv_7f6m6l16c8s4r__fma3)
  BENCHMARK_DWCONV(f32_dwconv_7f6m6l16c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f32_dwconv_7f6m6l32c8s4r__fma3)
  BENCHMARK_DWCONV(f32_dwconv_7f6m6l32c8s4r__fma3_acc2)

  BENCHMARK_DWCONV(f32_dwconv_25p16c__avx512f)
  BENCHMARK_DWCONV(f32_dwconv_25p16c__avx512f_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p32c__avx512f)
  BENCHMARK_DWCONV(f32_dwconv_25p32c__avx512f_acc2)

  BENCHMARK_DWCONV(f32_dwconv_5f5m5l16c16s1r__avx512f)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l16c16s1r__avx512f_acc2)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l32c16s1r__avx512f)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l32c16s1r__avx512f_acc2)
#endif  // XNN_ARCH_X88 || XNN_ARCH_X86_64


#if XNN_ARCH_WASM
  static void f32_dwconv_9p1c__wasm(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p1c__wasm,
      xnn_init_f32_minmax_scalar_params,
      1 /* channel tile */, 9 /* primary tile */);
  }
  static void f32_dwconv_9p1c__wasm_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_9p1c__wasm_acc2,
      xnn_init_f32_minmax_scalar_params,
      1 /* channel tile */, 9 /* primary tile */);
  }
  static void f32_dwconv_25p1c__wasm(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p1c__wasm,
      xnn_init_f32_minmax_scalar_params,
      1 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p1c__wasm_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p1c__wasm_acc2,
      xnn_init_f32_minmax_scalar_params,
      1 /* channel tile */, 25 /* primary tile */);
  }

  static void f32_dwconv_3f3m3l1c1s1r__wasm(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_3f3m3l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params,
               3 /* first pass tile */, 3 /* middle pass tile */, 3 /* last pass tile */,
               1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
  }
  static void f32_dwconv_3f3m3l1c1s1r__wasm_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_3f3m3l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params,
               3 /* first pass tile */, 3 /* middle pass tile */, 3 /* last pass tile */,
               1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
  }

  static void f32_dwconv_5f5m5l1c1s1r__wasm(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
  }
  static void f32_dwconv_5f5m5l1c1s1r__wasm_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
  }

  static void f32_dwconv_6f6m7l1c1s1r__wasm(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
  }
  static void f32_dwconv_6f6m7l1c1s1r__wasm_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params,
               6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
               1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
  }

  static void f32_dwconv_8f8m9l1c1s1r__wasm(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm, xnn_init_f32_minmax_scalar_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
  }
  static void f32_dwconv_8f8m9l1c1s1r__wasm_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__wasm_acc2, xnn_init_f32_minmax_scalar_params,
               8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
               1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
  }

  BENCHMARK_DWCONV(f32_dwconv_25p1c__wasm)
  BENCHMARK_DWCONV(f32_dwconv_25p1c__wasm_acc2)

  BENCHMARK_DWCONV(f32_dwconv_5f5m5l1c1s1r__wasm)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l1c1s1r__wasm_acc2)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l1c1s1r__wasm)
  BENCHMARK_DWCONV(f32_dwconv_6f6m7l1c1s1r__wasm_acc2)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l1c1s1r__wasm)
  BENCHMARK_DWCONV(f32_dwconv_8f8m9l1c1s1r__wasm_acc2)

#endif  // XNN_ARCH_WASMSIMD

#if XNN_ARCH_WASMSIMD
  static void f32_dwconv_25p4c__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p4c__wasmsimd_arm_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__wasmsimd_arm_acc2,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p4c__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p4c__wasmsimd_x86_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__wasmsimd_x86_acc2,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p8c__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__wasmsimd_arm,
      xnn_init_f32_minmax_wasmsimd_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p8c__wasmsimd_arm_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__wasmsimd_arm_acc2,
      xnn_init_f32_minmax_wasmsimd_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p8c__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__wasmsimd_x86,
      xnn_init_f32_minmax_wasmsimd_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p8c__wasmsimd_x86_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__wasmsimd_x86_acc2,
      xnn_init_f32_minmax_wasmsimd_params,
      8 /* channel tile */, 25 /* primary tile */);
  }

  static void f32_dwconv_5f5m5l4c4s4r__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm, xnn_init_f32_minmax_wasmsimd_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_5f5m5l4c4s4r__wasmsimd_arm_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_arm_acc2, xnn_init_f32_minmax_wasmsimd_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }

  static void f32_dwconv_5f5m5l4c4s4r__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86, xnn_init_f32_minmax_wasmsimd_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_5f5m5l4c4s4r__wasmsimd_x86_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmsimd_x86_acc2, xnn_init_f32_minmax_wasmsimd_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }

  BENCHMARK_DWCONV(f32_dwconv_25p4c__wasmsimd_arm)
  BENCHMARK_DWCONV(f32_dwconv_25p4c__wasmsimd_arm_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p4c__wasmsimd_x86)
  BENCHMARK_DWCONV(f32_dwconv_25p4c__wasmsimd_x86_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__wasmsimd_arm)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__wasmsimd_arm_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__wasmsimd_x86)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__wasmsimd_x86_acc2)

  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__wasmsimd_arm)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__wasmsimd_arm_acc2)

  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__wasmsimd_x86)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__wasmsimd_x86_acc2)
#endif

#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_dwconv_25p4c__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p4c__wasmrelaxedsimd_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__wasmrelaxedsimd_acc2,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p4c__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p4c__wasmrelaxedsimd_fma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p4c__wasmrelaxedsimd_fma_acc2,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p8c__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p8c__wasmrelaxedsimd_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_acc2,
      xnn_init_f32_minmax_wasmsimd_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p8c__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void f32_dwconv_25p8c__wasmrelaxedsimd_fma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
      xnn_f32_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_fma_acc2,
      xnn_init_f32_minmax_wasmsimd_params,
      8 /* channel tile */, 25 /* primary tile */);
  }

  static void f32_dwconv_5f5m5l4c4s4r__wasmrelaxedsimd(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd, xnn_init_f32_minmax_wasmsimd_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_5f5m5l4c4s4r__wasmrelaxedsimd_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_acc2, xnn_init_f32_minmax_wasmsimd_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_5f5m5l4c4s4r__wasmrelaxedsimd_fma(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma, xnn_init_f32_minmax_wasmsimd_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }
  static void f32_dwconv_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2(benchmark::State& state, const char* net) {
    f32_dwconv(state,
               xnn_f32_dwconv_minmax_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2, xnn_init_f32_minmax_wasmsimd_params,
               5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
               4 /* channel tile */, 4 /* channel subtile */, 4 /* channel round */);
  }

  BENCHMARK_DWCONV(f32_dwconv_25p4c__wasmrelaxedsimd)
  BENCHMARK_DWCONV(f32_dwconv_25p4c__wasmrelaxedsimd_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p4c__wasmrelaxedsimd_fma)
  BENCHMARK_DWCONV(f32_dwconv_25p4c__wasmrelaxedsimd_fma_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__wasmrelaxedsimd)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__wasmrelaxedsimd_acc2)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__wasmrelaxedsimd_fma)
  BENCHMARK_DWCONV(f32_dwconv_25p8c__wasmrelaxedsimd_fma_acc2)

  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__wasmrelaxedsimd)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__wasmrelaxedsimd_acc2)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__wasmrelaxedsimd_fma)
  BENCHMARK_DWCONV(f32_dwconv_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2)
#endif


static void f32_dwconv_4p1c__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_4p1c__scalar,
    xnn_init_f32_minmax_scalar_params,
    1 /* channel tile */, 4 /* primary tile */);
}
static void f32_dwconv_4p1c__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_4p1c__scalar_acc2,
    xnn_init_f32_minmax_scalar_params,
    1 /* channel tile */, 4 /* primary tile */);
}
static void f32_dwconv_4p2c__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_4p2c__scalar,
    xnn_init_f32_minmax_scalar_params,
    2 /* channel tile */, 4 /* primary tile */);
}
static void f32_dwconv_4p2c__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_4p2c__scalar_acc2,
    xnn_init_f32_minmax_scalar_params,
    2 /* channel tile */, 4 /* primary tile */);
}
static void f32_dwconv_9p1c__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_9p1c__scalar,
    xnn_init_f32_minmax_scalar_params,
    1 /* channel tile */, 9 /* primary tile */);
}
static void f32_dwconv_9p1c__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_9p1c__scalar_acc2,
    xnn_init_f32_minmax_scalar_params,
    1 /* channel tile */, 9 /* primary tile */);
}
static void f32_dwconv_9p2c__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_9p2c__scalar,
    xnn_init_f32_minmax_scalar_params,
    2 /* channel tile */, 9 /* primary tile */);
}
static void f32_dwconv_9p2c__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_9p2c__scalar_acc2,
    xnn_init_f32_minmax_scalar_params,
    2 /* channel tile */, 9 /* primary tile */);
}
static void f32_dwconv_25p1c__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_25p1c__scalar,
    xnn_init_f32_minmax_scalar_params,
    1 /* channel tile */, 25 /* primary tile */);
}
static void f32_dwconv_25p1c__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_25p1c__scalar_acc2,
    xnn_init_f32_minmax_scalar_params,
    1 /* channel tile */, 25 /* primary tile */);
}
static void f32_dwconv_25p2c__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_25p2c__scalar,
    xnn_init_f32_minmax_scalar_params,
    2 /* channel tile */, 25 /* primary tile */);
}
static void f32_dwconv_25p2c__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
    xnn_f32_dwconv_minmax_ukernel_25p2c__scalar_acc2,
    xnn_init_f32_minmax_scalar_params,
    2 /* channel tile */, 25 /* primary tile */);
}

static void f32_dwconv_2f2m2l1c1s1r__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
             xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params,
             2 /* first pass tile */, 2 /* middle pass tile */, 2 /* last pass tile */,
             1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
}
static void f32_dwconv_2f2m2l1c1s1r__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
             xnn_f32_dwconv_minmax_ukernel_2f2m2l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params,
             2 /* first pass tile */, 2 /* middle pass tile */, 2 /* last pass tile */,
             1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
}
static void f32_dwconv_2f2m2l4c1s1r__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
             xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar, xnn_init_f32_minmax_scalar_params,
             2 /* first pass tile */, 2 /* middle pass tile */, 2 /* last pass tile */,
             4 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
}
static void f32_dwconv_2f2m2l4c1s1r__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
             xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params,
             2 /* first pass tile */, 2 /* middle pass tile */, 2 /* last pass tile */,
             4 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
}
static void f32_dwconv_5f5m5l1c1s1r__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
             xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params,
             5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
             1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
}
static void f32_dwconv_5f5m5l1c1s1r__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
             xnn_f32_dwconv_minmax_ukernel_5f5m5l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params,
             5 /* first pass tile */, 5 /* middle pass tile */, 5 /* last pass tile */,
             1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
}
static void f32_dwconv_6f6m7l1c1s1r__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
             xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params,
             6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
             1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
}
static void f32_dwconv_6f6m7l1c1s1r__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
             xnn_f32_dwconv_minmax_ukernel_6f6m7l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params,
             6 /* first pass tile */, 6 /* middle pass tile */, 7 /* last pass tile */,
             1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
}
static void f32_dwconv_8f8m9l1c1s1r__scalar(benchmark::State& state, const char* net) {
  f32_dwconv(state,
             xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar, xnn_init_f32_minmax_scalar_params,
             8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
             1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
}
static void f32_dwconv_8f8m9l1c1s1r__scalar_acc2(benchmark::State& state, const char* net) {
  f32_dwconv(state,
             xnn_f32_dwconv_minmax_ukernel_8f8m9l1c1s1r__scalar_acc2, xnn_init_f32_minmax_scalar_params,
             8 /* first pass tile */, 8 /* middle pass tile */, 9 /* last pass tile */,
             1 /* channel tile */, 1 /* channel subtile */, 1 /* channel round */);
}

BENCHMARK_DWCONV(f32_dwconv_4p1c__scalar)
BENCHMARK_DWCONV(f32_dwconv_4p1c__scalar_acc2)
BENCHMARK_DWCONV(f32_dwconv_4p2c__scalar)
BENCHMARK_DWCONV(f32_dwconv_4p2c__scalar_acc2)

BENCHMARK_DWCONV(f32_dwconv_9p1c__scalar)
BENCHMARK_DWCONV(f32_dwconv_9p1c__scalar_acc2)
BENCHMARK_DWCONV(f32_dwconv_9p2c__scalar)
BENCHMARK_DWCONV(f32_dwconv_9p2c__scalar_acc2)

BENCHMARK_DWCONV(f32_dwconv_25p1c__scalar)
BENCHMARK_DWCONV(f32_dwconv_25p1c__scalar_acc2)
BENCHMARK_DWCONV(f32_dwconv_25p2c__scalar)
BENCHMARK_DWCONV(f32_dwconv_25p2c__scalar_acc2)

BENCHMARK_DWCONV(f32_dwconv_2f2m2l1c1s1r__scalar)
BENCHMARK_DWCONV(f32_dwconv_2f2m2l1c1s1r__scalar_acc2)
BENCHMARK_DWCONV(f32_dwconv_2f2m2l4c1s1r__scalar)
BENCHMARK_DWCONV(f32_dwconv_2f2m2l4c1s1r__scalar_acc2)
BENCHMARK_DWCONV(f32_dwconv_5f5m5l1c1s1r__scalar)
BENCHMARK_DWCONV(f32_dwconv_5f5m5l1c1s1r__scalar_acc2)
BENCHMARK_DWCONV(f32_dwconv_6f6m7l1c1s1r__scalar)
BENCHMARK_DWCONV(f32_dwconv_6f6m7l1c1s1r__scalar_acc2)
BENCHMARK_DWCONV(f32_dwconv_8f8m9l1c1s1r__scalar)
BENCHMARK_DWCONV(f32_dwconv_8f8m9l1c1s1r__scalar_acc2)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
