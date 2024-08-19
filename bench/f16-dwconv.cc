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

#include <benchmark/benchmark.h>
#include <fp16/fp16.h>
#include "bench/dwconv.h"
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/indirection.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microkernel-utils.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"


static void f16_dwconv(benchmark::State& state,
  xnn_f16_dwconv_minmax_unipass_ukernel_fn dwconv,
  xnn_init_f16_minmax_params_fn init_params,
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
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

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

  std::vector<uint16_t> a(channels * input_height * input_width + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::generate(a.begin(), a.end(), std::ref(f16rng));
  std::vector<uint16_t> k(channels * kernel_height * kernel_width);
  std::generate(k.begin(), k.end(), std::ref(f16rng));
  std::vector<uint16_t> b(channels);
  std::generate(b.begin(), b.end(), std::ref(f16rng));

  std::vector<uint16_t> z(channels + XNN_EXTRA_BYTES / sizeof(uint16_t));

  const size_t w_elements = (kernel_size + 1) * c_stride;
  // Can read (primary_tile - kernel_size) elements after end of indirection buffer.
  const size_t i_elements = (primary_tile - kernel_size) + output_height * step_height;
  const size_t c_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(uint16_t) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), UINT16_C(0));
  xnn_pack_f16_dwconv_ghw_w(primary_tile, 0, 0, kernel_height, kernel_width, channels,
                            channel_tile, channel_tile, /*channel_round=*/1,
                            k.data(), b.data(), /*scale=*/nullptr, w.data(),
                            /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, /*params=*/nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const uint16_t*> i(i_elements * num_buffers);
  xnn_indirection_init_dwconv2d(
    /*output_y_start=*/0, /*output_y_end=*/output_height,
    reinterpret_cast<const void**>(i.data()),
    a.data(),
    channels << XNN_LOG2_SIZEOF_HALF,
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

  std::vector<uint16_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);

  xnn_f16_minmax_params params;
  init_params(&params, UINT16_C(0xFC00) /* -inf */, UINT16_C(0x7C00) /* inf */);

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(uint16_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (size_t y = 0; y < output_height; y++) {
      dwconv(channels, output_width,
        reinterpret_cast<const void**>(i.data() + buffer_index * i_elements + step_height * y),
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
    uint64_t(state.iterations()) * 2 * output_size * channels * kernel_size, benchmark::Counter::kIsRate);

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) * (output_size + input_height * input_width + kernel_size + 1 /* bias */) * channels * sizeof(uint16_t),
    benchmark::Counter::kIsRate);
}

static void f16_dwconv(benchmark::State& state,
  xnn_f16_dwconv_minmax_multipass_ukernel_fn dwconv,
  xnn_init_f16_minmax_params_fn init_params,
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
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;
  const size_t output_size = output_height * output_width;
  const size_t step_width = dilation == 1 ? std::min(subsampling, kernel_width) : kernel_width;
  const size_t step_height = kernel_size + (output_width - 1) * step_width * kernel_height;

  std::vector<uint16_t> a(channels * input_height * input_width + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::generate(a.begin(), a.end(), std::ref(f16rng));
  std::vector<uint16_t> k(channels * kernel_size);
  std::generate(k.begin(), k.end(), std::ref(f16rng));
  std::vector<uint16_t> b(channels);
  std::generate(b.begin(), b.end(), std::ref(f16rng));

  std::vector<uint16_t> z(channels + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> buffer(channels + XNN_MULTIPASS_EXTRA_BYTES / sizeof(uint16_t));

  const size_t tile_size = xnn_dwconv_multipass_tile_size(
    kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile);
  const size_t w_elements =
    xnn_dwconv_multipass_weights_size(
      tile_size, channels, channel_tile, channel_subtile, channel_round, /*bias_element_size=*/sizeof(uint16_t),
      /*log2_filter_element_size=*/1, /*extra_weights_byte=*/0) /
    sizeof(uint16_t);
  // Can read (primary_tile - kernel_size) elements after end of indirection buffer.
  const size_t i_elements = tile_size - kernel_size + output_height * step_height;
  const size_t c_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(uint16_t) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), UINT16_C(0));
  xnn_pack_f16_dwconv_ghw_w(
    first_pass_tile, middle_pass_tile, last_pass_tile,
    kernel_height, kernel_width,
    channels, channel_tile, channel_subtile, channel_round,
    k.data(), b.data(), /*scale=*/nullptr, w.data(), /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const uint16_t*> i(i_elements * num_buffers);
  xnn_indirection_init_dwconv2d(
    /*output_y_start=*/0, /*output_y_end=*/output_height,
    reinterpret_cast<const void**>(i.data()),
    a.data(),
    channels << XNN_LOG2_SIZEOF_HALF,
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

  std::vector<uint16_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);

  xnn_f16_minmax_params params;
  init_params(&params, UINT16_C(0xFC00) /* -inf */, UINT16_C(0x7C00) /* inf */);

  const int input_advanced = tile_size - last_pass_tile;
  const int input_stride_elements = kernel_height * step_width - input_advanced;
  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(uint16_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (size_t y = 0; y < output_height; y++) {
      dwconv(channels, output_width,
        reinterpret_cast<const void**>(i.data() + buffer_index * i_elements + step_height * y),
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
    uint64_t(state.iterations()) * 2 * output_size * channels * kernel_size, benchmark::Counter::kIsRate);

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) * (output_size + input_height * input_width + kernel_size + 1 /* bias */) * channels * sizeof(uint16_t),
    benchmark::Counter::kIsRate);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void f16_dwconv_4p8c__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_4p8c__neonfp16arith_acc2,
      xnn_init_f16_minmax_fp16arith_params,
      8, 4, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_4p8c__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_4p8c__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      8, 4, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_9p8c__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_9p8c__neonfp16arith_acc2,
      xnn_init_f16_minmax_fp16arith_params,
      8, 9, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_9p8c__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_9p8c__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      8, 9, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_25p8c__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_25p8c__neonfp16arith_acc2,
      xnn_init_f16_minmax_fp16arith_params,
      8, 25, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_25p8c__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_25p8c__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      8, 25, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_4p16c__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_4p16c__neonfp16arith_acc2,
      xnn_init_f16_minmax_fp16arith_params,
      16, 4, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_4p16c__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_4p16c__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      16, 4, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_9p16c__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_9p16c__neonfp16arith_acc2,
      xnn_init_f16_minmax_fp16arith_params,
      16, 9, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_9p16c__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_9p16c__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      16, 9, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_25p16c__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_25p16c__neonfp16arith_acc2,
      xnn_init_f16_minmax_fp16arith_params,
      16, 25, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_25p16c__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_25p16c__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      16, 25, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_4p32c__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_4p32c__neonfp16arith_acc2,
      xnn_init_f16_minmax_fp16arith_params,
      32, 4, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_4p32c__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_4p32c__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      32, 4, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_9p32c__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_9p32c__neonfp16arith_acc2,
      xnn_init_f16_minmax_fp16arith_params,
      32, 9, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_9p32c__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_9p32c__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      32, 9, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_25p32c__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_25p32c__neonfp16arith_acc2,
      xnn_init_f16_minmax_fp16arith_params,
      32, 25, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_25p32c__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(state,
      xnn_f16_dwconv_minmax_ukernel_25p32c__neonfp16arith,
      xnn_init_f16_minmax_fp16arith_params,
      32, 25, benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_5f5m5l8c8s4r__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_5f5m5l8c8s4r__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_5f5m5l16c8s4r__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_5f5m5l16c8s4r__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_5f5m5l32c8s4r__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_5f5m5l32c8s4r__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_6f6m7l8c8s4r__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_6f6m7l8c8s4r__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_6f6m7l16c8s4r__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_6f6m7l16c8s4r__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_6f6m7l32c8s4r__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_6f6m7l32c8s4r__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_8f8m9l8c8s4r__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_8f8m9l8c8s4r__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_8f8m9l16c8s4r__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_8f8m9l16c8s4r__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_8f8m9l32c8s4r__neonfp16arith(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_8f8m9l32c8s4r__neonfp16arith_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_DWCONV(f16_dwconv_4p8c__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_4p8c__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_9p8c__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_9p8c__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_25p8c__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_25p8c__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_4p16c__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_4p16c__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_9p16c__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_9p16c__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_25p16c__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_25p16c__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_4p32c__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_4p32c__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_9p32c__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_9p32c__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_25p32c__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_25p32c__neonfp16arith)

  BENCHMARK_DWCONV(f16_dwconv_5f5m5l8c8s4r__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_5f5m5l8c8s4r__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_5f5m5l16c8s4r__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_5f5m5l16c8s4r__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_5f5m5l32c8s4r__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_5f5m5l32c8s4r__neonfp16arith_acc2)

  BENCHMARK_DWCONV(f16_dwconv_6f6m7l8c8s4r__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_6f6m7l8c8s4r__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_6f6m7l16c8s4r__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_6f6m7l16c8s4r__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_6f6m7l32c8s4r__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_6f6m7l32c8s4r__neonfp16arith_acc2)

  BENCHMARK_DWCONV(f16_dwconv_8f8m9l8c8s4r__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_8f8m9l8c8s4r__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_8f8m9l16c8s4r__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_8f8m9l16c8s4r__neonfp16arith_acc2)
  BENCHMARK_DWCONV(f16_dwconv_8f8m9l32c8s4r__neonfp16arith)
  BENCHMARK_DWCONV(f16_dwconv_8f8m9l32c8s4r__neonfp16arith_acc2)
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f16_dwconv_25p8c__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_25p8c__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/8, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_25p8c__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_25p8c__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/8, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_25p16c__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_25p16c__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/16, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_25p16c__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_25p16c__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/16, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_25p32c__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_25p32c__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/32, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_25p32c__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_25p32c__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/32, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }

  static void f16_dwconv_5f5m5l8c8s4r__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_5f5m5l8c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_5f5m5l16c8s4r__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_5f5m5l16c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_5f5m5l32c8s4r__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_5f5m5l32c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }

  static void f16_dwconv_6f6m7l8c8s4r__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_6f6m7l8c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_6f6m7l16c8s4r__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_6f6m7l16c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_6f6m7l32c8s4r__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_6f6m7l32c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }

  static void f16_dwconv_8f8m9l8c8s4r__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_8f8m9l8c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_8f8m9l16c8s4r__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_8f8m9l16c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_8f8m9l32c8s4r__fma3(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_8f8m9l32c8s4r__fma3_acc2(benchmark::State& state, const char* net) {
    f16_dwconv(
      state, xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }

  BENCHMARK_DWCONV(f16_dwconv_25p8c__fma3)
  BENCHMARK_DWCONV(f16_dwconv_25p8c__fma3_acc2)
  BENCHMARK_DWCONV(f16_dwconv_25p16c__fma3)
  BENCHMARK_DWCONV(f16_dwconv_25p16c__fma3_acc2)
  BENCHMARK_DWCONV(f16_dwconv_25p32c__fma3)
  BENCHMARK_DWCONV(f16_dwconv_25p32c__fma3_acc2)

  BENCHMARK_DWCONV(f16_dwconv_5f5m5l8c8s4r__fma3)
  BENCHMARK_DWCONV(f16_dwconv_5f5m5l8c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f16_dwconv_5f5m5l16c8s4r__fma3)
  BENCHMARK_DWCONV(f16_dwconv_5f5m5l16c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f16_dwconv_5f5m5l32c8s4r__fma3)
  BENCHMARK_DWCONV(f16_dwconv_5f5m5l32c8s4r__fma3_acc2)

  BENCHMARK_DWCONV(f16_dwconv_6f6m7l8c8s4r__fma3)
  BENCHMARK_DWCONV(f16_dwconv_6f6m7l8c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f16_dwconv_6f6m7l16c8s4r__fma3)
  BENCHMARK_DWCONV(f16_dwconv_6f6m7l16c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f16_dwconv_6f6m7l32c8s4r__fma3)
  BENCHMARK_DWCONV(f16_dwconv_6f6m7l32c8s4r__fma3_acc2)

  BENCHMARK_DWCONV(f16_dwconv_8f8m9l8c8s4r__fma3)
  BENCHMARK_DWCONV(f16_dwconv_8f8m9l8c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f16_dwconv_8f8m9l16c8s4r__fma3)
  BENCHMARK_DWCONV(f16_dwconv_8f8m9l16c8s4r__fma3_acc2)
  BENCHMARK_DWCONV(f16_dwconv_8f8m9l32c8s4r__fma3)
  BENCHMARK_DWCONV(f16_dwconv_8f8m9l32c8s4r__fma3_acc2)

#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
