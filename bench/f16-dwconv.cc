// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "dwconv.h"
#include "utils.h"
#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/indirection.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microkernel-utils.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include <benchmark/benchmark.h>

static void bench_impl(uint64_t arch_flags, benchmark::State& state,
                       xnn_f16_dwconv_minmax_unipass_ukernel_fn dwconv,
                       xnn_init_f16_minmax_params_fn init_params,
                       uint32_t channel_tile, uint32_t primary_tile) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
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

  xnnpack::Buffer<xnn_float16> a(channels * input_height * input_width + XNN_EXTRA_BYTES / sizeof(xnn_float16));
  std::generate(a.begin(), a.end(), f32rng);
  xnnpack::Buffer<xnn_float16> k(channels * kernel_height * kernel_width);
  std::generate(k.begin(), k.end(), f32rng);
  xnnpack::Buffer<xnn_float16> b(channels);
  std::generate(b.begin(), b.end(), f32rng);

  xnnpack::Buffer<xnn_float16> z(channels + XNN_EXTRA_BYTES / sizeof(xnn_float16));

  const size_t w_elements = (kernel_size + 1) * c_stride;
  // Can read (primary_tile - kernel_size) elements after end of indirection buffer.
  const size_t i_elements = (primary_tile - kernel_size) + output_height * step_height;
  const size_t c_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(xnn_float16) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> w(w_elements * num_buffers);
  xnn_pack_f16_dwconv_ghw_w(primary_tile, 0, 0, kernel_height, kernel_width, channels,
                            channel_tile, channel_tile, /*channel_round=*/1,
                            reinterpret_cast<const uint16_t*>(k.data()),
                            reinterpret_cast<const uint16_t*>(b.data()),
                            /*scale=*/nullptr, reinterpret_cast<uint16_t*>(w.data()),
                            /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, /*params=*/nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  xnnpack::Buffer<const xnn_float16*> i(i_elements * num_buffers);
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

  xnnpack::Buffer<xnn_float16> c(c_elements * num_buffers);

  xnn_f16_minmax_params params;
  init_params(&params, static_cast<xnn_float16>(-INFINITY), static_cast<xnn_float16>(INFINITY));

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(xnn_float16));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (size_t y = 0; y < output_height; y++) {
      dwconv(channels, output_width,
        reinterpret_cast<const xnn_float16**>(i.data() + buffer_index * i_elements + step_height * y),
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
    uint64_t(state.iterations()) * (output_size + input_height * input_width + kernel_size + 1 /* bias */) * channels * sizeof(xnn_float16),
    benchmark::Counter::kIsRate);
}

static void bench_impl(uint64_t arch_flags, benchmark::State& state,
                       xnn_f16_dwconv_minmax_multipass_ukernel_fn dwconv,
                       xnn_init_f16_minmax_params_fn init_params,
                       uint32_t first_pass_tile, uint32_t middle_pass_tile,
                       uint32_t last_pass_tile, uint32_t channel_tile,
                       uint32_t channel_subtile, uint32_t channel_round) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
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

  xnnpack::Buffer<xnn_float16> a(channels * input_height * input_width + XNN_EXTRA_BYTES / sizeof(xnn_float16));
  std::generate(a.begin(), a.end(), f32rng);
  xnnpack::Buffer<xnn_float16> k(channels * kernel_size);
  std::generate(k.begin(), k.end(), f32rng);
  xnnpack::Buffer<xnn_float16> b(channels);
  std::generate(b.begin(), b.end(), f32rng);

  xnnpack::Buffer<xnn_float16> z(channels + XNN_EXTRA_BYTES / sizeof(xnn_float16));
  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> buffer(
      channels + XNN_MULTIPASS_EXTRA_BYTES / sizeof(xnn_float16));

  const size_t tile_size = xnn_dwconv_multipass_tile_size(
    kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile);
  const size_t w_elements =
    xnn_dwconv_multipass_weights_size(
      tile_size, channels, channel_tile, channel_subtile, channel_round, /*bias_element_size=*/sizeof(xnn_float16),
      /*log2_filter_element_size=*/1, /*extra_weights_byte=*/0) /
    sizeof(xnn_float16);
  // Can read (primary_tile - kernel_size) elements after end of indirection buffer.
  const size_t i_elements = tile_size - kernel_size + output_height * step_height;
  const size_t c_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(xnn_float16) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> w(w_elements * num_buffers);
  xnn_pack_f16_dwconv_ghw_w(
    first_pass_tile, middle_pass_tile, last_pass_tile,
    kernel_height, kernel_width,
    channels, channel_tile, channel_subtile, channel_round,
    reinterpret_cast<const uint16_t*>(k.data()),
    reinterpret_cast<const uint16_t*>(b.data()),
    /*scale=*/nullptr, reinterpret_cast<uint16_t*>(w.data()),
    /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  xnnpack::Buffer<const xnn_float16*> i(i_elements * num_buffers);
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

  xnnpack::Buffer<xnn_float16> c(c_elements * num_buffers);

  xnn_f16_minmax_params params;
  init_params(&params, static_cast<xnn_float16>(-INFINITY), static_cast<xnn_float16>(INFINITY));

  const int input_advanced = tile_size - last_pass_tile;
  const int input_stride_elements = kernel_height * step_width - input_advanced;
  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(xnn_float16));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (size_t y = 0; y < output_height; y++) {
      dwconv(channels, output_width,
        reinterpret_cast<const xnn_float16**>(i.data() + buffer_index * i_elements + step_height * y),
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
    uint64_t(state.iterations()) * (output_size + input_height * input_width + kernel_size + 1 /* bias */) * channels * sizeof(xnn_float16),
    benchmark::Counter::kIsRate);
}

#define XNN_DWCONV_UNIPASS(arch_flags, ukernel, c_block, is_pipelined, cr, kr, \
                           datatype, weights_type, params_type, init_params)   \
  static void BM_##ukernel(benchmark::State& state, const char* net) {         \
    bench_impl(arch_flags, state, ukernel, init_params, cr, kr);               \
  }                                                                            \
  BENCHMARK_DWCONV(BM_##ukernel);

#define XNN_DWCONV_MULTIPASS(                                               \
    arch_flags, ukernel, first_pass_tile, middle_pass_tile, last_pass_tile, \
    channel_tile, channel_subtile, channel_round, datatype, weights_type,   \
    buffer_type, params_type, init_params)                                  \
  static void BM_##ukernel(benchmark::State& state, const char* net) {      \
    bench_impl(arch_flags, state, ukernel, init_params, first_pass_tile,    \
               middle_pass_tile, last_pass_tile, channel_tile,              \
               channel_subtile, channel_round);                             \
  }                                                                         \
  BENCHMARK_DWCONV(BM_##ukernel);

// #include "f16-dwconv/f16-dwconv-unipass.h"
// #include "f16-dwconv/f16-dwconv-multipass.h"
#include "f16-dwconv/f16-dwconv-minmax-multipass.h"
#include "f16-dwconv/f16-dwconv-minmax-unipass.h"

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
