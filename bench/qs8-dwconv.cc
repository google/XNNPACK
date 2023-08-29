// Copyright 2021 Google LLC
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

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/indirection.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microkernel-utils.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>


static void DWConvBenchmark(benchmark::State& state,
  xnn_qs8_dwconv_minmax_unipass_ukernel_fn dwconv,
  xnn_init_qs8_conv_minmax_params_fn init_params,
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
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()), std::ref(rng));

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

  std::vector<int8_t> a(channels * input_height * input_width + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<int8_t> k(channels * kernel_height * kernel_width);
  std::generate(k.begin(), k.end(), std::ref(i8rng));
  std::vector<int32_t> b(channels);
  std::generate(b.begin(), b.end(), std::ref(i32rng));

  std::vector<int8_t> z(channels + XNN_EXTRA_BYTES / sizeof(int8_t));

  const size_t k_elements = kernel_size * c_stride;
  const size_t b_elements = c_stride;
  const size_t w_size = k_elements * sizeof(int8_t) + b_elements * sizeof(int32_t);
  // Can read (primary_tile - kernel_size) elements after end of indirection buffer.
  const size_t i_elements = (primary_tile - kernel_size) + output_height * step_height;
  const size_t c_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      (c_elements * sizeof(int8_t) + w_size) + sizeof(void*) * i_elements);

  std::vector<char, AlignedAllocator<char, 64>> w(w_size * num_buffers);
  std::fill(w.begin(), w.end(), 0);
  struct xnn_qs8_packing_params packing_params;
  packing_params.input_zero_point = 0;
  xnn_pack_qs8_dwconv_ghw_w(primary_tile, 0, 0, kernel_height, kernel_width, channels,
                            channel_tile, channel_tile, /*channel_round=*/1,
                            k.data(), b.data(), /*scale=*/nullptr, w.data(),
                            /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, &packing_params);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_size, w.begin() + n * w_size);
  }

  std::vector<const int8_t*> i(i_elements * num_buffers);
  xnn_operator convolution_op = { };
  convolution_op.indirection_buffer = reinterpret_cast<const void**>(i.data());
  convolution_op.input              = a.data();
  convolution_op.input_pixel_stride = channels;
  convolution_op.zero_buffer        = z.data();
  convolution_op.input_height       = input_height;
  convolution_op.input_width        = input_width;
  convolution_op.output_height      = output_height;
  convolution_op.output_width       = output_width;
  convolution_op.kernel_height      = kernel_height;
  convolution_op.kernel_width       = kernel_width;
  convolution_op.stride_height      = subsampling;
  convolution_op.stride_width       = subsampling;
  convolution_op.dilation_height    = dilation;
  convolution_op.dilation_width     = dilation;
  convolution_op.padding_top        = padding_top;
  convolution_op.padding_left       = padding_left;

  xnn_indirection_init_dwconv2d(
    /*output_y_start=*/0, /*output_y_end=*/convolution_op.output_height,
    convolution_op.indirection_buffer,
    convolution_op.input,
    convolution_op.input_pixel_stride << XNN_LOG2_SIZEOF_INT8_T,
    convolution_op.zero_buffer,
    convolution_op.input_height, convolution_op.input_width,
    convolution_op.output_height, convolution_op.output_width,
    convolution_op.kernel_height, convolution_op.kernel_width,
    convolution_op.stride_height, convolution_op.stride_width,
    convolution_op.dilation_height, convolution_op.dilation_width,
    convolution_op.padding_top, convolution_op.padding_left,
    step_height, step_width, primary_tile);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  std::vector<int8_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), INT8_C(0));

  xnn_qs8_conv_minmax_params params;
  init_params(&params,
    0.5f /* scale */, 0 /* output zero point */, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(int8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (size_t y = 0; y < output_height; y++) {
      dwconv(channels, output_width,
        i.data() + buffer_index * i_elements + step_height * y,
        w.data() + buffer_index * w_size,
        c.data() + buffer_index * c_elements + y * output_width * channels,
        kernel_height * step_width * sizeof(void*), 0,
        0, z.data(), &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * output_size * channels * kernel_size,
    benchmark::Counter::kIsRate);

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) * channels * ((output_size + input_height * input_width + kernel_size) * sizeof(int8_t) + sizeof(int32_t)),
    benchmark::Counter::kIsRate);
}

static void DWConvBenchmark(benchmark::State& state,
  xnn_qs8_dwconv_minmax_multipass_ukernel_fn dwconv,
  xnn_init_qs8_conv_minmax_params_fn init_params,
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
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), std::ref(rng));
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(-std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::max()), std::ref(rng));

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

  std::vector<int8_t> a(channels * input_height * input_width + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::vector<int8_t> k(channels * kernel_size);
  std::generate(k.begin(), k.end(), std::ref(i8rng));
  std::vector<int32_t> b(channels);
  std::generate(b.begin(), b.end(), std::ref(i32rng));

  std::vector<int8_t> z(channels + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int32_t, AlignedAllocator<int32_t, 64>> buffer(
    channels + XNN_MULTIPASS_EXTRA_BYTES / sizeof(int8_t));

  const size_t tile_size = xnn_dwconv_multipass_tile_size(
    kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile);
  const size_t k_elements = xnn_dwconv_multipass_weights_size(
    tile_size, channels, channel_tile, channel_subtile, channel_round,
    /*bias_element_size=*/sizeof(int32_t), /*log2_filter_element_size=*/0, /*extra_weights_byte=*/0);

  const size_t b_elements = c_stride;
  const size_t w_size = k_elements * sizeof(int8_t) + b_elements * sizeof(int32_t);
  // Can read (primary_tile - kernel_size) elements after end of indirection buffer.
  const size_t i_elements = tile_size - kernel_size + output_height * step_height;
  const size_t c_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      (c_elements * sizeof(int8_t) + w_size) + sizeof(void*) * i_elements);

  std::vector<char, AlignedAllocator<char, 64>> w(w_size * num_buffers);
  std::fill(w.begin(), w.end(), 0);
  struct xnn_qs8_packing_params packing_params;
  packing_params.input_zero_point = 0;
  xnn_pack_qs8_dwconv_ghw_w(first_pass_tile, middle_pass_tile, last_pass_tile,
                            kernel_height, kernel_width,
                            channels, channel_tile, channel_tile, channel_round,
                            k.data(), b.data(), /*scale=*/nullptr, w.data(),
                            /*per_tile_extra_bytes=*/0, /*per_subtile_extra_bytes=*/0, &packing_params);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_size, w.begin() + n * w_size);
  }

  std::vector<const int8_t*> i(i_elements * num_buffers);
  xnn_operator convolution_op = { };
  convolution_op.indirection_buffer = reinterpret_cast<const void**>(i.data());
  convolution_op.input              = a.data();
  convolution_op.input_pixel_stride = channels;
  convolution_op.zero_buffer        = z.data();
  convolution_op.input_height       = input_height;
  convolution_op.input_width        = input_width;
  convolution_op.output_height      = output_height;
  convolution_op.output_width       = output_width;
  convolution_op.kernel_height      = kernel_height;
  convolution_op.kernel_width       = kernel_width;
  convolution_op.stride_height      = subsampling;
  convolution_op.stride_width       = subsampling;
  convolution_op.dilation_height    = dilation;
  convolution_op.dilation_width     = dilation;
  convolution_op.padding_top        = padding_top;
  convolution_op.padding_left       = padding_left;

  xnn_indirection_init_dwconv2d(
    /*output_y_start=*/0, /*output_y_end=*/convolution_op.output_height,
    convolution_op.indirection_buffer,
    convolution_op.input,
    convolution_op.input_pixel_stride << XNN_LOG2_SIZEOF_INT8_T,
    convolution_op.zero_buffer,
    convolution_op.input_height, convolution_op.input_width,
    convolution_op.output_height, convolution_op.output_width,
    convolution_op.kernel_height, convolution_op.kernel_width,
    convolution_op.stride_height, convolution_op.stride_width,
    convolution_op.dilation_height, convolution_op.dilation_width,
    convolution_op.padding_top, convolution_op.padding_left,
    step_height, step_width, tile_size);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  std::vector<int8_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), INT8_C(0));

  xnn_qs8_conv_minmax_params params;
  init_params(&params,
    0.5f /* scale */, 0 /* output zero point */, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

  const int input_advanced = tile_size - last_pass_tile;
  const int input_stride_elements = kernel_height * step_width - input_advanced;
  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(int8_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (size_t y = 0; y < output_height; y++) {
      dwconv(channels, output_width,
        i.data() + buffer_index * i_elements + step_height * y,
        w.data() + buffer_index * w_size,
        c.data() + buffer_index * c_elements + y * output_width * channels,
        input_stride_elements * sizeof(void*), 0,
        0, z.data(), kernel_size, buffer.data(), &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * output_size * channels * kernel_size,
    benchmark::Counter::kIsRate);

  state.counters["bytes"] = benchmark::Counter(
    uint64_t(state.iterations()) * channels * ((output_size + input_height * input_width + kernel_size) * sizeof(int8_t) + sizeof(int32_t)),
    benchmark::Counter::kIsRate);
}


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_dwconv_9p8c__neon_mul8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_9p16c__neon_mul8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_9p16c__neon_mul8_ld128(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul8_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_9p8c__neon_mla8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mla8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_9p16c__neon_mla8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_9p16c__neon_mla8_ld128(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mla8_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_9p8c__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_9p8c__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_9p16c__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_9p16c__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_9p32c__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_9p32c__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_25p8c__neon_mul8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      8 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_25p16c__neon_mul8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_25p16c__neon_mul8_ld128(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul8_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_25p8c__neon_mla8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mla8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      8 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_25p16c__neon_mla8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_25p16c__neon_mla8_ld128(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mla8_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_25p8c__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_25p8c__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      8 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_25p16c__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_25p16c__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_25p32c__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      32 /* channel tile */, 25 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void qs8_dwconv_5f5m5l8c8s8r__neon_mul8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l8c8s8r__neon_mul8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_5f5m5l8c8s8r__neon_mla8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l8c8s8r__neon_mla8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__neon_mul8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l16c8s8r__neon_mul8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__neon_mla8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l16c8s8r__neon_mla8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__neon_mul8_ld128(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l16c8s8r__neon_mul8_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__neon_mla8_ld128(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l16c8s8r__neon_mla8_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_5f5m5l8c8s8r__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l8c8s8r__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l16c8s8r__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_5f5m5l32c8s8r__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_5f5m5l32c8s8r__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }

  static void qs8_dwconv_6f6m7l8c8s8r__neon_mul8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l8c8s8r__neon_mul8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_6f6m7l8c8s8r__neon_mla8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l8c8s8r__neon_mla8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__neon_mul8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l16c8s8r__neon_mul8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__neon_mla8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l16c8s8r__neon_mla8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__neon_mul8_ld128(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l16c8s8r__neon_mul8_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__neon_mla8_ld128(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l16c8s8r__neon_mla8_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_6f6m7l8c8s8r__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l8c8s8r__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l16c8s8r__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_6f6m7l32c8s8r__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_6f6m7l32c8s8r__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }

  static void qs8_dwconv_8f8m9l8c8s8r__neon_mul8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l8c8s8r__neon_mul8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_8f8m9l8c8s8r__neon_mla8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l8c8s8r__neon_mla8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__neon_mul8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l16c8s8r__neon_mul8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__neon_mla8_ld64(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l16c8s8r__neon_mla8_ld64,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__neon_mul8_ld128(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l16c8s8r__neon_mul8_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__neon_mla8_ld128(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l16c8s8r__neon_mla8_ld128,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_8f8m9l8c8s8r__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l8c8s8r__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l16c8s8r__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }
  static void qs8_dwconv_8f8m9l32c8s8r__neon_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_rndnu_ukernel_8f8m9l32c8s8r__neon_mul16,
      xnn_init_qs8_conv_minmax_rndnu_neon_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_DWCONV(qs8_dwconv_9p8c__neon_mul8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__neon_mul8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__neon_mul8_ld128);
  BENCHMARK_DWCONV(qs8_dwconv_9p8c__neon_mla8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__neon_mla8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__neon_mla8_ld128);
  BENCHMARK_DWCONV(qs8_dwconv_9p8c__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_9p32c__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_25p8c__neon_mul8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__neon_mul8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__neon_mul8_ld128);
  BENCHMARK_DWCONV(qs8_dwconv_25p8c__neon_mla8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__neon_mla8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__neon_mla8_ld128);
  BENCHMARK_DWCONV(qs8_dwconv_25p8c__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_25p32c__neon_mul16);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c8s8r__neon_mul8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c8s8r__neon_mla8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__neon_mul8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__neon_mla8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__neon_mul8_ld128);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__neon_mla8_ld128);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c8s8r__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l32c8s8r__neon_mul16);

  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c8s8r__neon_mul8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c8s8r__neon_mla8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__neon_mul8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__neon_mla8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__neon_mul8_ld128);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__neon_mla8_ld128);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c8s8r__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l32c8s8r__neon_mul16);

  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c8s8r__neon_mul8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c8s8r__neon_mla8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__neon_mul8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__neon_mla8_ld64);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__neon_mul8_ld128);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__neon_mla8_ld128);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c8s8r__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__neon_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l32c8s8r__neon_mul16);

#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qs8_dwconv_9p16c__avx512skx_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx512skx_mul32,
      xnn_init_qs8_conv_minmax_fp32_avx512_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512SKX);
  }
  static void qs8_dwconv_9p32c__avx512skx_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p32c__avx512skx_mul32,
      xnn_init_qs8_conv_minmax_fp32_avx512_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512SKX);
  }
  static void qs8_dwconv_9p16c__avx2_mul16_vpmovsx(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul16_vpmovsx,
      xnn_init_qs8_conv_minmax_fp32_avx2_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_9p32c__avx2_mul16_vpmovsx(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p32c__avx2_mul16_vpmovsx,
      xnn_init_qs8_conv_minmax_fp32_avx2_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_9p16c__avx2_mul16_vpunpck(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul16_vpunpck,
      xnn_init_qs8_conv_minmax_fp32_avx2_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_9p32c__avx2_mul16_vpunpck(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p32c__avx2_mul16_vpunpck,
      xnn_init_qs8_conv_minmax_fp32_avx2_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_9p16c__avx2_mul16_add16_vpunpck(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul16_add16_vpunpck,
      xnn_init_qs8_conv_minmax_fp32_avx2_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_9p32c__avx2_mul16_add16_vpunpck(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p32c__avx2_mul16_add16_vpunpck,
      xnn_init_qs8_conv_minmax_fp32_avx2_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_9p8c__avx2_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__avx2_mul32,
      xnn_init_qs8_conv_minmax_fp32_avx2_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_9p16c__avx2_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx2_mul32,
      xnn_init_qs8_conv_minmax_fp32_avx2_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_9p32c__avx2_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p32c__avx2_mul32,
      xnn_init_qs8_conv_minmax_fp32_avx2_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_9p8c__xop_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__xop_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckXOP);
  }
  static void qs8_dwconv_9p16c__xop_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__xop_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckXOP);
  }
  static void qs8_dwconv_9p8c__avx_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__avx_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_9p16c__avx_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_9p8c__avx_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__avx_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_9p16c__avx_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_9p8c__avx_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__avx_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_9p16c__avx_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__avx_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_9p8c__sse41_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_9p16c__sse41_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__sse41_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_9p8c__sse41_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_9p16c__sse41_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__sse41_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_9p8c__sse41_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__sse41_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_9p16c__sse41_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__sse41_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_9p8c__sse2_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      8 /* channel tile */, 9 /* primary tile */);
  }
  static void qs8_dwconv_9p16c__sse2_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__sse2_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      16 /* channel tile */, 9 /* primary tile */);
  }
  static void qs8_dwconv_9p8c__sse2_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__sse2_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      8 /* channel tile */, 9 /* primary tile */);
  }
  static void qs8_dwconv_9p16c__sse2_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__sse2_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      16 /* channel tile */, 9 /* primary tile */);
  }

  static void qs8_dwconv_25p8c__sse2_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void qs8_dwconv_25p16c__sse2_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__sse2_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      16 /* channel tile */, 25 /* primary tile */);
  }

  static void qs8_dwconv_25p8c__sse2_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__sse2_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void qs8_dwconv_25p16c__sse2_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__sse2_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      16 /* channel tile */, 25 /* primary tile */);
  }

  static void qs8_dwconv_25p8c__sse41_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void qs8_dwconv_25p16c__sse41_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__sse41_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      16 /* channel tile */, 25 /* primary tile */);
  }
  static void qs8_dwconv_25p8c__sse41_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void qs8_dwconv_25p16c__sse41_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__sse41_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      16 /* channel tile */, 25 /* primary tile */);
  }

  static void qs8_dwconv_25p8c__sse41_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__sse41_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void qs8_dwconv_25p16c__sse41_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__sse41_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      16 /* channel tile */, 25 /* primary tile */);
  }

  static void qs8_dwconv_5f5m5l8c8s8r__sse2_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__sse2_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__sse2_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__sse2_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_6f6m7l8c8s8r__sse2_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__sse2_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__sse2_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__sse2_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_8f8m9l8c8s8r__sse2_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__sse2_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__sse2_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__sse2_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }

  static void qs8_dwconv_5f5m5l8c8s8r__sse2_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__sse2_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__sse2_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__sse2_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_6f6m7l8c8s8r__sse2_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__sse2_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__sse2_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__sse2_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_8f8m9l8c8s8r__sse2_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__sse2_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__sse2_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__sse2_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse2_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }

  static void qs8_dwconv_5f5m5l8c8s8r__sse41_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__sse41_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__sse41_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__sse41_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_6f6m7l8c8s8r__sse41_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__sse41_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__sse41_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__sse41_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_8f8m9l8c8s8r__sse41_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__sse41_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__sse41_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__sse41_mul16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_dwconv_5f5m5l8c8s8r__sse41_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__sse41_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__sse41_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__sse41_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_6f6m7l8c8s8r__sse41_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__sse41_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__sse41_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__sse41_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_8f8m9l8c8s8r__sse41_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__sse41_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__sse41_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__sse41_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_dwconv_5f5m5l8c4s4r__sse41_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l8c4s4r__sse41_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_5f5m5l16c4s4r__sse41_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l16c4s4r__sse41_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_6f6m7l8c4s4r__sse41_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l8c4s4r__sse41_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_6f6m7l16c4s4r__sse41_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l16c4s4r__sse41_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_8f8m9l8c4s4r__sse41_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l8c4s4r__sse41_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_dwconv_8f8m9l16c4s4r__sse41_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l16c4s4r__sse41_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_dwconv_5f5m5l8c4s4r__avx_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l8c4s4r__avx_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_5f5m5l16c4s4r__avx_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l16c4s4r__avx_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_6f6m7l8c4s4r__avx_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l8c4s4r__avx_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_6f6m7l16c4s4r__avx_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l16c4s4r__avx_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_8f8m9l8c4s4r__avx_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l8c4s4r__avx_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckAVX);
  }
  static void qs8_dwconv_8f8m9l16c4s4r__avx_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l16c4s4r__avx_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckAVX);
  }

  static void qs8_dwconv_5f5m5l8c4s4r__xop_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l8c4s4r__xop_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckXOP);
  }
  static void qs8_dwconv_5f5m5l16c4s4r__xop_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l16c4s4r__xop_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckXOP);
  }
  static void qs8_dwconv_6f6m7l8c4s4r__xop_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l8c4s4r__xop_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckXOP);
  }
  static void qs8_dwconv_6f6m7l16c4s4r__xop_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l16c4s4r__xop_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckXOP);
  }
  static void qs8_dwconv_8f8m9l8c4s4r__xop_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l8c4s4r__xop_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckXOP);
  }
  static void qs8_dwconv_8f8m9l16c4s4r__xop_mul32(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l16c4s4r__xop_mul32,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/4, /*channel_round=*/4,
      benchmark::utils::CheckXOP);
  }

  static void qs8_dwconv_5f5m5l16c16s16r__avx2_mul16_vpmovsx(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l16c16s16r__avx2_mul16_vpmovsx,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/16, /*channel_round=*/16,
      benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_5f5m5l32c16s16r__avx2_mul16_vpmovsx(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l32c16s16r__avx2_mul16_vpmovsx,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/5, /*middle_pass_tile=*/5, /*last_pass_tile=*/5,
      /*channel_tile=*/32, /*channel_subtile=*/16, /*channel_round=*/16,
      benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_6f6m7l16c16s16r__avx2_mul16_vpmovsx(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l16c16s16r__avx2_mul16_vpmovsx,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/16, /*channel_round=*/16,
      benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_6f6m7l32c16s16r__avx2_mul16_vpmovsx(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l32c16s16r__avx2_mul16_vpmovsx,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/6, /*middle_pass_tile=*/6, /*last_pass_tile=*/7,
      /*channel_tile=*/32, /*channel_subtile=*/16, /*channel_round=*/16,
      benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_8f8m9l16c16s16r__avx2_mul16_vpmovsx(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l16c16s16r__avx2_mul16_vpmovsx,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/16, /*channel_round=*/16,
      benchmark::utils::CheckAVX2);
  }
  static void qs8_dwconv_8f8m9l32c16s16r__avx2_mul16_vpmovsx(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l32c16s16r__avx2_mul16_vpmovsx,
      xnn_init_qs8_conv_minmax_fp32_sse4_params,
      /*first_pass_tile=*/8, /*middle_pass_tile=*/8, /*last_pass_tile=*/9,
      /*channel_tile=*/32, /*channel_subtile=*/16, /*channel_round=*/16,
      benchmark::utils::CheckAVX2);
  }

  BENCHMARK_DWCONV(qs8_dwconv_9p16c__avx512skx_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_9p32c__avx512skx_mul32);

  BENCHMARK_DWCONV(qs8_dwconv_9p16c__avx2_mul16_vpmovsx);
  BENCHMARK_DWCONV(qs8_dwconv_9p32c__avx2_mul16_vpmovsx);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__avx2_mul16_vpunpck);
  BENCHMARK_DWCONV(qs8_dwconv_9p32c__avx2_mul16_vpunpck);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__avx2_mul16_add16_vpunpck);
  BENCHMARK_DWCONV(qs8_dwconv_9p32c__avx2_mul16_add16_vpunpck);
  BENCHMARK_DWCONV(qs8_dwconv_9p8c__avx2_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__avx2_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_9p32c__avx2_mul32);

  BENCHMARK_DWCONV(qs8_dwconv_9p8c__xop_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__xop_mul16_add16);

  BENCHMARK_DWCONV(qs8_dwconv_9p8c__avx_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__avx_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_9p8c__avx_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__avx_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_9p8c__avx_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__avx_mul32);

  BENCHMARK_DWCONV(qs8_dwconv_9p8c__sse41_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__sse41_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_9p8c__sse41_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__sse41_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_9p8c__sse41_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__sse41_mul32);

  BENCHMARK_DWCONV(qs8_dwconv_9p8c__sse2_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__sse2_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_9p8c__sse2_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__sse2_mul16_add16);

  BENCHMARK_DWCONV(qs8_dwconv_25p8c__sse2_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__sse2_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_25p8c__sse2_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__sse2_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_25p8c__sse41_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__sse41_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_25p8c__sse41_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__sse41_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_25p8c__sse41_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__sse41_mul32);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c8s8r__sse2_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__sse2_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c8s8r__sse2_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__sse2_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c8s8r__sse2_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__sse2_mul16);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c8s8r__sse2_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__sse2_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c8s8r__sse2_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__sse2_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c8s8r__sse2_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__sse2_mul16_add16);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c8s8r__sse41_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__sse41_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c8s8r__sse41_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__sse41_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c8s8r__sse41_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__sse41_mul16);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c8s8r__sse41_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__sse41_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c8s8r__sse41_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__sse41_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c8s8r__sse41_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__sse41_mul16_add16);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c4s4r__sse41_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c4s4r__sse41_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c4s4r__sse41_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c4s4r__sse41_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c4s4r__sse41_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c4s4r__sse41_mul32);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c4s4r__avx_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c4s4r__avx_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c4s4r__avx_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c4s4r__avx_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c4s4r__avx_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c4s4r__avx_mul32);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c4s4r__xop_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c4s4r__xop_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c4s4r__xop_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c4s4r__xop_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c4s4r__xop_mul32);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c4s4r__xop_mul32);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c16s16r__avx2_mul16_vpmovsx);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l32c16s16r__avx2_mul16_vpmovsx);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c16s16r__avx2_mul16_vpmovsx);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l32c16s16r__avx2_mul16_vpmovsx);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c16s16r__avx2_mul16_vpmovsx);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l32c16s16r__avx2_mul16_vpmovsx);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qs8_dwconv_9p8c__wasmsimd_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__wasmsimd_mul16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      8 /* channel tile */, 9 /* primary tile */);
  }
  static void qs8_dwconv_9p16c__wasmsimd_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__wasmsimd_mul16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      16 /* channel tile */, 9 /* primary tile */);
  }

  static void qs8_dwconv_9p8c__wasmsimd_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p8c__wasmsimd_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      8 /* channel tile */, 9 /* primary tile */);
  }
  static void qs8_dwconv_9p16c__wasmsimd_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p16c__wasmsimd_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      16 /* channel tile */, 9 /* primary tile */);
  }

  static void qs8_dwconv_25p8c__wasmsimd_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__wasmsimd_mul16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void qs8_dwconv_25p16c__wasmsimd_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__wasmsimd_mul16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      16 /* channel tile */, 25 /* primary tile */);
  }

  static void qs8_dwconv_25p8c__wasmsimd_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p8c__wasmsimd_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      8 /* channel tile */, 25 /* primary tile */);
  }
  static void qs8_dwconv_25p16c__wasmsimd_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p16c__wasmsimd_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      16 /* channel tile */, 25 /* primary tile */);
  }

  static void qs8_dwconv_5f5m5l8c8s8r__wasmsimd_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__wasmsimd_mul16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__wasmsimd_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__wasmsimd_mul16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_5f5m5l8c8s8r__wasmsimd_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l8c8s8r__wasmsimd_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_5f5m5l16c8s8r__wasmsimd_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l16c8s8r__wasmsimd_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }

  static void qs8_dwconv_6f6m7l8c8s8r__wasmsimd_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__wasmsimd_mul16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__wasmsimd_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__wasmsimd_mul16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_6f6m7l8c8s8r__wasmsimd_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l8c8s8r__wasmsimd_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_6f6m7l16c8s8r__wasmsimd_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l16c8s8r__wasmsimd_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }

  static void qs8_dwconv_8f8m9l8c8s8r__wasmsimd_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__wasmsimd_mul16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__wasmsimd_mul16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__wasmsimd_mul16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_8f8m9l8c8s8r__wasmsimd_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l8c8s8r__wasmsimd_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/8);
  }
  static void qs8_dwconv_8f8m9l16c8s8r__wasmsimd_mul16_add16(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l16c8s8r__wasmsimd_mul16_add16,
      xnn_init_qs8_conv_minmax_fp32_wasmsimd_params,
      /*first_pass_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/8);
  }

  BENCHMARK_DWCONV(qs8_dwconv_9p8c__wasmsimd_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__wasmsimd_mul16);

  BENCHMARK_DWCONV(qs8_dwconv_9p8c__wasmsimd_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_9p16c__wasmsimd_mul16_add16);

  BENCHMARK_DWCONV(qs8_dwconv_25p8c__wasmsimd_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__wasmsimd_mul16);

  BENCHMARK_DWCONV(qs8_dwconv_25p8c__wasmsimd_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_25p16c__wasmsimd_mul16_add16);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c8s8r__wasmsimd_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__wasmsimd_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l8c8s8r__wasmsimd_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l16c8s8r__wasmsimd_mul16_add16);

  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c8s8r__wasmsimd_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__wasmsimd_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l8c8s8r__wasmsimd_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l16c8s8r__wasmsimd_mul16_add16);

  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c8s8r__wasmsimd_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__wasmsimd_mul16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l8c8s8r__wasmsimd_mul16_add16);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l16c8s8r__wasmsimd_mul16_add16);

#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qs8_dwconv_9p1c__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p1c__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      1 /* channel tile */, 9 /* primary tile */);
  }
  static void qs8_dwconv_9p2c__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p2c__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      2 /* channel tile */, 9 /* primary tile */);
  }
  static void qs8_dwconv_9p4c__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_9p4c__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void qs8_dwconv_25p1c__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p1c__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      1 /* channel tile */, 25 /* primary tile */);
  }
  static void qs8_dwconv_25p2c__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p2c__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      2 /* channel tile */, 25 /* primary tile */);
  }
  static void qs8_dwconv_25p4c__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_25p4c__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      4 /* channel tile */, 25 /* primary tile */);
  }

  static void qs8_dwconv_5f5m5l1c1s1r__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l1c1s1r__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      /*first_pass_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*channel_tile=*/1, /*channel_subtile=*/1, /*channel_round=*/1);
  }
  static void qs8_dwconv_5f5m5l2c1s1r__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l2c1s1r__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      /*first_pass_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*channel_tile=*/2, /*channel_subtile=*/1, /*channel_round=*/1);
  }
  static void qs8_dwconv_5f5m5l4c1s1r__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l4c1s1r__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      /*first_pass_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*channel_tile=*/4, /*channel_subtile=*/1, /*channel_round=*/1);
  }

  static void qs8_dwconv_6f6m7l1c1s1r__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l1c1s1r__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      /*first_pass_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*channel_tile=*/1, /*channel_subtile=*/1, /*channel_round=*/1);
  }
  static void qs8_dwconv_6f6m7l2c1s1r__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l2c1s1r__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      /*first_pass_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*channel_tile=*/2, /*channel_subtile=*/1, /*channel_round=*/1);
  }
  static void qs8_dwconv_6f6m7l4c1s1r__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_6f6m7l4c1s1r__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      /*first_pass_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*channel_tile=*/4, /*channel_subtile=*/1, /*channel_round=*/1);
  }

  static void qs8_dwconv_8f8m9l1c1s1r__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l1c1s1r__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      /*first_pass_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*channel_tile=*/1, /*channel_subtile=*/1, /*channel_round=*/1);
  }
  static void qs8_dwconv_8f8m9l2c1s1r__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l2c1s1r__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      /*first_pass_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*channel_tile=*/2, /*channel_subtile=*/1, /*channel_round=*/1);
  }
  static void qs8_dwconv_8f8m9l4c1s1r__wasm_fmagic(benchmark::State& state, const char* net) {
    DWConvBenchmark(state,
      xnn_qs8_dwconv_minmax_fp32_ukernel_8f8m9l4c1s1r__wasm_fmagic,
      xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
      /*first_pass_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*channel_tile=*/4, /*channel_subtile=*/1, /*channel_round=*/1);
  }

  BENCHMARK_DWCONV(qs8_dwconv_9p1c__wasm_fmagic);
  BENCHMARK_DWCONV(qs8_dwconv_9p2c__wasm_fmagic);
  BENCHMARK_DWCONV(qs8_dwconv_9p4c__wasm_fmagic);

  BENCHMARK_DWCONV(qs8_dwconv_25p1c__wasm_fmagic);
  BENCHMARK_DWCONV(qs8_dwconv_25p2c__wasm_fmagic);
  BENCHMARK_DWCONV(qs8_dwconv_25p4c__wasm_fmagic);

  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l1c1s1r__wasm_fmagic);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l2c1s1r__wasm_fmagic);
  BENCHMARK_DWCONV(qs8_dwconv_5f5m5l4c1s1r__wasm_fmagic);

  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l1c1s1r__wasm_fmagic);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l2c1s1r__wasm_fmagic);
  BENCHMARK_DWCONV(qs8_dwconv_6f6m7l4c1s1r__wasm_fmagic);

  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l1c1s1r__wasm_fmagic);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l2c1s1r__wasm_fmagic);
  BENCHMARK_DWCONV(qs8_dwconv_8f8m9l4c1s1r__wasm_fmagic);
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


static void qs8_dwconv_9p1c__scalar_fmagic(benchmark::State& state, const char* net) {
  DWConvBenchmark(state,
    xnn_qs8_dwconv_minmax_fp32_ukernel_9p1c__scalar_fmagic,
    xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
    1 /* channel tile */, 9 /* primary tile */);
}
static void qs8_dwconv_9p2c__scalar_fmagic(benchmark::State& state, const char* net) {
  DWConvBenchmark(state,
    xnn_qs8_dwconv_minmax_fp32_ukernel_9p2c__scalar_fmagic,
    xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
    2 /* channel tile */, 9 /* primary tile */);
}
static void qs8_dwconv_9p4c__scalar_fmagic(benchmark::State& state, const char* net) {
  DWConvBenchmark(state,
    xnn_qs8_dwconv_minmax_fp32_ukernel_9p4c__scalar_fmagic,
    xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params,
    4 /* channel tile */, 9 /* primary tile */);
}

static void qs8_dwconv_9p1c__scalar_imagic(benchmark::State& state, const char* net) {
  DWConvBenchmark(state,
    xnn_qs8_dwconv_minmax_fp32_ukernel_9p1c__scalar_imagic,
    xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params,
    1 /* channel tile */, 9 /* primary tile */);
}
static void qs8_dwconv_9p2c__scalar_imagic(benchmark::State& state, const char* net) {
  DWConvBenchmark(state,
    xnn_qs8_dwconv_minmax_fp32_ukernel_9p2c__scalar_imagic,
    xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params,
    2 /* channel tile */, 9 /* primary tile */);
}
static void qs8_dwconv_9p4c__scalar_imagic(benchmark::State& state, const char* net) {
  DWConvBenchmark(state,
    xnn_qs8_dwconv_minmax_fp32_ukernel_9p4c__scalar_imagic,
    xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params,
    4 /* channel tile */, 9 /* primary tile */);
}

static void qs8_dwconv_9p1c__scalar_lrintf(benchmark::State& state, const char* net) {
  DWConvBenchmark(state,
    xnn_qs8_dwconv_minmax_fp32_ukernel_9p1c__scalar_lrintf,
    xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params,
    1 /* channel tile */, 9 /* primary tile */);
}
static void qs8_dwconv_9p2c__scalar_lrintf(benchmark::State& state, const char* net) {
  DWConvBenchmark(state,
    xnn_qs8_dwconv_minmax_fp32_ukernel_9p2c__scalar_lrintf,
    xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params,
    2 /* channel tile */, 9 /* primary tile */);
}
static void qs8_dwconv_9p4c__scalar_lrintf(benchmark::State& state, const char* net) {
  DWConvBenchmark(state,
    xnn_qs8_dwconv_minmax_fp32_ukernel_9p4c__scalar_lrintf,
    xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params,
    4 /* channel tile */, 9 /* primary tile */);
}

BENCHMARK_DWCONV(qs8_dwconv_9p1c__scalar_fmagic);
BENCHMARK_DWCONV(qs8_dwconv_9p2c__scalar_fmagic);
BENCHMARK_DWCONV(qs8_dwconv_9p4c__scalar_fmagic);

BENCHMARK_DWCONV(qs8_dwconv_9p1c__scalar_imagic);
BENCHMARK_DWCONV(qs8_dwconv_9p2c__scalar_imagic);
BENCHMARK_DWCONV(qs8_dwconv_9p4c__scalar_imagic);

BENCHMARK_DWCONV(qs8_dwconv_9p1c__scalar_lrintf);
BENCHMARK_DWCONV(qs8_dwconv_9p2c__scalar_lrintf);
BENCHMARK_DWCONV(qs8_dwconv_9p4c__scalar_lrintf);


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
