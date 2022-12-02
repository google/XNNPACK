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

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/indirection.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>


static void f16_dwconv(benchmark::State& state,
  xnn_f16_dwconv_minmax_unipass_ukernel_fn dwconv,
  xnn_init_f16_minmax_params_fn init_params,
  uint32_t channel_tile, uint32_t primary_tile,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
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
  xnn_pack_f16_dwconv_ghw_w(primary_tile, kernel_height, kernel_width, channels, channel_tile,
      k.data(), b.data(), w.data(), 0 /* extra bytes */, nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const uint16_t*> i(i_elements * num_buffers);
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

  xnn_indirection_init_dwconv2d(&convolution_op, step_height, step_width, primary_tile, 1 /* log2(sizeof(uint16_t)) */);
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
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
