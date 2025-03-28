// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>

#include "bench/dwconv.h"
#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/indirection.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microkernel-utils.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/pack.h"
#include <benchmark/benchmark.h>

static void bench_impl(uint64_t arch_flags, benchmark::State& state,
                       xnn_qs8_dwconv_minmax_ukernel_fn dwconv,
                       xnn_init_qs8_conv_minmax_params_fn init_params,
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
  auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000),
                          std::ref(rng));
  auto i8rng = std::bind(std::uniform_int_distribution<int32_t>(
                             -std::numeric_limits<int8_t>::max(),
                             std::numeric_limits<int8_t>::max()),
                         std::ref(rng));

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t output_height =
      (input_height + padding_height - effective_kernel_height) / subsampling +
      1;
  const size_t output_width =
      (input_width + padding_width - effective_kernel_width) / subsampling + 1;
  const size_t output_size = output_height * output_width;
  const size_t step_width =
      dilation == 1 ? std::min(subsampling, kernel_width) : kernel_width;
  const size_t step_height =
      kernel_size + (output_width - 1) * step_width * kernel_height;

  const size_t c_stride =
      benchmark::utils::RoundUp<size_t>(channels, channel_tile);

  xnnpack::Buffer<int8_t> a(channels * input_height * input_width +
                            XNN_EXTRA_BYTES / sizeof(int8_t));
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  xnnpack::Buffer<int8_t> k(channels * kernel_height * kernel_width);
  std::generate(k.begin(), k.end(), std::ref(i8rng));
  xnnpack::Buffer<int32_t> b(channels);
  std::generate(b.begin(), b.end(), std::ref(i32rng));

  // Zero buffer needs to be initialized with zeros.
  xnnpack::Buffer<int8_t> z(channels + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::fill(z.begin(), z.end(), 0);

  const size_t k_elements = kernel_size * c_stride;
  const size_t b_elements = c_stride;
  const size_t w_size =
      k_elements * sizeof(int8_t) + b_elements * sizeof(int32_t);
  // Can read (primary_tile - kernel_size) elements after end of indirection
  // buffer.
  const size_t i_elements =
      (primary_tile - kernel_size) + output_height * step_height;
  const size_t c_elements = output_size * channels;
  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     (c_elements * sizeof(int8_t) + w_size) +
                                         sizeof(void*) * i_elements);

  // Explicitly initialize the weights buffer since `num_buffers` may be larger
  // than the number of buffers that are actually initialized/needed.
  xnnpack::Buffer<char, XNN_ALLOCATION_ALIGNMENT> w(w_size * num_buffers);
  std::fill(w.begin(), w.end(), 0);

  // Pack the weights buffer.
  struct xnn_qs8_packing_params packing_params;
  packing_params.input_zero_point = 0;
  xnn_pack_qs8_dwconv_ghw_w(primary_tile, kernel_height, kernel_width, channels,
                            channel_tile, k.data(), b.data(),
                            /*scale=*/nullptr, w.data(),
                            /*per_tile_extra_bytes=*/0, &packing_params);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_size, w.begin() + n * w_size);
  }

  xnnpack::Buffer<const int8_t*> i(i_elements * num_buffers);
  xnn_indirection_init_dwconv2d(
      /*output_y_start=*/0, /*output_y_end=*/output_height,
      reinterpret_cast<const void**>(i.data()), a.data(),
      channels << XNN_LOG2_SIZEOF_INT8_T, z.data(), input_height, input_width,
      output_height, output_width, kernel_height, kernel_width, subsampling,
      subsampling, dilation, dilation, padding_top, padding_left, step_height,
      step_width, primary_tile);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  xnnpack::Buffer<int8_t> c(c_elements * num_buffers);

  xnn_qs8_conv_minmax_params params;
  init_params(&params, 0.5f /* scale */, 0 /* output zero point */,
              std::numeric_limits<int8_t>::min(),
              std::numeric_limits<int8_t>::max());

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
             kernel_height * step_width * sizeof(void*), 0, 0, z.data(),
             &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] =
      benchmark::Counter(static_cast<uint64_t>(state.iterations()) * 2 *
                             output_size * channels * kernel_size,
                         benchmark::Counter::kIsRate);

  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * channels *
          ((output_size + input_height * input_width + kernel_size) *
               sizeof(int8_t) +
           sizeof(int32_t)),
      benchmark::Counter::kIsRate);
}

#define XNN_UKERNEL(arch_flags, ukernel, c_block, is_pipelined, cr, kr, \
                    datatype, weights_type, params_type, init_params)   \
  static void BM_##ukernel(benchmark::State& state, const char* net) {  \
    bench_impl(arch_flags, state, ukernel, init_params, cr, kr);        \
  }                                                                     \
  BENCHMARK_DWCONV(BM_##ukernel);

#include "src/qs8-dwconv/qs8-dwconv-minmax-fp32.h"
#include "src/qs8-dwconv/qs8-dwconv-minmax-rndnu.h"

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
