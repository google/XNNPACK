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
#include "bench/conv.h"
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/igemm.h"
#include "xnnpack/indirection.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"


static void f16_igemm(benchmark::State& state,
  xnn_f16_igemm_minmax_ukernel_fn igemm,
  uint32_t mr, uint32_t nr, uint32_t kr, uint32_t sr,
  xnn_init_f16_minmax_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t input_height = state.range(0);
  const size_t input_width = state.range(1);
  const size_t kernel_height = state.range(2);
  const size_t kernel_width = state.range(3);
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t padding_height = state.range(4);
  const size_t padding_width = state.range(5);
  const size_t subsampling = state.range(6);
  const size_t dilation = state.range(7);
  const size_t group_input_channels = state.range(8);
  const size_t group_output_channels = state.range(9);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  const size_t output_pixel_stride = group_output_channels;
  const size_t input_pixel_stride = group_input_channels;
  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;
  const size_t output_size = output_height * output_width;

  const size_t mc_stride = benchmark::utils::RoundUp<size_t>(output_size, mr);
  const size_t nc_stride = benchmark::utils::RoundUp<size_t>(group_output_channels, nr);
  const size_t kc_stride = benchmark::utils::RoundUp<size_t>(group_input_channels, kr * sr);

  std::vector<uint16_t> a(input_height * input_width * input_pixel_stride + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::generate(a.begin(), a.end(), std::ref(f16rng));
  std::vector<uint16_t> k(group_output_channels * kernel_height * kernel_width * group_input_channels);
  std::generate(k.begin(), k.end(), std::ref(f16rng));
  std::vector<uint16_t> b(group_output_channels);
  std::generate(b.begin(), b.end(), std::ref(f16rng));

  std::vector<uint16_t> z(group_input_channels + XNN_EXTRA_BYTES / sizeof(uint16_t));

  const size_t w_elements = (kernel_size * kc_stride + 1) * nc_stride;
  const size_t i_elements = mc_stride * kernel_size;
  const size_t c_elements = output_height * output_width * output_pixel_stride;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(uint16_t) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);
  xnn_pack_f16_conv_goki_w(
    /*groups=*/1, group_output_channels, kernel_size, group_input_channels,
    nr, kr, sr, k.data(), b.data(), /*scale=*/nullptr, w.data(), /*extra_bytes=*/0, /*params=*/nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const uint16_t*> i(i_elements * num_buffers);
  const size_t tiled_output_size = round_up(output_size, mr);
  xnn_indirection_init_conv2d(
      /*output_tile_size=*/mr,
      /*output_start=*/0,
      /*output_end=*/tiled_output_size,
      reinterpret_cast<const void**>(i.data()),
      a.data(),
      z.data(),
      input_pixel_stride << XNN_LOG2_SIZEOF_HALF,
      input_height, input_width,
      output_height, output_width,
      kernel_height, kernel_width,
      subsampling, subsampling,
      dilation, dilation,
      padding_top, padding_left);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  std::vector<uint16_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), UINT16_C(0x7E00) /* NaN */);

  // Prepare minmax parameters.
  xnn_f16_minmax_params params;
  init_params(&params,
    UINT16_C(0x7C00) /* inf */, UINT16_C(0xFC00) /* -inf */);

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(uint16_t));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < output_size; m += mr) {
      const uint32_t mb = min(output_size - m, mr);
      for (uint32_t n = 0; n < group_output_channels; n += nr) {
        const uint32_t nb = min(group_output_channels - n, nr);
        igemm(
          mb, nb, group_input_channels * sizeof(uint16_t), kernel_size * mr * sizeof(void*),
          reinterpret_cast<const void**>(i.data()) + buffer_index * i_elements + m,
          w.data() + buffer_index * w_elements + n * (kc_stride * kernel_size + 1),
          c.data() + buffer_index * c_elements + m * group_output_channels + n, group_output_channels * sizeof(uint16_t), nr * sizeof(uint16_t),
          0, z.data(), &params);
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      output_height * output_width *
      group_input_channels * group_output_channels *
      kernel_height * kernel_width,
    benchmark::Counter::kIsRate);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f16_f32acc_igemm_1x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_igemm(state, xnn_f16_f32acc_igemm_minmax_ukernel_1x8__avx2_broadcast, 1, 8, 1, 1,
      xnn_init_f16_minmax_avx_params, benchmark::utils::CheckAVX2);
  }
  static void f16_f32acc_igemm_4x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_igemm(state, xnn_f16_f32acc_igemm_minmax_ukernel_4x8__avx2_broadcast, 4, 8, 1, 1,
      xnn_init_f16_minmax_avx_params, benchmark::utils::CheckAVX2);
  }
  static void f16_f32acc_igemm_5x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_igemm(state, xnn_f16_f32acc_igemm_minmax_ukernel_5x8__avx2_broadcast, 5, 8, 1, 1,
      xnn_init_f16_minmax_avx_params, benchmark::utils::CheckAVX2);
  }
  static void f16_f32acc_igemm_6x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_igemm(state, xnn_f16_f32acc_igemm_minmax_ukernel_6x8__avx2_broadcast, 6, 8, 1, 1,
      xnn_init_f16_minmax_avx_params, benchmark::utils::CheckAVX2);
  }
  static void f16_f32acc_igemm_7x8__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_igemm(state, xnn_f16_f32acc_igemm_minmax_ukernel_7x8__avx2_broadcast, 7, 8, 1, 1,
      xnn_init_f16_minmax_avx_params, benchmark::utils::CheckAVX2);
  }
  static void f16_f32acc_igemm_1x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_igemm(state, xnn_f16_f32acc_igemm_minmax_ukernel_1x16__avx2_broadcast, 1, 16, 1, 1,
      xnn_init_f16_minmax_avx_params, benchmark::utils::CheckAVX2);
  }
  static void f16_f32acc_igemm_3x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_igemm(state, xnn_f16_f32acc_igemm_minmax_ukernel_3x16__avx2_broadcast, 3, 16, 1, 1,
      xnn_init_f16_minmax_avx_params, benchmark::utils::CheckAVX2);
  }
  static void f16_f32acc_igemm_4x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_igemm(state, xnn_f16_f32acc_igemm_minmax_ukernel_4x16__avx2_broadcast, 4, 16, 1, 1,
      xnn_init_f16_minmax_avx_params, benchmark::utils::CheckAVX2);
  }
  static void f16_f32acc_igemm_5x16__avx2_broadcast(benchmark::State& state, const char* net) {
    f16_igemm(state, xnn_f16_f32acc_igemm_minmax_ukernel_5x16__avx2_broadcast, 5, 16, 1, 1,
      xnn_init_f16_minmax_avx_params, benchmark::utils::CheckAVX2);
  }

  BENCHMARK_CONV(f16_f32acc_igemm_1x8__avx2_broadcast)
  BENCHMARK_CONV(f16_f32acc_igemm_4x8__avx2_broadcast)
  BENCHMARK_CONV(f16_f32acc_igemm_5x8__avx2_broadcast)
  BENCHMARK_CONV(f16_f32acc_igemm_6x8__avx2_broadcast)
  BENCHMARK_CONV(f16_f32acc_igemm_7x8__avx2_broadcast)
  BENCHMARK_CONV(f16_f32acc_igemm_1x16__avx2_broadcast)
  BENCHMARK_CONV(f16_f32acc_igemm_3x16__avx2_broadcast)
  BENCHMARK_CONV(f16_f32acc_igemm_4x16__avx2_broadcast)
  BENCHMARK_CONV(f16_f32acc_igemm_5x16__avx2_broadcast)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
