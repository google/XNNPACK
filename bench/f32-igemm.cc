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

#include "conv.h"
#include "utils.h"
#include "xnnpack/common.h"
#include "xnnpack/igemm.h"
#include "xnnpack/indirection.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/pack.h"
#include "xnnpack/buffer.h"
#include <benchmark/benchmark.h>

static void f32_igemm(benchmark::State& state,
  xnn_f32_igemm_minmax_ukernel_fn igemm,
  xnn_init_f32_minmax_params_fn init_params,
  uint32_t mr, uint32_t nr, uint32_t kr, uint32_t sr,
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
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), std::ref(rng));

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

  xnnpack::Buffer<float> a(input_height * input_width * input_pixel_stride + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  xnnpack::Buffer<float> k(group_output_channels * kernel_height * kernel_width * group_input_channels);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  xnnpack::Buffer<float> b(group_output_channels);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  xnnpack::Buffer<float> z(group_input_channels + XNN_EXTRA_BYTES / sizeof(float));

  const size_t w_elements = kernel_size * kc_stride * nc_stride + nc_stride;
  const size_t i_elements = mc_stride * kernel_size;
  const size_t c_elements = output_height * output_width * output_pixel_stride;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> w(w_elements * num_buffers);
  xnn_pack_f32_conv_goki_w(
    /*groups=*/1, group_output_channels, kernel_size, group_input_channels,
    nr, kr, sr, k.data(), b.data(), /*scale=*/nullptr, w.data(), /*extra_bytes=*/0, /*params=*/nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  xnnpack::Buffer<const float*> i(i_elements * num_buffers);
  const size_t tiled_output_size = round_up(output_size, mr);
  xnn_indirection_init_conv2d(
      /*output_tile_size=*/mr,
      /*output_start=*/0,
      /*output_end=*/tiled_output_size,
      reinterpret_cast<const void**>(i.data()),
      a.data(),
      z.data(),
      input_pixel_stride << XNN_LOG2_SIZEOF_FLOAT,
      input_height, input_width,
      output_height, output_width,
      kernel_height, kernel_width,
      subsampling, subsampling,
      dilation, dilation,
      padding_top, padding_left);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  xnnpack::Buffer<float> c(c_elements * num_buffers);

  xnn_f32_minmax_params params;
  init_params(&params,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < output_size; m += mr) {
      const uint32_t mb = min(output_size - m, mr);
      igemm(
        mb, group_output_channels, group_input_channels * sizeof(float), kernel_size * mr * sizeof(void*),
        i.data() + buffer_index * i_elements + m,
        w.data() + buffer_index * w_elements,
        c.data() + buffer_index * c_elements + m * group_output_channels, group_output_channels * sizeof(float), nr * sizeof(float),
        0, z.data(), &params);
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

#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void f32_igemm_4x8__asm_aarch32_neon_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_cortex_a7(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_cortex_a53_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_cortex_a55(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_cortex_a75_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x8__asm_aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x8__asm_aarch32_neon_cortex_a53_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_ld64)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_cortex_a7)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_cortex_a53)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_cortex_a53_prfm)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_cortex_a55)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_cortex_a75_prfm)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_cortex_a75)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch32_neon_cortex_a53)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch32_neon_cortex_a53_prfm)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_igemm_1x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x8__asm_aarch64_neonfma_ld64_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x12__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/12, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x2__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x2__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x2__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_5x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_5x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x12__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/12, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_cortex_a73(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x2__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x2__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x4__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x4__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_ld64_prfm)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_cortex_a53_prfm)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_CONV(f32_igemm_1x12__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_CONV(f32_igemm_4x2__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(f32_igemm_4x2__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_CONV(f32_igemm_4x2__asm_aarch64_neonfma_ld64)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_cortex_a53_prfm)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_cortex_a55)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_ld128)
  BENCHMARK_CONV(f32_igemm_4x12__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_CONV(f32_igemm_5x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(f32_igemm_5x8__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_cortex_a53_prfm)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_cortex_a55)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_cortex_a73)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_ld128)
  BENCHMARK_CONV(f32_igemm_1x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_CONV(f32_igemm_4x2__aarch64_neonfma_lane_ld64)
  BENCHMARK_CONV(f32_igemm_6x2__aarch64_neonfma_lane_ld64)
  BENCHMARK_CONV(f32_igemm_4x4__aarch64_neonfma_lane_ld64)
  BENCHMARK_CONV(f32_igemm_4x8__aarch64_neonfma_lane_ld128)
  BENCHMARK_CONV(f32_igemm_4x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_CONV(f32_igemm_6x8__aarch64_neonfma_lane_ld64)
  BENCHMARK_CONV(f32_igemm_6x8__aarch64_neonfma_lane_ld128)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_igemm_1x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x2__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x2__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x2__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/2, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x4__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x4__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_1x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x8__neon_dup_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__neon_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x8__neon_dup_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__neon_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_1x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_4x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__neonfma_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_4x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_6x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_6x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld128,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_1x8s4__neon(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x8s4__neon(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x8s4__neon(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_8x8s4__neon(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_8x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
  static void f32_igemm_1x8s4__neonfma(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_4x8s4__neonfma(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_6x8s4__neonfma(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_8x8s4__neonfma(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_8x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/4,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_CONV(f32_igemm_1x8__neon_lane_ld64)
  BENCHMARK_CONV(f32_igemm_4x2__neon_lane_ld64)
  BENCHMARK_CONV(f32_igemm_6x2__neon_lane_ld64)
  BENCHMARK_CONV(f32_igemm_4x4__neon_lane_ld64)
  BENCHMARK_CONV(f32_igemm_4x8__neon_lane_ld128)
  BENCHMARK_CONV(f32_igemm_4x8__neon_lane_ld64)
  BENCHMARK_CONV(f32_igemm_6x8__neon_lane_ld64)
  BENCHMARK_CONV(f32_igemm_6x8__neon_lane_ld128)
  BENCHMARK_CONV(f32_igemm_1x8__neon_dup_ld64)
  BENCHMARK_CONV(f32_igemm_4x8__neon_dup_ld128)
  BENCHMARK_CONV(f32_igemm_4x8__neon_dup_ld64)
  BENCHMARK_CONV(f32_igemm_6x8__neon_dup_ld64)
  BENCHMARK_CONV(f32_igemm_6x8__neon_dup_ld128)
  BENCHMARK_CONV(f32_igemm_1x8__neonfma_dup_ld64)
  BENCHMARK_CONV(f32_igemm_4x8__neonfma_dup_ld128)
  BENCHMARK_CONV(f32_igemm_4x8__neonfma_dup_ld64)
  BENCHMARK_CONV(f32_igemm_6x8__neonfma_dup_ld64)
  BENCHMARK_CONV(f32_igemm_6x8__neonfma_dup_ld128)

  BENCHMARK_CONV(f32_igemm_1x8s4__neon)
  BENCHMARK_CONV(f32_igemm_4x8s4__neon)
  BENCHMARK_CONV(f32_igemm_6x8s4__neon)
  BENCHMARK_CONV(f32_igemm_8x8s4__neon)
  BENCHMARK_CONV(f32_igemm_1x8s4__neonfma)
  BENCHMARK_CONV(f32_igemm_4x8s4__neonfma)
  BENCHMARK_CONV(f32_igemm_6x8s4__neonfma)
  BENCHMARK_CONV(f32_igemm_8x8s4__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_igemm_1x8__sse_load1(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_load1,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_3x8__sse_load1(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_3x8__sse_load1,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__sse_load1(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__sse_load1,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_5x8__sse_load1(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8__sse_load1,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__sse_load1(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__sse_load1,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x8__sse_dup(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_dup,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_3x8__sse_dup(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_3x8__sse_dup,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_4x8__sse_dup(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__sse_dup,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_5x8__sse_dup(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8__sse_dup,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_6x8__sse_dup(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__sse_dup,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_1x8s4__sse(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8s4__sse,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_igemm_3x8s4__sse(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_3x8s4__sse,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_igemm_4x8s4__sse(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8s4__sse,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_igemm_5x8s4__sse(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8s4__sse,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_igemm_6x8s4__sse(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8s4__sse,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }
  static void f32_igemm_1x8__avx_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_igemm_4x8__avx_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_igemm_5x8__avx_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_igemm_6x8__avx_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }
  static void f32_igemm_7x8__avx_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_7x8__avx_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  static void f32_igemm_1x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_4x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_5x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_6x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_7x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_7x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/7, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_8x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_8x8__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  static void f32_igemm_1x16__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_3x16__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_3x16__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_4x16__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x16__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_5x16__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  static void f32_igemm_5x16__fma3_broadcast_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_6x16__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x16__fma3_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }

  static void f32_igemm_6x16__fma3_broadcast_prfm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x16__fma3_broadcast_prfm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_1x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_igemm_4x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_igemm_5x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_igemm_6x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_igemm_7x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/7, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_igemm_8x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_8x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/16, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_CONV(f32_igemm_1x8__sse_load1)
  BENCHMARK_CONV(f32_igemm_3x8__sse_load1)
  BENCHMARK_CONV(f32_igemm_4x8__sse_load1)
  BENCHMARK_CONV(f32_igemm_5x8__sse_load1)
  BENCHMARK_CONV(f32_igemm_6x8__sse_load1)
  BENCHMARK_CONV(f32_igemm_1x8__sse_dup)
  BENCHMARK_CONV(f32_igemm_3x8__sse_dup)
  BENCHMARK_CONV(f32_igemm_4x8__sse_dup)
  BENCHMARK_CONV(f32_igemm_5x8__sse_dup)
  BENCHMARK_CONV(f32_igemm_6x8__sse_dup)
  BENCHMARK_CONV(f32_igemm_1x8s4__sse)
  BENCHMARK_CONV(f32_igemm_3x8s4__sse)
  BENCHMARK_CONV(f32_igemm_4x8s4__sse)
  BENCHMARK_CONV(f32_igemm_5x8s4__sse)
  BENCHMARK_CONV(f32_igemm_6x8s4__sse)
  BENCHMARK_CONV(f32_igemm_1x8__avx_broadcast)
  BENCHMARK_CONV(f32_igemm_4x8__avx_broadcast)
  BENCHMARK_CONV(f32_igemm_5x8__avx_broadcast)
  BENCHMARK_CONV(f32_igemm_6x8__avx_broadcast)
  BENCHMARK_CONV(f32_igemm_7x8__avx_broadcast)
  BENCHMARK_CONV(f32_igemm_1x8__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_4x8__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_5x8__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_6x8__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_7x8__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_8x8__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_1x16__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_3x16__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_4x16__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_5x16__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_6x16__fma3_broadcast)
  BENCHMARK_CONV(f32_igemm_5x16__fma3_broadcast_prfm)
  BENCHMARK_CONV(f32_igemm_6x16__fma3_broadcast_prfm)
  BENCHMARK_CONV(f32_igemm_1x16__avx512f_broadcast)
  BENCHMARK_CONV(f32_igemm_4x16__avx512f_broadcast)
  BENCHMARK_CONV(f32_igemm_5x16__avx512f_broadcast)
  BENCHMARK_CONV(f32_igemm_6x16__avx512f_broadcast)
  BENCHMARK_CONV(f32_igemm_7x16__avx512f_broadcast)
  BENCHMARK_CONV(f32_igemm_8x16__avx512f_broadcast)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_igemm_3x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_4x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_5x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_6x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_1x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }
  static void f32_igemm_3x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_4x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_5x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_6x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_3x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_4x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_5x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_6x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_1x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_3x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_4x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_5x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_6x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void f32_igemm_3x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  static void f32_igemm_4x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  static void f32_igemm_5x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  static void f32_igemm_6x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  static void f32_igemm_1x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  static void f32_igemm_3x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  static void f32_igemm_4x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_4x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  static void f32_igemm_5x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_5x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  static void f32_igemm_6x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/1, /*sr=*/4);
  }

  BENCHMARK_CONV(f32_igemm_3x8__wasmsimd_arm_loadsplat)
  BENCHMARK_CONV(f32_igemm_4x8__wasmsimd_arm_loadsplat)
  BENCHMARK_CONV(f32_igemm_5x8__wasmsimd_arm_loadsplat)
  BENCHMARK_CONV(f32_igemm_6x8__wasmsimd_arm_loadsplat)
  BENCHMARK_CONV(f32_igemm_1x8__wasmsimd_x86_loadsplat)
  BENCHMARK_CONV(f32_igemm_3x8__wasmsimd_x86_loadsplat)
  BENCHMARK_CONV(f32_igemm_4x8__wasmsimd_x86_loadsplat)
  BENCHMARK_CONV(f32_igemm_5x8__wasmsimd_x86_loadsplat)
  BENCHMARK_CONV(f32_igemm_6x8__wasmsimd_x86_loadsplat)
  BENCHMARK_CONV(f32_igemm_3x8__wasmsimd_arm_splat)
  BENCHMARK_CONV(f32_igemm_4x8__wasmsimd_arm_splat)
  BENCHMARK_CONV(f32_igemm_5x8__wasmsimd_arm_splat)
  BENCHMARK_CONV(f32_igemm_6x8__wasmsimd_arm_splat)
  BENCHMARK_CONV(f32_igemm_1x8__wasmsimd_x86_splat)
  BENCHMARK_CONV(f32_igemm_3x8__wasmsimd_x86_splat)
  BENCHMARK_CONV(f32_igemm_4x8__wasmsimd_x86_splat)
  BENCHMARK_CONV(f32_igemm_5x8__wasmsimd_x86_splat)
  BENCHMARK_CONV(f32_igemm_6x8__wasmsimd_x86_splat)
  BENCHMARK_CONV(f32_igemm_3x8s4__wasmsimd_arm)
  BENCHMARK_CONV(f32_igemm_4x8s4__wasmsimd_arm)
  BENCHMARK_CONV(f32_igemm_5x8s4__wasmsimd_arm)
  BENCHMARK_CONV(f32_igemm_6x8s4__wasmsimd_arm)
  BENCHMARK_CONV(f32_igemm_1x8s4__wasmsimd_x86)
  BENCHMARK_CONV(f32_igemm_3x8s4__wasmsimd_x86)
  BENCHMARK_CONV(f32_igemm_4x8s4__wasmsimd_x86)
  BENCHMARK_CONV(f32_igemm_5x8s4__wasmsimd_x86)
  BENCHMARK_CONV(f32_igemm_6x8s4__wasmsimd_x86)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

static void f32_igemm_1x4__scalar(benchmark::State& state, const char* net) {
  f32_igemm(state,
    xnn_f32_igemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

static void f32_igemm_2x4__scalar(benchmark::State& state, const char* net) {
  f32_igemm(state,
    xnn_f32_igemm_minmax_ukernel_2x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

static void f32_igemm_4x4__scalar(benchmark::State& state, const char* net) {
  f32_igemm(state,
    xnn_f32_igemm_minmax_ukernel_4x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

BENCHMARK_CONV(f32_igemm_1x4__scalar)
BENCHMARK_CONV(f32_igemm_2x4__scalar)
BENCHMARK_CONV(f32_igemm_4x4__scalar)

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  static void f32_igemm_1x4v__rvv(benchmark::State& state, const char* net) {
    size_t vlenb = 0;
    asm volatile("csrr %0, vlenb" : "=r"(vlenb));
    // LMUL=4
    size_t nr = vlenb * 1.0;
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_1x4v__rvv,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/nr, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }

  static void f32_igemm_7x4v__rvv(benchmark::State& state, const char* net) {
    size_t vlenb = 0;
    asm volatile("csrr %0, vlenb" : "=r"(vlenb));
    // LMUL=4
    size_t nr = vlenb * 1.0;
    f32_igemm(state,
      xnn_f32_igemm_minmax_ukernel_7x4v__rvv,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/7, /*nr=*/nr, /*kr=*/1, /*sr=*/1,
      benchmark::utils::CheckRVV);
  }

  BENCHMARK_CONV(f32_igemm_1x4v__rvv)
  BENCHMARK_CONV(f32_igemm_7x4v__rvv)
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
