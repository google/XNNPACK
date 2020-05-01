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


static void DWConvBenchmark(benchmark::State& state,
  xnn_f32_dwconv_minmax_unipass_ukernel_function dwconv,
  uint32_t cr, uint32_t kr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
    return;
  }
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
  if (kernel_size != kr) {
    state.SkipWithError("kernel size mismatch");
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(0.0f, 1.0f), rng);

  const size_t effective_kernel_height = (kernel_height - 1) * dilation + 1;
  const size_t effective_kernel_width = (kernel_width - 1) * dilation + 1;
  const size_t padding_left = padding_width / 2;
  const size_t padding_top = padding_height / 2;
  const size_t output_height = (input_height + padding_height - effective_kernel_height) / subsampling + 1;
  const size_t output_width = (input_width + padding_width - effective_kernel_width) / subsampling + 1;
  const size_t output_size = output_height * output_width;
  const size_t step_width = dilation == 1 ? subsampling : kernel_width;
  const size_t step_height = kernel_size + (output_width - 1) * step_width * kernel_height;

  const size_t c_stride = benchmark::utils::RoundUp<size_t>(channels, cr);

  std::vector<float> a(channels * input_height * input_width + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(channels * kernel_height * kernel_width);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(channels);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  std::vector<float> z(channels + XNN_EXTRA_BYTES / sizeof(float));

  const size_t w_elements = (kernel_size + 1) * c_stride;
  const size_t i_elements = output_height * step_height;
  const size_t c_elements = output_size * channels;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  std::vector<float, AlignedAllocator<float, 32>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_dwconv_ghw_w(kernel_height, kernel_width, channels, cr,
      k.data(), b.data(), w.data());
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const float*> i(i_elements * num_buffers);
  xnn_operator convolution_op = { };
  convolution_op.indirection_buffer = reinterpret_cast<const void**>(i.data());
  convolution_op.input              = a.data();
  convolution_op.input_pixel_stride = channels;
  convolution_op.zero_buffer        = z.data();
  convolution_op.batch_size         = 1;
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

  xnn_indirection_init_dwconv2d(&convolution_op, 0, step_height, step_width, 2 /* log2(sizeof(float)) */);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params =
    xnn_init_f32_minmax_params(-std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size() * sizeof(float));
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t y = 0; y < output_height; y++) {
      dwconv(channels, output_width,
        i.data() + buffer_index * i_elements + step_height * y,
        w.data() + buffer_index * w_elements,
        c.data() + buffer_index * c_elements + y * output_width * channels,
        kernel_height * step_width * sizeof(void*), 0,
        &params);
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * output_size * channels * kernel_size,
    benchmark::Counter::kIsRate);

  state.counters["BYTES"] = benchmark::Counter(
    uint64_t(state.iterations()) * (output_size + input_height * input_width + kernel_size + 1 /* bias */) * channels * sizeof(float),
    benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_dwconv_4x9__aarch64_neonfma(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up4x9__neon, 4, 9);
  }

  static void f32_dwconv_4x9__aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, 4, 9);
  }

  BENCHMARK_DWCONV(f32_dwconv_4x9__aarch64_neonfma)
  BENCHMARK_DWCONV(f32_dwconv_4x9__aarch64_neonfma_cortex_a55)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_dwconv_4x9__neon(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up4x9__neon, 4, 9,
      benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_4x9__neonfma(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma, 4, 9,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_dwconv_8x9__neonfma(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma, 8, 9,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_DWCONV(f32_dwconv_4x9__neon)
  BENCHMARK_DWCONV(f32_dwconv_4x9__neonfma)
  BENCHMARK_DWCONV(f32_dwconv_8x9__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_dwconv_4x4__sse(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up4x4__sse, 4, 4);
  }

  static void f32_dwconv_4x9__sse(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up4x9__sse, 4, 9);
  }

  static void f32_dwconv_4x25__sse(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up4x25__sse, 4, 25);
  }

  BENCHMARK_DWCONV(f32_dwconv_4x4__sse)
  BENCHMARK_DWCONV(f32_dwconv_4x9__sse)
  BENCHMARK_DWCONV(f32_dwconv_4x25__sse)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  static void f32_dwconv_4x4__psimd(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up4x4__psimd, 4, 4);
  }

  static void f32_dwconv_4x9__psimd(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up4x9__psimd, 4, 9);
  }

  static void f32_dwconv_4x25__psimd(benchmark::State& state, const char* net) {
    DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up4x25__psimd, 4, 25);
  }

  BENCHMARK_DWCONV(f32_dwconv_4x4__psimd)
  BENCHMARK_DWCONV(f32_dwconv_4x9__psimd)
  BENCHMARK_DWCONV(f32_dwconv_4x25__psimd)
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


static void f32_dwconv_1x4__scalar(benchmark::State& state, const char* net) {
  DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up1x4__scalar, 1, 4);
}

static void f32_dwconv_1x9__scalar(benchmark::State& state, const char* net) {
  DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up1x9__scalar, 1, 9);
}

static void f32_dwconv_1x25__scalar(benchmark::State& state, const char* net) {
  DWConvBenchmark(state, xnn_f32_dwconv_minmax_ukernel_up1x25__scalar, 1, 25);
}

BENCHMARK_DWCONV(f32_dwconv_1x4__scalar)
BENCHMARK_DWCONV(f32_dwconv_1x9__scalar)
BENCHMARK_DWCONV(f32_dwconv_1x25__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
