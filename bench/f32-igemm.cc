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

#include <benchmark/benchmark.h>
#include "bench/conv.h"
#include "bench/utils.h"

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/igemm.h>
#include <xnnpack/indirection.h>
#include <xnnpack/operator.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/pack.h>


static void f32_igemm(benchmark::State& state,
  xnn_f32_igemm_minmax_ukernel_fn igemm,
  uint32_t mr, uint32_t nr, uint32_t kr, uint32_t sr,
  xnn_init_f32_minmax_params_fn init_params,
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

  std::vector<float> a(input_height * input_width * input_pixel_stride + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(group_output_channels * kernel_height * kernel_width * group_input_channels);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(group_output_channels);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  std::vector<float> z(group_input_channels + XNN_EXTRA_BYTES / sizeof(float));

  const size_t w_elements = kernel_size * kc_stride * nc_stride + nc_stride;
  const size_t i_elements = mc_stride * kernel_size;
  const size_t c_elements = output_height * output_width * output_pixel_stride;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_conv_goki_w(
    1 /* groups */, group_output_channels, kernel_size, group_input_channels,
    nr, kr, sr, k.data(), b.data(), w.data(), 0 /* extra bytes */, nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const float*> i(i_elements * num_buffers);
  xnn_operator convolution_op = { };
  convolution_op.indirection_buffer   = reinterpret_cast<const void**>(i.data());
  convolution_op.input                = a.data();
  convolution_op.input_pixel_stride   = input_pixel_stride;
  convolution_op.zero_buffer          = z.data();
  convolution_op.groups               = 1;
  convolution_op.group_input_channels = group_input_channels;
  convolution_op.batch_size           = 1;
  convolution_op.input_height         = input_height;
  convolution_op.input_width          = input_width;
  convolution_op.output_height        = output_height;
  convolution_op.output_width         = output_width;
  convolution_op.kernel_height        = kernel_height;
  convolution_op.kernel_width         = kernel_width;
  convolution_op.stride_height        = subsampling;
  convolution_op.stride_width         = subsampling;
  convolution_op.dilation_height      = dilation;
  convolution_op.dilation_width       = dilation;
  convolution_op.padding_top          = padding_top;
  convolution_op.padding_left         = padding_left;
  xnn_indirection_init_conv2d(&convolution_op, mr, 2 /* log2(sizeof(float)) */);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

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

#if XNN_PLATFORM_JIT
  static void f32_igemm(benchmark::State& state,
    xnn_jit_igemm_code_generator_fn generator,
    size_t mr, size_t nr, size_t kr, size_t sr,
    xnn_init_f32_minmax_params_fn init_params,
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

  std::vector<float> a(input_height * input_width * input_pixel_stride + XNN_EXTRA_BYTES / sizeof(float));
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<float> k(group_output_channels * kernel_height * kernel_width * group_input_channels);
  std::generate(k.begin(), k.end(), std::ref(f32rng));
  std::vector<float> b(group_output_channels);
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  std::vector<float> z(group_input_channels + XNN_EXTRA_BYTES / sizeof(float));

  const size_t w_elements = kernel_size * kc_stride * nc_stride + nc_stride;
  const size_t i_elements = mc_stride * kernel_size;
  const size_t c_elements = output_height * output_width * output_pixel_stride;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  std::vector<float, AlignedAllocator<float, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  xnn_pack_f32_conv_goki_w(
    1 /* groups */, group_output_channels, kernel_size, group_input_channels,
    nr, kr, sr, k.data(), b.data(), w.data(), 0 /* extra bytes */, nullptr);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const float*> i(i_elements * num_buffers);
  xnn_operator convolution_op = { };
  convolution_op.indirection_buffer   = reinterpret_cast<const void**>(i.data());
  convolution_op.input                = a.data();
  convolution_op.input_pixel_stride   = input_pixel_stride;
  convolution_op.zero_buffer          = z.data();
  convolution_op.groups               = 1;
  convolution_op.group_input_channels = group_input_channels;
  convolution_op.batch_size           = 1;
  convolution_op.input_height         = input_height;
  convolution_op.input_width          = input_width;
  convolution_op.output_height        = output_height;
  convolution_op.output_width         = output_width;
  convolution_op.kernel_height        = kernel_height;
  convolution_op.kernel_width         = kernel_width;
  convolution_op.stride_height        = subsampling;
  convolution_op.stride_width         = subsampling;
  convolution_op.dilation_height      = dilation;
  convolution_op.dilation_width       = dilation;
  convolution_op.padding_top          = padding_top;
  convolution_op.padding_left         = padding_left;
  xnn_indirection_init_conv2d(&convolution_op, mr, 2 /* log2(sizeof(float)) */);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  xnn_f32_minmax_params params;
  init_params(&params,
    -std::numeric_limits<float>::infinity(), +std::numeric_limits<float>::infinity());

  jit_gemm_params jit_params = {};
  jit_params.f32_minmax.min = -std::numeric_limits<float>::infinity();
  jit_params.f32_minmax.max = +std::numeric_limits<float>::infinity();

  xnn_code_buffer code_buffer;
  xnn_allocate_code_memory(&code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE);
  generator(&code_buffer,
            mr,
            group_output_channels % nr,
            group_input_channels * sizeof(float),
            kernel_size * mr * sizeof(void *),
            &jit_params);
  xnn_finalize_code_memory(&code_buffer);
  auto igemm = reinterpret_cast<xnn_f32_igemm_minmax_ukernel_fn>(code_buffer.start);

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
  xnn_release_code_memory(&code_buffer);

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
#endif  // XNN_PLATFORM_JIT

#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  static void jit_f32_igemm_1x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void jit_f32_igemm_1x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void jit_f32_igemm_4x8__aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void jit_f32_igemm_4x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void jit_f32_igemm_6x8__aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  BENCHMARK_CONV(jit_f32_igemm_1x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(jit_f32_igemm_1x8__aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_CONV(jit_f32_igemm_4x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(jit_f32_igemm_4x8__aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_CONV(jit_f32_igemm_6x8__aarch64_neonfma_ld128)

#define BENCHMARK_UPTO_MR_IGEMM(name, max_mr, nr)                            \
  static void name(benchmark::State &state, const char *net) {               \
    f32_igemm(                                                               \
        state,                                                               \
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75, \
        max_mr, nr, 1, 1, xnn_init_f32_minmax_scalar_params);                \
  }                                                                          \
  BENCHMARK_CONV(name)
  BENCHMARK_UPTO_MR_IGEMM(jit_f32_igemm_6x8_1x8__aarch64_neonfma_prfm_cortex_a75, 1, 8);
  BENCHMARK_UPTO_MR_IGEMM(jit_f32_igemm_6x8_2x8__aarch64_neonfma_prfm_cortex_a75, 2, 8);
  BENCHMARK_UPTO_MR_IGEMM(jit_f32_igemm_6x8_3x8__aarch64_neonfma_prfm_cortex_a75, 3, 8);
  BENCHMARK_UPTO_MR_IGEMM(jit_f32_igemm_6x8_4x8__aarch64_neonfma_prfm_cortex_a75, 4, 8);
  BENCHMARK_UPTO_MR_IGEMM(jit_f32_igemm_6x8_5x8__aarch64_neonfma_prfm_cortex_a75, 5, 8);
  BENCHMARK_UPTO_MR_IGEMM(jit_f32_igemm_6x8_6x8__aarch64_neonfma_prfm_cortex_a75, 6, 8);
#undef BENCHMARK_UPTO_MR_IGEMM

#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT

#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
  static void jit_f32_igemm_4x8__aarch32_neon_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void jit_f32_igemm_4x8__aarch32_neon_cortex_a7(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a7, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void jit_f32_igemm_4x8__aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void jit_f32_igemm_4x8__aarch32_neon_cortex_a55(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void jit_f32_igemm_4x8__aarch32_neon_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_prfm_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void jit_f32_igemm_4x8__aarch32_neon_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  BENCHMARK_CONV(jit_f32_igemm_4x8__aarch32_neon_ld64)
  BENCHMARK_CONV(jit_f32_igemm_4x8__aarch32_neon_cortex_a7)
  BENCHMARK_CONV(jit_f32_igemm_4x8__aarch32_neon_cortex_a53)
  BENCHMARK_CONV(jit_f32_igemm_4x8__aarch32_neon_cortex_a55)
  BENCHMARK_CONV(jit_f32_igemm_4x8__aarch32_neon_prfm_cortex_a75)
  BENCHMARK_CONV(jit_f32_igemm_4x8__aarch32_neon_cortex_a75)
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT

#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void f32_igemm_4x8__asm_aarch32_neon_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_cortex_a7(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_prfm_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a53, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_cortex_a55(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch32_neon_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_1x8__asm_aarch32_neon_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_1x8__asm_aarch32_neon_prfm_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_ld64)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_cortex_a7)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_cortex_a53)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_prfm_cortex_a53)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_cortex_a55)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_prfm_cortex_a75)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch32_neon_cortex_a75)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch32_neon_cortex_a53)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch32_neon_prfm_cortex_a53)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_igemm_1x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_1x8__asm_aarch64_neonfma_prfm_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_ld64, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_1x12__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53, 1, 12, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_1x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_1x8__asm_aarch64_neonfma_prfm_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_1x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_1x8__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x2__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75, 4, 2, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x2__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_prfm_cortex_a75, 4, 2, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x2__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64, 4, 2, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_prfm_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a53, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a75, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_5x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_5x8__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__asm_aarch64_neonfma_prfm_cortex_a75, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x12__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53, 4, 12, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_prfm_cortex_a53(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a53, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_cortex_a73(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a75, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x8__asm_aarch64_neonfma_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_1x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x2__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64, 4, 2, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x2__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64, 6, 2, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x4__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x4__aarch64_neonfma_lane_ld64, 4, 4, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_4x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x8__aarch64_neonfma_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }
  static void f32_igemm_6x8__aarch64_neonfma_lane_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_prfm_ld64)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_prfm_cortex_a53)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(f32_igemm_1x8__asm_aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_CONV(f32_igemm_1x12__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_CONV(f32_igemm_4x2__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(f32_igemm_4x2__asm_aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_CONV(f32_igemm_4x2__asm_aarch64_neonfma_ld64)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_prfm_cortex_a53)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_cortex_a55)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_CONV(f32_igemm_4x8__asm_aarch64_neonfma_ld128)
  BENCHMARK_CONV(f32_igemm_4x12__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_CONV(f32_igemm_5x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(f32_igemm_5x8__asm_aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_prfm_cortex_a53)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_cortex_a55)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_cortex_a73)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_CONV(f32_igemm_6x8__asm_aarch64_neonfma_prfm_cortex_a75)
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
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x2__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64, 4, 2, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x2__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x2__neon_lane_ld64, 6, 2, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x4__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x4__neon_lane_ld64, 4, 4, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x8__neon_lane_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld64, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x8__neon_lane_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld128, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_1x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__neon_dup_ld64, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x8__neon_dup_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__neon_dup_ld128, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__neon_dup_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x8__neon_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__neon_dup_ld64, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x8__neon_dup_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__neon_dup_ld128, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_1x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__neonfma_dup_ld64, 1, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_4x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__neonfma_dup_ld128, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_4x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__neonfma_dup_ld64, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_6x8__neonfma_dup_ld64(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld64, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_6x8__neonfma_dup_ld128(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld128, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_1x8s4__neon(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8s4__neon, 1, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_4x8s4__neon(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8s4__neon, 4, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_6x8s4__neon(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8s4__neon, 6, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_8x8s4__neon(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_8x8s4__neon, 8, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEON);
  }
  static void f32_igemm_1x8s4__neonfma(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma, 1, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_4x8s4__neonfma(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8s4__neonfma, 4, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_6x8s4__neonfma(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8s4__neonfma, 6, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
  }
  static void f32_igemm_8x8s4__neonfma(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_8x8s4__neonfma, 8, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckNEONFMA);
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
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__sse_load1, 1, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_3x8__sse_load1(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_3x8__sse_load1, 3, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_4x8__sse_load1(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__sse_load1, 4, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_5x8__sse_load1(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__sse_load1, 5, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }

  static void f32_igemm_1x8__sse_dup(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__sse_dup, 1, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_3x8__sse_dup(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_3x8__sse_dup, 3, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_4x8__sse_dup(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__sse_dup, 4, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_5x8__sse_dup(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__sse_dup, 5, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }

  static void f32_igemm_1x8s4__sse(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8s4__sse, 1, 8, 1, 4,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_3x8s4__sse(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_3x8s4__sse, 3, 8, 1, 4,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_4x8s4__sse(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8s4__sse, 4, 8, 1, 4,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_5x8s4__sse(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8s4__sse, 5, 8, 1, 4,
      xnn_init_f32_minmax_sse_params);
  }

  static void f32_igemm_1x8__sse2_dup(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__sse2_dup, 1, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_3x8__sse2_dup(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_3x8__sse2_dup, 3, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_4x8__sse2_dup(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__sse2_dup, 4, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }
  static void f32_igemm_5x8__sse2_dup(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__sse2_dup, 5, 8, 1, 1,
      xnn_init_f32_minmax_sse_params);
  }

  static void f32_igemm_1x8__avx_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__avx_broadcast, 1, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_igemm_4x8__avx_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__avx_broadcast, 4, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_igemm_5x8__avx_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__avx_broadcast, 5, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_igemm_6x8__avx_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__avx_broadcast, 6, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }
  static void f32_igemm_7x8__avx_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_7x8__avx_broadcast, 7, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckAVX);
  }

  static void f32_igemm_1x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x8__fma3_broadcast, 1, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_4x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__fma3_broadcast, 4, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_5x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__fma3_broadcast, 5, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_6x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__fma3_broadcast, 6, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_7x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_7x8__fma3_broadcast, 7, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_8x8__fma3_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_8x8__fma3_broadcast, 8, 8, 1, 1,
      xnn_init_f32_minmax_avx_params, benchmark::utils::CheckFMA3);
  }

  static void f32_igemm_1x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast, 1, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_4x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x16__avx512f_broadcast, 4, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_5x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x16__avx512f_broadcast, 5, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_6x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x16__avx512f_broadcast, 6, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_7x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast, 7, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckFMA3);
  }
  static void f32_igemm_8x16__avx512f_broadcast(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_8x16__avx512f_broadcast, 8, 16, 1, 1,
      xnn_init_f32_minmax_scalar_params, benchmark::utils::CheckFMA3);
  }

  BENCHMARK_CONV(f32_igemm_1x8__sse_load1)
  BENCHMARK_CONV(f32_igemm_3x8__sse_load1)
  BENCHMARK_CONV(f32_igemm_4x8__sse_load1)
  BENCHMARK_CONV(f32_igemm_5x8__sse_load1)

  BENCHMARK_CONV(f32_igemm_1x8__sse_dup)
  BENCHMARK_CONV(f32_igemm_3x8__sse_dup)
  BENCHMARK_CONV(f32_igemm_4x8__sse_dup)
  BENCHMARK_CONV(f32_igemm_5x8__sse_dup)

  BENCHMARK_CONV(f32_igemm_1x8s4__sse)
  BENCHMARK_CONV(f32_igemm_3x8s4__sse)
  BENCHMARK_CONV(f32_igemm_4x8s4__sse)
  BENCHMARK_CONV(f32_igemm_5x8s4__sse)

  BENCHMARK_CONV(f32_igemm_1x8__sse2_dup)
  BENCHMARK_CONV(f32_igemm_3x8__sse2_dup)
  BENCHMARK_CONV(f32_igemm_4x8__sse2_dup)
  BENCHMARK_CONV(f32_igemm_5x8__sse2_dup)

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

  BENCHMARK_CONV(f32_igemm_1x16__avx512f_broadcast)
  BENCHMARK_CONV(f32_igemm_4x16__avx512f_broadcast)
  BENCHMARK_CONV(f32_igemm_5x16__avx512f_broadcast)
  BENCHMARK_CONV(f32_igemm_6x16__avx512f_broadcast)
  BENCHMARK_CONV(f32_igemm_7x16__avx512f_broadcast)
  BENCHMARK_CONV(f32_igemm_8x16__avx512f_broadcast)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_igemm_3x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat, 3, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_4x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_5x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_6x8__wasmsimd_arm_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_3x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat, 3, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_4x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_5x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_6x8__wasmsimd_x86_loadsplat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_3x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_arm_splat, 3, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_4x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_arm_splat, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_5x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_splat, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_6x8__wasmsimd_arm_splat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_splat, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_3x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_splat, 3, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_4x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_splat, 4, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_5x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_x86_splat, 5, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_6x8__wasmsimd_x86_splat(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_splat, 6, 8, 1, 1,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_3x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_arm, 3, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_4x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8s4__wasmsimd_arm, 4, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_5x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8s4__wasmsimd_arm, 5, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_6x8s4__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_arm, 6, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_3x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_x86, 3, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_4x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x8s4__wasmsimd_x86, 4, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_5x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_5x8s4__wasmsimd_x86, 5, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }

  static void f32_igemm_6x8s4__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_igemm(state, xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_x86, 6, 8, 1, 4,
      xnn_init_f32_minmax_scalar_params);
  }

  BENCHMARK_CONV(f32_igemm_3x8__wasmsimd_arm_loadsplat)
  BENCHMARK_CONV(f32_igemm_4x8__wasmsimd_arm_loadsplat)
  BENCHMARK_CONV(f32_igemm_5x8__wasmsimd_arm_loadsplat)
  BENCHMARK_CONV(f32_igemm_6x8__wasmsimd_arm_loadsplat)
  BENCHMARK_CONV(f32_igemm_3x8__wasmsimd_x86_loadsplat)
  BENCHMARK_CONV(f32_igemm_4x8__wasmsimd_x86_loadsplat)
  BENCHMARK_CONV(f32_igemm_5x8__wasmsimd_x86_loadsplat)
  BENCHMARK_CONV(f32_igemm_6x8__wasmsimd_x86_loadsplat)
  BENCHMARK_CONV(f32_igemm_3x8__wasmsimd_arm_splat)
  BENCHMARK_CONV(f32_igemm_4x8__wasmsimd_arm_splat)
  BENCHMARK_CONV(f32_igemm_5x8__wasmsimd_arm_splat)
  BENCHMARK_CONV(f32_igemm_6x8__wasmsimd_arm_splat)
  BENCHMARK_CONV(f32_igemm_3x8__wasmsimd_x86_splat)
  BENCHMARK_CONV(f32_igemm_4x8__wasmsimd_x86_splat)
  BENCHMARK_CONV(f32_igemm_5x8__wasmsimd_x86_splat)
  BENCHMARK_CONV(f32_igemm_6x8__wasmsimd_x86_splat)
  BENCHMARK_CONV(f32_igemm_3x8s4__wasmsimd_arm)
  BENCHMARK_CONV(f32_igemm_4x8s4__wasmsimd_arm)
  BENCHMARK_CONV(f32_igemm_5x8s4__wasmsimd_arm)
  BENCHMARK_CONV(f32_igemm_6x8s4__wasmsimd_arm)
  BENCHMARK_CONV(f32_igemm_3x8s4__wasmsimd_x86)
  BENCHMARK_CONV(f32_igemm_4x8s4__wasmsimd_x86)
  BENCHMARK_CONV(f32_igemm_5x8s4__wasmsimd_x86)
  BENCHMARK_CONV(f32_igemm_6x8s4__wasmsimd_x86)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


static void f32_igemm_1x4__scalar(benchmark::State& state, const char* net) {
  f32_igemm(state, xnn_f32_igemm_minmax_ukernel_1x4__scalar, 1, 4, 1, 1,
    xnn_init_f32_minmax_scalar_params);
}

static void f32_igemm_2x4__scalar(benchmark::State& state, const char* net) {
  f32_igemm(state, xnn_f32_igemm_minmax_ukernel_2x4__scalar, 2, 4, 1, 1,
    xnn_init_f32_minmax_scalar_params);
}

static void f32_igemm_4x4__scalar(benchmark::State& state, const char* net) {
  f32_igemm(state, xnn_f32_igemm_minmax_ukernel_4x4__scalar, 4, 4, 1, 1,
    xnn_init_f32_minmax_scalar_params);
}

BENCHMARK_CONV(f32_igemm_1x4__scalar)
BENCHMARK_CONV(f32_igemm_2x4__scalar)
BENCHMARK_CONV(f32_igemm_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
