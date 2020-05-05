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
#include "bench/conv.h"
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/igemm.h>
#include <xnnpack/indirection.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static void IGEMMBenchmark(benchmark::State& state,
  xnn_f16_igemm_minmax_ukernel_function f16_igemm,
  uint32_t mr, uint32_t nr, uint32_t kr, uint32_t sr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (!cpuinfo_initialize()) {
    state.SkipWithError("cpuinfo initialization failed");
  }
  if (isa_check && !isa_check(state)) {
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
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);
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
  const size_t kc_stride = benchmark::utils::RoundUp<size_t>(group_input_channels, kr);

  std::vector<uint16_t> a(input_height * input_width * input_pixel_stride);
  std::generate(a.begin(), a.end(), std::ref(f16rng));
  std::vector<uint16_t> k(group_output_channels * kernel_height * kernel_width * group_input_channels);
  std::generate(k.begin(), k.end(), std::ref(f16rng));
  std::vector<uint16_t> b(group_output_channels);
  std::generate(b.begin(), b.end(), std::ref(f16rng));

  std::vector<uint16_t> z(group_input_channels);

  const size_t w_elements = (kernel_size * kc_stride + 1) * nc_stride;
  const size_t i_elements = mc_stride * kernel_size;
  const size_t c_elements = output_height * output_width * output_pixel_stride;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(uint16_t) * (w_elements + c_elements) + sizeof(void*) * i_elements);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 32>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0);
  xnn_pack_f16_conv_goki_w(
    1 /* groups */, group_output_channels, kernel_size, group_input_channels,
    nr, kr, sr, k.data(), b.data(), w.data());
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(w.cbegin(), w.cbegin() + w_elements, w.begin() + n * w_elements);
  }

  std::vector<const uint16_t*> i(i_elements * num_buffers);
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
  xnn_indirection_init_conv2d(&convolution_op, mr, 2 /* log2(sizeof(uint16_t)) */);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(i.cbegin(), i.cbegin() + i_elements, i.begin() + n * i_elements);
  }

  std::vector<uint16_t> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Prepare minmax parameters.
  xnn_f16_scaleminmax_params params;
  params = xnn_init_f16_scaleminmax_params(
    UINT16_C(0x3C00),  /* 1.0 */
    UINT16_C(0x7C00),  /* inf */
    UINT16_C(0xFC00)); /* -inf */

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
        f16_igemm(
          mb, nb, group_input_channels * sizeof(uint16_t), kernel_size * mr * sizeof(void*),
          reinterpret_cast<const void**>(i.data()) + buffer_index * i_elements + m,
          w.data() + buffer_index * w_elements + n * (kc_stride * kernel_size + 1),
          c.data() + buffer_index * c_elements + m * group_output_channels + n, group_output_channels * sizeof(uint16_t), nr * sizeof(uint16_t),
          0, z.data(), &params);
      }
    }
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["FLOPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 *
      output_height * output_width *
      group_input_channels * group_output_channels *
      kernel_height * kernel_width,
    benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64
  static void f16_igemm_1x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    IGEMMBenchmark(state, xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64, 1, 8, 1, 1);
  }

  static void f16_igemm_4x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    IGEMMBenchmark(state, xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64, 4, 8, 1, 1);
  }

  static void f16_igemm_6x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    IGEMMBenchmark(state, xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64, 6, 8, 1, 1);
  }

  static void f16_igemm_8x8__neonfp16arith_ld64(benchmark::State& state, const char* net) {
    IGEMMBenchmark(state, xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64, 8, 8, 1, 1);
  }

  BENCHMARK_CONV(f16_igemm_1x8__neonfp16arith_ld64)
  BENCHMARK_CONV(f16_igemm_4x8__neonfp16arith_ld64)
  BENCHMARK_CONV(f16_igemm_6x8__neonfp16arith_ld64)
  BENCHMARK_CONV(f16_igemm_8x8__neonfp16arith_ld64)
#endif  /* XNN_ARCH_ARM64 */

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
