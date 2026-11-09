// Copyright 2026 Kalray
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

#include "bench/utils.h"
#include "src/xnnpack/aligned-allocator.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/ibilinear.h"
#include "src/xnnpack/indirection.h"
#include "src/xnnpack/microfnptr.h"
#include "test/replicable_random_device.h"

static void f32_ibilinear_chw(benchmark::State& state,
                              xnn_f32_ibilinear_chw_ukernel_fn ibilinear,
                              uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t input_width = state.range(0);
  const size_t input_height = state.range(1);
  const size_t output_width = state.range(2);
  const size_t output_height = state.range(3);
  const size_t channels = state.range(4);

  const size_t input_pixels = input_height * input_width;
  const size_t output_pixels = output_height * output_width;

  // Channel planes are disjoint and exactly tiled: the operator advances by one
  // whole input plane per channel. See resize-bilinear-nchw.c:250.
  const size_t input_increment = input_pixels * sizeof(float);

  benchmark::utils::DisableDenormals();

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist;

  // Two pointers (top-left, bottom-left) and two weights (alpha_h, alpha_v) per
  // output pixel, interleaved, shared by every channel.
  const size_t i_elements = output_pixels * 2;
  const size_t w_elements = output_pixels * 2;
  const size_t input_elements = channels * input_pixels;
  const size_t output_elements = channels * output_pixels;

  const size_t image_stride = benchmark::utils::RoundUp<size_t>(
      input_elements * sizeof(float), XNN_ALLOCATION_ALIGNMENT);

  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(float) * (input_elements + output_elements + w_elements) +
                  sizeof(void*) * i_elements);

  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> input(
      num_buffers * (image_stride / sizeof(float)));
  std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

  xnnpack::Buffer<const float*> indirection(i_elements * num_buffers);
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> packed_weights(w_elements *
                                                                  num_buffers);

  // Build the indirection buffer and the interpolation coefficients with the
  // operator's own code.
  xnn_indirection_init_resize_bilinear2d_chw_f32(
      /*input_pixel_stride=*/sizeof(float), input_height, input_width,
      output_height, output_width, input.data(),
      reinterpret_cast<const void**>(indirection.data()), packed_weights.data(),
      /*align_corners=*/false, /*tensorflow_legacy=*/false);
  for (size_t n = 1; n < num_buffers; n++) {
    std::copy(indirection.cbegin(), indirection.cbegin() + i_elements,
              indirection.begin() + n * i_elements);
    std::copy(packed_weights.cbegin(), packed_weights.cbegin() + w_elements,
              packed_weights.begin() + n * w_elements);
  }

  xnnpack::Buffer<float> output(output_elements * num_buffers);

  size_t buffer_index = 0;
  for (auto _ : state) {
    buffer_index = (buffer_index + 1) % num_buffers;
    ibilinear(output_pixels, channels,
              indirection.data() + buffer_index * i_elements,
              /*input_offset=*/buffer_index * image_stride,
              packed_weights.data() + buffer_index * w_elements,
              output.data() + buffer_index * output_elements, input_increment);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
      benchmark::Counter(static_cast<double>(state.iterations()) *
                             static_cast<double>(output_elements),
                         benchmark::Counter::kIsRate);

  // 3 subtractions and 3 FMAs per output pixel per channel.
  state.counters["FLOPS"] =
      benchmark::Counter(static_cast<double>(state.iterations()) *
                             static_cast<double>(output_elements) * 9,
                         benchmark::Counter::kIsRate);

  // Compulsory traffic only: understates untiled kernels, which re-read the
  // indirection and weights for every channel.
  state.counters["bytes"] = benchmark::Counter(
      static_cast<double>(state.iterations()) *
          static_cast<double>(
              channels * (input_pixels + output_pixels) * sizeof(float) +
              output_pixels * (2 * sizeof(void*) + 2 * sizeof(float))),
      benchmark::Counter::kIsRate);
}

// {InW, InH, OutW, OutH, C}.
static void BilinearArguments(benchmark::Benchmark* b) {
  b->ArgNames({"InW", "InH", "OutW", "OutH", "C"});

  // MobileNet/DeepLab-style decoder ladder: channels halve as the spatial size
  // doubles. Same shapes as bench/operators/resize-bilinear-nhwc.cc, so the CHW
  // microkernel numbers stay comparable with the operator benchmark.
  b->Args({12, 8, 24, 16, 256});
  b->Args({5, 33, 10, 33, 192});
  b->Args({20, 33, 40, 33, 96});
  b->Args({6, 5, 12, 10, 96});
  b->Args({24, 20, 48, 40, 48});
  b->Args({80, 66, 160, 66, 24});

  // 2x image upsampling, RGB and RGBA.
  for (int c : {3, 4}) {
    b->Args({256, 256, 512, 512, c});
  }

  // Working sets larger than cache, where amortizing the indirection and weight
  // re-reads across channels actually matters.
  b->Args({64, 64, 128, 128, 128});   // 2x, 2 MB in / 8 MB out
  b->Args({32, 32, 128, 128, 256});   // 4x, ASPP decoder
  b->Args({128, 128, 256, 256, 32});  // 2x, segmentation head
}

#define BENCHMARK_IBILINEAR_CHW(name, arch_flags)                      \
  BENCHMARK_CAPTURE(f32_ibilinear_chw, name,                           \
                    xnn_f32_ibilinear_chw_ukernel__##name, arch_flags) \
      ->Apply(BilinearArguments)                                       \
      ->UseRealTime();

BENCHMARK_IBILINEAR_CHW(scalar_p1, 0);
BENCHMARK_IBILINEAR_CHW(scalar_p2, 0);
BENCHMARK_IBILINEAR_CHW(scalar_p4, 0);

#if XNN_ARCH_WASMRELAXEDSIMD || XNN_ARCH_WASMSIMD
BENCHMARK_IBILINEAR_CHW(wasmsimd_p4, 0);
BENCHMARK_IBILINEAR_CHW(wasmsimd_p8, 0);
#endif

#if XNN_ARCH_ARM
BENCHMARK_IBILINEAR_CHW(neon_p4, xnn_arch_arm_neon);
BENCHMARK_IBILINEAR_CHW(neon_p8, xnn_arch_arm_neon);
BENCHMARK_IBILINEAR_CHW(neon_p16, xnn_arch_arm_neon);
#endif

#if XNN_ARCH_ARM64
BENCHMARK_IBILINEAR_CHW(neonfma_p4, xnn_arch_arm_neon_fma);
BENCHMARK_IBILINEAR_CHW(neonfma_p8, xnn_arch_arm_neon_fma);
BENCHMARK_IBILINEAR_CHW(neonfma_p16, xnn_arch_arm_neon_fma);
#endif

#if (XNN_ARCH_X86 || XNN_ARCH_X86_64) && XNN_ENABLE_SSE
BENCHMARK_IBILINEAR_CHW(sse_p4, xnn_arch_x86_sse);
BENCHMARK_IBILINEAR_CHW(sse_p8, xnn_arch_x86_sse);
#endif

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
