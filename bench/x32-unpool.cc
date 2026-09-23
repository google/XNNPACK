// Copyright 2026 Léandre Le Duc
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>
#include <src/xnnpack/hardware-config.h>

#include <cstddef>
#include <cstdint>
#include <random>

#include "bench/utils.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/indirection.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/unpool.h"
#include "test/replicable_random_device.h"

static void x32_unpool(benchmark::State& state,
                       xnn_x32_unpool_ukernel_fn unpool,
                       uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t input_width = state.range(0);
  const size_t input_height = state.range(1);
  const size_t channels = state.range(2);
  const size_t pooling_width = state.range(3);
  const size_t pooling_height = state.range(4);

  const size_t pooling_size = pooling_height * pooling_width;

  // Unpadded unpooling expands each input pixel into a whole pooling window.
  const size_t output_height = input_height * pooling_height;
  const size_t output_width = input_width * pooling_width;

  const size_t input_pixels = input_height * input_width;
  const size_t output_pixels = output_height * output_width;

  const size_t input_elements = channels * input_pixels;
  const size_t output_elements = channels * output_pixels;
  // One argmax index per input element, and one output pointer per pooling
  // window position.
  const size_t index_elements = input_elements;
  const size_t indirection_elements = input_pixels * pooling_size;

  xnnpack::ReplicableRandomDevice rng;

  const size_t num_buffers =
      1 + benchmark::utils::DivideRoundUp<size_t>(
              benchmark::utils::GetMaxCacheSize(),
              sizeof(uint32_t) *
                      (input_elements + output_elements + index_elements) +
                  sizeof(void*) * indirection_elements);

  xnnpack::Buffer<uint32_t, XNN_ALLOCATION_ALIGNMENT> input(input_elements *
                                                            num_buffers);
  xnnpack::Buffer<uint32_t, XNN_ALLOCATION_ALIGNMENT> output(output_elements *
                                                             num_buffers);
  xnnpack::Buffer<uint32_t> index(index_elements * num_buffers);
  xnnpack::Buffer<uint32_t*> indirection(indirection_elements * num_buffers);

  xnnpack::fill_uniform_random_bits(input.data(), input.size(), rng);

  // The indices are the argmax positions produced by the forward max-pooling
  // pass, so each one selects a position inside its own pooling window.
  std::uniform_int_distribution<uint32_t> index_dist(
      0, static_cast<uint32_t>(pooling_size - 1));
  for (size_t n = 0; n < index.size(); n++) {
    index[n] = index_dist(rng);
  }

  // Build the output pointers with the operator's own code: the ordering it
  // produces is what the kernel actually walks.
  for (size_t n = 0; n < num_buffers; n++) {
    xnn_indirection_init_unpool2d(
        reinterpret_cast<const void**>(
            (uintptr_t*)(indirection.data() + n * indirection_elements)),
        output.data() + n * output_elements,
        /*output_pixel_stride=*/channels * sizeof(uint32_t),
        /*batch_size=*/1, input_height, input_width, output_height,
        output_width, pooling_height, pooling_width, /*output_padding_top=*/0,
        /*output_padding_left=*/0, /*batch_start=*/0);
  }

  size_t buffer_index = 0;
  for (auto _ : state) {
    buffer_index = (buffer_index + 1) % num_buffers;

    const uint32_t* i = input.data() + buffer_index * input_elements;
    const uint32_t* idx = index.data() + buffer_index * index_elements;
    uint32_t** o = indirection.data() + buffer_index * indirection_elements;

    // The microkernel handles a single input pixel per call.
    for (size_t p = input_pixels; p != 0; p--) {
      unpool(pooling_size, channels, /*fill=*/0, i, idx, o);
      i += channels;
      idx += channels;
      o += pooling_size;
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["elements"] =
      benchmark::Counter(static_cast<double>(state.iterations()) *
                             static_cast<double>(output_elements),
                         benchmark::Counter::kIsRate);

  // output is written once, input, indices and
  // output pointers are each streamed once.
  state.counters["bytes"] = benchmark::Counter(
      static_cast<double>(state.iterations()) *
          static_cast<double>(
              (input_elements + index_elements + output_elements) *
                  sizeof(uint32_t) +
              indirection_elements * sizeof(void*)),
      benchmark::Counter::kIsRate);
}

// {InW, InH, C, PoolW, PoolH}.
static void UnpoolArguments(benchmark::Benchmark* b) {
  b->ArgNames({"InW", "InH", "C", "PoolW", "PoolH"});
  // SegNet-style decoder stages, 2x2 max-unpooling.
  b->Args({7, 7, 512, 2, 2});
  b->Args({14, 14, 512, 2, 2});
  b->Args({28, 28, 256, 2, 2});
  b->Args({56, 56, 128, 2, 2});
  b->Args({112, 112, 64, 2, 2});
  // Wider window, to shift the fill/scatter ratio.
  b->Args({28, 28, 256, 3, 3});
}

#define BENCHMARK_UNPOOL(name, arch_flags)                            \
  BENCHMARK_CAPTURE(x32_unpool, name, xnn_x32_unpool_ukernel__##name, \
                    arch_flags)                                       \
      ->Apply(UnpoolArguments)                                        \
      ->UseRealTime();

BENCHMARK_UNPOOL(scalar, 0);

#if XNN_ARCH_WASMRELAXEDSIMD || XNN_ARCH_WASMSIMD
BENCHMARK_UNPOOL(wasmsimd, 0);
#endif

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_UNPOOL(neon, xnn_arch_arm_neon);
#endif

#if (XNN_ARCH_X86 || XNN_ARCH_X86_64) && XNN_ENABLE_SSE
BENCHMARK_UNPOOL(sse2, xnn_arch_x86_sse2);
#endif

#if XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR
BENCHMARK_UNPOOL(rvv, xnn_arch_riscv_vector);
#endif

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
