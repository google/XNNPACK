// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>

#include "bench/utils.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/reduce.h"
#include <benchmark/benchmark.h>

static void s8_rminmax(benchmark::State& state,
                       xnn_s8_reduce_ukernel_fn rminmax,
                       benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto s8rng = std::bind(std::uniform_int_distribution<int32_t>(-128, 127),
                         std::ref(rng));

  xnnpack::Buffer<int8_t, XNN_ALLOCATION_ALIGNMENT> input(elements);
  std::generate(input.begin(), input.end(), std::ref(s8rng));

  int8_t output[2] = {std::numeric_limits<int8_t>::max(),
                      std::numeric_limits<int8_t>::min()};
  for (auto _ : state) {
    rminmax(elements * sizeof(int8_t), input.data(), output,
            /*params=*/nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = elements * sizeof(float);
  state.counters["bytes"] = benchmark::Counter(
      static_cast<uint64_t>(state.iterations()) * bytes_per_iteration,
      benchmark::Counter::kIsRate);
}

BENCHMARK_CAPTURE(s8_rminmax, scalar_u1, xnn_s8_rminmax_ukernel__scalar_u1)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();

BENCHMARK_CAPTURE(s8_rminmax, scalar_u2_acc2,
                  xnn_s8_rminmax_ukernel__scalar_u2_acc2)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();

BENCHMARK_CAPTURE(s8_rminmax, scalar_u3_acc3,
                  xnn_s8_rminmax_ukernel__scalar_u3_acc3)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();

BENCHMARK_CAPTURE(s8_rminmax, scalar_u4_acc2,
                  xnn_s8_rminmax_ukernel__scalar_u4_acc2)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();

BENCHMARK_CAPTURE(s8_rminmax, scalar_u4_acc4,
                  xnn_s8_rminmax_ukernel__scalar_u4_acc4)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(s8_rminmax, neon_u16, xnn_s8_rminmax_ukernel__neon_u16,
                  benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(s8_rminmax, neon_u32_acc2,
                  xnn_s8_rminmax_ukernel__neon_u32_acc2,
                  benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(s8_rminmax, neon_u48_acc3,
                  xnn_s8_rminmax_ukernel__neon_u48_acc3,
                  benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(s8_rminmax, neon_u64_acc2,
                  xnn_s8_rminmax_ukernel__neon_u64_acc2,
                  benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
BENCHMARK_CAPTURE(s8_rminmax, neon_u64_acc4,
                  xnn_s8_rminmax_ukernel__neon_u64_acc4,
                  benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
BENCHMARK_CAPTURE(s8_rminmax, sse41_u16, xnn_s8_rminmax_ukernel__sse41_u16,
                  benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
BENCHMARK_CAPTURE(s8_rminmax, sse41_u32_acc2,
                  xnn_s8_rminmax_ukernel__sse41_u32_acc2,
                  benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
BENCHMARK_CAPTURE(s8_rminmax, sse41_u48_acc3,
                  xnn_s8_rminmax_ukernel__sse41_u48_acc3,
                  benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
BENCHMARK_CAPTURE(s8_rminmax, sse41_u64_acc2,
                  xnn_s8_rminmax_ukernel__sse41_u64_acc2,
                  benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
BENCHMARK_CAPTURE(s8_rminmax, sse41_u64_acc4,
                  xnn_s8_rminmax_ukernel__sse41_u64_acc4,
                  benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_HEXAGON && XNN_ENABLE_HVX
BENCHMARK_CAPTURE(s8_rminmax, hvx_u256_acc2,
                  xnn_s8_rminmax_ukernel__hvx_u256_acc2,
                  benchmark::utils::CheckHVX)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_HEXAGON && XNN_ENABLE_HVX

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
BENCHMARK_CAPTURE(s8_rminmax, wasmsimd_u16,
                  xnn_s8_rminmax_ukernel__wasmsimd_u16)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
BENCHMARK_CAPTURE(s8_rminmax, wasmsimd_u32_acc2,
                  xnn_s8_rminmax_ukernel__wasmsimd_u32_acc2)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
BENCHMARK_CAPTURE(s8_rminmax, wasmsimd_u48_acc3,
                  xnn_s8_rminmax_ukernel__wasmsimd_u48_acc3)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
BENCHMARK_CAPTURE(s8_rminmax, wasmsimd_u64_acc2,
                  xnn_s8_rminmax_ukernel__wasmsimd_u64_acc2)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
BENCHMARK_CAPTURE(s8_rminmax, wasmsimd_u64_acc4,
                  xnn_s8_rminmax_ukernel__wasmsimd_u64_acc4)
    ->Apply(benchmark::utils::ReductionParameters<int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
