// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/vunary.h>


static void f32_vrelu(
  benchmark::State& state,
  xnn_f32_vrelu_ukernel_fn f32_vrelu,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::vector<float, AlignedAllocator<float, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));

  for (auto _ : state) {
    f32_vrelu(num_elements * sizeof(float), x.data(), y.data(), nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * num_elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if (XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD) && XNN_PLATFORM_JIT
static void f32_vrelu(
  benchmark::State& state,
  xnn_vrelu_generator_fn generator,
  int k_unroll,
  bool use_local,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  generator(&b, k_unroll, use_local);
  xnn_finalize_code_memory(&b);
  auto kernel = (xnn_f32_vrelu_ukernel_fn)(xnn_first_function_ptr(&b));
  f32_vrelu(state, kernel, isa_check);
  xnn_release_code_memory(&b);
}
#endif  // (XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD) && XNN_PLATFORM_JIT

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vrelu, sse_u4,
                    xnn_f32_vrelu_ukernel__sse_u4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, sse_u8,
                    xnn_f32_vrelu_ukernel__sse_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, avx_u8,
                    xnn_f32_vrelu_ukernel__avx_u8,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, avx_u16,
                    xnn_f32_vrelu_ukernel__avx_u16,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, avx512f_u16,
                    xnn_f32_vrelu_ukernel__avx512f_u16,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, avx512f_u32,
                    xnn_f32_vrelu_ukernel__avx512f_u32,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_vrelu, neon_u4,
                    xnn_f32_vrelu_ukernel__neon_u4,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_vrelu, neon_u8,
                    xnn_f32_vrelu_ukernel__neon_u8,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vrelu, wasm_u1,
                    xnn_f32_vrelu_ukernel__wasm_u1)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, wasm_u2,
                    xnn_f32_vrelu_ukernel__wasm_u2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, wasm_u4,
                    xnn_f32_vrelu_ukernel__wasm_u4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, wasm_u8,
                    xnn_f32_vrelu_ukernel__wasm_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, wasm32_shr_u1,
                    xnn_f32_vrelu_ukernel__wasm32_shr_u1)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, wasm32_shr_u2,
                    xnn_f32_vrelu_ukernel__wasm32_shr_u2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, wasm32_shr_u4,
                    xnn_f32_vrelu_ukernel__wasm32_shr_u4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if (XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD) && XNN_PLATFORM_JIT
  BENCHMARK_CAPTURE(f32_vrelu, jit_wasm32_shr_u1,
                    xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 1, false)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, jit_wasm32_shr_u2,
                    xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, false)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, jit_wasm32_shr_u4,
                    xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, false)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, jit_wasm32_shr_u8,
                    xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, false)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, jit_wasm32_shr_local_u1,
                    xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 1, true)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, jit_wasm32_shr_local_u2,
                    xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 2, true)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, jit_wasm32_shr_local_u4,
                    xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 4, true)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, jit_wasm32_shr_local_u8,
                    xnn_generate_f32_vrelu_ukernel__jit_wasm32_shr, 8, true)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // (XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD) && XNN_PLATFORM_JIT

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_vrelu, wasmsimd_u4,
                    xnn_f32_vrelu_ukernel__wasmsimd_u4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, wasmsimd_u8,
                    xnn_f32_vrelu_ukernel__wasmsimd_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_vrelu, wasmsimd_u16,
                    xnn_f32_vrelu_ukernel__wasmsimd_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_vrelu, scalar_u1,
                  xnn_f32_vrelu_ukernel__scalar_u1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_vrelu, scalar_u2,
                  xnn_f32_vrelu_ukernel__scalar_u2)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_vrelu, scalar_u4,
                  xnn_f32_vrelu_ukernel__scalar_u4)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_vrelu, scalar_u8,
                  xnn_f32_vrelu_ukernel__scalar_u8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
