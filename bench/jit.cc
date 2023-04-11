// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include <benchmark/benchmark.h>

#include <xnnpack/gemm.h>
#include <xnnpack/memory.h>
#include <xnnpack/microparams.h>


void reset_code_buffer(xnn_code_buffer* code_buffer, void* start) {
  code_buffer->start = start; // Reset to beginning so we don't constantly grow the buffer.
  code_buffer->size = 0;
}

#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
static void f32_gemm_6x8__aarch64_neonfma_ld128(benchmark::State& state) {
  xnn_code_buffer code_buffer;
  if (xnn_allocate_code_memory(&code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE) != xnn_status_success) {
    state.SkipWithError("failed to allocate code memory");
  }
  void* start = code_buffer.start;
  jit_gemm_params params = {};
  for (auto _ : state) {
    reset_code_buffer(&code_buffer, start);
    const xnn_status_t status =
      xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128(&code_buffer, 6, 0, sizeof(float), &params);
    if (status != xnn_status_success) {
      state.SkipWithError("code generation failed");
    }
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * code_buffer.size);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * code_buffer.size / sizeof(uint32_t));
  xnn_release_code_memory(&code_buffer);
}

BENCHMARK(f32_gemm_6x8__aarch64_neonfma_ld128);
#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT

#if XNN_ARCH_ARM && XNN_PLATFORM_JIT
static void f32_gemm_4x8__aarch32_neon_ld64(benchmark::State& state) {
  xnn_code_buffer code_buffer;
  if (xnn_allocate_code_memory(&code_buffer, XNN_DEFAULT_CODE_BUFFER_SIZE) != xnn_status_success) {
    state.SkipWithError("failed to allocate code memory");
  }
  void* start = code_buffer.start;
  jit_gemm_params params = {};
  for (auto _ : state) {
    reset_code_buffer(&code_buffer, start);
    const xnn_status_t status =
      xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_ld64(&code_buffer, 4, 0, sizeof(float), &params);
    if (status != xnn_status_success) {
      state.SkipWithError("code generation failed");
    }
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * code_buffer.size);
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * code_buffer.size / sizeof(uint32_t));
  xnn_release_code_memory(&code_buffer);
}

BENCHMARK(f32_gemm_4x8__aarch32_neon_ld64);
#endif  // XNN_ARCH_ARM && XNN_PLATFORM_JIT

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
