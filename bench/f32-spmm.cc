// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-spmm-minmax.yaml
//   Generator: tools/generate-spmm-test.py

#include <benchmark/benchmark.h>
#include "bench/spmm-benchmark.h"
#include "bench/utils.h"

#include <xnnpack/gemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_pipelined, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_pipelined)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_pipelined_x2, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_pipelined_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_x2, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_x4, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_arm_x4)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_pipelined, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_pipelined)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_pipelined_x2, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_pipelined_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_x2, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_x4, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmrelaxedsimd_x86_x4)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_pipelined, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_pipelined)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_pipelined_x2, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_pipelined_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_x2, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_x4, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_arm_x4)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_pipelined, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_pipelined)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_pipelined_x2, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_pipelined_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_x2, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_x4, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmrelaxedsimd_x86_x4)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_pipelined, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_pipelined)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_pipelined_x2, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_pipelined_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_x2, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_x4, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_arm_x4)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_pipelined, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_pipelined)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_pipelined_x2, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_pipelined_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_x2, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_x4, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmrelaxedsimd_x86_x4)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_pipelined, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_pipelined)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_pipelined_x2, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_pipelined_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_x2, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_x4, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_arm_x4)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_pipelined, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_pipelined)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_pipelined_x2, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_pipelined_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_x2, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_x2)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_x4, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmrelaxedsimd_x86_x4)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmsimd_arm)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_pipelined, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_pipelined)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_pipelined_x2, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_pipelined_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_x2, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_x4, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmsimd_arm_x4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmsimd_x86)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_pipelined, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_pipelined)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_pipelined_x2, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_pipelined_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_x2, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_x4, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__wasmsimd_x86_x4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmsimd_arm)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_pipelined, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_pipelined)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_pipelined_x2, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_pipelined_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_x2, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_x4, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmsimd_arm_x4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmsimd_x86)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_pipelined, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_pipelined)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_pipelined_x2, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_pipelined_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x2, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x4, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__wasmsimd_x86_x4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmsimd_arm)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_pipelined, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_pipelined)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_pipelined_x2, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_pipelined_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_x2, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_x4, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmsimd_arm_x4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmsimd_x86)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_pipelined, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_pipelined)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_pipelined_x2, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_pipelined_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_x2, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_x4, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__wasmsimd_x86_x4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmsimd_arm(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmsimd_arm)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_pipelined, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_pipelined)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_pipelined_x2, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_pipelined_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_x2, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_x4, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmsimd_arm_x4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmsimd_x86(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmsimd_x86)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_pipelined, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_pipelined)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_pipelined_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_pipelined_x2, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_pipelined_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x2, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x2)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x4(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x4, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_wasmsimd_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__wasmsimd_x86_x4)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_4x1__neon(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__neon, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__neon)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_4x1__neon_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__neon_pipelined, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__neon_pipelined)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_4x1__neon_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__neon_x2, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__neon_x2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_8x1__neon(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__neon, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__neon)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_8x1__neon_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__neon_pipelined, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__neon_pipelined)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_8x1__neon_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__neon_x2, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__neon_x2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_12x1__neon(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_12x1__neon, 12, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_12x1__neon)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_16x1__neon(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__neon, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__neon)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_16x1__neon_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__neon_pipelined, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__neon_pipelined)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_16x1__neon_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__neon_x2, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__neon_x2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_32x1__neon(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__neon, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__neon)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_32x1__neon_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__neon_pipelined, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__neon_pipelined)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_32x1__neon_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__neon_x2, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEON
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__neon_x2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_4x1__neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__neonfma, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_4x1__neonfma_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__neonfma_pipelined, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__neonfma_pipelined)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_4x1__neonfma_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__neonfma_x2, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__neonfma_x2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_4x2__aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x2__aarch64_neonfma, 4, 2,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x2__aarch64_neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_4x4__aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x4__aarch64_neonfma, 4, 4,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x4__aarch64_neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_8x1__neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__neonfma, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_8x1__neonfma_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__neonfma_pipelined, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__neonfma_pipelined)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_8x1__neonfma_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__neonfma_x2, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__neonfma_x2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_8x2__aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x2__aarch64_neonfma, 8, 2,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x2__aarch64_neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_8x4__aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x4__aarch64_neonfma, 8, 4,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x4__aarch64_neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_12x1__neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_12x1__neonfma, 12, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_12x1__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_12x2__aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_12x2__aarch64_neonfma, 12, 2,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_12x2__aarch64_neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_12x4__aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_12x4__aarch64_neonfma, 12, 4,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_12x4__aarch64_neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_16x1__neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__neonfma, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_16x1__neonfma_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__neonfma_pipelined, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__neonfma_pipelined)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_16x1__neonfma_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__neonfma_x2, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__neonfma_x2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_16x2__aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x2__aarch64_neonfma, 16, 2,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x2__aarch64_neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_16x4__aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x4__aarch64_neonfma, 16, 4,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x4__aarch64_neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_32x1__neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__neonfma, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__neonfma)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_32x1__neonfma_pipelined(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__neonfma_pipelined, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__neonfma_pipelined)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_32x1__neonfma_x2(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__neonfma_x2, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__neonfma_x2)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_32x2__aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x2__aarch64_neonfma, 32, 2,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x2__aarch64_neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  static void f32_spmm_minmax_ukernel_32x4__aarch64_neonfma(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x4__aarch64_neonfma, 32, 4,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
    benchmark::utils::CheckNEONFMA
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x4__aarch64_neonfma)
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_spmm_minmax_ukernel_4x1__sse(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__sse, 4, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_sse_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__sse)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_spmm_minmax_ukernel_8x1__sse(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__sse, 8, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_sse_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__sse)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_spmm_minmax_ukernel_16x1__sse(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_16x1__sse, 16, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_sse_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_16x1__sse)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_spmm_minmax_ukernel_32x1__sse(benchmark::State& state, const char* net) {
    f32_spmm(state, xnn_f32_spmm_minmax_ukernel_32x1__sse, 32, 1,
      /*sparsity=*/0.8f, xnn_init_f32_minmax_sse_params,
    /*isa_check=*/nullptr
    );
  }

  BENCHMARK_SPMM(f32_spmm_minmax_ukernel_32x1__sse)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


static void f32_spmm_minmax_ukernel_1x1__scalar(benchmark::State& state, const char* net) {
  f32_spmm(state, xnn_f32_spmm_minmax_ukernel_1x1__scalar, 1, 1,
    /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
  /*isa_check=*/nullptr
  );
}

BENCHMARK_SPMM(f32_spmm_minmax_ukernel_1x1__scalar)

static void f32_spmm_minmax_ukernel_1x1__scalar_pipelined(benchmark::State& state, const char* net) {
  f32_spmm(state, xnn_f32_spmm_minmax_ukernel_1x1__scalar_pipelined, 1, 1,
    /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
  /*isa_check=*/nullptr
  );
}

BENCHMARK_SPMM(f32_spmm_minmax_ukernel_1x1__scalar_pipelined)

static void f32_spmm_minmax_ukernel_2x1__scalar(benchmark::State& state, const char* net) {
  f32_spmm(state, xnn_f32_spmm_minmax_ukernel_2x1__scalar, 2, 1,
    /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
  /*isa_check=*/nullptr
  );
}

BENCHMARK_SPMM(f32_spmm_minmax_ukernel_2x1__scalar)

static void f32_spmm_minmax_ukernel_2x1__scalar_pipelined(benchmark::State& state, const char* net) {
  f32_spmm(state, xnn_f32_spmm_minmax_ukernel_2x1__scalar_pipelined, 2, 1,
    /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
  /*isa_check=*/nullptr
  );
}

BENCHMARK_SPMM(f32_spmm_minmax_ukernel_2x1__scalar_pipelined)

static void f32_spmm_minmax_ukernel_4x1__scalar(benchmark::State& state, const char* net) {
  f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__scalar, 4, 1,
    /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
  /*isa_check=*/nullptr
  );
}

BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__scalar)

static void f32_spmm_minmax_ukernel_4x1__scalar_pipelined(benchmark::State& state, const char* net) {
  f32_spmm(state, xnn_f32_spmm_minmax_ukernel_4x1__scalar_pipelined, 4, 1,
    /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
  /*isa_check=*/nullptr
  );
}

BENCHMARK_SPMM(f32_spmm_minmax_ukernel_4x1__scalar_pipelined)

static void f32_spmm_minmax_ukernel_8x1__scalar(benchmark::State& state, const char* net) {
  f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__scalar, 8, 1,
    /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
  /*isa_check=*/nullptr
  );
}

BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__scalar)

static void f32_spmm_minmax_ukernel_8x1__scalar_pipelined(benchmark::State& state, const char* net) {
  f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x1__scalar_pipelined, 8, 1,
    /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
  /*isa_check=*/nullptr
  );
}

BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x1__scalar_pipelined)

static void f32_spmm_minmax_ukernel_8x2__scalar(benchmark::State& state, const char* net) {
  f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x2__scalar, 8, 2,
    /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
  /*isa_check=*/nullptr
  );
}

BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x2__scalar)

static void f32_spmm_minmax_ukernel_8x4__scalar(benchmark::State& state, const char* net) {
  f32_spmm(state, xnn_f32_spmm_minmax_ukernel_8x4__scalar, 8, 4,
    /*sparsity=*/0.8f, xnn_init_f32_minmax_scalar_params,
  /*isa_check=*/nullptr
  );
}

BENCHMARK_SPMM(f32_spmm_minmax_ukernel_8x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
