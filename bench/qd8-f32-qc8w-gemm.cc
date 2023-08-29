// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
//
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/gemm.h"
#include "bench/utils.h"

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/math.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/pack.h>


void GEMMBenchmark(benchmark::State& state,
  xnn_qd8_f32_qc8w_gemm_ukernel_fn gemm,
  xnn_init_f32_minmax_params_fn init_params,
  size_t mr, size_t nr, size_t kr, size_t sr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t mc = state.range(0);
  const size_t nc = state.range(1);
  const size_t kc = state.range(2);

  const size_t nc_stride = benchmark::utils::RoundUp(nc, nr);
  const size_t kc_stride = benchmark::utils::RoundUp(kc, kr * sr);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), std::ref(rng));

  std::vector<int8_t> a(mc * kc + XNN_EXTRA_BYTES);
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::vector<int8_t> k(nc * kc);
  std::generate(k.begin(), k.end(), std::ref(f32rng));

  std::vector<xnn_qd8_quantization_params> quantization_params(mr + XNN_EXTRA_QUANTIZATION_PARAMS);
  const size_t w_elements = nc_stride * (sizeof(float) * 2 + sizeof(int32_t)) + kc_stride * nc_stride;

  const size_t c_elements = mc * nc;
  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(),
      sizeof(float) * (w_elements + c_elements));

  std::vector<char, AlignedAllocator<char, 64>> w(w_elements * num_buffers);
  std::fill(w.begin(), w.end(), 0.0f);
  const xnn_qs8_packing_params packing_params = { /*input_zero_point=*/1 };
  xnn_pack_qs8_gemm_goi_w(1, nc, kc, nr, kr, sr,
                          k.data(), /*bias=*/nullptr, /*scale=*/nullptr, w.data(), sizeof(float) * 2 * nr, &packing_params);
  std::vector<float> c(c_elements * num_buffers);
  std::fill(c.begin(), c.end(), std::nanf(""));

  // Prepare parameters.
  xnn_f32_minmax_params params;
  init_params(&params, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

  size_t buffer_index = 0;
  for (auto _ : state) {
    // Use circular buffers (exceeding cache size) and prefetch to control cache state:
    // - A is always in L1 cache (if fits, otherwise L2, L3, etc)
    // - W is not in cache (for any cache level)
    // - C is not in cache (for any cache level)
    state.PauseTiming();
    benchmark::utils::PrefetchToL1(a.data(), a.size());
    buffer_index = (buffer_index + 1) % num_buffers;
    state.ResumeTiming();

    for (uint32_t m = 0; m < mc; m += mr) {
      const uint32_t mb = min(mc - m, mr);
      gemm(
        mb, nc, kc,
        a.data() + m * kc, kc * sizeof(int8_t),
        w.data() + w_elements * buffer_index,
        c.data() + (buffer_index * mc + m) * nc, nc * sizeof(float), nr * sizeof(float),
        &params, quantization_params.data());
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  state.counters["OPS"] = benchmark::Counter(
    uint64_t(state.iterations()) * 2 * mc * nc * kc, benchmark::Counter::kIsRate);
}

static void qd8_f32_qc8w_gemm_ukernel_1x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar, xnn_init_f32_minmax_scalar_params,
    /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc8w_gemm_ukernel_1x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__scalar, xnn_init_f32_minmax_scalar_params,
    /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc8w_gemm_ukernel_1x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__scalar, xnn_init_f32_minmax_scalar_params,
    /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc8w_gemm_ukernel_2x2__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__scalar, xnn_init_f32_minmax_scalar_params,
    /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc8w_gemm_ukernel_2x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4__scalar, xnn_init_f32_minmax_scalar_params,
    /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc8w_gemm_ukernel_2x8__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__scalar, xnn_init_f32_minmax_scalar_params,
    /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1);
}

static void qd8_f32_qc8w_gemm_ukernel_4x4__scalar(benchmark::State& state, const char* net) {
  GEMMBenchmark(state,
    xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__scalar, xnn_init_f32_minmax_scalar_params,
    /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_ukernel_1x8c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal, xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_ukernel_1x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot, xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_ukernel_1x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot, xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qd8_f32_qc8w_gemm_ukernel_2x8c2s4__neon_mlal(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal, xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/2, /*sr=*/4,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_ukernel_2x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__neondot, xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_ukernel_2x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__neondot, xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_ukernel_3x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__neondot, xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_ukernel_4x8c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot, xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qd8_f32_qc8w_gemm_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55, xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qd8_f32_qc8w_gemm_ukernel_4x16c4__asm_aarch64_neondot_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64, xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qd8_f32_qc8w_gemm_ukernel_4x16c4__asm_aarch64_neondot_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128, xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qd8_f32_qc8w_gemm_ukernel_4x16c4__neondot(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot, xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/4, /*sr=*/1,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x4c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x4c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x4c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x4c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/8, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckNEONI8MM);
  }
#endif //XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qd8_f32_qc8w_gemm_ukernel_1x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void qd8_f32_qc8w_gemm_ukernel_1x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void qd8_f32_qc8w_gemm_ukernel_1x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  static void qd8_f32_qc8w_gemm_ukernel_1x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  static void qd8_f32_qc8w_gemm_ukernel_2x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void qd8_f32_qc8w_gemm_ukernel_2x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void qd8_f32_qc8w_gemm_ukernel_2x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  static void qd8_f32_qc8w_gemm_ukernel_2x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  static void qd8_f32_qc8w_gemm_ukernel_3x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void qd8_f32_qc8w_gemm_ukernel_3x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void qd8_f32_qc8w_gemm_ukernel_3x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  static void qd8_f32_qc8w_gemm_ukernel_3x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  static void qd8_f32_qc8w_gemm_ukernel_4x4c8__sse2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void qd8_f32_qc8w_gemm_ukernel_4x4c8__sse2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void qd8_f32_qc8w_gemm_ukernel_4x4c8__sse41_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  static void qd8_f32_qc8w_gemm_ukernel_4x4c8__sse41_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckSSE41);
  }

  static void qd8_f32_qc8w_gemm_ukernel_1x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  static void qd8_f32_qc8w_gemm_ukernel_1x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  static void qd8_f32_qc8w_gemm_ukernel_1x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  static void qd8_f32_qc8w_gemm_ukernel_1x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  static void qd8_f32_qc8w_gemm_ukernel_2x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  static void qd8_f32_qc8w_gemm_ukernel_2x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  static void qd8_f32_qc8w_gemm_ukernel_2x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  static void qd8_f32_qc8w_gemm_ukernel_2x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  static void qd8_f32_qc8w_gemm_ukernel_3x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  static void qd8_f32_qc8w_gemm_ukernel_3x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  static void qd8_f32_qc8w_gemm_ukernel_3x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__xop_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  static void qd8_f32_qc8w_gemm_ukernel_3x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__xop_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  static void qd8_f32_qc8w_gemm_ukernel_4x4c8__avx_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  static void qd8_f32_qc8w_gemm_ukernel_4x4c8__avx_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX);
  }

  static void qd8_f32_qc8w_gemm_ukernel_4x4c8__xop_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__xop_ld64, xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  static void qd8_f32_qc8w_gemm_ukernel_4x4c8__xop_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__xop_ld128, xnn_init_f32_minmax_sse_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckXOP);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx2, xnn_init_f32_minmax_avx_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx2, xnn_init_f32_minmax_avx_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx2(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx2, xnn_init_f32_minmax_avx_params,
      /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx, xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx, xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx, xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx, xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16, /*kr=*/8, /*sr=*/1,
      benchmark::utils::CheckAVX2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__wasm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__wasm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__wasm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4__wasm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__wasm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__wasm(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__wasm, xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1);
  }
#endif // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/2, /*sr=*/4);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/1, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/2, /*sr=*/4);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*kr=*/2, /*sr=*/4);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/1);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/2, /*sr=*/4);
  }

  static void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, const char* net) {
    GEMMBenchmark(state,
      xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld128, xnn_init_f32_minmax_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*kr=*/8, /*sr=*/1);
  }
#endif // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x2__scalar)
BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x4__scalar)
BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x8__scalar)
BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x2__scalar)
BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x4__scalar)
BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x8__scalar)
BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x4__scalar)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x8c2s4__neon_mlal)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x8c2s4__neon_mlal)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x8c4__neondot)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x16c4__neondot)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x8c4__neondot)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x16c4__neondot)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_3x16c4__neondot)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x8c4__neondot)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x16c4__neondot)
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x16c4__asm_aarch64_neondot_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x16c4__asm_aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x4c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x4c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x4c8__sse2_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x4c8__sse2_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x4c8__sse41_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x4c8__sse41_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x4c8__sse2_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x4c8__sse2_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x4c8__sse41_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x4c8__sse41_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_3x4c8__sse2_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_3x4c8__sse2_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_3x4c8__sse41_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_3x4c8__sse41_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x4c8__sse2_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x4c8__sse2_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x4c8__sse41_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x4c8__sse41_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x4c8__avx_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x4c8__avx_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x4c8__xop_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_1x4c8__xop_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x4c8__avx_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x4c8__avx_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x4c8__xop_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_2x4c8__xop_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_3x4c8__avx_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_3x4c8__avx_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_3x4c8__xop_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_3x4c8__xop_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x4c8__avx_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x4c8__avx_ld128)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x4c8__xop_ld64)
  BENCHMARK_GEMM(qd8_f32_qc8w_gemm_ukernel_4x4c8__xop_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx2)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx2)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx2)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__wasm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__wasm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__wasm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4__wasm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__wasm)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__wasm)
#endif // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64)
  BENCHMARK_GEMM(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld128)
#endif // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
