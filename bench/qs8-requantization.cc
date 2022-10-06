// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2020 Google LLC
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
#include "bench/utils.h"

#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/requantization-stubs.h>


class Requantization : public benchmark::Fixture {
 public:
  inline Requantization()
  {
#if XNN_ARCH_HEXAGON
    const size_t l1d_size = 16384;
#else
    cpuinfo_initialize();
    const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size;
#endif
    const size_t l1d_reserve = 1024;
    n_ = (l1d_size - l1d_reserve) / (sizeof(int32_t) + sizeof(int8_t));
    n_ = n_ / 16 * 16;
  }

  virtual void SetUp(benchmark::State& state) override
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto i32rng = std::bind(std::uniform_int_distribution<int32_t>(), std::ref(rng));

    input_.resize(n());
    std::generate(input_.begin(), input_.end(), std::ref(i32rng));
    output_.resize(n());
    std::fill(output_.begin(), output_.end(), 0xA5);

    const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
    if (cpu_frequency != 0) {
      state.counters["cpufreq"] = cpu_frequency;
    }
  }
  virtual void TearDown(benchmark::State& state) override
  {
    state.SetItemsProcessed(uint64_t(state.iterations()) * n());
    state.SetBytesProcessed(uint64_t(state.iterations()) * n() * (sizeof(int32_t) + sizeof(int8_t)));
    input_.clear();
    output_.clear();
  }

  inline const int32_t* input() const
  {
    return input_.data();
  }

  inline int8_t* output()
  {
    return output_.data();
  }

  inline size_t n() const
  {
    return n_;
  }

 protected:
  std::vector<int32_t, AlignedAllocator<int32_t, 64>> input_;
  std::vector<int8_t> output_;
  size_t n_;
};


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_F(Requantization, fp32__neon)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_fp32__neon(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, gemmlowp__neon)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_gemmlowp__neon(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, rndna__neon)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_rndna__neon(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, rndnu__neon_mull)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_rndnu__neon_mull(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, rndnu__neon_qdmulh)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_rndnu__neon_qdmulh(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_F(Requantization, fp32__sse2)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_fp32__sse2(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, fp32__sse4)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_fp32__sse4(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, gemmlowp__sse2)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_gemmlowp__sse2(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, gemmlowp__ssse3)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_gemmlowp__ssse3(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, gemmlowp__sse4)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_gemmlowp__sse4(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, rndna__sse2)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_rndna__sse2(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, rndna__ssse3)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_rndna__ssse3(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, rndna__sse4)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_rndna__sse4(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, rndnu__sse4_sra)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_rndnu__sse4_sra(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, rndnu__sse4_srl)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_rndnu__sse4_srl(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_F(Requantization, fp32__wasmsimd)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_fp32__wasmsimd(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, gemmlowp__wasmsimd)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_qs8_requantize_gemmlowp__wasmsimd(
          n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


BENCHMARK_F(Requantization, fp32__scalar_lrintf)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_qs8_requantize_fp32__scalar_lrintf(
        n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, fp32__scalar_fmagic)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_qs8_requantize_fp32__scalar_fmagic(
        n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, gemmlowp__scalar)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_qs8_requantize_gemmlowp__scalar(
        n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, rndna__scalar_signed64)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_qs8_requantize_rndna__scalar_signed64(
        n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, rndna__scalar_unsigned32)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_qs8_requantize_rndna__scalar_unsigned32(
        n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, rndna__scalar_unsigned64)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_qs8_requantize_rndna__scalar_unsigned64(
        n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, rndnu__scalar)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_qs8_requantize_rndnu__scalar(
        n(), input(), 0x1.0p-12f /* scale */, -1 /* zero point */, -127 /* qmin */, 126 /* qmax */, output());
  }
}


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
