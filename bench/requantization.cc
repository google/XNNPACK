// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/requantization-stubs.h>


inline uint32_t divide_round_up(uint32_t x, uint32_t q) {
  return x / q + uint32_t(x % q != 0);
}

inline uint32_t round_up(uint32_t x, uint32_t q) {
  return q * divide_round_up(x, q);
}

inline uint32_t min(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

class Requantization : public benchmark::Fixture {
 public:
  inline Requantization()
  {
    cpuinfo_initialize();
    const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size;
    const size_t l1d_reserve = 1024;
    n_ = (l1d_size - l1d_reserve) / (sizeof(int32_t) + sizeof(uint8_t));
    n_ = n_ / 16 * 16;
  }

  virtual void SetUp(const benchmark::State&) override
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(), rng);

    input_.resize(n());
    std::generate(input_.begin(), input_.end(), std::ref(s32rng));
    output_.resize(n());
    std::fill(output_.begin(), output_.end(), 0xA5);
  }

  virtual void TearDown(benchmark::State& state) override
  {
    state.SetItemsProcessed(uint64_t(state.iterations()) * n());
    state.SetBytesProcessed(uint64_t(state.iterations()) * n() * (sizeof(int32_t) + sizeof(uint8_t)));
    input_.clear();
    output_.clear();
  }

  inline const int32_t* input() const
  {
    return input_.data();
  }

  inline uint8_t* output()
  {
    return output_.data();
  }

  inline size_t n() const
  {
    return n_;
  }

 protected:
  std::vector<int32_t, AlignedAllocator<int32_t, 32>> input_;
  std::vector<uint8_t> output_;
  size_t n_;
};

BENCHMARK_F(Requantization, precise__scalar_unsigned32)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_requantize_precise__scalar_unsigned32(
        n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, precise__scalar_unsigned64)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_requantize_precise__scalar_unsigned64(
        n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, precise__scalar_signed64)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_requantize_precise__scalar_signed64(
        n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, fp32__scalar_lrintf)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_requantize_fp32__scalar_lrintf(
        n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, fp32__scalar_magic)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_requantize_fp32__scalar_magic(
        n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
  }
}

BENCHMARK_F(Requantization, gemmlowp__scalar)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_requantize_gemmlowp__scalar(
        n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
  }
}

#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM
  BENCHMARK_F(Requantization, precise__psimd)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_precise__psimd(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, fp32__psimd)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_fp32__psimd(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_F(Requantization, precise__neon)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_precise__neon(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, fp32__neon)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_fp32__neon(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, q31__neon)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_q31__neon(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, gemmlowp__neon)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_gemmlowp__neon(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }
#endif

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_F(Requantization, precise__sse2)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_precise__sse2(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, precise__ssse3)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_precise__ssse3(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, precise__sse4)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_precise__sse4(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, fp32__sse2)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_fp32__sse2(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, q31__sse2)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_q31__sse2(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, q31__ssse3)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_q31__ssse3(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, q31__sse4)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_q31__sse4(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, gemmlowp__sse2)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_gemmlowp__sse2(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, gemmlowp__ssse3)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_gemmlowp__ssse3(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }

  BENCHMARK_F(Requantization, gemmlowp__sse4)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_requantize_gemmlowp__sse4(
          n(), input(), 0x1.0p-12f /* scale */, 128 /* zero point */, 1 /* qmin */, 254 /* qmax */, output());
    }
  }
#endif

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
