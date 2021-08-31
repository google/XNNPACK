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
#include <xnnpack/math-stubs.h>


class Rounding : public benchmark::Fixture {
 public:
  inline Rounding()
  {
    cpuinfo_initialize();
    const size_t l1d_size = cpuinfo_get_l1d_cache(0)->size;
    const size_t l1d_reserve = 1024;
    n_ = (l1d_size - l1d_reserve) / (2 * sizeof(float));
    n_ = n_ / 16 * 16;
  }

  virtual void SetUp(const benchmark::State&) override
  {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

    input_.resize(n());
    std::generate(input_.begin(), input_.end(), std::ref(f32rng));
    output_.resize(n());
    std::fill(output_.begin(), output_.end(), 0xA5);
  }

  virtual void TearDown(benchmark::State& state) override
  {
    state.SetItemsProcessed(uint64_t(state.iterations()) * n());
    state.SetBytesProcessed(uint64_t(state.iterations()) * n() * 2 * sizeof(float));
    input_.clear();
    output_.clear();
  }

  inline const float* input() const
  {
    return input_.data();
  }

  inline float* output()
  {
    return output_.data();
  }

  inline size_t n() const
  {
    return n_;
  }

 protected:
  std::vector<float, AlignedAllocator<float, 64>> input_;
  std::vector<float, AlignedAllocator<float, 64>> output_;
  size_t n_;
};

class RoundingToNearestEven : public Rounding { };
class RoundingDown : public Rounding { };
class RoundingUp : public Rounding { };
class RoundingTowardsZero : public Rounding { };

BENCHMARK_F(RoundingToNearestEven, scalar_addsub)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundne__scalar_addsub(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingToNearestEven, scalar_nearbyint)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundne__scalar_nearbyint(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingToNearestEven, scalar_rint)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundne__scalar_rint(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingDown, scalar_addsub)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundd__scalar_addsub(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingDown, scalar_cvt)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundd__scalar_cvt(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingDown, scalar_floor)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundd__scalar_floor(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingUp, scalar_addsub)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundu__scalar_addsub(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingUp, scalar_cvt)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundu__scalar_cvt(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingUp, scalar_ceil)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundu__scalar_ceil(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingTowardsZero, scalar_addsub)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundz__scalar_addsub(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingTowardsZero, scalar_cvt)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundz__scalar_cvt(
        n() * sizeof(float), input(), output());
  }
}

BENCHMARK_F(RoundingTowardsZero, scalar_trunc)(benchmark::State& state) {
  for (auto _ : state) {
    xnn_math_f32_roundz__scalar_trunc(
        n() * sizeof(float), input(), output());
  }
}

#if XNN_ARCH_WASMSIMD
  BENCHMARK_F(RoundingToNearestEven, wasmsimd_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundne__wasmsimd_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingDown, wasmsimd_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundd__wasmsimd_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingDown, wasmsimd_cvt)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundd__wasmsimd_cvt(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingUp, wasmsimd_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundu__wasmsimd_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingUp, wasmsimd_cvt)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundu__wasmsimd_cvt(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingTowardsZero, wasmsimd_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundz__wasmsimd_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingTowardsZero, wasmsimd_cvt)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundz__wasmsimd_cvt(
          n() * sizeof(float), input(), output());
    }
  }
#endif  // XNN_ARCH_WASMSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_F(RoundingToNearestEven, neon_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundne__neon_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingToNearestEven, neonv8)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundne__neonv8(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingDown, neon_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundd__neon_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingDown, neon_cvt)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundd__neon_cvt(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingDown, neonv8)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundd__neonv8(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingUp, neon_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundu__neon_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingUp, neon_cvt)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundu__neon_cvt(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingUp, neonv8)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundu__neonv8(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingTowardsZero, neon_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundz__neon_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingTowardsZero, neon_cvt)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundz__neon_cvt(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingTowardsZero, neonv8)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundz__neonv8(
          n() * sizeof(float), input(), output());
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_F(RoundingToNearestEven, sse_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundne__sse_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingToNearestEven, sse2_cvt)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundne__sse2_cvt(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingToNearestEven, sse4)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundne__sse41(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingDown, sse_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundd__sse_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingDown, sse2_cvt)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundd__sse2_cvt(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingDown, sse4)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundd__sse41(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingUp, sse_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundu__sse_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingUp, sse2_cvt)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundu__sse2_cvt(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingUp, sse4)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundu__sse41(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingTowardsZero, sse_addsub)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundz__sse_addsub(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingTowardsZero, sse2_cvt)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundz__sse2_cvt(
          n() * sizeof(float), input(), output());
    }
  }

  BENCHMARK_F(RoundingTowardsZero, sse4)(benchmark::State& state) {
    for (auto _ : state) {
      xnn_math_f32_roundz__sse41(
          n() * sizeof(float), input(), output());
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
