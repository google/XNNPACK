// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SIMD_BENCH_GENERIC_H_
#define XNNPACK_YNNPACK_BASE_SIMD_BENCH_GENERIC_H_

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/vec.h"
#include <benchmark/benchmark.h>

namespace ynn {

namespace simd {

using u8 = uint8_t;
using s8 = int8_t;
using s16 = int16_t;
using f16 = half;
using bf16 = bfloat16;
using f32 = float;
using s32 = int32_t;

template <typename T, typename Init>
YNN_NO_INLINE static auto load_no_inline(const T* src, size_t n, Init init) {
  return load(src, n, init);
}

template <typename T, size_t N>
YNN_NO_INLINE static void store_no_inline(T* dst, vec<T, N> v, size_t n) {
  return store(dst, v, n);
}

template <typename scalar, size_t N, typename Init>
static void BM_partial_load(benchmark::State& state) {
  const size_t n = state.range(0);
  const size_t align = state.range(1);
  using vector = vec<scalar, N>;

  alignas(vector) scalar src_aligned[N * 2];
  for (size_t i = 0; i < N * 2; ++i) {
    src_aligned[i] = static_cast<scalar>(i);
  }
  scalar* src = &src_aligned[N + align - n];

  Init init = {};
  benchmark::DoNotOptimize(src);
  for (auto _ : state) {
    benchmark::DoNotOptimize(load_no_inline(src, n, init));
  }
}

template <typename scalar, size_t N>
static void BM_partial_store(benchmark::State& state) {
  const size_t n = state.range(0);
  const size_t align = state.range(1);
  using vector = vec<scalar, N>;

  alignas(vector) scalar dst_aligned[N * 2];
  scalar* dst = &dst_aligned[align];
  vector v = broadcast<N>(scalar{1});
  benchmark::DoNotOptimize(v);
  for (auto _ : state) {
    store_no_inline(dst, v, n);
    benchmark::DoNotOptimize(dst);
  }
}

template <int N>
static void partial_load_store_params(benchmark::Benchmark* b) {
  b->ArgNames({"n", "offset"});
  for (int offset : {0, 1}) {
    for (int n : {1, N - 1}) {
      b->Args({n, offset});
    }
  }
}

#define BENCH_PARTIAL_LOAD_STORE(arch, type, N)                               \
  void BM_partial_load_##type##x##N##_##arch(benchmark::State& state) {       \
    BM_partial_load<type, N, vec<type, N>>(state);                            \
  }                                                                           \
  void BM_partial_load_zero_##type##x##N##_##arch(benchmark::State& state) {  \
    BM_partial_load<type, N, zeros<N>>(state);                                \
  }                                                                           \
  void BM_partial_load_undef_##type##x##N##_##arch(benchmark::State& state) { \
    BM_partial_load<type, N, undef<N>>(state);                                \
  }                                                                           \
  void BM_partial_store_##type##x##N##_##arch(benchmark::State& state) {      \
    BM_partial_store<type, N>(state);                                         \
  }                                                                           \
  BENCHMARK(BM_partial_load_##type##x##N##_##arch)                            \
      ->Apply(partial_load_store_params<N>);                                  \
  BENCHMARK(BM_partial_load_zero_##type##x##N##_##arch)                       \
      ->Apply(partial_load_store_params<N>);                                  \
  BENCHMARK(BM_partial_load_undef_##type##x##N##_##arch)                      \
      ->Apply(partial_load_store_params<N>);                                  \
  BENCHMARK(BM_partial_store_##type##x##N##_##arch)                           \
      ->Apply(partial_load_store_params<N>);

template <typename T>
YNN_NO_INLINE static T fma_no_inline(T a, T b, T acc) {
  return fma(a, b, acc);
}

template <typename scalar, size_t N>
static void BM_fma(benchmark::State& state) {
  using vector = vec<scalar, N>;

  vector a{1};
  vector b{2};
  vector acc{3};

  benchmark::DoNotOptimize(a);
  benchmark::DoNotOptimize(b);
  benchmark::DoNotOptimize(acc);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fma_no_inline(a, b, acc));
  }
}

#define BENCH_FMA(arch, type, N)                               \
  void BM_fma_##type##x##N##_##arch(benchmark::State& state) { \
    BM_fma<type, N>(state);                                    \
  }                                                            \
  BENCHMARK(BM_fma_##type##x##N##_##arch);

}  // namespace simd

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SIMD_BENCH_GENERIC_H_
