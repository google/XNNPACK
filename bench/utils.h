// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_BENCH_UTILS_H_
#define XNNPACK_BENCH_UTILS_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

#include "src/xnnpack/common.h"
#include <benchmark/benchmark.h>
#include <pthreadpool.h>

#ifdef BENCHMARK_ARGS_BOTTLENECK
#define XNN_BENCHMARK_MAIN()                            \
  extern "C" {                                          \
  int BenchmarkArgBottleneck(int& argc, char**& argv) { \
    return benchmark::utils::ProcessArgs(argc, argv);   \
  }                                                     \
  }
#else
#define XNN_BENCHMARK_MAIN()                                            \
  int main(int argc, char** argv) {                                     \
    ::benchmark::Initialize(&argc, argv);                               \
    int status = benchmark::utils::ProcessArgs(argc, argv);             \
    if (status != 0) return status;                                     \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; \
    ::benchmark::RunSpecifiedBenchmarks();                              \
  }                                                                     \
  int main(int, char**)
#endif  // BENCHMARK_ARGS_BOTTLENECK

// Common flags for all benchmarks.
extern int FLAGS_num_threads;
extern int FLAGS_batch_size;
extern uint32_t FLAGS_xnn_runtime_flags;
extern uint32_t FLAGS_benchmark_min_iters;

namespace benchmark {
namespace utils {

int ProcessArgs(int& argc, char**& argv);

uint32_t WipeCache();
uint32_t PrefetchToL1(const void* ptr, size_t size);

// Clear the L2 cache in each thread of the given `threadpool`, calls
// `state.PauseTiming()` while doing so.
void WipePthreadpoolL2Caches(benchmark::State& state, pthreadpool_t threadpool);

// Disable support for denormalized numbers in floating-point units.
void DisableDenormals();

// Return clock rate, in Hz, for the currently used logical processor.
uint64_t GetCurrentCpuFrequency();

// Return maximum (across all cores/clusters/sockets) last level cache size.
// Can overestimate, but not underestimate LLC size.
size_t GetMaxCacheSize();

template <class InType>
static void ReduceParameters(benchmark::internal::Benchmark* b) {
  b->ArgNames({"channels", "rows"});
  b->Args({1, 512});
  b->Args({1, 1024});
  b->Args({1, 8000});
  b->Args({512, 512});
  b->Args({512, 1024});
  b->Args({512, 8000});
  b->Args({1024, 64});
  b->Args({32768, 1});
}

template <class InType>
static void ReduceDiscontiguousParameters(benchmark::internal::Benchmark* b) {
  b->ArgNames({"rows", "channels"});
  b->Args({8, 1024});
  b->Args({16, 1024});
  b->Args({1024, 1024});
  b->Args({32768, 5});
}

// Set number of elements for a unary elementwise microkernel such that:
// - It is divisible by 2, 3, 4, 5, 6.
// - It is divisible by AVX512 width.
// - Total memory footprint does not exceed the characteristic cache size for
//   the architecture.
template <class InType, class OutType>
void UnaryElementwiseParameters(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgName("N");

  size_t characteristic_l1 = 32 * 1024;
  size_t characteristic_l2 = 256 * 1024;
#if XNN_ARCH_ARM
  characteristic_l1 = 16 * 1024;
  characteristic_l2 = 128 * 1024;
#endif  // XNN_ARCH_ARM

  const size_t elementwise_size = sizeof(InType) + sizeof(OutType);
  benchmark->Arg(characteristic_l1 / elementwise_size / 960 * 960);
  benchmark->Arg(characteristic_l2 / elementwise_size / 960 * 960);
}

// Set number of elements for a binary elementwise microkernel such that:
// - It is divisible by 2, 3, 4, 5, 6.
// - It is divisible by AVX512 width.
// - Total memory footprint does not exceed the characteristic cache size for
//   the architecture.
template <class InType, class OutType>
void BinaryElementwiseParameters(benchmark::internal::Benchmark* benchmark) {
  benchmark->ArgName("N");

  size_t characteristic_l1 = 32 * 1024;
  size_t characteristic_l2 = 256 * 1024;
#if XNN_ARCH_ARM
  characteristic_l1 = 16 * 1024;
  characteristic_l2 = 128 * 1024;
#endif  // XNN_ARCH_ARM

  const size_t elementwise_size = 2 * sizeof(InType) + sizeof(OutType);
  benchmark->Arg(
      std::max<size_t>(1, characteristic_l1 / elementwise_size / 960) * 960);
  benchmark->Arg(
      std::max<size_t>(1, characteristic_l2 / elementwise_size / 960) * 960);
}

// Check if the architecture flags are supported.
// If unsupported, report error in benchmark state, and return false.
bool CheckArchFlags(benchmark::State& state, uint64_t arch_flags);

template <class T>
inline T DivideRoundUp(T x, T q) {
  return x / q + T(x % q != 0);
}

template <class T>
inline T RoundUp(T x, T q) {
  return q * DivideRoundUp(x, q);
}

template <class T>
inline T Doz(T a, T b) {
  return a >= b ? a - b : T(0);
}

}  // namespace utils
}  // namespace benchmark

#endif  // XNNPACK_BENCH_UTILS_H_
