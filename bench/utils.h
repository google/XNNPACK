// Copyright 2019-2025 Google LLC
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

#include "include/experimental.h"
#include "src/xnnpack/common.h"
#include <benchmark/benchmark.h>
#include <pthreadpool.h>

// This might be provided by google benchmark
#ifndef BENCHMARK_NAMED
#define BENCHMARK_NAMED(func, test_case_name)                          \
  BENCHMARK_PRIVATE_DECLARE(_benchmark_) =                             \
      (::benchmark::internal::RegisterBenchmarkInternal(               \
          std::make_unique< ::benchmark::internal::FunctionBenchmark>( \
              #func "/" #test_case_name, func)))
#endif  // BENCHMARK_NAMED

#if defined(BENCHMARK_ARGS_BOTTLENECK)
#define XNN_BENCHMARK_MAIN()                            \
  extern "C" {                                          \
  int BenchmarkArgBottleneck(int& argc, char**& argv) { \
    return benchmark::utils::ProcessArgs(argc, argv);   \
  }                                                     \
  }
#elif defined(__hexagon__)
#define XNN_BENCHMARK_MAIN()                                            \
  int __attribute__((weak)) main(int argc, char** argv) {               \
    ::benchmark::Initialize(&argc, argv);                               \
    int status = benchmark::utils::ProcessArgs(argc, argv);             \
    if (status != 0) return status;                                     \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; \
    ::benchmark::RunSpecifiedBenchmarks();                              \
  }
#else
#define XNN_BENCHMARK_MAIN()                                            \
  int main(int argc, char** argv) {                                     \
    ::benchmark::Initialize(&argc, argv);                               \
    int status = benchmark::utils::ProcessArgs(argc, argv);             \
    if (status != 0) return status;                                     \
    benchmark::utils::ApplyDeferredArgs();                              \
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; \
    ::benchmark::RunSpecifiedBenchmarks();                              \
  }
#endif  // BENCHMARK_ARGS_BOTTLENECK

// Common flags for all benchmarks.
extern int FLAGS_num_threads;
extern int FLAGS_batch_size;
extern uint32_t FLAGS_xnn_runtime_flags;
extern uint32_t FLAGS_benchmark_min_iters;
extern bool FLAGS_wipe_caches;

namespace benchmark {
namespace utils {

extern std::vector<int64_t> FLAGS_shapes;
extern int64_t FLAGS_gemm_block_size;

int ProcessArgs(int& argc, char**& argv);

typedef void (*ArgsFn)(benchmark::Benchmark*);
void DeferArgs(benchmark::Benchmark* b, ArgsFn fn);
void ApplyDeferredArgs();

uint32_t WipeCache();
uint32_t PrefetchToL1(const void* ptr, size_t size);

// Clear the L2 cache in each thread of the given `threadpool`, calls
// `state.PauseTiming()` while doing so.
void WipePthreadpoolL2Caches(benchmark::State& state, pthreadpool_t threadpool);
void WipeSchedulerL2Caches(benchmark::State& state, xnn_scheduler_v2 scheduler,
                           void* scheduler_context);

// Disable support for denormalized numbers in floating-point units.
void DisableDenormals();

// Return clock rate, in Hz, for the currently used logical processor.
uint64_t GetCurrentCpuFrequency();

// Return maximum (across all cores/clusters/sockets) last level cache size.
// Can overestimate, but not underestimate LLC size.
size_t GetMaxCacheSize();

template <class InType>
static void ReduceParameters(benchmark::Benchmark* b) {
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
static void ReduceDiscontiguousParameters(benchmark::Benchmark* b) {
  b->ArgNames({"rows", "channels"});
  b->Args({8, 1024});
  b->Args({16, 1024});
  b->Args({1024, 1024});
  b->Args({32768, 5});
}

// Set number of elements for a unary elementwise microkernel.
// Use a consistent number of elements to allow comparisons between different
// data types and architectures.
template <class InType, class OutType>
void UnaryElementwiseParameters(benchmark::Benchmark* benchmark) {
  benchmark->ArgName("N");
  benchmark->Arg(16 * 1024);
  benchmark->Arg(128 * 1024);
}

// Set number of elements for a binary elementwise microkernel.
// Use a consistent number of elements to allow comparisons between different
// data types and architectures.
template <class InType, class OutType>
void BinaryElementwiseParameters(benchmark::Benchmark* benchmark) {
  benchmark->ArgName("N");
  benchmark->Arg(8 * 1024);
  benchmark->Arg(64 * 1024);
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
