// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>

#include <benchmark/benchmark.h>

namespace benchmark {
namespace utils {

uint32_t WipeCache();
uint32_t PrefetchToL1(const void* ptr, size_t size);

// Disable support for denormalized numbers in floating-point units.
void DisableDenormals();

// Return clock rate, in Hz, for the currently used logical processor.
uint64_t GetCurrentCpuFrequency();

// Return maximum (across all cores/clusters/sockets) last level cache size.
// Can overestimate, but not underestimate LLC size.
size_t GetMaxCacheSize();

// Set multi-threading parameters appropriate for the processor.
void MultiThreadingParameters(benchmark::internal::Benchmark* benchmark);

typedef bool (*IsaCheckFunction)(benchmark::State& state);

// Check if ARM NEON extension is supported.
// If NEON is unsupported, report error in benchmark state, and return false.
bool CheckNEON(benchmark::State& state);

// Check if ARM NEON-FMA extension is supported.
// If NEON-FMA is unsupported, report error in benchmark state, and return false.
bool CheckNEONFMA(benchmark::State& state);

// Check if x86 SSE4.1 extension is supported.
// If SSE4.1 is unsupported, report error in benchmark state, and return false.
bool CheckSSE41(benchmark::State& state);

// Check if x86 AVX extension is supported.
// If AVX is unsupported, report error in benchmark state, and return false.
bool CheckAVX(benchmark::State& state);

// Check if x86 FMA3 extension is supported.
// If FMA3 is unsupported, report error in benchmark state, and return false.
bool CheckFMA3(benchmark::State& state);

// Check if x86 AVX2 extension is supported.
// If AVX2 is unsupported, report error in benchmark state, and return false.
bool CheckAVX2(benchmark::State& state);

// Check if x86 AVX512F extension is supported.
// If AVX512F is unsupported, report error in benchmark state, and return false.
bool CheckAVX512F(benchmark::State& state);

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
