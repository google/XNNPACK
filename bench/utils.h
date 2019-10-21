// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>

namespace benchmark {
namespace utils {

uint32_t wipeCache();
uint32_t prefetchToL1(const void* ptr, size_t size);

// Disable support for denormalized numbers in floating-point units.
void DisableDenormals();

// Return clock rate, in Hz, for the currently used logical processor.
uint64_t GetCurrentCpuFrequency();

// Return maximum (across all cores/clusters/sockets) last level cache size.
// Can overestimate, but not underestimate LLC size.
size_t GetMaxCacheSize();

template <class T>
inline T divideRoundUp(T x, T q) {
  return x / q + T(x % q != 0);
}

template <class T>
inline T roundUp(T x, T q) {
  return q * divideRoundUp(x, q);
}

template <class T>
inline T doz(T a, T b) {
  return a >= b ? a - b : T(0);
}

}  // namespace utils
}  // namespace benchmark
