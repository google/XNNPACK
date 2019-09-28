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
uint64_t GetCurrentCpuFrequency(void);  // Return clockrate of current cpu

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
