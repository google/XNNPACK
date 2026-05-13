// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_ALGORITHM_H_
#define XNNPACK_YNNPACK_BASE_ALGORITHM_H_

#include <cstddef>

namespace ynn {

// Like std::any/std::all, but works for a sequence of integers.
template <typename Predicate>
bool any_n(size_t n, Predicate&& pred) {
  for (size_t i = 0; i < n; ++i) {
    if (pred(i)) return true;
  }
  return false;
}

template <typename Predicate>
bool all_n(size_t n, Predicate&& pred) {
  for (size_t i = 0; i < n; ++i) {
    if (!pred(i)) return false;
  }
  return true;
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_ALGORITHM_H_
