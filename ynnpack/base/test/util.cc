// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/test/util.h"

#include <cstddef>
#include <set>
#include <vector>

namespace ynn {

std::vector<size_t> simd_sizes_up_to(size_t max_size, size_t alignment) {
  std::set<size_t> widths;
  widths.insert(alignment);
  // First N powers of 2, include the first 4 multiples of that power, +/-1.
  for (size_t i = alignment;; i *= 2) {
    for (size_t j = 1; j < 4; ++j) {
      if (i * j > alignment) {
        widths.insert(i * j - alignment);
      }
      widths.insert(i * j);
      widths.insert(i * j + alignment);
    }
    if (*widths.rbegin() >= max_size) break;
  }
  widths.erase(widths.upper_bound(max_size), widths.end());
  return std::vector<size_t>(widths.begin(), widths.end());
}

}  // namespace ynn
