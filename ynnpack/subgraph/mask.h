// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_MASK_H_
#define XNNPACK_YNNPACK_SUBGRAPH_MASK_H_

#include <algorithm>
#include <cstdint>

#include "ynnpack/base/arithmetic.h"
#include "slinky/runtime/buffer.h"

namespace ynn {

inline bool is_zero_mask(const void* p, slinky::index_t elem_size) {
  const auto* bytes = static_cast<const uint8_t*>(p);
  for (slinky::index_t i = 0; i < elem_size; ++i) {
    if (bytes[i] != 0) return false;
  }
  return true;
}

// Partitions [0, m) into runs of active (any row unmasked) and inactive (all
// rows masked) groups of `granularity` rows, calling `active(begin, end)` or
// `inactive(begin, end)` for each run. Because rows are grouped by
// `granularity`, `active` runs may include masked rows.
template <typename Active, typename Inactive>
void for_each_mask_run(const void* mask, slinky::index_t mask_stride_m,
                       slinky::index_t mask_elem_size, slinky::index_t m,
                       slinky::index_t granularity, const Active& active,
                       const Inactive& inactive) {
  if (m <= 0) return;
  if (!mask) {
    active(0, m);
    return;
  }
  if (mask_stride_m == 0) {
    if (!is_zero_mask(mask, mask_elem_size)) {
      active(0, m);
    } else {
      inactive(0, m);
    }
    return;
  }
  auto is_group_active = [&](slinky::index_t begin) {
    const slinky::index_t end = std::min(begin + granularity, m);
    for (slinky::index_t i = begin; i < end; ++i) {
      if (!is_zero_mask(offset_bytes(mask, i * mask_stride_m),
                        mask_elem_size)) {
        return true;
      }
    }
    return false;
  };
  slinky::index_t begin = 0;
  bool run_active = is_group_active(0);
  for (slinky::index_t i = granularity; i < m; i += granularity) {
    const bool group_active = is_group_active(i);
    if (group_active != run_active) {
      if (run_active) {
        active(begin, i);
      } else {
        inactive(begin, i);
      }
      begin = i;
      run_active = group_active;
    }
  }
  if (run_active) {
    active(begin, m);
  } else {
    inactive(begin, m);
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_MASK_H_
