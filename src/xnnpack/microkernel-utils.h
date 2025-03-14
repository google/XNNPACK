// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_SRC_XNNPACK_MICROKERNEL_UTILS_H_
#define THIRD_PARTY_XNNPACK_SRC_XNNPACK_MICROKERNEL_UTILS_H_

#include <stddef.h>

#include "src/xnnpack/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// When parallelizing GEMMs, try to tile the computation such that we have at
// least this many tiles per thread.
#define XNN_GEMM_MIN_TILES_PER_THREAD 5

// Compute the optimal tile size (integer multiple `nr`) for a GEMM such that
// the number of tiles is minimized, but such that the data needed for each tile
// fits in either the L1 or L2 cache.
size_t xnn_gemm_best_tile_size(size_t num_groups, size_t m, size_t n,
                               size_t m_stride, size_t n_stride,
                               size_t cm_stride, size_t cn_stride, size_t mr,
                               size_t nr, size_t num_threads);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_MICROKERNEL_UTILS_H_
