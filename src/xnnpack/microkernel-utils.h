// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_MICROKERNEL_UTILS_H_
#define XNNPACK_SRC_XNNPACK_MICROKERNEL_UTILS_H_

#include <stdbool.h>
#include <stddef.h>

#include "src/xnnpack/config-types.h"

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
                               size_t cn_stride, size_t mr, size_t nr,
                               size_t num_threads);

// Checks wheter it is worthwhile to inline the lhs packing for a GEMM with the
// given parameters.
bool xnn_should_inline_lhs_packing(const struct xnn_gemm_config* gemm_config,
                                   size_t m_packed_stride, size_t n_stride,
                                   size_t cn_stride, size_t mc, size_t nc);

// Checks whether to use the `nr2` config or not.
bool xnn_use_nr2(size_t nr, size_t nr2, size_t output_channels);

#ifdef __cplusplus
}
#endif

#endif  // XNNPACK_SRC_XNNPACK_MICROKERNEL_UTILS_H_
