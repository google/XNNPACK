// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <benchmark/benchmark.h>
#include "bgemm.h"
#include "packw-benchmark.h"
#include "utils.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/packw.h"

static void qb4_packw_wrapper(benchmark::State& state,
                              xnn_qb4_packw_gemm_goi_ukernel_fn ukernel,
                              uint64_t arch_flags, size_t nr, size_t kr,
                              size_t sr, size_t bl) {
  qb4_packw(state, ukernel, nr, kr, sr, bl, /*null_bias=*/false, arch_flags);
}

#define XNN_QB4_UKERNEL(arch_flags, ukernel, nr, kr, sr, bl, kblock, nr_scale, \
                        izp)                                                   \
  BENCHMARK_CAPTURE_BGEMM(qb4_packw_wrapper, ukernel##_, ukernel, arch_flags,  \
                          nr, kr, sr, bl);

#include "src/qb4-packw/qb4-packw.inc"

#undef XNN_QB4_UKERNEL


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
