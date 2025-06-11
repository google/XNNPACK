// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "bench/bgemm.h"
#include "bench/packw-benchmark.h"
#include "bench/utils.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/packw.h"
#include <benchmark/benchmark.h>

static void qs8_qc4w_packw(benchmark::State& state, const char* net,
                           xnn_qs8_qc4w_packw_gemm_goi_ukernel_fn ukernel,
                           uint64_t arch_flags, size_t nr, size_t kr,
                           size_t sr) {
  benchmark::utils::CheckArchFlags(state, arch_flags);
  qs8_qc4w_packw(state, ukernel, nr, kr, sr);
}

#define XNN_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale)         \
  BENCHMARK_CAPTURE_BGEMM(qs8_qc4w_packw, ukernel##_, ukernel, arch_flags, nr, \
                          kr, sr);

#include "src/qs8-qc4w-packw/qs8-qc4w-packw.inc"

#undef XNN_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
