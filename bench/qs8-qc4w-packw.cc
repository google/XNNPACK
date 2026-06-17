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

#define XNN_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale, izp)     \
  BENCHMARK_CAPTURE_BGEMM(qs8_qc4w_packw, ukernel##_, ukernel, nr, kr, sr, \
                          arch_flags);

#define XNN_QC4UW_UKERNEL XNN_UKERNEL
#include "src/qs8-qc4w-packw/qs8-qc4w-packw.inc"

#undef XNN_UKERNEL
#undef XNN_QC4UW_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
