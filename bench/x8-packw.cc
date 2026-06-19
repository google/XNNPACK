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

#define XNN_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale) \
  BENCHMARK_CAPTURE_BGEMM(x8_packw, ukernel##_, ukernel, nr, kr, sr,   \
                          arch_flags);

#define XNN_GIO_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale) \
  BENCHMARK_CAPTURE_BGEMM(x8_gio_packw, ukernel##_, ukernel, nr, kr, sr,   \
                          arch_flags);

#include "src/x8-packw/x8-packw.inc"
#undef XNN_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
