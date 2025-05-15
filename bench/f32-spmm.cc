// clang-format off
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-spmm-minmax
//   Generator: tools/generate-spmm-test.py

#include <benchmark/benchmark.h>

#include "spmm-benchmark.h"
#include "utils.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, mr, nr, pipelined, kblock, datatype, params_type, init_params)\
static void ukernel(benchmark::State& state, const char* net) {                                                    \
  f32_spmm(state, arch_flags, ukernel, mr, nr,                                                                                 \
    /*sparsity=*/0.8f, init_params);                                                                               \
}                                                                                                                  \
                                                                                                                   \
BENCHMARK_SPMM(ukernel)
#include "src/f32-spmm/f32-spmm-minmax.h"
#undef XNN_UKERNEL_WITH_PARAMS


#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
