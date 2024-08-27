// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-vhswish
//   Generator: tools/generate-vunary-benchmark.py

#include <stddef.h>
#include <stdint.h>

#include <benchmark/benchmark.h>
#include "bench/f32-vunary-benchmark.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"

void f32_vhswish(benchmark::State& state, uint64_t arch_flags, xnn_f32_vhswish_ukernel_fn ukernel,
              xnn_init_f32_hswish_params_fn init_params = nullptr) {
  f32_vunary_benchmark<xnn_f32_hswish_params>(
      state, ukernel,
      init_params,
      arch_flags,
      /*range_min=*/-10.0,
      /*range_max=*/10.0);
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,              \
                                datatype, params_type, init_params)                        \
BENCHMARK_CAPTURE(f32_vhswish, ukernel, arch_flags, ukernel, init_params)                  \
  ->Apply(benchmark::utils::UnaryElementwiseParameters<datatype, datatype>)                \
  ->UseRealTime();
#include "src/f32-vhswish/f32-vhswish.h"
#undef XNN_UKERNEL_WITH_PARAMS


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
