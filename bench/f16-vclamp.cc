// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f16-vclamp
//   Generator: tools/generate-vunary-benchmark.py

#include <stddef.h>
#include <stdint.h>

#include <benchmark/benchmark.h>
#include "bench/f16-vunary-benchmark.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"

void f16_vclamp(benchmark::State& state, uint64_t arch_flags, xnn_f16_vclamp_ukernel_fn ukernel,
              xnn_init_f16_minmax_params_fn init_params = nullptr) {
  f16_vunary_benchmark<xnn_f16_minmax_params>(
      state, ukernel,
      [init_params](xnn_f16_minmax_params* params) -> size_t {
        init_params(params, -1.0f, 1.0f);
        return sizeof(*params);
      },
      arch_flags,
      /*range_min=*/-10.0,
      /*range_max=*/10.0);
}
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,              \
                                datatype, params_type, init_params)                        \
BENCHMARK_CAPTURE(f16_vclamp, ukernel, arch_flags, ukernel, init_params)                   \
  ->Apply(benchmark::utils::UnaryElementwiseParameters<datatype, datatype>)                \
  ->UseRealTime();
#include "src/f16-vclamp/f16-vclamp.h"
#undef XNN_UKERNEL_WITH_PARAMS


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
