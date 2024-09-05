// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f16-velu
//   Generator: tools/generate-vunary-benchmark.py

#include <stddef.h>
#include <stdint.h>

#include <benchmark/benchmark.h>
#include "bench/f16-vunary-benchmark.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"

void f16_velu(benchmark::State& state, uint64_t arch_flags, xnn_f16_velu_ukernel_fn ukernel,
              xnn_init_f16_elu_params_fn init_params = nullptr) {
  f16_vunary_benchmark<xnn_f16_elu_params>(
      state, ukernel,
      [init_params](xnn_f16_elu_params* params) -> size_t {
        init_params(params,
                    /*prescale=*/1.0f,
                    /*alpha=*/1.0f,
                    /*beta=*/1.0f);
        return sizeof(*params);
      },
      arch_flags,
      /*range_min=*/-9.0,
      /*range_max=*/9.0);
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,              \
                                datatype, params_type, init_params)                        \
BENCHMARK_CAPTURE(f16_velu, ukernel, arch_flags, ukernel, init_params)                     \
  ->Apply(benchmark::utils::UnaryElementwiseParameters<datatype, datatype>)                \
  ->UseRealTime();
#include "src/f16-velu/f16-velu.h"
#undef XNN_UKERNEL_WITH_PARAMS


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
