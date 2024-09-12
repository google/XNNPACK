// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>
#include "bench/utils.h"
#include "bench/vcvt-benchmark.h"
#include "xnnpack.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vcvt.h"

static void f32_qu8_vcvt(
  benchmark::State& state,
  uint64_t arch_flags,
  xnn_f32_qu8_vcvt_ukernel_fn cvt,
  xnn_init_f32_qu8_cvt_params_fn init_params)
{
  xnn_f32_qu8_cvt_params params;
  init_params(&params,
    25.0f /* scale */,
    127 /* output zero point */,
    std::numeric_limits<uint8_t>::min() + 1 /* output min */,
    std::numeric_limits<uint8_t>::max() - 1 /* output max */);

  cvt_benchmark<float, uint8_t>(state, arch_flags, cvt, &params);
}

#define XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,   \
                                datatype_in, datatype_out, params_type, init_params)\
BENCHMARK_CAPTURE(f32_qu8_vcvt, ukernel, arch_flags, ukernel, init_params)          \
  ->Apply(benchmark::utils::UnaryElementwiseParameters<datatype_in, datatype_out>)  \
  ->UseRealTime();
#include "src/f32-qu8-vcvt/f32-qu8-vcvt.h"
#undef XNN_CVT_UKERNEL_WITH_PARAMS


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
