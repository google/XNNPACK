// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>
#include "utils.h"
#include "vcvt-benchmark.h"
#include "xnnpack.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vcvt.h"

static void qs16_qs8_vcvt(
  benchmark::State& state,
  uint64_t arch_flags,
  xnn_qs16_qs8_vcvt_ukernel_fn cvt,
  xnn_init_qs16_qs8_cvt_params_fn init_params)
{
  xnn_qs16_qs8_cvt_params params;
  init_params(&params, 1.25f /* scale */, 1 /* output zero point */);

  cvt_benchmark<int16_t, int8_t>(state, arch_flags, cvt, &params);
}

#define XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,   \
                                datatype_in, datatype_out, params_type, init_params)\
BENCHMARK_CAPTURE(qs16_qs8_vcvt, ukernel, arch_flags, ukernel, init_params)         \
  ->Apply(benchmark::utils::UnaryElementwiseParameters<datatype_in, datatype_out>)  \
  ->UseRealTime();
#include "qs16-qs8-vcvt/qs16-qs8-vcvt.h"
#undef XNN_CVT_UKERNEL_WITH_PARAMS


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
