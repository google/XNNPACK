// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "utils.h"
#include "vcvt-benchmark.h"
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vcvt.h"
#include <benchmark/benchmark.h>

static void qs8_f16_vcvt(
  benchmark::State& state,
  uint64_t arch_flags,
  xnn_qs8_f16_vcvt_ukernel_fn cvt,
  xnn_init_qs8_f16_cvt_params_fn init_params)
{
  xnn_qs8_f16_cvt_params params;
  init_params(&params,
    0.25f /* scale */,
    1 /* output zero point */);

  cvt_benchmark<int8_t, xnn_float16>(state, arch_flags, cvt, &params);
}

#define XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile,   \
                                datatype_in, datatype_out, params_type, init_params)\
BENCHMARK_CAPTURE(qs8_f16_vcvt, ukernel, arch_flags, ukernel, init_params)          \
  ->Apply(benchmark::utils::UnaryElementwiseParameters<datatype_in, datatype_out>)  \
  ->UseRealTime();
#include "qs8-f16-vcvt/qs8-f16-vcvt.h"
#undef XNN_CVT_UKERNEL_WITH_PARAMS


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
