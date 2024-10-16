// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <benchmark/benchmark.h>

#include "bench/rsum-benchmark.h"
#include "bench/rw-benchmark.h"
#include "bench/utils.h"
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/reduce.h"

//Scalar rdsum
BENCHMARK_CAPTURE(f32_rdsum, scalar_c4,
                  xnn_f32_rdsum_ukernel_7p7x__scalar_c4,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRDSUM)
  ->UseRealTime();

//Scalar rsum 
BENCHMARK_CAPTURE(f32_rsum, scalar_u1,
                  xnn_f32_rsum_ukernel__scalar_u1,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rsum, scalar_u2_acc2,
                  xnn_f32_rsum_ukernel__scalar_u2_acc2,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rsum, scalar_u3_acc3,
                  xnn_f32_rsum_ukernel__scalar_u3_acc3,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rsum, scalar_u4_acc2,
                  xnn_f32_rsum_ukernel__scalar_u4_acc2,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rsum, scalar_u4_acc4,
                  xnn_f32_rsum_ukernel__scalar_u4_acc4,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();


//Scalar rwdsum 
BENCHMARK_CAPTURE(f32_rwdsum, scalar_c1,
                  xnn_f32_rwdsum_ukernel_1p1x__scalar_c1)
  ->Apply(BenchmarkRWDSUM)
  ->UseRealTime();

//Scalar rwsum 
BENCHMARK_CAPTURE(f32_rwsum, scalar_u1,
                  xnn_f32_rwsum_ukernel__scalar_u1)
  ->Apply(BenchmarkRWSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rwsum, scalar_u2_acc2,
                  xnn_f32_rwsum_ukernel__scalar_u2_acc2)
  ->Apply(BenchmarkRWSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rwsum, scalar_u3_acc3,
                  xnn_f32_rwsum_ukernel__scalar_u3_acc3)
  ->Apply(BenchmarkRWSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rwsum, scalar_u4_acc4,
                  xnn_f32_rwsum_ukernel__scalar_u4_acc4)
  ->Apply(BenchmarkRWSUM)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
