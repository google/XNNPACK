// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef __XNNPACK_BENCH_PACKQ_BENCHMARK_H
#define __XNNPACK_BENCH_PACKQ_BENCHMARK_H

#include <cstddef>

#include "bench/utils.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packq.h"
#include <benchmark/benchmark.h>

void x8_packq(benchmark::State& state, xnn_x8_packq_f32qp8_ukernel_fn packq,
              size_t mr, size_t kr, size_t sr,
              benchmark::utils::IsaCheckFunction isa_check = nullptr);

#endif  // __XNNPACK_TEST_PACKQ_MICROKERNEL_TESTER_H
