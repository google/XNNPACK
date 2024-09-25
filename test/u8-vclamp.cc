// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: u8-vclamp
//   Generator: tools/generate-vunary-test.py


#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"
#include "next_prime.h"
#include "vunary-microkernel-tester.h"

using TestInfo = Clamp;

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)       \
  TEST(ukernel, batch_eq) { TestBatchEq<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, batch_div) { TestBatchDiv<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }\
  TEST(ukernel, batch_lt) { TestBatchLT<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, batch_gt) { TestBatchGT<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, inplace) { TestInPlace<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }   \
TEST(ukernel, clamp_min) {                                                                                              \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                                 \
  const size_t batch_scale = get_batch_scale<datatype>();                                                               \
  const size_t batch_end = batch_tile * batch_scale;                                                                    \
  const size_t batch_step =                                                                                             \
      batch_scale == 1 ? std::max(1, batch_tile - 1) : batch_end - 1;                                                   \
  for (size_t min = 1; min < 255; min = xnnpack::NextPrime(min)) {                                                      \
    for (size_t batch_size = 1; batch_size <= 5 * batch_end;                                                            \
         batch_size += batch_step) {                                                                                    \
      xnn_unary_params params;                                                                                          \
      params.clamp.min = min;                                                                                           \
      params.clamp.max = 255;                                                                                           \
      VUnaryMicrokernelTester()                                                                                         \
          .batch_size(batch_size)                                                                                       \
          .Test<TestInfo>(ukernel, init_params, params);                                                                \
    }                                                                                                                   \
  }                                                                                                                     \
}                                                                                                                       \
                                                                                                                        \
TEST(ukernel, clamp_max) {                                                                                              \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                                 \
  const size_t batch_scale = get_batch_scale<datatype>();                                                               \
  const size_t batch_end = batch_tile * batch_scale;                                                                    \
  const size_t batch_step =                                                                                             \
      batch_scale == 1 ? std::max(1, batch_tile - 1) : batch_end - 1;                                                   \
  for (size_t max = 1; max < 255; max = xnnpack::NextPrime(max)) {                                                      \
    for (size_t batch_size = 1; batch_size <= 5 * batch_end;                                                            \
         batch_size += batch_step) {                                                                                    \
      xnn_unary_params params;                                                                                          \
      params.clamp.min = 0;                                                                                             \
      params.clamp.max = max;                                                                                           \
      VUnaryMicrokernelTester()                                                                                         \
          .batch_size(batch_size)                                                                                       \
          .Test<TestInfo>(ukernel, init_params, params);                                                                \
    }                                                                                                                   \
  }                                                                                                                     \
}
#include "u8-vclamp/u8-vclamp.h"
#undef XNN_UKERNEL_WITH_PARAMS
