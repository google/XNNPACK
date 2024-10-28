// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-vclamp
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
  const size_t batch_size = batch_tile * batch_scale;                                                                   \
  for (int16_t min : {-128, -20, -1, 0, 1, 30, 127, 255}) {                                                             \
    xnn_unary_params params;                                                                                            \
    params.clamp.min = min;                                                                                             \
    params.clamp.max = 255;                                                                                             \
    VUnaryMicrokernelTester()                                                                                           \
        .batch_size(batch_size)                                                                                         \
        .Test<TestInfo>(ukernel, init_params, params);                                                                  \
  }                                                                                                                     \
}                                                                                                                       \
                                                                                                                        \
TEST(ukernel, clamp_max) {                                                                                              \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                                 \
  const size_t batch_scale = get_batch_scale<datatype>();                                                               \
  const size_t batch_size = batch_tile * batch_scale;                                                                   \
  for (int16_t max : {-127, -11, 0, 40, 127, 255}) {                                                                    \
    xnn_unary_params params;                                                                                            \
    params.clamp.min = -128;                                                                                            \
    params.clamp.max = max;                                                                                             \
    VUnaryMicrokernelTester()                                                                                           \
        .batch_size(batch_size)                                                                                         \
        .Test<TestInfo>(ukernel, init_params, params);                                                                  \
  }                                                                                                                     \
}
#include "f32-vclamp/f32-vclamp.h"
#undef XNN_UKERNEL_WITH_PARAMS
