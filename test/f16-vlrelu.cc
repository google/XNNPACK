// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f16-vlrelu
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

using TestInfo = LeakyReLU;

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)       \
  TEST(ukernel, batch_eq) { TestBatchEq<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, batch_div) { TestBatchDiv<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }\
  TEST(ukernel, batch_lt) { TestBatchLT<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, batch_gt) { TestBatchGT<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, inplace) { TestInPlace<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }   \
TEST(ukernel, negative_slope) {                                                                                         \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                                 \
  const size_t batch_scale = get_batch_scale<datatype>();                                                               \
  const size_t batch_size = batch_tile * batch_scale;                                                                   \
  for (float negative_slope : {0.01f, 0.3f, 1.3f}) {                                                                    \
    xnn_unary_params params;                                                                                            \
    params.leaky_relu.negative_slope = negative_slope;                                                                  \
    VUnaryMicrokernelTester()                                                                                           \
      .batch_size(batch_size)                                                                                           \
      .Test<TestInfo>(ukernel, init_params, params);                                                                    \
  }                                                                                                                     \
}
#include "f16-vlrelu/f16-vlrelu.h"
#undef XNN_UKERNEL_WITH_PARAMS
