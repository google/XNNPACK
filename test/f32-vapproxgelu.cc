// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: f32-vapproxgelu
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

using TestInfo = ApproxGELU;

#define XNN_QUANTIZED(T) xnnpack::quantized<T>
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)       \
  TEST(ukernel, batch_eq) { TestBatchEq<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, batch_div) { TestBatchDiv<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }\
  TEST(ukernel, batch_lt) { TestBatchLT<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, batch_gt) { TestBatchGT<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }  \
  TEST(ukernel, inplace) { TestInPlace<TestInfo, datatype, datatype>(arch_flags, batch_tile, ukernel, init_params); }   \
TEST(ukernel, special_values) {                                                                                         \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                                 \
  VUnaryMicrokernelTester().Test<TestInfo, datatype, datatype>(ukernel, init_params,                                    \
    /*inputs=*/{-6.0f, 6.0f, 0.0f},                                                                                     \
    /*outputs=*/{0.0f, 6.0f, 0.0f},                                                                                     \
    /*tolerance_ulp=*/1);                                                                                               \
}
#include "f32-vapproxgelu/f32-vapproxgelu.h"
#undef XNN_UKERNEL_WITH_PARAMS
#undef XNN_QUANTIZED
