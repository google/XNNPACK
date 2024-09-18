// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: u64-u32-vsqrtshift
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
#include "src/xnnpack/vunary.h"
#include "next_prime.h"
#include "vunary-microkernel-tester.h"

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)\
                                                                                                                 \
XNN_TEST_UNARY_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                        \
XNN_TEST_UNARY_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                       \
XNN_TEST_UNARY_BATCH_LT(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                        \
XNN_TEST_UNARY_BATCH_GT(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                        \
                                                                                                                 \
TEST(ukernel, shift) {                                                                                           \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                          \
  const size_t batch_scale = get_batch_scale<datatype>();                                                        \
  const size_t batch_end = batch_tile * batch_scale;                                                             \
  const size_t batch_step = std::max(1, batch_tile - 1);                                                         \
  for (uint32_t shift = 0; shift < 32; shift++) {                                                                \
    for (size_t batch_size = 1; batch_size <= 5 * batch_end; batch_size += batch_step) {                         \
      VUnaryMicrokernelTester()                                                                                  \
        .batch_size(batch_size)                                                                                  \
        .shift(shift)                                                                                            \
        .Test(ukernel, init_params);                                                                             \
    }                                                                                                            \
  }                                                                                                              \
}
#include "src/u64-u32-vsqrtshift/u64-u32-vsqrtshift.h"
#undef XNN_UKERNEL_WITH_PARAMS
