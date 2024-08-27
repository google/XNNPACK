// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Microkernel: qu8-vlrelu
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
#include "xnnpack/vlrelu.h"
#include "next_prime.h"
#include "vlrelu-microkernel-tester.h"

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params)                    \
                                                                                                                                     \
XNN_TEST_UNARY_BATCH_EQ(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                                            \
XNN_TEST_UNARY_BATCH_DIV(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                                           \
XNN_TEST_UNARY_BATCH_LT(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                                            \
XNN_TEST_UNARY_BATCH_GT(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                                            \
                                                                                                                                     \
XNN_TEST_UNARY_INPLACE(ukernel, arch_flags, batch_tile, datatype, ukernel, init_params);                                             \
TEST(ukernel, positive_scale) {                                                                                                      \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                                              \
  for (size_t batch_size = 1; batch_size <= batch_tile * 5; batch_size += std::max(1, batch_tile - 1)) {                             \
    for (float positive_scale : std::vector<float>({1.0f / 256.0f, 0.3f, 1.3f, 128.0f})) {                                           \
      VLReLUMicrokernelTester()                                                                                                      \
        .batch_size(batch_size)                                                                                                      \
        .positive_scale(positive_scale)                                                                                              \
        .Test(ukernel, init_params);                                                                                                 \
      }                                                                                                                              \
  }                                                                                                                                  \
}                                                                                                                                    \
                                                                                                                                     \
TEST(ukernel, negative_scale) {                                                                                                      \
  TEST_REQUIRES_ARCH_FLAGS(arch_flags);                                                                                              \
  for (size_t batch_size = 1; batch_size <= batch_tile * 5; batch_size += std::max(1, batch_tile - 1)) {                             \
    for (float negative_scale : std::vector<float>({-127.99609375f, -1.3f, -0.3f, -1.0f / 256.0f, 1 / 256.0f, 0.3f, 1.3f, 128.0f})) {\
      VLReLUMicrokernelTester()                                                                                                      \
        .batch_size(batch_size)                                                                                                      \
        .negative_scale(negative_scale)                                                                                              \
        .Test(ukernel, init_params);                                                                                                 \
      }                                                                                                                              \
  }                                                                                                                                  \
}
#include "src/qu8-vlrelu/qu8-vlrelu.h"
#undef XNN_UKERNEL_WITH_PARAMS
