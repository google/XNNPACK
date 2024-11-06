// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include <gtest/gtest.h>
#include "xnnpack/lut.h"
#include "lut-norm-microkernel-tester.h"


#define XNN_TEST_LUTNORM_N_EQ_1(ukernel, arch_flags, ...)                                                              \
  TEST(ukernel, n_eq_1)                                                                                                \
  {                                                                                                                    \
    LUTNormMicrokernelTester().n(1).Test(ukernel);                                                                     \
  }
#define XNN_TEST_LUTNORM_SMALL_N(ukernel, arch_flags, ...)                                                             \
  TEST(ukernel, small_n)                                                                                               \
  {                                                                                                                    \
    for (size_t i = 2; i <= 16; i++) {                                                                                 \
      LUTNormMicrokernelTester().n(i).Test(ukernel);                                                                   \
    }                                                                                                                  \
  }
#define XNN_TEST_LUTNORM_LARGE_N(ukernel, arch_flags, ...)                                                             \
  TEST(ukernel, large_n)                                                                                               \
  {                                                                                                                    \
    for (size_t i = 16; i <= 128; i += 2) {                                                                            \
      LUTNormMicrokernelTester().n(i).Test(ukernel);                                                                   \
    }                                                                                                                  \
  }
#define XNN_TEST_LUTNORM_N_EQ_1_INPLACE(ukernel, arch_flags, ...)                                                      \
  TEST(ukernel, n_eq_1_inplace)                                                                                        \
  {                                                                                                                    \
    LUTNormMicrokernelTester().n(1).inplace(true).Test(ukernel);                                                       \
  }
#define XNN_TEST_LUTNORM_SMALL_N_INPLACE(ukernel, arch_flags, ...)                                                     \
  TEST(ukernel, small_n_inplace)                                                                                       \
  {                                                                                                                    \
    for (size_t i = 2; i <= 16; i++) {                                                                                 \
      LUTNormMicrokernelTester().n(i).inplace(true).Test(ukernel);                                                     \
    }                                                                                                                  \
  }
#define XNN_TEST_LUTNORM_LARGE_N_INPLACE(ukernel, arch_flags, ...)                                                     \
  TEST(ukernel, large_n_inplace)                                                                                       \
  {                                                                                                                    \
    for (size_t i = 16; i <= 128; i += 2) {                                                                            \
      LUTNormMicrokernelTester().n(i).inplace(true).Test(ukernel);                                                     \
    }                                                                                                                  \
  }

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, datatype, params_type, init_params)                               \
  XNN_TEST_LUTNORM_N_EQ_1(ukernel, arch_flags, init_params);                                                           \
  XNN_TEST_LUTNORM_SMALL_N(ukernel, arch_flags, init_params);                                                          \
  XNN_TEST_LUTNORM_LARGE_N(ukernel, arch_flags, init_params);                                                          \
  XNN_TEST_LUTNORM_N_EQ_1_INPLACE(ukernel, arch_flags, init_params);                                                   \
  XNN_TEST_LUTNORM_SMALL_N_INPLACE(ukernel, arch_flags, init_params);                                                  \
  XNN_TEST_LUTNORM_LARGE_N_INPLACE(ukernel, arch_flags, init_params);
#include "u8-lut32norm/u8-lut32norm.h"
#undef XNN_UKERNEL_WITH_PARAMS
