// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/raddextexp.h"
#include "raddextexp-microkernel-tester.h"

#define XNN_TEST_RADDEXTEXP_ELEMENT_EQ(ukernel, arch_flags, element_tile, ...)                                         \
  TEST(ukernel, element_eq)                                                                                            \
  {                                                                                                                    \
    RAddExtExpMicrokernelTester().elements(element_tile).Test(ukernel);                                                \
  }
#define XNN_TEST_RADDEXTEXP_ELEMENT_DIV(ukernel, arch_flags, element_tile, ...)                                        \
  TEST(ukernel, element_gt)                                                                                            \
  {                                                                                                                    \
    for (size_t element_size = element_tile * 2; element_size < element_tile * 10; element_size += element_tile) {     \
      RAddExtExpMicrokernelTester().elements(element_size).Test(ukernel);                                              \
    }                                                                                                                  \
  }
#define XNN_TEST_RADDEXTEXP_ELEMENT_LT(ukernel, arch_flags, element_tile, ...)                                         \
  TEST(ukernel, element_lt)                                                                                            \
  {                                                                                                                    \
    for (size_t element_size = 1; element_size < element_tile; element_size++) {                                       \
      RAddExtExpMicrokernelTester().elements(element_size).Test(ukernel);                                              \
    }                                                                                                                  \
  }
#define XNN_TEST_RADDEXTEXP_ELEMENT_GT(ukernel, arch_flags, element_tile, ...)                                         \
  TEST(ukernel, element_div)                                                                                           \
  {                                                                                                                    \
    for (size_t element_size = element_tile + 1; element_size < (element_tile == 1 ? 10 : element_tile * 2);           \
         element_size++) {                                                                                             \
      RAddExtExpMicrokernelTester().elements(element_size).Test(ukernel);                                              \
    }                                                                                                                  \
  }

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params) XNN_TEST_RADDEXTEXP_ELEMENT_EQ(ukernel,arch_flags, element_tile, init_params);  \
XNN_TEST_RADDEXTEXP_ELEMENT_DIV(ukernel, arch_flags, element_tile, init_params);                                                                                                        \
XNN_TEST_RADDEXTEXP_ELEMENT_LT(ukernel, arch_flags, element_tile, init_params);                                                                                                         \
XNN_TEST_RADDEXTEXP_ELEMENT_GT(ukernel, arch_flags, element_tile, init_params);
#include "f32-raddextexp/f32-raddextexp.h"
#undef XNN_UKERNEL_WITH_PARAMS
