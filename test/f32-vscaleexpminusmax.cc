// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
//   Specification: f32-vscaleexpminusmax


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/vscaleexpminusmax.h"
#include "vscaleexpminusmax-microkernel-tester.h"


#define XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_EQ(ukernel, arch_flags, element_tile, ...)                                  \
  TEST(ukernel, element_eq)                                                                                            \
  {                                                                                                                    \
    VScaleExpMinusMaxMicrokernelTester().elements(element_tile).Test(ukernel);                                         \
  }
#define XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_GT(ukernel, arch_flags, element_tile, ...)                                  \
  TEST(ukernel, element_gt)                                                                                            \
  {                                                                                                                    \
    for (size_t element_size = element_tile + 1; element_size < ((element_tile == 1) ? 10 : element_tile * 2);         \
         element_size++) {                                                                                             \
      VScaleExpMinusMaxMicrokernelTester().elements(element_size).Test(ukernel);                                       \
    }                                                                                                                  \
  }
#define XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_LT(ukernel, arch_flags, element_tile, ...)                                  \
  TEST(ukernel, element_lt)                                                                                            \
  {                                                                                                                    \
    for (size_t element_size = 1; element_size < element_tile; element_size++) {                                       \
      VScaleExpMinusMaxMicrokernelTester().elements(element_size).Test(ukernel);                                       \
    }                                                                                                                  \
  }
#define XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_DIV(ukernel, arch_flags, element_tile, ...)                                 \
  TEST(ukernel, element_div)                                                                                           \
  {                                                                                                                    \
    for (size_t element_size = 2 * element_tile; element_size < 10 * element_tile; element_size += element_tile) {     \
      VScaleExpMinusMaxMicrokernelTester().elements(element_size).Test(ukernel);                                       \
    }                                                                                                                  \
  }

#define XNN_TEST_VSCALEEXPMINUSMAX_SCALE(ukernel, arch_flags, element_tile, ...)                                       \
  TEST(ukernel, scale)                                                                                                 \
  {                                                                                                                    \
    for (size_t element_size = 1; element_size <= 5 * element_tile; element_size += max(1, element_tile - 1)) {        \
      VScaleExpMinusMaxMicrokernelTester().elements(element_size).scale(0.01f).Test(ukernel);                          \
      VScaleExpMinusMaxMicrokernelTester().elements(element_size).scale(100.0f).Test(ukernel);                         \
    }                                                                                                                  \
  }

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, element_tile, datatype, params_type, init_params) XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_EQ(ukernel,arch_flags, element_tile, init_params);  \
XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_DIV(ukernel,arch_flags, element_tile, init_params);                                                                                                       \
XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_LT(ukernel,arch_flags, element_tile, init_params);                                                                                                        \
XNN_TEST_VSCALEEXPMINUSMAX_ELEMENT_GT(ukernel,arch_flags, element_tile, init_params);                                                                                                        \
XNN_TEST_VSCALEEXPMINUSMAX_SCALE(ukernel,arch_flags, element_tile, init_params);
#include "f32-vscaleexpminusmax/f32-vscaleexpminusmax.h"
#undef XNN_UKERNEL_WITH_PARAMS
