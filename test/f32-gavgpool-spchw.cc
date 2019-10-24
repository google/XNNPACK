// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gavgpool.h>
#include "gavgpool-spchw-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_GAVGPOOL_SPCHW__NEON_X4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolSpCHWMicrokernelTester()
      .elements(4)
      .channels(4)
      .Test(xnn_f32_gavgpool_spchw_ukernel__neon_x4);
  }

  TEST(F32_GAVGPOOL_SPCHW__NEON_X4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 8; elements < 32; elements += 4) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .Test(xnn_f32_gavgpool_spchw_ukernel__neon_x4);
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__NEON_X4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 4; elements++) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .Test(xnn_f32_gavgpool_spchw_ukernel__neon_x4);
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__NEON_X4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 5; elements < 8; elements++) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .Test(xnn_f32_gavgpool_spchw_ukernel__neon_x4);
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__NEON_X4, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t elements = 1; elements < 16; elements += 3) {
        GAvgPoolSpCHWMicrokernelTester()
          .elements(elements)
          .channels(channels)
          .Test(xnn_f32_gavgpool_spchw_ukernel__neon_x4);
      }
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__NEON_X4, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t elements = 1; elements < 16; elements += 3) {
        GAvgPoolSpCHWMicrokernelTester()
          .elements(elements)
          .channels(channels)
          .Test(xnn_f32_gavgpool_spchw_ukernel__neon_x4);
      }
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__NEON_X4, channels_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t channels = 8; channels <= 16; channels += 4) {
      for (size_t elements = 1; elements < 16; elements += 3) {
        GAvgPoolSpCHWMicrokernelTester()
          .elements(elements)
          .channels(channels)
          .Test(xnn_f32_gavgpool_spchw_ukernel__neon_x4);
      }
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__NEON_X4, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements += 3) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_spchw_ukernel__neon_x4);
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__NEON_X4, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements += 3) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_spchw_ukernel__neon_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_GAVGPOOL_SPCHW__SSE_X4, elements_eq_4) {
    TEST_REQUIRES_X86_SSE;
    GAvgPoolSpCHWMicrokernelTester()
      .elements(4)
      .channels(4)
      .Test(xnn_f32_gavgpool_spchw_ukernel__sse_x4);
  }

  TEST(F32_GAVGPOOL_SPCHW__SSE_X4, elements_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t elements = 8; elements < 32; elements += 4) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .Test(xnn_f32_gavgpool_spchw_ukernel__sse_x4);
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__SSE_X4, elements_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t elements = 1; elements < 4; elements++) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .Test(xnn_f32_gavgpool_spchw_ukernel__sse_x4);
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__SSE_X4, elements_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t elements = 5; elements < 8; elements++) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .Test(xnn_f32_gavgpool_spchw_ukernel__sse_x4);
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__SSE_X4, channels_lt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t elements = 1; elements < 16; elements += 3) {
        GAvgPoolSpCHWMicrokernelTester()
          .elements(elements)
          .channels(channels)
          .Test(xnn_f32_gavgpool_spchw_ukernel__sse_x4);
      }
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__SSE_X4, channels_gt_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t elements = 1; elements < 16; elements += 3) {
        GAvgPoolSpCHWMicrokernelTester()
          .elements(elements)
          .channels(channels)
          .Test(xnn_f32_gavgpool_spchw_ukernel__sse_x4);
      }
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__SSE_X4, channels_div_4) {
    TEST_REQUIRES_X86_SSE;
    for (size_t channels = 8; channels <= 16; channels += 4) {
      for (size_t elements = 1; elements < 16; elements += 3) {
        GAvgPoolSpCHWMicrokernelTester()
          .elements(elements)
          .channels(channels)
          .Test(xnn_f32_gavgpool_spchw_ukernel__sse_x4);
      }
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__SSE_X4, qmin) {
    TEST_REQUIRES_X86_SSE;
    for (size_t elements = 1; elements < 16; elements += 3) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f32_gavgpool_spchw_ukernel__sse_x4);
    }
  }

  TEST(F32_GAVGPOOL_SPCHW__SSE_X4, qmax) {
    TEST_REQUIRES_X86_SSE;
    for (size_t elements = 1; elements < 16; elements += 3) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f32_gavgpool_spchw_ukernel__sse_x4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


TEST(F32_GAVGPOOL_SPCHW__SCALAR_X1, elements_eq_4) {
  GAvgPoolSpCHWMicrokernelTester()
    .elements(4)
    .channels(1)
    .Test(xnn_f32_gavgpool_spchw_ukernel__scalar_x1, GAvgPoolSpCHWMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_SPCHW__SCALAR_X1, elements_div_4) {
  for (size_t elements = 8; elements < 32; elements += 4) {
    GAvgPoolSpCHWMicrokernelTester()
      .elements(elements)
      .channels(1)
      .Test(xnn_f32_gavgpool_spchw_ukernel__scalar_x1, GAvgPoolSpCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_SPCHW__SCALAR_X1, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    GAvgPoolSpCHWMicrokernelTester()
      .elements(elements)
      .channels(1)
      .Test(xnn_f32_gavgpool_spchw_ukernel__scalar_x1, GAvgPoolSpCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_SPCHW__SCALAR_X1, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    GAvgPoolSpCHWMicrokernelTester()
      .elements(elements)
      .channels(1)
      .Test(xnn_f32_gavgpool_spchw_ukernel__scalar_x1, GAvgPoolSpCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_SPCHW__SCALAR_X1, channels_gt_1) {
  for (size_t channels = 2; channels < 5; channels++) {
    for (size_t elements = 1; elements < 16; elements += 3) {
      GAvgPoolSpCHWMicrokernelTester()
        .elements(elements)
        .channels(channels)
        .Test(xnn_f32_gavgpool_spchw_ukernel__scalar_x1, GAvgPoolSpCHWMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_GAVGPOOL_SPCHW__SCALAR_X1, qmin) {
  for (size_t elements = 1; elements < 16; elements += 3) {
    GAvgPoolSpCHWMicrokernelTester()
      .elements(elements)
      .channels(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_spchw_ukernel__scalar_x1, GAvgPoolSpCHWMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_SPCHW__SCALAR_X1, qmax) {
  for (size_t elements = 1; elements < 16; elements += 3) {
    GAvgPoolSpCHWMicrokernelTester()
      .elements(elements)
      .channels(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_spchw_ukernel__scalar_x1, GAvgPoolSpCHWMicrokernelTester::Variant::Scalar);
  }
}
