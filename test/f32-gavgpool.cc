// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gavgpool.h>
#include "gavgpool-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_GAVGPOOL_UP7__NEON, n_eq_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .Test(xnn_f32_gavgpool_ukernel_up7__neon);
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_eq_4_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .x_stride(11)
      .Test(xnn_f32_gavgpool_ukernel_up7__neon);
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_eq_4_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_ukernel_up7__neon);
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_eq_4_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_ukernel_up7__neon);
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_div_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 12) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_lt_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_lt_4_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_lt_4_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_gt_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_gt_4_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(F32_GAVGPOOL_UP7__NEON, n_gt_4_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_eq_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_eq_4_2pass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .x_stride(11)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_eq_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_eq_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_eq_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_eq_4_2pass_subtile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(4)
        .x_stride(11)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_eq_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_eq_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_div_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 12) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_div_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_div_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_div_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(131)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_lt_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_lt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_lt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_lt_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_lt_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_lt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(23)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_gt_4_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_gt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_gt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_gt_4_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_gt_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__NEON, n_gt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(23)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_GAVGPOOL_UP7__SSE2, n_eq_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .Test(xnn_f32_gavgpool_ukernel_up7__sse);
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_eq_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_up7__sse);
    }
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_eq_4_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .x_stride(11)
      .Test(xnn_f32_gavgpool_ukernel_up7__sse);
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_eq_4_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_ukernel_up7__sse);
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_eq_4_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_ukernel_up7__sse);
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_div_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 12) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_up7__sse);
    }
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_up7__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_lt_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_up7__sse);
    }
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_lt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_up7__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_lt_4_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__sse);
    }
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_lt_4_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__sse);
    }
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_gt_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_up7__sse);
    }
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_up7__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_gt_4_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__sse);
    }
  }

  TEST(F32_GAVGPOOL_UP7__SSE2, n_gt_4_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_eq_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_eq_4_2pass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .x_stride(11)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_eq_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_eq_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_eq_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_eq_4_2pass_subtile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(4)
        .x_stride(11)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_eq_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_eq_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_div_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 12) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_div_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_div_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_div_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(131)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_lt_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_lt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_lt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_lt_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_lt_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_lt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(23)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_gt_4_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_gt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_gt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_gt_4_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_gt_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__SSE2, n_gt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(23)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__sse);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_WASM && !XNN_ARCH_ASMJS
  TEST(F32_GAVGPOOL_UP7__PSIMD, n_eq_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_eq_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_eq_4_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .x_stride(11)
      .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_eq_4_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_eq_4_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_div_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 12) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_div_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_lt_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_lt_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_lt_4_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_lt_4_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_gt_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_gt_4_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_gt_4_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_UP7__PSIMD, n_gt_4_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_up7__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_eq_4_2pass_fulltile) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_eq_4_2pass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .x_stride(11)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_eq_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .qmax(128)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_eq_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(4)
      .qmin(128)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_eq_4_2pass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_eq_4_2pass_subtile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(4)
        .x_stride(11)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_eq_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_eq_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(4)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_div_4_2pass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 12) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_div_4_2pass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_div_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_div_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 4; n < 64; n += 12) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(131)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_lt_4_2pass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_lt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_lt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_lt_4_2pass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_lt_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_lt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n < 4; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(23)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_gt_4_2pass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_gt_4_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmax(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_gt_4_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .qmin(128)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_gt_4_2pass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_gt_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_GAVGPOOL_MP7p7q__PSIMD, n_gt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 5; n < 8; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(23)
          .Test(xnn_f32_gavgpool_ukernel_mp7p7q__psimd, GAvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
#endif  // !XNN_ARCH_WASM && !XNN_ARCH_ASMJS


TEST(F32_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile) {
  GAvgPoolMicrokernelTester()
    .m(7)
    .n(1)
    .Test(xnn_f32_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_UP7__SCALAR, n_eq_1_subtile) {
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester()
      .m(m)
      .n(1)
      .Test(xnn_f32_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile_with_x_stride) {
  GAvgPoolMicrokernelTester()
    .m(7)
    .n(1)
    .x_stride(11)
    .Test(xnn_f32_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .m(7)
    .n(1)
    .qmax(128)
    .Test(xnn_f32_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .m(7)
    .n(1)
    .qmin(128)
    .Test(xnn_f32_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_UP7__SCALAR, n_gt_1_fulltile) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(n)
      .Test(xnn_f32_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_UP7__SCALAR, n_gt_1_subtile) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(n)
        .Test(xnn_f32_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_GAVGPOOL_UP7__SCALAR, n_gt_1_fulltile_with_qmax) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(n)
      .qmax(128)
      .Test(xnn_f32_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_UP7__SCALAR, n_gt_1_fulltile_with_qmin) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(n)
      .qmin(128)
      .Test(xnn_f32_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .m(14)
    .n(1)
    .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile_with_x_stride) {
  GAvgPoolMicrokernelTester()
    .m(14)
    .n(1)
    .x_stride(11)
    .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .m(14)
    .n(1)
    .qmax(128)
    .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .m(14)
    .n(1)
    .qmin(128)
    .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_subtile) {
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester()
      .m(7 + m)
      .n(1)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_subtile_with_x_stride) {
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester()
      .m(7 + m)
      .n(1)
        .x_stride(11)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_multipass_fulltile) {
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester()
      .m(m)
      .n(1)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_multipass_fulltile_with_x_stride) {
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester()
      .m(m)
      .n(1)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_fulltile) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(n)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_fulltile_with_qmax) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(n)
        .qmax(128)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_fulltile_with_qmin) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(n)
        .qmin(128)
      .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_subtile) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(n)
            .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_multipass_fulltile) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(n)
            .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_multipass_fulltile_with_x_stride) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(n)
            .x_stride(23)
        .Test(xnn_f32_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}
