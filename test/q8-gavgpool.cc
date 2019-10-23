// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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
  TEST(Q8_GAVGPOOL_UP7__NEON, n_eq_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .Test(xnn_q8_gavgpool_ukernel_up7__neon);
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_eq_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(8)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_eq_8_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .x_stride(11)
      .Test(xnn_q8_gavgpool_ukernel_up7__neon);
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_eq_8_fulltile_with_x_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .x_scale(x_scale)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_eq_8_fulltile_with_x_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .x_zero_point(x_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_eq_8_fulltile_with_y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .y_scale(y_scale)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_eq_8_fulltile_with_y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .y_zero_point(y_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_ukernel_up7__neon);
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_ukernel_up7__neon);
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_div_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 24) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_div_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 24) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_lt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_lt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_lt_8_fulltile_with_x_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .x_scale(x_scale)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_lt_8_fulltile_with_x_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .x_zero_point(x_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_lt_8_fulltile_with_y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .y_scale(y_scale)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_lt_8_fulltile_with_y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .y_zero_point(y_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_gt_8_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_gt_8_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_gt_8_fulltile_with_x_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .x_scale(x_scale)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_gt_8_fulltile_with_x_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .x_zero_point(x_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_gt_8_fulltile_with_y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .y_scale(y_scale)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_gt_8_fulltile_with_y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .y_zero_point(y_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_up7__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__NEON, n_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_ukernel_up7__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_2pass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .x_stride(11)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_2pass_fulltile_with_x_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .x_scale(x_scale)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_2pass_fulltile_with_x_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .x_zero_point(x_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_2pass_fulltile_with_y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .y_scale(y_scale)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_2pass_fulltile_with_y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .y_zero_point(y_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(8)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_2pass_subtile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(8)
        .x_stride(11)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(8)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_eq_8_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(8)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_div_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 24) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_div_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 24) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_div_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 24) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_div_8_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 8; n < 128; n += 24) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(131)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_lt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_lt_8_2pass_fulltile_with_x_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
      for (size_t n = 1; n < 8; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .x_scale(x_scale)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_lt_8_2pass_fulltile_with_x_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
      for (size_t n = 1; n < 8; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .x_zero_point(x_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_lt_8_2pass_fulltile_with_y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
      for (size_t n = 1; n < 8; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .y_scale(y_scale)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_lt_8_2pass_fulltile_with_y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      for (size_t n = 1; n < 8; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .y_zero_point(y_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_lt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_lt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_lt_8_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n < 8; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(23)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_gt_8_2pass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_gt_8_2pass_fulltile_with_x_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
      for (size_t n = 9; n < 16; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .x_scale(x_scale)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_gt_8_2pass_fulltile_with_x_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
      for (size_t n = 9; n < 16; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .x_zero_point(x_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_gt_8_2pass_fulltile_with_y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
      for (size_t n = 9; n < 16; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .y_scale(y_scale)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_gt_8_2pass_fulltile_with_y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      for (size_t n = 9; n < 16; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .y_zero_point(y_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_gt_8_2pass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_gt_8_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__NEON, n_gt_8_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 9; n < 16; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(23)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__neon);
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(Q8_GAVGPOOL_UP7__SSE2, n_eq_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_eq_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(8)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_eq_8_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .x_stride(11)
      .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_eq_8_fulltile_with_x_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .x_scale(x_scale)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_eq_8_fulltile_with_x_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .x_zero_point(x_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_eq_8_fulltile_with_y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .y_scale(y_scale)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_eq_8_fulltile_with_y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(8)
        .y_zero_point(y_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_eq_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_eq_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(8)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_div_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 8; n < 128; n += 24) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_div_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 8; n < 128; n += 24) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_lt_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_lt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_lt_8_fulltile_with_x_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .x_scale(x_scale)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_lt_8_fulltile_with_x_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .x_zero_point(x_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_lt_8_fulltile_with_y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .y_scale(y_scale)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_lt_8_fulltile_with_y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .y_zero_point(y_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_lt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_lt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_gt_8_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_gt_8_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_gt_8_fulltile_with_x_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .x_scale(x_scale)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_gt_8_fulltile_with_x_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .x_zero_point(x_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_gt_8_fulltile_with_y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .y_scale(y_scale)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_gt_8_fulltile_with_y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
        GAvgPoolMicrokernelTester()
          .m(7)
          .n(n)
          .y_zero_point(y_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_gt_8_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_UP7__SSE2, n_gt_8_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_ukernel_up7__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_2pass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .x_stride(11)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_2pass_fulltile_with_x_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .x_scale(x_scale)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_2pass_fulltile_with_x_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .x_zero_point(x_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_2pass_fulltile_with_y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .y_scale(y_scale)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_2pass_fulltile_with_y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(8)
        .y_zero_point(y_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(8)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(8)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_2pass_subtile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(8)
        .x_stride(11)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(8)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_eq_8_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(8)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_div_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 8; n < 128; n += 24) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_div_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 8; n < 128; n += 24) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_div_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 8; n < 128; n += 24) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_div_8_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 8; n < 128; n += 24) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(131)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_lt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_lt_8_2pass_fulltile_with_x_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
      for (size_t n = 1; n < 8; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .x_scale(x_scale)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_lt_8_2pass_fulltile_with_x_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
      for (size_t n = 1; n < 8; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .x_zero_point(x_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_lt_8_2pass_fulltile_with_y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
      for (size_t n = 1; n < 8; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .y_scale(y_scale)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_lt_8_2pass_fulltile_with_y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      for (size_t n = 1; n < 8; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .y_zero_point(y_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_lt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_lt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_lt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_lt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_lt_8_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n < 8; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(23)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_gt_8_2pass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_gt_8_2pass_fulltile_with_x_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
      for (size_t n = 9; n < 16; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .x_scale(x_scale)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_gt_8_2pass_fulltile_with_x_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
      for (size_t n = 9; n < 16; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .x_zero_point(x_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_gt_8_2pass_fulltile_with_y_scale) {
    TEST_REQUIRES_X86_SSE2;
    for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
      for (size_t n = 9; n < 16; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .y_scale(y_scale)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_gt_8_2pass_fulltile_with_y_zero_point) {
    TEST_REQUIRES_X86_SSE2;
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      for (size_t n = 9; n < 16; n++) {
        GAvgPoolMicrokernelTester()
          .m(14)
          .n(n)
          .y_zero_point(y_zero_point)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_gt_8_2pass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmax(128)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_gt_8_2pass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .x_zero_point(128)
        .y_zero_point(128)
        .x_scale(1.0f)
        .y_scale(1.0f)
        .qmin(128)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_gt_8_2pass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      for (size_t m = 1; m < 7; m++) {
        GAvgPoolMicrokernelTester()
          .m(7 + m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_gt_8_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }

  TEST(Q8_GAVGPOOL_MP7p7q__SSE2, n_gt_8_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 9; n < 16; n++) {
      for (size_t m = 14; m <= 35; m += 7) {
        GAvgPoolMicrokernelTester()
          .m(m)
          .n(n)
          .x_stride(23)
          .Test(xnn_q8_gavgpool_ukernel_mp7p7q__sse2);
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile) {
  GAvgPoolMicrokernelTester()
    .m(7)
    .n(1)
    .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_eq_1_subtile) {
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester()
      .m(m)
      .n(1)
      .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile_with_x_stride) {
  GAvgPoolMicrokernelTester()
    .m(7)
    .n(1)
    .x_stride(11)
    .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile_with_x_scale) {
  for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(1)
      .x_scale(x_scale)
      .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile_with_x_zero_point) {
  for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(1)
      .x_zero_point(x_zero_point)
      .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile_with_y_scale) {
  for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(1)
      .y_scale(y_scale)
      .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile_with_y_zero_point) {
  for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(1)
      .y_zero_point(y_zero_point)
      .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .m(7)
    .n(1)
    .x_zero_point(128)
    .y_zero_point(128)
    .x_scale(1.0f)
    .y_scale(1.0f)
    .qmax(128)
    .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_eq_1_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .m(7)
    .n(1)
    .x_zero_point(128)
    .y_zero_point(128)
    .x_scale(1.0f)
    .y_scale(1.0f)
    .qmin(128)
    .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_gt_1_fulltile) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(n)
      .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_gt_1_subtile) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_gt_1_fulltile_with_x_scale) {
  for (size_t n = 2; n < 8; n++) {
    for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .x_scale(x_scale)
        .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_gt_1_fulltile_with_x_zero_point) {
  for (size_t n = 2; n < 8; n++) {
    for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .x_zero_point(x_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_gt_1_fulltile_with_y_scale) {
  for (size_t n = 2; n < 8; n++) {
    for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .y_scale(y_scale)
        .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_gt_1_fulltile_with_y_zero_point) {
  for (size_t n = 2; n < 8; n++) {
    for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
      GAvgPoolMicrokernelTester()
        .m(7)
        .n(n)
        .y_zero_point(y_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_gt_1_fulltile_with_qmax) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(n)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_UP7__SCALAR, n_gt_1_fulltile_with_qmin) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(7)
      .n(n)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_ukernel_up7__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile) {
  GAvgPoolMicrokernelTester()
    .m(14)
    .n(1)
    .nr(8)
    .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile_with_x_stride) {
  GAvgPoolMicrokernelTester()
    .m(14)
    .n(1)
    .nr(8)
    .x_stride(11)
    .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile_with_x_scale) {
  for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(1)
      .x_scale(x_scale)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile_with_x_zero_point) {
  for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(1)
      .x_zero_point(x_zero_point)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile_with_y_scale) {
  for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(1)
      .y_scale(y_scale)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile_with_y_zero_point) {
  for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(1)
      .y_zero_point(y_zero_point)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile_with_qmax) {
  GAvgPoolMicrokernelTester()
    .m(14)
    .n(1)
    .nr(8)
    .x_zero_point(128)
    .y_zero_point(128)
    .x_scale(1.0f)
    .y_scale(1.0f)
    .qmax(128)
    .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_fulltile_with_qmin) {
  GAvgPoolMicrokernelTester()
    .m(14)
    .n(1)
    .nr(8)
    .x_zero_point(128)
    .y_zero_point(128)
    .x_scale(1.0f)
    .y_scale(1.0f)
    .qmin(128)
    .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_subtile) {
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester()
      .m(7 + m)
      .n(1)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_2pass_subtile_with_x_stride) {
  for (size_t m = 1; m < 7; m++) {
    GAvgPoolMicrokernelTester()
      .m(7 + m)
      .n(1)
      .x_stride(11)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_multipass_fulltile) {
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester()
      .m(m)
      .n(1)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_eq_1_multipass_fulltile_with_x_stride) {
  for (size_t m = 14; m <= 35; m += 7) {
    GAvgPoolMicrokernelTester()
      .m(m)
      .n(1)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_fulltile) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(n)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_fulltile_with_x_scale) {
  for (float x_scale = 0.01f; x_scale < 100.0f; x_scale *= 3.14159265f) {
    for (size_t n = 2; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .x_scale(x_scale)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_fulltile_with_x_zero_point) {
  for (int32_t x_zero_point = 0; x_zero_point <= 255; x_zero_point += 51) {
    for (size_t n = 2; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .x_zero_point(x_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_fulltile_with_y_scale) {
  for (float y_scale = 0.01f; y_scale < 100.0f; y_scale *= 3.14159265f) {
    for (size_t n = 2; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .y_scale(y_scale)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_fulltile_with_y_zero_point) {
  for (int32_t y_zero_point = 0; y_zero_point <= 255; y_zero_point += 51) {
    for (size_t n = 2; n < 8; n++) {
      GAvgPoolMicrokernelTester()
        .m(14)
        .n(n)
        .y_zero_point(y_zero_point)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_fulltile_with_qmax) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(n)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmax(128)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_fulltile_with_qmin) {
  for (size_t n = 2; n < 8; n++) {
    GAvgPoolMicrokernelTester()
      .m(14)
      .n(n)
      .x_zero_point(128)
      .y_zero_point(128)
      .x_scale(1.0f)
      .y_scale(1.0f)
      .qmin(128)
      .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_2pass_subtile) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t m = 1; m < 7; m++) {
      GAvgPoolMicrokernelTester()
        .m(7 + m)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_multipass_fulltile) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(n)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(Q8_GAVGPOOL_MP7p7q__SCALAR, n_gt_1_multipass_fulltile_with_x_stride) {
  for (size_t n = 2; n < 8; n++) {
    for (size_t m = 14; m <= 35; m += 7) {
      GAvgPoolMicrokernelTester()
        .m(m)
        .n(n)
        .x_stride(23)
        .Test(xnn_q8_gavgpool_ukernel_mp7p7q__scalar, GAvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}
