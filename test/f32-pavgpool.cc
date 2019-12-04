// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/pavgpool.h>
#include "avgpool-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_PAVGPOOL_UP9__NEON, kc_eq_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_pavgpool_ukernel_up9__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_eq_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .kc(4);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            tester
              .kh(kh)
              .kw(kw)
              .Test(xnn_f32_pavgpool_ukernel_up9__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_div_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_div_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 4; kc < 64; kc += 12) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_up9__neon);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_div_4_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(131)
              .Test(xnn_f32_pavgpool_ukernel_up9__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_lt_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_lt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 1; kc < 4; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_up9__neon);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_lt_4_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_up9__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_gt_4_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_gt_4_subtile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 5; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_up9__neon);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_gt_4_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_up9__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .qmax(128)
          .Test(xnn_f32_pavgpool_ukernel_up9__neon);
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, kc_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .qmin(128)
          .Test(xnn_f32_pavgpool_ukernel_up9__neon);
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, small_n) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .Test(xnn_f32_pavgpool_ukernel_up9__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, small_n_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .x_stride(29)
            .Test(xnn_f32_pavgpool_ukernel_up9__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, small_n_with_y_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .y_stride(31)
            .Test(xnn_f32_pavgpool_ukernel_up9__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__NEON, small_n_with_s) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          for (size_t s = 2; s <= ks; s++) {
            AvgPoolMicrokernelTester()
              .mr(9)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .Test(xnn_f32_pavgpool_ukernel_up9__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_eq_4_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_eq_4_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_eq_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .kc(4);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            tester
              .kh(kh)
              .kw(kw)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_eq_4_multipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .kc(4);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        tester
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        tester
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_div_4_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = 17;
    for (size_t kc = 4; kc < 64; kc += 12) {
      tester
        .kc(kc)
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
      tester
        .kc(kc)
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_div_4_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_div_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(131)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_div_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 4; kc < 64; kc += 12) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_div_4_multipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        for (size_t kc = 4; kc < 64; kc += 12) {
          tester
            .kc(kc)
            .kh(ks)
            .kw(1)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
          tester
            .kc(kc)
            .kh(1)
            .kw(ks)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_div_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 4; kc < 64; kc += 12) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .x_stride(131)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_lt_4_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_lt_4_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_lt_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_lt_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 1; kc < 4; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_lt_4_multipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        for (size_t kc = 1; kc < 4; kc++) {
          tester
            .kc(kc)
            .kh(ks)
            .kw(1)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
          tester
            .kc(kc)
            .kh(1)
            .kw(ks)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_lt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 1; kc < 4; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .x_stride(23)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_gt_4_twopass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_gt_4_twopass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_gt_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_gt_4_multipass_fulltile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 5; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_gt_4_multipass_subtile) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        for (size_t kc = 5; kc < 8; kc++) {
          tester
            .kc(kc)
            .kh(ks)
            .kw(1)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
          tester
            .kc(kc)
            .kh(1)
            .kw(ks)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_gt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 5; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .x_stride(23)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_div_4_with_qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .qmax(128)
          .iterations(3)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, kc_div_4_with_qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .qmin(128)
          .iterations(3)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, small_n) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, small_n_with_x_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .x_stride(29)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, small_n_with_y_stride) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .y_stride(31)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__NEON, small_n_with_s) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t s = 2; s <= 5; s++) {
          for (size_t kc = 8; kc < 25; kc += 5) {
            AvgPoolMicrokernelTester()
              .mr(9)
              .qr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__neon);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_PAVGPOOL_UP9__SSE2, kc_eq_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_pavgpool_ukernel_up9__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_eq_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .kc(4);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            tester
              .kh(kh)
              .kw(kw)
              .Test(xnn_f32_pavgpool_ukernel_up9__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_div_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_div_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 4; kc < 64; kc += 12) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_up9__sse);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_div_4_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(131)
              .Test(xnn_f32_pavgpool_ukernel_up9__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_lt_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_lt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 1; kc < 4; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_up9__sse);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_lt_4_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_up9__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_gt_4_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_gt_4_subtile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 5; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_up9__sse);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_gt_4_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_up9__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .qmax(128)
          .Test(xnn_f32_pavgpool_ukernel_up9__sse);
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, kc_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .qmin(128)
          .Test(xnn_f32_pavgpool_ukernel_up9__sse);
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, small_n) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .Test(xnn_f32_pavgpool_ukernel_up9__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, small_n_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .x_stride(29)
            .Test(xnn_f32_pavgpool_ukernel_up9__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, small_n_with_y_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .y_stride(31)
            .Test(xnn_f32_pavgpool_ukernel_up9__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__SSE2, small_n_with_s) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          for (size_t s = 2; s <= ks; s++) {
            AvgPoolMicrokernelTester()
              .mr(9)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .Test(xnn_f32_pavgpool_ukernel_up9__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_eq_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_eq_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_eq_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .kc(4);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            tester
              .kh(kh)
              .kw(kw)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_eq_4_multipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .kc(4);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        tester
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_div_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = 17;
    for (size_t kc = 4; kc < 64; kc += 12) {
      tester
        .kc(kc)
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
      tester
        .kc(kc)
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_div_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_div_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(131)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_div_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 4; kc < 64; kc += 12) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_div_4_multipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        for (size_t kc = 4; kc < 64; kc += 12) {
          tester
            .kc(kc)
            .kh(ks)
            .kw(1)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
          tester
            .kc(kc)
            .kh(1)
            .kw(ks)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_div_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 4; kc < 64; kc += 12) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .x_stride(131)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_lt_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_lt_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_lt_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_lt_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 1; kc < 4; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_lt_4_multipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        for (size_t kc = 1; kc < 4; kc++) {
          tester
            .kc(kc)
            .kh(ks)
            .kw(1)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
          tester
            .kc(kc)
            .kh(1)
            .kw(ks)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_lt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 1; kc < 4; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .x_stride(23)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_gt_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_gt_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_gt_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_gt_4_multipass_fulltile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 5; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_gt_4_multipass_subtile) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        for (size_t kc = 5; kc < 8; kc++) {
          tester
            .kc(kc)
            .kh(ks)
            .kw(1)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
          tester
            .kc(kc)
            .kh(1)
            .kw(ks)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_gt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 5; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .x_stride(23)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_div_4_with_qmax) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .qmax(128)
          .iterations(3)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, kc_div_4_with_qmin) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .qmin(128)
          .iterations(3)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, small_n) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, small_n_with_x_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .x_stride(29)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, small_n_with_y_stride) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .y_stride(31)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__SSE2, small_n_with_s) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t s = 2; s <= 5; s++) {
          for (size_t kc = 8; kc < 25; kc += 5) {
            AvgPoolMicrokernelTester()
              .mr(9)
              .qr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__sse);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_WASM && !XNN_ARCH_ASMJS
  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_eq_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_eq_4_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .kc(4);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            tester
              .kh(kh)
              .kw(kw)
              .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_div_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_div_4_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 4; kc < 64; kc += 12) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_div_4_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(131)
              .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_lt_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_lt_4_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 1; kc < 4; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_lt_4_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_gt_4_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_gt_4_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 5; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_gt_4_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_div_4_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .qmax(128)
          .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, kc_div_4_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .qmin(128)
          .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, small_n) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, small_n_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .x_stride(29)
            .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, small_n_with_y_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .y_stride(31)
            .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__PSIMD, small_n_with_s) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          for (size_t s = 2; s <= ks; s++) {
            AvgPoolMicrokernelTester()
              .mr(9)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .Test(xnn_f32_pavgpool_ukernel_up9__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_eq_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_eq_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_eq_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .kc(4);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            tester
              .kh(kh)
              .kw(kw)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_eq_4_multipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .kc(4);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        tester
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_div_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = 17;
    for (size_t kc = 4; kc < 64; kc += 12) {
      tester
        .kc(kc)
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
      tester
        .kc(kc)
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_div_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_div_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(131)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_div_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 4; kc < 64; kc += 12) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_div_4_multipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        for (size_t kc = 4; kc < 64; kc += 12) {
          tester
            .kc(kc)
            .kh(ks)
            .kw(1)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          tester
            .kc(kc)
            .kh(1)
            .kw(ks)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_div_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 4; kc < 64; kc += 12) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .x_stride(131)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_lt_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_lt_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_lt_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_lt_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 1; kc < 4; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_lt_4_multipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        for (size_t kc = 1; kc < 4; kc++) {
          tester
            .kc(kc)
            .kh(ks)
            .kw(1)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          tester
            .kc(kc)
            .kh(1)
            .kw(ks)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_lt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 1; kc < 4; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .x_stride(23)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_gt_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_gt_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_gt_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_gt_4_multipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 5; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_gt_4_multipass_subtile) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        for (size_t kc = 5; kc < 8; kc++) {
          tester
            .kc(kc)
            .kh(ks)
            .kw(1)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          tester
            .kc(kc)
            .kh(1)
            .kw(ks)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_gt_4_multipass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 5; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .x_stride(23)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_div_4_with_qmax) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .qmax(128)
          .iterations(3)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, kc_div_4_with_qmin) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .qmin(128)
          .iterations(3)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, small_n) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, small_n_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .x_stride(29)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, small_n_with_y_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 8; kc < 25; kc += 5) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .y_stride(31)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__PSIMD, small_n_with_s) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t s = 2; s <= 5; s++) {
          for (size_t kc = 8; kc < 25; kc += 5) {
            AvgPoolMicrokernelTester()
              .mr(9)
              .qr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__psimd, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }
#endif  // !XNN_ARCH_WASM && !XNN_ARCH_ASMJS


#if XNN_ARCH_WASM
  TEST(F32_PAVGPOOL_UP9__WASM, kc_eq_1_fulltile) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .kc(1);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__WASM, kc_eq_1_subtile) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .kc(1);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            tester
              .kh(kh)
              .kw(kw)
              .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__WASM, kc_gt_1_fulltile) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 2; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__WASM, kc_gt_1_subtile) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 2; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__WASM, kc_gt_1_fulltile_with_x_stride) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 2; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__WASM, qmax) {
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .qmax(128)
          .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__WASM, qmin) {
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(3)
          .kw(3)
          .kc(kc)
          .qmin(128)
          .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__WASM, small_n) {
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 1; kc < 8; kc += 3) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__WASM, small_n_with_x_stride) {
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 1; kc < 8; kc += 3) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .x_stride(29)
            .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__WASM, small_n_with_y_stride) {
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 1; kc < 8; kc += 3) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .y_stride(31)
            .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_UP9__WASM, small_n_with_s) {
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3}}) {
        for (size_t kc = 1; kc < 8; kc += 3) {
          for (size_t s = 2; s <= ks; s++) {
            AvgPoolMicrokernelTester()
              .mr(9)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .Test(xnn_f32_pavgpool_ukernel_up9__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, kc_eq_1_twopass_fulltile) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(1);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, kc_eq_1_twopass_subtile) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(1);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, kc_eq_1_multipass_fulltile) {
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .kc(1);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            tester
              .kh(kh)
              .kw(kw)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, kc_eq_1_multipass_subtile) {
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .kc(1);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        tester
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, kc_gt_1_twopass_fulltile) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 2; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, kc_gt_1_twopass_subtile) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 2; kc < 8; kc++) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, kc_gt_1_twopass_fulltile_with_x_stride) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    const size_t ks = tester.mr() + tester.qr();
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 2; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, kc_gt_1_multipass_fulltile) {
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 2; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, kc_gt_1_multipass_subtile) {
    for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
        for (size_t kc = 2; kc < 8; kc++) {
          tester
            .kc(kc)
            .kh(ks)
            .kw(1)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
          tester
            .kc(kc)
            .kh(1)
            .kw(ks)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, kc_gt_1_multipass_fulltile_with_x_stride) {
    for (size_t ks : std::vector<size_t>{{25, 49}}) {
      auto tester = AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .iterations(3);
      for (size_t kh = 1; kh <= ks; kh++) {
        for (size_t kw = 1; kw <= ks; kw++) {
          if (kh * kw == ks) {
            for (size_t kc = 2; kc < 8; kc++) {
              tester
                .kh(kh)
                .kw(kw)
                .kc(kc)
                .x_stride(23)
                .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
            }
          }
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, qmax) {
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .qmax(128)
          .iterations(3)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, qmin) {
    for (size_t n = 1; n <= 5; n += 2) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(5)
          .kw(5)
          .kc(kc)
          .qmin(128)
          .iterations(3)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, small_n) {
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 1; kc < 8; kc += 3) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, small_n_with_x_stride) {
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 1; kc < 8; kc += 3) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .x_stride(29)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, small_n_with_y_stride) {
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t kc = 1; kc < 8; kc += 3) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .y_stride(31)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(F32_PAVGPOOL_MP9P8Q__WASM, small_n_with_s) {
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{5, 7}}) {
        for (size_t s = 2; s <= 5; s++) {
          for (size_t kc = 1; kc < 8; kc += 3) {
            AvgPoolMicrokernelTester()
              .mr(9)
              .qr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__wasm, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }
#endif  // XNN_ARCH_WASM


TEST(F32_PAVGPOOL_UP9__SCALAR, kc_eq_1_fulltile) {
  auto tester = AvgPoolMicrokernelTester()
    .mr(9)
    .kc(1);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester
          .kh(kh)
          .kw(kw)
          .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PAVGPOOL_UP9__SCALAR, kc_eq_1_subtile) {
  auto tester = AvgPoolMicrokernelTester()
    .mr(9)
    .kc(1);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(F32_PAVGPOOL_UP9__SCALAR, kc_gt_1_fulltile) {
  auto tester = AvgPoolMicrokernelTester()
    .mr(9);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 2; kc < 8; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(F32_PAVGPOOL_UP9__SCALAR, kc_gt_1_subtile) {
  auto tester = AvgPoolMicrokernelTester()
    .mr(9)
    .iterations(3);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 2; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }
}

TEST(F32_PAVGPOOL_UP9__SCALAR, kc_gt_1_fulltile_with_x_stride) {
  auto tester = AvgPoolMicrokernelTester()
    .mr(9)
    .iterations(3);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 2; kc < 8; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .x_stride(23)
            .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(F32_PAVGPOOL_UP9__SCALAR, qmax) {
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 1; kc < 8; kc += 3) {
      AvgPoolMicrokernelTester()
        .mr(9)
        .n(n)
        .kh(3)
        .kw(3)
        .kc(kc)
        .qmax(128)
        .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PAVGPOOL_UP9__SCALAR, qmin) {
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 1; kc < 8; kc += 3) {
      AvgPoolMicrokernelTester()
        .mr(9)
        .n(n)
        .kh(3)
        .kw(3)
        .kc(kc)
        .qmin(128)
        .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PAVGPOOL_UP9__SCALAR, small_n) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(ks)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PAVGPOOL_UP9__SCALAR, small_n_with_x_stride) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(ks)
          .kw(ks)
          .kc(kc)
          .x_stride(29)
          .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PAVGPOOL_UP9__SCALAR, small_n_with_y_stride) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .n(n)
          .kh(ks)
          .kw(ks)
          .kc(kc)
          .y_stride(31)
          .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PAVGPOOL_UP9__SCALAR, small_n_with_s) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3}}) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        for (size_t s = 2; s <= ks; s++) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .s(s)
            .Test(xnn_f32_pavgpool_ukernel_up9__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, kc_eq_1_twopass_fulltile) {
  auto tester = AvgPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        tester
          .kh(kh)
          .kw(kw)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, kc_eq_1_twopass_subtile) {
  auto tester = AvgPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
    tester
      .kh(ks)
      .kw(1)
      .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
    tester
      .kh(1)
      .kw(ks)
      .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, kc_eq_1_multipass_fulltile) {
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(1);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, kc_eq_1_multipass_subtile) {
  for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(1);
    for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, kc_gt_1_twopass_fulltile) {
  auto tester = AvgPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .iterations(3);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        for (size_t kc = 2; kc < 8; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, kc_gt_1_twopass_subtile) {
  auto tester = AvgPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .iterations(3);
  for (size_t ks = 10; ks < tester.mr() + tester.qr(); ks++) {
    for (size_t kc = 2; kc < 8; kc++) {
      tester
        .kc(kc)
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      tester
        .kc(kc)
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, kc_gt_1_twopass_fulltile_with_x_stride) {
  auto tester = AvgPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .iterations(3);
  const size_t ks = tester.mr() + tester.qr();
  for (size_t kh = 1; kh <= ks; kh++) {
    for (size_t kw = 1; kw <= ks; kw++) {
      if (kh * kw == ks) {
        for (size_t kc = 2; kc < 8; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .x_stride(23)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, kc_gt_1_multipass_fulltile) {
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 2; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, kc_gt_1_multipass_subtile) {
  for (size_t ks_max : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = ks_max - tester.qr() + 1; ks < ks_max; ks++) {
      for (size_t kc = 2; kc < 8; kc++) {
        tester
          .kc(kc)
          .kh(ks)
          .kw(1)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
        tester
          .kc(kc)
          .kh(1)
          .kw(ks)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, kc_gt_1_multipass_fulltile_with_x_stride) {
  for (size_t ks : std::vector<size_t>{{25, 49}}) {
    auto tester = AvgPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= ks; kh++) {
      for (size_t kw = 1; kw <= ks; kw++) {
        if (kh * kw == ks) {
          for (size_t kc = 2; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(23)
              .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, qmax) {
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 1; kc < 8; kc += 3) {
      AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .n(n)
        .kh(5)
        .kw(5)
        .kc(kc)
        .qmax(128)
        .iterations(3)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, qmin) {
  for (size_t n = 1; n <= 5; n += 2) {
    for (size_t kc = 1; kc < 8; kc += 3) {
      AvgPoolMicrokernelTester()
        .mr(9)
        .qr(8)
        .n(n)
        .kh(5)
        .kw(5)
        .kc(kc)
        .qmin(128)
        .iterations(3)
        .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, small_n) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(ks)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, small_n_with_x_stride) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(ks)
          .kw(ks)
          .kc(kc)
          .x_stride(29)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, small_n_with_y_stride) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t kc = 1; kc < 8; kc += 3) {
        AvgPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(ks)
          .kw(ks)
          .kc(kc)
          .y_stride(31)
          .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(F32_PAVGPOOL_MP9P8Q__SCALAR, small_n_with_s) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{5, 7}}) {
      for (size_t s = 2; s <= 5; s++) {
        for (size_t kc = 1; kc < 8; kc += 3) {
          AvgPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .s(s)
            .Test(xnn_f32_pavgpool_ukernel_mp9p8q__scalar, AvgPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}
