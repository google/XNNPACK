// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cpuinfo.h>
#include <gtest/gtest.h>

#include <xnnpack/isa-checks.h>
#include <xnnpack/maxpool.h>

#include "maxpool-microkernel-tester.h"


#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          tester
            .kh(kh)
            .kw(kw)
            .qmin(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          tester
            .kh(kh)
            .kw(kw)
            .qmax(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_unipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_unipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_unipass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_unipass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_unipass_subtile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          tester
            .kh(kh)
            .kw(kw)
            .qmin(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          tester
            .kh(kh)
            .kw(kw)
            .qmax(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_twopass_fulltile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_twopass_subtile) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      tester
        .kh(ks)
        .kw(1)
        .qmin(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      tester
        .kh(1)
        .kw(ks)
        .qmin(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_eq_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      tester
        .kh(ks)
        .kw(1)
        .qmax(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      tester
        .kh(1)
        .kw(ks)
        .qmax(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_div_4_multipass_with_x_stride) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_lt_4_multipass_with_x_stride) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_multipass) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_multipass_with_qmin) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_multipass_with_qmax) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, kc_gt_4_multipass_with_x_stride) {
    TEST_REQUIRES_X86_SSE;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, small_n) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
        for (size_t kc = 1; kc < 51; kc += 5) {
          MaxPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .iterations(3)
            .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, small_n_with_x_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
        for (size_t kc = 1; kc < 51; kc += 5) {
          MaxPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .x_stride(101)
            .iterations(1)
            .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, small_n_with_y_stride) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
        for (size_t kc = 1; kc < 51; kc += 5) {
          MaxPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .y_stride(103)
            .iterations(1)
            .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__SSE, small_n_with_s) {
    TEST_REQUIRES_X86_SSE;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        for (size_t kc = 1; kc < 51; kc += 5) {
          for (size_t s = 2; s <= ks; s++) {
            MaxPoolMicrokernelTester()
              .mr(9)
              .qr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .iterations(1)
              .Test(xnn_f32_maxpool_ukernel_9p8q__sse);
          }
        }
      }
    }
  }
#endif  // CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64


#if !CPUINFO_ARCH_WASM && !CPUINFO_ARCH_ASMJS
  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          tester
            .kh(kh)
            .kw(kw)
            .qmin(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          tester
            .kh(kh)
            .kw(kw)
            .qmax(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_unipass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_unipass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_unipass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_unipass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_unipass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_unipass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr(); kw++) {
        if (kh * kw == tester.mr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_unipass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = 2; ks < tester.mr(); ks++) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          tester
            .kh(kh)
            .kw(kw)
            .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          tester
            .kh(kh)
            .kw(kw)
            .qmin(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          tester
            .kh(kh)
            .kw(kw)
            .qmax(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 4; kc < 64; kc += 12) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 1; kc < 4; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_twopass_fulltile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_twopass_fulltile_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmin(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_twopass_fulltile_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .qmax(192)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_twopass_fulltile_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
      for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
        if (kh * kw == tester.mr() + tester.qr()) {
          for (size_t kc = 5; kc < 8; kc++) {
            tester
              .kh(kh)
              .kw(kw)
              .kc(kc)
              .x_stride(257)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_twopass_subtile) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_multipass) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      tester
        .kh(ks)
        .kw(1)
        .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_multipass_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      tester
        .kh(ks)
        .kw(1)
        .qmin(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .qmin(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_eq_4_multipass_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .kc(4);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      tester
        .kh(ks)
        .kw(1)
        .qmax(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .qmax(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_multipass) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_multipass_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_multipass_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_div_4_multipass_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 4; kc < 64; kc += 12) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_multipass) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_multipass_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_multipass_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_lt_4_multipass_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 1; kc < 4; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_multipass) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_multipass_with_qmin) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_multipass_with_qmax) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, kc_gt_4_multipass_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    auto tester = MaxPoolMicrokernelTester()
      .mr(9)
      .qr(8)
      .iterations(3);
    for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
      for (size_t kc = 5; kc < 8; kc++) {
        tester
          .kh(ks)
          .kw(1)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        tester
          .kh(1)
          .kw(ks)
          .kc(kc)
          .x_stride(257)
          .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, small_n) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
        for (size_t kc = 1; kc < 51; kc += 5) {
          MaxPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .iterations(3)
            .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, small_n_with_x_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
        for (size_t kc = 1; kc < 51; kc += 5) {
          MaxPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .x_stride(101)
            .iterations(1)
            .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, small_n_with_y_stride) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
        for (size_t kc = 1; kc < 51; kc += 5) {
          MaxPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .y_stride(103)
            .iterations(1)
            .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }

  TEST(SMAXPOOL_9P8Q__PSIMD, small_n_with_s) {
    TEST_REQUIRES_PSIMD;
    for (size_t n = 2; n < 5; n++) {
      for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
        for (size_t kc = 1; kc < 51; kc += 5) {
          for (size_t s = 2; s <= ks; s++) {
            MaxPoolMicrokernelTester()
              .mr(9)
              .qr(8)
              .n(n)
              .kh(ks)
              .kw(ks)
              .kc(kc)
              .s(s)
              .iterations(1)
              .Test(xnn_f32_maxpool_ukernel_9p8q__psimd, MaxPoolMicrokernelTester::Variant::Scalar);
          }
        }
      }
    }
  }
#endif  // !CPUINFO_ARCH_WASM && !CPUINFO_ARCH_ASMJS


TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_unipass_fulltile) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester
          .kh(kh)
          .kw(kw)
          .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_unipass_fulltile_with_qmin) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester
          .kh(kh)
          .kw(kw)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_unipass_fulltile_with_qmax) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        tester
          .kh(kh)
          .kw(kw)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_unipass_subtile) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    tester
      .kh(ks)
      .kw(1)
      .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    tester
      .kh(1)
      .kw(ks)
      .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_unipass_fulltile) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 2; kc < 5; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_unipass_fulltile_with_qmin) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 2; kc < 5; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .qmin(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_unipass_fulltile_with_qmax) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 2; kc < 5; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .qmax(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_unipass_fulltile_with_x_stride) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .iterations(3);
  for (size_t kh = 1; kh <= tester.mr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr(); kw++) {
      if (kh * kw == tester.mr()) {
        for (size_t kc = 2; kc < 5; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .x_stride(257)
            .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_unipass_subtile) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .iterations(3);
  for (size_t ks = 2; ks < tester.mr(); ks++) {
    for (size_t kc = 2; kc < 5; kc++) {
      tester
        .kh(ks)
        .kw(1)
        .kc(kc)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .kc(kc)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_twopass_fulltile) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        tester
          .kh(kh)
          .kw(kw)
          .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_twopass_fulltile_with_qmin) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        tester
          .kh(kh)
          .kw(kw)
          .qmin(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_twopass_fulltile_with_qmax) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        tester
          .kh(kh)
          .kw(kw)
          .qmax(192)
          .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_twopass_subtile) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
    tester
      .kh(ks)
      .kw(1)
      .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    tester
      .kh(1)
      .kw(ks)
      .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_twopass_fulltile) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 2; kc < 5; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_twopass_fulltile_with_qmin) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 2; kc < 5; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .qmin(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_twopass_fulltile_with_qmax) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 2; kc < 5; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .qmax(192)
            .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_twopass_fulltile_with_x_stride) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .iterations(3);
  for (size_t kh = 1; kh <= tester.mr() + tester.qr(); kh++) {
    for (size_t kw = 1; kw <= tester.mr() + tester.qr(); kw++) {
      if (kh * kw == tester.mr() + tester.qr()) {
        for (size_t kc = 2; kc < 5; kc++) {
          tester
            .kh(kh)
            .kw(kw)
            .kc(kc)
            .x_stride(257)
            .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_twopass_subtile) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .iterations(3);
  for (size_t ks = tester.mr() + 1; ks < tester.mr() + tester.qr(); ks++) {
    for (size_t kc = 2; kc < 5; kc++) {
      tester
        .kh(ks)
        .kw(1)
        .kc(kc)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .kc(kc)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_multipass) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
    tester
      .kh(ks)
      .kw(1)
      .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    tester
      .kh(1)
      .kw(ks)
      .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_multipass_with_qmin) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
    tester
      .kh(ks)
      .kw(1)
      .qmin(192)
      .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    tester
      .kh(1)
      .kw(ks)
      .qmin(192)
      .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_eq_1_multipass_with_qmax) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .kc(1);
  for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
    tester
      .kh(ks)
      .kw(1)
      .qmax(192)
      .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    tester
      .kh(1)
      .kw(ks)
      .qmax(192)
      .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_multipass) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8);
  for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
    for (size_t kc = 2; kc < 5; kc++) {
      tester
        .kh(ks)
        .kw(1)
        .kc(kc)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .kc(kc)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_multipass_with_qmin) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8);
  for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
    for (size_t kc = 2; kc < 5; kc++) {
      tester
        .kh(ks)
        .kw(1)
        .kc(kc)
        .qmin(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .kc(kc)
        .qmin(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_multipass_with_qmax) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8);
  for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
    for (size_t kc = 2; kc < 5; kc++) {
      tester
        .kh(ks)
        .kw(1)
        .kc(kc)
        .qmax(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .kc(kc)
        .qmax(192)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, kc_gt_1_multipass_with_x_stride) {
  auto tester = MaxPoolMicrokernelTester()
    .mr(9)
    .qr(8)
    .iterations(3);
  for (size_t ks = tester.mr() + tester.qr() + 1; ks < tester.mr() + 3 * tester.qr(); ks += 3) {
    for (size_t kc = 2; kc < 5; kc++) {
      tester
        .kh(ks)
        .kw(1)
        .kc(kc)
        .x_stride(257)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      tester
        .kh(1)
        .kw(ks)
        .kc(kc)
        .x_stride(257)
        .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, small_n) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
      for (size_t kc = 1; kc < 5; kc++) {
        MaxPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(ks)
          .kw(ks)
          .kc(kc)
          .iterations(3)
          .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, small_n_with_x_stride) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
      for (size_t kc = 1; kc < 5; kc++) {
        MaxPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(ks)
          .kw(ks)
          .kc(kc)
          .x_stride(101)
          .iterations(1)
          .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, small_n_with_y_stride) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5, 10}}) {
      for (size_t kc = 1; kc < 5; kc++) {
        MaxPoolMicrokernelTester()
          .mr(9)
          .qr(8)
          .n(n)
          .kh(ks)
          .kw(ks)
          .kc(kc)
          .y_stride(103)
          .iterations(1)
          .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
      }
    }
  }
}

TEST(SMAXPOOL_9P8Q__SCALAR, small_n_with_s) {
  for (size_t n = 2; n < 5; n++) {
    for (size_t ks : std::vector<size_t>{{2, 3, 5}}) {
      for (size_t kc = 1; kc < 5; kc++) {
        for (size_t s = 2; s <= ks; s++) {
          MaxPoolMicrokernelTester()
            .mr(9)
            .qr(8)
            .n(n)
            .kh(ks)
            .kw(ks)
            .kc(kc)
            .s(s)
            .iterations(1)
            .Test(xnn_f32_maxpool_ukernel_9p8q__scalar, MaxPoolMicrokernelTester::Variant::Scalar);
        }
      }
    }
  }
}
