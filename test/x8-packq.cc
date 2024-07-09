// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x8-packq.yaml
//   Generator: tools/generate-packq-test.py


#include <cstddef>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packq.h"
#include "packq-microkernel-tester.h"


namespace xnnpack {

TEST(X8_PACKQ_F32QP8__SCALAR_U1, k_div_kr_m_div_mr) {
  for (size_t kr = 1; kr <= 4; kr++) {
    for (size_t mr = 1; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(mr * 10)
          .k(kr * 10)
          .mr(mr)
          .kr(kr)
          .Test(xnn_x8_packq_f32qp8_ukernel__scalar_u1);
    }
  }
}

TEST(X8_PACKQ_F32QP8__SCALAR_U1, k_div_kr_m_div_mr_kr_div_sr) {
  for (size_t sr = 1; sr <= 4; sr++) {
    for (size_t kr = sr; kr <= 4 * sr; kr += sr) {
      for (size_t mr = 1; mr <= 4; mr++) {
        PackQMicrokernelTester()
            .m(mr * 10)
            .k(kr * 10)
            .mr(mr)
            .kr(kr)
            .sr(sr)
            .Test(xnn_x8_packq_f32qp8_ukernel__scalar_u1);
      }
    }
  }
}

TEST(X8_PACKQ_F32QP8__SCALAR_U1, k_div_kr_m_lt_mr) {
  for (size_t kr = 1; kr <= 4; kr++) {
    for (size_t mr = 2; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(mr - 1)
          .k(kr * 10)
          .mr(mr)
          .kr(kr)
          .Test(xnn_x8_packq_f32qp8_ukernel__scalar_u1);
    }
  }
}

TEST(X8_PACKQ_F32QP8__SCALAR_U1, k_div_kr_m_gt_mr) {
  for (size_t kr = 1; kr <= 4; kr++) {
    for (size_t mr = 2; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(2 * mr + 1)
          .k(kr * 10)
          .mr(mr)
          .kr(kr)
          .Test(xnn_x8_packq_f32qp8_ukernel__scalar_u1);
    }
  }
}

#if XNN_ARCH_ARM64
  #if XNN_ENABLE_KLEIDIAI
  TEST(X8_PACKQ_F32QP8__AARCH64_NEON_U2, k_div_kr_m_div_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t kr = 1; kr <= 4; kr++) {
      for (size_t mr = 1; mr <= 4; mr++) {
        PackQMicrokernelTester()
            .m(mr * 20)
            .k(kr * 20)
            .mr(mr)
            .kr(kr)
            .Test(xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2);
      }
    }
  }

  TEST(X8_PACKQ_F32QP8__AARCH64_NEON_U2, k_div_kr_m_div_mr_kr_div_sr) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t sr = 1; sr <= 4; sr++) {
      for (size_t kr = sr; kr <= 4 * sr; kr += sr) {
        for (size_t mr = 1; mr <= 4; mr++) {
          PackQMicrokernelTester()
              .m(mr * 20)
              .k(kr * 20)
              .mr(mr)
              .kr(kr)
              .sr(sr)
              .Test(xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2);
        }
      }
    }
  }

  TEST(X8_PACKQ_F32QP8__AARCH64_NEON_U2, k_div_kr_m_lt_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t kr = 1; kr <= 4; kr++) {
      for (size_t mr = 2; mr <= 4; mr++) {
        PackQMicrokernelTester()
            .m(mr - 1)
            .k(kr * 20)
            .mr(mr)
            .kr(kr)
            .Test(xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2);
      }
    }
  }

  TEST(X8_PACKQ_F32QP8__AARCH64_NEON_U2, k_div_kr_m_gt_mr) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t kr = 1; kr <= 4; kr++) {
      for (size_t mr = 2; mr <= 4; mr++) {
        PackQMicrokernelTester()
            .m(2 * mr + 1)
            .k(kr * 20)
            .mr(mr)
            .kr(kr)
            .Test(xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2);
      }
    }
  }
  #endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ARCH_ARM64


};  // namespace xnnpack
