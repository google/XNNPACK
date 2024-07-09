// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x32-packb.yaml
//   Generator: tools/generate-packb-test.py


#include <cstddef>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packb.h"
#include "packb-microkernel-tester.h"


TEST(X32_PACKB_GEMM_2C1S1R__SCALAR_FLOAT, n_eq_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(2)
      .kernel_tile(k)
      .channel_tile(2)
      .channel_subtile(1)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_2C1S1R__SCALAR_FLOAT, n_div_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(4)
      .kernel_tile(k)
      .channel_tile(2)
      .channel_subtile(1)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_2C1S1R__SCALAR_FLOAT, n_lt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 2; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(2)
        .channel_subtile(1)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_2C1S1R__SCALAR_FLOAT, n_gt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 3; n < 4; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(2)
        .channel_subtile(1)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_2C1S1R__SCALAR_FLOAT, groups_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = 3; n < 4; n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(2)
          .channel_subtile(1)
          .channel_round(1)
          .Test(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_float);
      }
    }
  }
}

TEST(X32_PACKB_GEMM_2C1S1R__SCALAR_INT, n_eq_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(2)
      .kernel_tile(k)
      .channel_tile(2)
      .channel_subtile(1)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_2C1S1R__SCALAR_INT, n_div_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(4)
      .kernel_tile(k)
      .channel_tile(2)
      .channel_subtile(1)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_2C1S1R__SCALAR_INT, n_lt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 2; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(2)
        .channel_subtile(1)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_2C1S1R__SCALAR_INT, n_gt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 3; n < 4; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(2)
        .channel_subtile(1)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_2C1S1R__SCALAR_INT, groups_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = 3; n < 4; n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(2)
          .channel_subtile(1)
          .channel_round(1)
          .Test(xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_int);
      }
    }
  }
}

TEST(X32_PACKB_GEMM_2C2S1R__SCALAR_FLOAT, n_eq_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(2)
      .kernel_tile(k)
      .channel_tile(2)
      .channel_subtile(2)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_2C2S1R__SCALAR_FLOAT, n_div_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(4)
      .kernel_tile(k)
      .channel_tile(2)
      .channel_subtile(2)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_2C2S1R__SCALAR_FLOAT, n_lt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 2; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(2)
        .channel_subtile(2)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_2C2S1R__SCALAR_FLOAT, n_gt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 3; n < 4; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(2)
        .channel_subtile(2)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_2C2S1R__SCALAR_FLOAT, groups_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = 3; n < 4; n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(2)
          .channel_subtile(2)
          .channel_round(1)
          .Test(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_float);
      }
    }
  }
}

TEST(X32_PACKB_GEMM_2C2S1R__SCALAR_INT, n_eq_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(2)
      .kernel_tile(k)
      .channel_tile(2)
      .channel_subtile(2)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_2C2S1R__SCALAR_INT, n_div_2) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(4)
      .kernel_tile(k)
      .channel_tile(2)
      .channel_subtile(2)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_2C2S1R__SCALAR_INT, n_lt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 2; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(2)
        .channel_subtile(2)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_2C2S1R__SCALAR_INT, n_gt_2) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 3; n < 4; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(2)
        .channel_subtile(2)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_2C2S1R__SCALAR_INT, groups_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = 3; n < 4; n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(2)
          .channel_subtile(2)
          .channel_round(1)
          .Test(xnn_x32_packb_gemm_ukernel_2c2s1r__scalar_int);
      }
    }
  }
}

TEST(X32_PACKB_GEMM_4C1S1R__SCALAR_FLOAT, n_eq_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(4)
      .kernel_tile(k)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_4C1S1R__SCALAR_FLOAT, n_div_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(8)
      .kernel_tile(k)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_4C1S1R__SCALAR_FLOAT, n_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 4; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_4C1S1R__SCALAR_FLOAT, n_gt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 5; n < 8; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_4C1S1R__SCALAR_FLOAT, groups_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = 5; n < 8; n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(4)
          .channel_subtile(1)
          .channel_round(1)
          .Test(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_float);
      }
    }
  }
}

TEST(X32_PACKB_GEMM_4C1S1R__SCALAR_INT, n_eq_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(4)
      .kernel_tile(k)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_4C1S1R__SCALAR_INT, n_div_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(8)
      .kernel_tile(k)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_4C1S1R__SCALAR_INT, n_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 4; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_4C1S1R__SCALAR_INT, n_gt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 5; n < 8; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_4C1S1R__SCALAR_INT, groups_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = 5; n < 8; n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(4)
          .channel_subtile(1)
          .channel_round(1)
          .Test(xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_int);
      }
    }
  }
}

TEST(X32_PACKB_GEMM_4C4S1R__SCALAR_FLOAT, n_eq_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(4)
      .kernel_tile(k)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_4C4S1R__SCALAR_FLOAT, n_div_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(8)
      .kernel_tile(k)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_float);
  }
}

TEST(X32_PACKB_GEMM_4C4S1R__SCALAR_FLOAT, n_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 4; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_4C4S1R__SCALAR_FLOAT, n_gt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 5; n < 8; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_float);
    }
  }
}

TEST(X32_PACKB_GEMM_4C4S1R__SCALAR_FLOAT, groups_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = 5; n < 8; n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(1)
          .Test(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_float);
      }
    }
  }
}

TEST(X32_PACKB_GEMM_4C4S1R__SCALAR_INT, n_eq_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(4)
      .kernel_tile(k)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_4C4S1R__SCALAR_INT, n_div_4) {
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(8)
      .kernel_tile(k)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(1)
      .Test(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_int);
  }
}

TEST(X32_PACKB_GEMM_4C4S1R__SCALAR_INT, n_lt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < 4; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_4C4S1R__SCALAR_INT, n_gt_4) {
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 5; n < 8; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(1)
        .Test(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_int);
    }
  }
}

TEST(X32_PACKB_GEMM_4C4S1R__SCALAR_INT, groups_gt_1) {
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = 5; n < 8; n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(1)
          .Test(xnn_x32_packb_gemm_ukernel_4c4s1r__scalar_int);
      }
    }
  }
}