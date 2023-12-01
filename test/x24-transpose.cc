// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x24-transpose.yaml
//   Generator: tools/generate-transpose-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/transpose.h>
#include "transpose-microkernel-tester.h"


TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_1_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x2_scalar);
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_1_2_bw_1_4) {
  for (size_t i = 1; i <= 2; ++i) {
    for (size_t j = 1; j <= 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__1x2_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_1_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x2_scalar);
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_1_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(1)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__1x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_2_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__1x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(7)
    .block_width(2)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x2_scalar);
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_2_2_bw_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(19)
      .output_stride(i)
      .block_width(5)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__1x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_2_2_bw_4) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__1x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_2_2_bw_3_4) {
  for (size_t i = 2; i < 2; ++i) {
    for (size_t j = 3; j < 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__1x2_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_1_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x2_scalar);
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_1_bw_2_os_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x2_scalar);
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_1_bw_2_is_4_os_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x2_scalar);
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_17_bw_38_ies_14) {
  TransposeMicrokernelTester()
    .input_stride(38)
    .output_stride(17)
    .block_width(38)
    .block_height(17)
    .element_size(3)
    .input_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x2_scalar);
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_3_bw_10_oes_14) {
  TransposeMicrokernelTester()
    .input_stride(10)
    .output_stride(3)
    .block_width(10)
    .block_height(3)
    .element_size(3)
    .output_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x2_scalar);
}

TEST(X24_TRANSPOSEC__1X2_SCALAR_3, bh_7_bw_46_ies_20_oes_16) {
  TransposeMicrokernelTester()
    .input_stride(51)
    .output_stride(13)
    .block_width(46)
    .block_height(7)
    .element_size(3)
    .input_element_stride(20)
    .output_element_stride(16)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x2_scalar);
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_1_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x4_scalar);
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_1_2_bw_1_8) {
  for (size_t i = 1; i <= 2; ++i) {
    for (size_t j = 1; j <= 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__1x4_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_1_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(1)
    .block_width(8)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x4_scalar);
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_1_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(1)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__1x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_2_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__1x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(7)
    .block_width(4)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x4_scalar);
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_2_2_bw_4) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(21)
      .output_stride(i)
      .block_width(7)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__1x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_2_2_bw_8) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__1x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_2_2_bw_5_8) {
  for (size_t i = 2; i < 2; ++i) {
    for (size_t j = 5; j < 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__1x4_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_1_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x4_scalar);
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_1_bw_4_os_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x4_scalar);
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_1_bw_4_is_8_os_2) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x4_scalar);
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_17_bw_76_ies_14) {
  TransposeMicrokernelTester()
    .input_stride(76)
    .output_stride(17)
    .block_width(76)
    .block_height(17)
    .element_size(3)
    .input_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x4_scalar);
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_3_bw_20_oes_14) {
  TransposeMicrokernelTester()
    .input_stride(20)
    .output_stride(3)
    .block_width(20)
    .block_height(3)
    .element_size(3)
    .output_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x4_scalar);
}

TEST(X24_TRANSPOSEC__1X4_SCALAR_3, bh_7_bw_92_ies_20_oes_16) {
  TransposeMicrokernelTester()
    .input_stride(97)
    .output_stride(13)
    .block_width(92)
    .block_height(7)
    .element_size(3)
    .input_element_stride(20)
    .output_element_stride(16)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__1x4_scalar);
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_2_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x1_scalar);
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_1_4_bw_1_2) {
  for (size_t i = 1; i <= 4; ++i) {
    for (size_t j = 1; j <= 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__2x1_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x1_scalar);
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_2_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x1_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_4_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x1_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_4_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(10)
    .block_width(1)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x1_scalar);
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_3_4_bw_1) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(18)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x1_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_3_4_bw_2) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x1_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_3_4_bw_2_2) {
  for (size_t i = 3; i < 4; ++i) {
    for (size_t j = 2; j < 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__2x1_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_2_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x1_scalar);
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_2_bw_1_os_4) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x1_scalar);
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_2_bw_1_is_2_os_4) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x1_scalar);
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_34_bw_19_ies_14) {
  TransposeMicrokernelTester()
    .input_stride(19)
    .output_stride(34)
    .block_width(19)
    .block_height(34)
    .element_size(3)
    .input_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x1_scalar);
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_6_bw_5_oes_14) {
  TransposeMicrokernelTester()
    .input_stride(5)
    .output_stride(6)
    .block_width(5)
    .block_height(6)
    .element_size(3)
    .output_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x1_scalar);
}

TEST(X24_TRANSPOSEC__2X1_SCALAR_3, bh_14_bw_23_ies_20_oes_16) {
  TransposeMicrokernelTester()
    .input_stride(28)
    .output_stride(20)
    .block_width(23)
    .block_height(14)
    .element_size(3)
    .input_element_stride(20)
    .output_element_stride(16)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x1_scalar);
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x2_scalar);
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_1_4_bw_1_4) {
  for (size_t i = 1; i <= 4; ++i) {
    for (size_t j = 1; j <= 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__2x2_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x2_scalar);
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_2_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_4_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(10)
    .block_width(2)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x2_scalar);
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_3_4_bw_2) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(19)
      .output_stride(i)
      .block_width(5)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_3_4_bw_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_3_4_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    for (size_t j = 3; j < 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__2x2_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_2_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x2_scalar);
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_2_bw_2_os_4) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x2_scalar);
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_2_bw_2_is_4_os_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x2_scalar);
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_34_bw_38_ies_14) {
  TransposeMicrokernelTester()
    .input_stride(38)
    .output_stride(34)
    .block_width(38)
    .block_height(34)
    .element_size(3)
    .input_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x2_scalar);
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_6_bw_10_oes_14) {
  TransposeMicrokernelTester()
    .input_stride(10)
    .output_stride(6)
    .block_width(10)
    .block_height(6)
    .element_size(3)
    .output_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x2_scalar);
}

TEST(X24_TRANSPOSEC__2X2_SCALAR_3, bh_14_bw_46_ies_20_oes_16) {
  TransposeMicrokernelTester()
    .input_stride(51)
    .output_stride(20)
    .block_width(46)
    .block_height(14)
    .element_size(3)
    .input_element_stride(20)
    .output_element_stride(16)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x2_scalar);
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x4_scalar);
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_1_4_bw_1_8) {
  for (size_t i = 1; i <= 4; ++i) {
    for (size_t j = 1; j <= 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__2x4_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_2_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(8)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x4_scalar);
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_2_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_4_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(10)
    .block_width(4)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x4_scalar);
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_3_4_bw_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(21)
      .output_stride(i)
      .block_width(7)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_3_4_bw_8) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_3_4_bw_5_8) {
  for (size_t i = 3; i < 4; ++i) {
    for (size_t j = 5; j < 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__2x4_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_2_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x4_scalar);
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_2_bw_4_os_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x4_scalar);
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_2_bw_4_is_8_os_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x4_scalar);
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_34_bw_76_ies_14) {
  TransposeMicrokernelTester()
    .input_stride(76)
    .output_stride(34)
    .block_width(76)
    .block_height(34)
    .element_size(3)
    .input_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x4_scalar);
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_6_bw_20_oes_14) {
  TransposeMicrokernelTester()
    .input_stride(20)
    .output_stride(6)
    .block_width(20)
    .block_height(6)
    .element_size(3)
    .output_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x4_scalar);
}

TEST(X24_TRANSPOSEC__2X4_SCALAR_3, bh_14_bw_92_ies_20_oes_16) {
  TransposeMicrokernelTester()
    .input_stride(97)
    .output_stride(20)
    .block_width(92)
    .block_height(14)
    .element_size(3)
    .input_element_stride(20)
    .output_element_stride(16)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__2x4_scalar);
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_4_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x1_scalar);
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_1_8_bw_1_2) {
  for (size_t i = 1; i <= 8; ++i) {
    for (size_t j = 1; j <= 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x1_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x1_scalar);
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_4_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x1_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_8_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x1_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_8_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(16)
    .block_width(1)
    .block_height(8)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x1_scalar);
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_5_8_bw_1) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(18)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x1_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_5_8_bw_2) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x1_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_5_8_bw_2_2) {
  for (size_t i = 5; i < 8; ++i) {
    for (size_t j = 2; j < 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x1_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_4_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x1_scalar);
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_4_bw_1_os_8) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x1_scalar);
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_4_bw_1_is_2_os_8) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x1_scalar);
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_68_bw_19_ies_14) {
  TransposeMicrokernelTester()
    .input_stride(19)
    .output_stride(68)
    .block_width(19)
    .block_height(68)
    .element_size(3)
    .input_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x1_scalar);
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_12_bw_5_oes_14) {
  TransposeMicrokernelTester()
    .input_stride(5)
    .output_stride(12)
    .block_width(5)
    .block_height(12)
    .element_size(3)
    .output_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x1_scalar);
}

TEST(X24_TRANSPOSEC__4X1_SCALAR_3, bh_28_bw_23_ies_20_oes_16) {
  TransposeMicrokernelTester()
    .input_stride(28)
    .output_stride(34)
    .block_width(23)
    .block_height(28)
    .element_size(3)
    .input_element_stride(20)
    .output_element_stride(16)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x1_scalar);
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x2_scalar);
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_1_8_bw_1_4) {
  for (size_t i = 1; i <= 8; ++i) {
    for (size_t j = 1; j <= 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x2_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x2_scalar);
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_4_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_8_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_8_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(16)
    .block_width(2)
    .block_height(8)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x2_scalar);
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_5_8_bw_2) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(19)
      .output_stride(i)
      .block_width(5)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_5_8_bw_4) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x2_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_5_8_bw_3_4) {
  for (size_t i = 5; i < 8; ++i) {
    for (size_t j = 3; j < 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x2_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_4_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x2_scalar);
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_4_bw_2_os_8) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x2_scalar);
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_4_bw_2_is_4_os_8) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x2_scalar);
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_68_bw_38_ies_14) {
  TransposeMicrokernelTester()
    .input_stride(38)
    .output_stride(68)
    .block_width(38)
    .block_height(68)
    .element_size(3)
    .input_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x2_scalar);
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_12_bw_10_oes_14) {
  TransposeMicrokernelTester()
    .input_stride(10)
    .output_stride(12)
    .block_width(10)
    .block_height(12)
    .element_size(3)
    .output_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x2_scalar);
}

TEST(X24_TRANSPOSEC__4X2_SCALAR_3, bh_28_bw_46_ies_20_oes_16) {
  TransposeMicrokernelTester()
    .input_stride(51)
    .output_stride(34)
    .block_width(46)
    .block_height(28)
    .element_size(3)
    .input_element_stride(20)
    .output_element_stride(16)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x2_scalar);
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x4_scalar);
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_1_8_bw_1_8) {
  for (size_t i = 1; i <= 8; ++i) {
    for (size_t j = 1; j <= 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x4_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_4_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(8)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x4_scalar);
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_4_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_8_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_8_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(16)
    .block_width(4)
    .block_height(8)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x4_scalar);
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_5_8_bw_4) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(21)
      .output_stride(i)
      .block_width(7)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_5_8_bw_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_scalar);
  }
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_5_8_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    for (size_t j = 5; j < 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x4_scalar);
    }
  }
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_4_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x4_scalar);
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_4_bw_4_os_8) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x4_scalar);
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_4_bw_4_is_8_os_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .element_size(3)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x4_scalar);
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_68_bw_76_ies_14) {
  TransposeMicrokernelTester()
    .input_stride(76)
    .output_stride(68)
    .block_width(76)
    .block_height(68)
    .element_size(3)
    .input_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x4_scalar);
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_12_bw_20_oes_14) {
  TransposeMicrokernelTester()
    .input_stride(20)
    .output_stride(12)
    .block_width(20)
    .block_height(12)
    .element_size(3)
    .output_element_stride(14)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x4_scalar);
}

TEST(X24_TRANSPOSEC__4X4_SCALAR_3, bh_28_bw_92_ies_20_oes_16) {
  TransposeMicrokernelTester()
    .input_stride(97)
    .output_stride(34)
    .block_width(92)
    .block_height(28)
    .element_size(3)
    .input_element_stride(20)
    .output_element_stride(16)
    .iterations(1)
    .Test(xnn_x24_transposec_ukernel__4x4_scalar);
}

#if (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ASSEMBLY
  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_2_bw_2) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(2)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_1_4_bw_1_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 4; ++i) {
      for (size_t j = 1; j <= 4; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(3)
          .iterations(1)
          .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
      }
    }
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_2_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(2)
      .block_width(4)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_2_bw_3_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 3; i < 4; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(2)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
    }
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_4_bw_3_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 3; i < 4; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(4)
        .block_width(i)
        .block_height(4)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
    }
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_4_bw_2) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(10)
      .block_width(2)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_3_4_bw_2) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 3; i < 4; ++i) {
      TransposeMicrokernelTester()
        .input_stride(19)
        .output_stride(i)
        .block_width(5)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
    }
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_3_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 3; i < 4; ++i) {
      TransposeMicrokernelTester()
        .input_stride(4)
        .output_stride(i)
        .block_width(4)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
    }
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_3_4_bw_3_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 3; i < 4; ++i) {
      for (size_t j = 3; j < 4; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(3)
          .iterations(1)
          .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
      }
    }
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_2_bw_2_is_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(2)
      .block_width(2)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_2_bw_2_os_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(4)
      .block_width(2)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_2_bw_2_is_4_os_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(4)
      .block_width(2)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_34_bw_38_ies_14) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(38)
      .output_stride(34)
      .block_width(38)
      .block_height(34)
      .element_size(3)
      .input_element_stride(14)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_6_bw_10_oes_14) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(10)
      .output_stride(6)
      .block_width(10)
      .block_height(6)
      .element_size(3)
      .output_element_stride(14)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
  }

  TEST(X24_TRANSPOSEC__2X2_NEON_TBL64_3, bh_14_bw_46_ies_20_oes_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(51)
      .output_stride(20)
      .block_width(46)
      .block_height(14)
      .element_size(3)
      .input_element_stride(20)
      .output_element_stride(16)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__2x2_neon_tbl64, xnn_init_x24_transpose_neon_tbl64_params);
  }
#endif  // (XNN_ARCH_ARM || XNN_ARCH_ARM64) && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(3)
          .iterations(1)
          .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
      }
    }
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
    }
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
    }
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(16)
      .block_width(4)
      .block_height(8)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_5_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(21)
        .output_stride(i)
        .block_width(7)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
    }
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_5_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
    }
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 5; j < 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(3)
          .iterations(1)
          .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
      }
    }
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_68_bw_76_ies_14) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(76)
      .output_stride(68)
      .block_width(76)
      .block_height(68)
      .element_size(3)
      .input_element_stride(14)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_12_bw_20_oes_14) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(20)
      .output_stride(12)
      .block_width(20)
      .block_height(12)
      .element_size(3)
      .output_element_stride(14)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
  }

  TEST(X24_TRANSPOSEC__4X4_AARCH64_NEON_TBL128_3, bh_28_bw_92_ies_20_oes_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(97)
      .output_stride(34)
      .block_width(92)
      .block_height(28)
      .element_size(3)
      .input_element_stride(20)
      .output_element_stride(16)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128, xnn_init_x24_transpose_neon_tbl128_params);
  }
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if (XNN_ARCH_X86 || XNN_ARCH_X86_64) && XNN_ENABLE_ASSEMBLY
  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_4_bw_4) {
    TEST_REQUIRES_X86_SSSE3;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_1_8_bw_1_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(3)
          .iterations(1)
          .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
      }
    }
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_4_bw_8) {
    TEST_REQUIRES_X86_SSSE3;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_4_bw_5_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
    }
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_8_bw_5_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
    }
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_8_bw_4) {
    TEST_REQUIRES_X86_SSSE3;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(16)
      .block_width(4)
      .block_height(8)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_5_8_bw_4) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(21)
        .output_stride(i)
        .block_width(7)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
    }
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_5_8_bw_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
    }
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_5_8_bw_5_8) {
    TEST_REQUIRES_X86_SSSE3;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 5; j < 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(3)
          .iterations(1)
          .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
      }
    }
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_4_bw_4_is_8) {
    TEST_REQUIRES_X86_SSSE3;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_4_bw_4_os_8) {
    TEST_REQUIRES_X86_SSSE3;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_X86_SSSE3;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(3)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_68_bw_76_ies_14) {
    TEST_REQUIRES_X86_SSSE3;
    TransposeMicrokernelTester()
      .input_stride(76)
      .output_stride(68)
      .block_width(76)
      .block_height(68)
      .element_size(3)
      .input_element_stride(14)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_12_bw_20_oes_14) {
    TEST_REQUIRES_X86_SSSE3;
    TransposeMicrokernelTester()
      .input_stride(20)
      .output_stride(12)
      .block_width(20)
      .block_height(12)
      .element_size(3)
      .output_element_stride(14)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
  }

  TEST(X24_TRANSPOSEC__4X4_SSSE3_3, bh_28_bw_92_ies_20_oes_16) {
    TEST_REQUIRES_X86_SSSE3;
    TransposeMicrokernelTester()
      .input_stride(97)
      .output_stride(34)
      .block_width(92)
      .block_height(28)
      .element_size(3)
      .input_element_stride(20)
      .output_element_stride(16)
      .iterations(1)
      .Test(xnn_x24_transposec_ukernel__4x4_ssse3, xnn_init_x24_transpose_ssse3_params);
  }
#endif  // (XNN_ARCH_X86 || XNN_ARCH_X86_64) && XNN_ENABLE_ASSEMBLY
