// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x16-transpose.yaml
//   Generator: tools/generate-transpose-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/transpose.h"
#include "transpose-microkernel-tester.h"


TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_1_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_1_2_bw_1_4) {
  for (size_t i = 1; i <= 2; ++i) {
    for (size_t j = 1; j <= 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_1_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_1_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(1)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_2_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(7)
    .block_width(2)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_2_2_bw_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(19)
      .output_stride(i)
      .block_width(5)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_2_2_bw_4) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_2_2_bw_3_4) {
  for (size_t i = 2; i < 2; ++i) {
    for (size_t j = 3; j < 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_1_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_1_bw_2_os_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_1_bw_2_is_4_os_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(1)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_17_bw_38_ies_13) {
  TransposeMicrokernelTester()
    .input_stride(38)
    .output_stride(17)
    .block_width(38)
    .block_height(17)
    .element_size(2)
    .input_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_3_bw_10_oes_13) {
  TransposeMicrokernelTester()
    .input_stride(10)
    .output_stride(3)
    .block_width(10)
    .block_height(3)
    .element_size(2)
    .output_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
}

TEST(X16_TRANSPOSEC__1X2_SCALAR_INT_2, bh_7_bw_46_ies_19_oes_15) {
  TransposeMicrokernelTester()
    .input_stride(51)
    .output_stride(13)
    .block_width(46)
    .block_height(7)
    .element_size(2)
    .input_element_stride(19)
    .output_element_stride(15)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x2_scalar_int);
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_1_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_1_2_bw_1_8) {
  for (size_t i = 1; i <= 2; ++i) {
    for (size_t j = 1; j <= 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_1_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(1)
    .block_width(8)
    .block_height(1)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_1_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(1)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_2_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(7)
    .block_width(4)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_2_2_bw_4) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(21)
      .output_stride(i)
      .block_width(7)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_2_2_bw_8) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_2_2_bw_5_8) {
  for (size_t i = 2; i < 2; ++i) {
    for (size_t j = 5; j < 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_1_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(1)
    .block_width(4)
    .block_height(1)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_1_bw_4_os_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_1_bw_4_is_8_os_2) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(1)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_17_bw_76_ies_13) {
  TransposeMicrokernelTester()
    .input_stride(76)
    .output_stride(17)
    .block_width(76)
    .block_height(17)
    .element_size(2)
    .input_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_3_bw_20_oes_13) {
  TransposeMicrokernelTester()
    .input_stride(20)
    .output_stride(3)
    .block_width(20)
    .block_height(3)
    .element_size(2)
    .output_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
}

TEST(X16_TRANSPOSEC__1X4_SCALAR_INT_2, bh_7_bw_92_ies_19_oes_15) {
  TransposeMicrokernelTester()
    .input_stride(97)
    .output_stride(13)
    .block_width(92)
    .block_height(7)
    .element_size(2)
    .input_element_stride(19)
    .output_element_stride(15)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__1x4_scalar_int);
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_2_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_1_4_bw_1_2) {
  for (size_t i = 1; i <= 4; ++i) {
    for (size_t j = 1; j <= 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_2_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(2)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_4_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_4_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(10)
    .block_width(1)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_3_4_bw_1) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(18)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_3_4_bw_2) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_3_4_bw_2_2) {
  for (size_t i = 3; i < 4; ++i) {
    for (size_t j = 2; j < 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_2_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_2_bw_1_os_4) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_2_bw_1_is_2_os_4) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_34_bw_19_ies_13) {
  TransposeMicrokernelTester()
    .input_stride(19)
    .output_stride(34)
    .block_width(19)
    .block_height(34)
    .element_size(2)
    .input_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_6_bw_5_oes_13) {
  TransposeMicrokernelTester()
    .input_stride(5)
    .output_stride(6)
    .block_width(5)
    .block_height(6)
    .element_size(2)
    .output_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
}

TEST(X16_TRANSPOSEC__2X1_SCALAR_INT_2, bh_14_bw_23_ies_19_oes_15) {
  TransposeMicrokernelTester()
    .input_stride(28)
    .output_stride(20)
    .block_width(23)
    .block_height(14)
    .element_size(2)
    .input_element_stride(19)
    .output_element_stride(15)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x1_scalar_int);
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_2_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_1_4_bw_1_4) {
  for (size_t i = 1; i <= 4; ++i) {
    for (size_t j = 1; j <= 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_2_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(2)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_4_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(10)
    .block_width(2)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_3_4_bw_2) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(19)
      .output_stride(i)
      .block_width(5)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_3_4_bw_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_3_4_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    for (size_t j = 3; j < 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_2_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(2)
    .block_width(2)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_2_bw_2_os_4) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_2_bw_2_is_4_os_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_34_bw_38_ies_13) {
  TransposeMicrokernelTester()
    .input_stride(38)
    .output_stride(34)
    .block_width(38)
    .block_height(34)
    .element_size(2)
    .input_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_6_bw_10_oes_13) {
  TransposeMicrokernelTester()
    .input_stride(10)
    .output_stride(6)
    .block_width(10)
    .block_height(6)
    .element_size(2)
    .output_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
}

TEST(X16_TRANSPOSEC__2X2_SCALAR_INT_2, bh_14_bw_46_ies_19_oes_15) {
  TransposeMicrokernelTester()
    .input_stride(51)
    .output_stride(20)
    .block_width(46)
    .block_height(14)
    .element_size(2)
    .input_element_stride(19)
    .output_element_stride(15)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x2_scalar_int);
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_2_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_1_4_bw_1_8) {
  for (size_t i = 1; i <= 4; ++i) {
    for (size_t j = 1; j <= 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_2_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(8)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_2_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(2)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_4_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(4)
      .block_width(i)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(10)
    .block_width(4)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_3_4_bw_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(21)
      .output_stride(i)
      .block_width(7)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_3_4_bw_8) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_3_4_bw_5_8) {
  for (size_t i = 3; i < 4; ++i) {
    for (size_t j = 5; j < 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_2_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(2)
    .block_width(4)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_2_bw_4_os_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_2_bw_4_is_8_os_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(2)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_34_bw_76_ies_13) {
  TransposeMicrokernelTester()
    .input_stride(76)
    .output_stride(34)
    .block_width(76)
    .block_height(34)
    .element_size(2)
    .input_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_6_bw_20_oes_13) {
  TransposeMicrokernelTester()
    .input_stride(20)
    .output_stride(6)
    .block_width(20)
    .block_height(6)
    .element_size(2)
    .output_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
}

TEST(X16_TRANSPOSEC__2X4_SCALAR_INT_2, bh_14_bw_92_ies_19_oes_15) {
  TransposeMicrokernelTester()
    .input_stride(97)
    .output_stride(20)
    .block_width(92)
    .block_height(14)
    .element_size(2)
    .input_element_stride(19)
    .output_element_stride(15)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__2x4_scalar_int);
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_4_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_1_8_bw_1_2) {
  for (size_t i = 1; i <= 8; ++i) {
    for (size_t j = 1; j <= 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_4_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_8_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_8_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(16)
    .block_width(1)
    .block_height(8)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_5_8_bw_1) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(18)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_5_8_bw_2) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_5_8_bw_2_2) {
  for (size_t i = 5; i < 8; ++i) {
    for (size_t j = 2; j < 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_4_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(4)
    .block_width(1)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_4_bw_1_os_8) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_4_bw_1_is_2_os_8) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(1)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_68_bw_19_ies_13) {
  TransposeMicrokernelTester()
    .input_stride(19)
    .output_stride(68)
    .block_width(19)
    .block_height(68)
    .element_size(2)
    .input_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_12_bw_5_oes_13) {
  TransposeMicrokernelTester()
    .input_stride(5)
    .output_stride(12)
    .block_width(5)
    .block_height(12)
    .element_size(2)
    .output_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
}

TEST(X16_TRANSPOSEC__4X1_SCALAR_INT_2, bh_28_bw_23_ies_19_oes_15) {
  TransposeMicrokernelTester()
    .input_stride(28)
    .output_stride(34)
    .block_width(23)
    .block_height(28)
    .element_size(2)
    .input_element_stride(19)
    .output_element_stride(15)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x1_scalar_int);
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_4_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_1_8_bw_1_4) {
  for (size_t i = 1; i <= 8; ++i) {
    for (size_t j = 1; j <= 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_4_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_8_bw_3_4) {
  for (size_t i = 3; i < 4; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_8_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(16)
    .block_width(2)
    .block_height(8)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_5_8_bw_2) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(19)
      .output_stride(i)
      .block_width(5)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_5_8_bw_4) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_5_8_bw_3_4) {
  for (size_t i = 5; i < 8; ++i) {
    for (size_t j = 3; j < 4; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_4_bw_2_is_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(4)
    .block_width(2)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_4_bw_2_os_8) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_4_bw_2_is_4_os_8) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(2)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_68_bw_38_ies_13) {
  TransposeMicrokernelTester()
    .input_stride(38)
    .output_stride(68)
    .block_width(38)
    .block_height(68)
    .element_size(2)
    .input_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_12_bw_10_oes_13) {
  TransposeMicrokernelTester()
    .input_stride(10)
    .output_stride(12)
    .block_width(10)
    .block_height(12)
    .element_size(2)
    .output_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
}

TEST(X16_TRANSPOSEC__4X2_SCALAR_INT_2, bh_28_bw_46_ies_19_oes_15) {
  TransposeMicrokernelTester()
    .input_stride(51)
    .output_stride(34)
    .block_width(46)
    .block_height(28)
    .element_size(2)
    .input_element_stride(19)
    .output_element_stride(15)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x2_scalar_int);
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_4_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_1_8_bw_1_8) {
  for (size_t i = 1; i <= 8; ++i) {
    for (size_t j = 1; j <= 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_4_bw_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(8)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_4_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_8_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(8)
      .block_width(i)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_8_bw_4) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(16)
    .block_width(4)
    .block_height(8)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_5_8_bw_4) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(21)
      .output_stride(i)
      .block_width(7)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_5_8_bw_8) {
  for (size_t i = 5; i < 8; ++i) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(i)
      .block_width(8)
      .block_height(i)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
  }
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_5_8_bw_5_8) {
  for (size_t i = 5; i < 8; ++i) {
    for (size_t j = 5; j < 8; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
    }
  }
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_4_bw_4_is_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(4)
    .block_width(4)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_4_bw_4_os_8) {
  TransposeMicrokernelTester()
    .input_stride(4)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_4_bw_4_is_8_os_8) {
  TransposeMicrokernelTester()
    .input_stride(8)
    .output_stride(8)
    .block_width(4)
    .block_height(4)
    .element_size(2)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_68_bw_76_ies_13) {
  TransposeMicrokernelTester()
    .input_stride(76)
    .output_stride(68)
    .block_width(76)
    .block_height(68)
    .element_size(2)
    .input_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_12_bw_20_oes_13) {
  TransposeMicrokernelTester()
    .input_stride(20)
    .output_stride(12)
    .block_width(20)
    .block_height(12)
    .element_size(2)
    .output_element_stride(13)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
}

TEST(X16_TRANSPOSEC__4X4_SCALAR_INT_2, bh_28_bw_92_ies_19_oes_15) {
  TransposeMicrokernelTester()
    .input_stride(97)
    .output_stride(34)
    .block_width(92)
    .block_height(28)
    .element_size(2)
    .input_element_stride(19)
    .output_element_stride(15)
    .iterations(1)
    .Test(xnn_x16_transposec_ukernel__4x4_scalar_int);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_4_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_1_8_bw_1_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x8_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_4_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(4)
      .block_width(16)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_4_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x8_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_8_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x8_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_8_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_5_8_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x8_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_5_8_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x8_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_5_8_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x8_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_4_bw_8_is_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_4_bw_8_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_4_bw_8_is_16_os_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_68_bw_152_ies_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(68)
      .block_width(152)
      .block_height(68)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_12_bw_40_oes_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(12)
      .block_width(40)
      .block_height(12)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x8_sse2);
  }

  TEST(X16_TRANSPOSEC__4X8_SSE2_2, bh_28_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(34)
      .block_width(184)
      .block_height(28)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x8_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_8_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_8_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_8_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_9_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_9_16_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_SSE2_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_8_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_8_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_8_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_9_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_9_16_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_SSE2_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_8_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_8_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_8_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_9_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_9_16_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_SSE2_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_8_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_8_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_8_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_9_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_9_16_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_SSE2_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_8_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_8_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_8_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_9_16_bw_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_9_16_bw_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_SSE2_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_X86_SSE2;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_16_bw_16) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_1_32_bw_1_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 1; i <= 32; ++i) {
      for (size_t j = 1; j <= 32; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_16_bw_32) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(32)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_16_bw_17_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 17; i < 32; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(32)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_32_bw_17_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 17; i < 32; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(32)
        .block_width(i)
        .block_height(32)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_32_bw_16) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(52)
      .block_width(16)
      .block_height(32)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_17_32_bw_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 17; i < 32; ++i) {
      TransposeMicrokernelTester()
        .input_stride(33)
        .output_stride(i)
        .block_width(19)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_17_32_bw_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 17; i < 32; ++i) {
      TransposeMicrokernelTester()
        .input_stride(32)
        .output_stride(i)
        .block_width(32)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_17_32_bw_17_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 17; i < 32; ++i) {
      for (size_t j = 17; j < 32; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_16_bw_16_is_32) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_16_bw_16_os_32) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_16_bw_16_is_32_os_32) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_272_bw_304_ies_13) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(304)
      .output_stride(272)
      .block_width(304)
      .block_height(272)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_48_bw_80_oes_13) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(80)
      .output_stride(48)
      .block_width(80)
      .block_height(48)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_MOV_AVX2_2, bh_112_bw_368_ies_19_oes_15) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(373)
      .output_stride(118)
      .block_width(368)
      .block_height(112)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_16_bw_16) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_1_32_bw_1_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 1; i <= 32; ++i) {
      for (size_t j = 1; j <= 32; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_16_bw_32) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(32)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_16_bw_17_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 17; i < 32; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(32)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_32_bw_17_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 17; i < 32; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(32)
        .block_width(i)
        .block_height(32)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_32_bw_16) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(52)
      .block_width(16)
      .block_height(32)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_17_32_bw_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 17; i < 32; ++i) {
      TransposeMicrokernelTester()
        .input_stride(33)
        .output_stride(i)
        .block_width(19)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_17_32_bw_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 17; i < 32; ++i) {
      TransposeMicrokernelTester()
        .input_stride(32)
        .output_stride(i)
        .block_width(32)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_17_32_bw_17_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t i = 17; i < 32; ++i) {
      for (size_t j = 17; j < 32; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
      }
    }
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_16_bw_16_is_32) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(16)
      .block_width(16)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_16_bw_16_os_32) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_16_bw_16_is_32_os_32) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(32)
      .output_stride(32)
      .block_width(16)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_272_bw_304_ies_13) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(304)
      .output_stride(272)
      .block_width(304)
      .block_height(272)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_48_bw_80_oes_13) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(80)
      .output_stride(48)
      .block_width(80)
      .block_height(48)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
  }

  TEST(X16_TRANSPOSEC__16X16_REUSE_SWITCH_AVX2_2, bh_112_bw_368_ies_19_oes_15) {
    TEST_REQUIRES_X86_AVX2;
    TransposeMicrokernelTester()
      .input_stride(373)
      .output_stride(118)
      .block_width(368)
      .block_height(112)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_8_bw_8) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_1_16_bw_1_16) {
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_8_bw_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_8_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_16_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_16_bw_8) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_9_16_bw_8) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_9_16_bw_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_9_16_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_8_bw_8_is_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_8_bw_8_os_16) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_8_bw_8_is_16_os_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_136_bw_152_ies_13) {
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_24_bw_40_oes_13) {
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_WASMSIMD_2, bh_56_bw_184_ies_19_oes_15) {
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_8_bw_8) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_1_16_bw_1_16) {
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_8_bw_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_8_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_16_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_16_bw_8) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_9_16_bw_8) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_9_16_bw_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_9_16_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_8_bw_8_is_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_8_bw_8_os_16) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_8_bw_8_is_16_os_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_136_bw_152_ies_13) {
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_24_bw_40_oes_13) {
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_WASMSIMD_2, bh_56_bw_184_ies_19_oes_15) {
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_8_bw_8) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_1_16_bw_1_16) {
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_8_bw_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_8_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_16_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_16_bw_8) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_9_16_bw_8) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_9_16_bw_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_9_16_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_8_bw_8_is_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_8_bw_8_os_16) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_8_bw_8_is_16_os_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_136_bw_152_ies_13) {
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_24_bw_40_oes_13) {
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_WASMSIMD_2, bh_56_bw_184_ies_19_oes_15) {
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_8_bw_8) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_1_16_bw_1_16) {
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_8_bw_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_8_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_16_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_16_bw_8) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_9_16_bw_8) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_9_16_bw_16) {
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_9_16_bw_9_16) {
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_8_bw_8_is_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_8_bw_8_os_16) {
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_8_bw_8_is_16_os_16) {
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_136_bw_152_ies_13) {
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_24_bw_40_oes_13) {
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_WASMSIMD_2, bh_56_bw_184_ies_19_oes_15) {
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(16)
      .block_width(4)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_5_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(21)
        .output_stride(i)
        .block_width(7)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_5_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 5; j < 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_68_bw_76_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(76)
      .output_stride(68)
      .block_width(76)
      .block_height(68)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_12_bw_20_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(20)
      .output_stride(12)
      .block_width(20)
      .block_height(12)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_DEC_ZIP_NEON_2, bh_28_bw_92_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(97)
      .output_stride(34)
      .block_width(92)
      .block_height(28)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(16)
      .block_width(4)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_5_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(21)
        .output_stride(i)
        .block_width(7)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_5_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 5; j < 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_68_bw_76_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(76)
      .output_stride(68)
      .block_width(76)
      .block_height(68)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_12_bw_20_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(20)
      .output_stride(12)
      .block_width(20)
      .block_height(12)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MOV_ZIP_NEON_2, bh_28_bw_92_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(97)
      .output_stride(34)
      .block_width(92)
      .block_height(28)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(16)
      .block_width(4)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_5_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(21)
        .output_stride(i)
        .block_width(7)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_5_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 5; j < 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_68_bw_76_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(76)
      .output_stride(68)
      .block_width(76)
      .block_height(68)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_12_bw_20_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(20)
      .output_stride(12)
      .block_width(20)
      .block_height(12)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_MULTI_ZIP_NEON_2, bh_28_bw_92_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(97)
      .output_stride(34)
      .block_width(92)
      .block_height(28)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(16)
      .block_width(4)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_5_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(21)
        .output_stride(i)
        .block_width(7)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_5_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 5; j < 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_68_bw_76_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(76)
      .output_stride(68)
      .block_width(76)
      .block_height(68)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_12_bw_20_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(20)
      .output_stride(12)
      .block_width(20)
      .block_height(12)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_MULTI_SWITCH_ZIP_NEON_2, bh_28_bw_92_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(97)
      .output_stride(34)
      .block_width(92)
      .block_height(28)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(16)
      .block_width(4)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_5_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(21)
        .output_stride(i)
        .block_width(7)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_5_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 5; j < 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_68_bw_76_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(76)
      .output_stride(68)
      .block_width(76)
      .block_height(68)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_12_bw_20_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(20)
      .output_stride(12)
      .block_width(20)
      .block_height(12)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_DEC_ZIP_NEON_2, bh_28_bw_92_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(97)
      .output_stride(34)
      .block_width(92)
      .block_height(28)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(16)
      .block_width(4)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_5_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(21)
        .output_stride(i)
        .block_width(7)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_5_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 5; j < 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_68_bw_76_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(76)
      .output_stride(68)
      .block_width(76)
      .block_height(68)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_12_bw_20_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(20)
      .output_stride(12)
      .block_width(20)
      .block_height(12)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MOV_ZIP_NEON_2, bh_28_bw_92_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(97)
      .output_stride(34)
      .block_width(92)
      .block_height(28)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(16)
      .block_width(4)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_5_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(21)
        .output_stride(i)
        .block_width(7)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_5_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 5; j < 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_68_bw_76_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(76)
      .output_stride(68)
      .block_width(76)
      .block_height(68)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_12_bw_20_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(20)
      .output_stride(12)
      .block_width(20)
      .block_height(12)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_MULTI_ZIP_NEON_2, bh_28_bw_92_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(97)
      .output_stride(34)
      .block_width(92)
      .block_height(28)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_4_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_1_8_bw_1_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 8; ++i) {
      for (size_t j = 1; j <= 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_4_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(8)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_4_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(4)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(8)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(16)
      .block_width(4)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_5_8_bw_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(21)
        .output_stride(i)
        .block_width(7)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_5_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      TransposeMicrokernelTester()
        .input_stride(8)
        .output_stride(i)
        .block_width(8)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_5_8_bw_5_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 5; i < 8; ++i) {
      for (size_t j = 5; j < 8; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_4_bw_4_is_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(4)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_4_bw_4_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(4)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_4_bw_4_is_8_os_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(8)
      .block_width(4)
      .block_height(4)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_68_bw_76_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(76)
      .output_stride(68)
      .block_width(76)
      .block_height(68)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_12_bw_20_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(20)
      .output_stride(12)
      .block_width(20)
      .block_height(12)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__4X4_REUSE_SWITCH_ZIP_NEON_2, bh_28_bw_92_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(97)
      .output_stride(34)
      .block_width(92)
      .block_height(28)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_8_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_8_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_9_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_9_16_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_DEC_ZIP_NEON_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_8_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_8_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_9_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_9_16_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_MOV_ZIP_NEON_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_8_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_8_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_9_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_9_16_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_MULTI_SWITCH_ZIP_NEON_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_8_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_8_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_9_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_9_16_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_DEC_ZIP_NEON_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_8_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_8_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_9_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_9_16_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MOV_ZIP_NEON_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_8_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_8_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_9_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_9_16_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_MULTI_ZIP_NEON_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_8_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_1_16_bw_1_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 1; i <= 16; ++i) {
      for (size_t j = 1; j <= 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j * 3)
          .output_stride(i * 7)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_8_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(16)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_8_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(8)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(i)
        .output_stride(16)
        .block_width(i)
        .block_height(16)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(28)
      .block_width(8)
      .block_height(16)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_9_16_bw_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(25)
        .output_stride(i)
        .block_width(11)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_9_16_bw_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      TransposeMicrokernelTester()
        .input_stride(16)
        .output_stride(i)
        .block_width(16)
        .block_height(i)
        .element_size(2)
        .iterations(1)
        .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_9_16_bw_9_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t i = 9; i < 16; ++i) {
      for (size_t j = 9; j < 16; ++j) {
        TransposeMicrokernelTester()
          .input_stride(j)
          .output_stride(i)
          .block_width(j)
          .block_height(i)
          .element_size(2)
          .iterations(1)
          .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
      }
    }
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_8_bw_8_is_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(8)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_8_bw_8_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(8)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_8_bw_8_is_16_os_16) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(16)
      .output_stride(16)
      .block_width(8)
      .block_height(8)
      .element_size(2)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_136_bw_152_ies_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(152)
      .output_stride(136)
      .block_width(152)
      .block_height(136)
      .element_size(2)
      .input_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_24_bw_40_oes_13) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(40)
      .output_stride(24)
      .block_width(40)
      .block_height(24)
      .element_size(2)
      .output_element_stride(13)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
  }

  TEST(X16_TRANSPOSEC__8X8_REUSE_SWITCH_ZIP_NEON_2, bh_56_bw_184_ies_19_oes_15) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .input_stride(189)
      .output_stride(62)
      .block_width(184)
      .block_height(56)
      .element_size(2)
      .input_element_stride(19)
      .output_element_stride(15)
      .iterations(1)
      .Test(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
