// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/xx-transposev.yaml
//   Generator: tools/generate-transpose-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/transpose.h"
#include "transpose-microkernel-tester.h"


TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_1_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(1)
    .element_size(1)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_1_2_bw_1_2) {
  for (size_t i = 1; i <= 2; ++i) {
    for (size_t j = 1; j <= 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(1)
        .iterations(1)
        .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
    }
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_1_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .element_size(1)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_1_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(1)
      .element_size(1)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_2_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .element_size(1)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_2_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(7)
    .block_width(1)
    .block_height(2)
    .element_size(1)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_2_2_bw_1) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(18)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(1)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_2_2_bw_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .element_size(1)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_2_2_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    for (size_t j = 2; j < 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(1)
        .iterations(1)
        .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
    }
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_1_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(1)
    .block_width(1)
    .block_height(1)
    .element_size(1)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_1_bw_1_os_2) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(2)
    .block_width(1)
    .block_height(1)
    .element_size(1)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_1_bw_1_is_2_os_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(1)
    .element_size(1)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_17_bw_19_ies_12) {
  TransposeMicrokernelTester()
    .input_stride(19)
    .output_stride(17)
    .block_width(19)
    .block_height(17)
    .element_size(1)
    .input_element_stride(12)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_3_bw_5_oes_12) {
  TransposeMicrokernelTester()
    .input_stride(5)
    .output_stride(3)
    .block_width(5)
    .block_height(3)
    .element_size(1)
    .output_element_stride(12)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_1, bh_7_bw_23_ies_18_oes_14) {
  TransposeMicrokernelTester()
    .input_stride(28)
    .output_stride(13)
    .block_width(23)
    .block_height(7)
    .element_size(1)
    .input_element_stride(18)
    .output_element_stride(14)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}
TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_1_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_1_2_bw_1_2) {
  for (size_t i = 1; i <= 2; ++i) {
    for (size_t j = 1; j <= 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
    }
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_1_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_1_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(1)
      .element_size(3)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_2_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .element_size(3)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_2_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(7)
    .block_width(1)
    .block_height(2)
    .element_size(3)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_2_2_bw_1) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(18)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_2_2_bw_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .element_size(3)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_2_2_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    for (size_t j = 2; j < 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(3)
        .iterations(1)
        .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
    }
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_1_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(1)
    .block_width(1)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_1_bw_1_os_2) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(2)
    .block_width(1)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_1_bw_1_is_2_os_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(1)
    .element_size(3)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_17_bw_19_ies_14) {
  TransposeMicrokernelTester()
    .input_stride(19)
    .output_stride(17)
    .block_width(19)
    .block_height(17)
    .element_size(3)
    .input_element_stride(14)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_3_bw_5_oes_14) {
  TransposeMicrokernelTester()
    .input_stride(5)
    .output_stride(3)
    .block_width(5)
    .block_height(3)
    .element_size(3)
    .output_element_stride(14)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_3, bh_7_bw_23_ies_20_oes_16) {
  TransposeMicrokernelTester()
    .input_stride(28)
    .output_stride(13)
    .block_width(23)
    .block_height(7)
    .element_size(3)
    .input_element_stride(20)
    .output_element_stride(16)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}
TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_1_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(1)
    .element_size(5)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_1_2_bw_1_2) {
  for (size_t i = 1; i <= 2; ++i) {
    for (size_t j = 1; j <= 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j * 3)
        .output_stride(i * 7)
        .block_width(j)
        .block_height(i)
        .element_size(5)
        .iterations(1)
        .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
    }
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_1_bw_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(1)
    .block_width(2)
    .block_height(1)
    .element_size(5)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_1_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(1)
      .element_size(5)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_2_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(i)
      .output_stride(2)
      .block_width(i)
      .block_height(2)
      .element_size(5)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_2_bw_1) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(7)
    .block_width(1)
    .block_height(2)
    .element_size(5)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_2_2_bw_1) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(18)
      .output_stride(i)
      .block_width(4)
      .block_height(i)
      .element_size(5)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_2_2_bw_2) {
  for (size_t i = 2; i < 2; ++i) {
    TransposeMicrokernelTester()
      .input_stride(2)
      .output_stride(i)
      .block_width(2)
      .block_height(i)
      .element_size(5)
      .iterations(1)
      .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_2_2_bw_2_2) {
  for (size_t i = 2; i < 2; ++i) {
    for (size_t j = 2; j < 2; ++j) {
      TransposeMicrokernelTester()
        .input_stride(j)
        .output_stride(i)
        .block_width(j)
        .block_height(i)
        .element_size(5)
        .iterations(1)
        .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
    }
  }
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_1_bw_1_is_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(1)
    .block_width(1)
    .block_height(1)
    .element_size(5)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_1_bw_1_os_2) {
  TransposeMicrokernelTester()
    .input_stride(1)
    .output_stride(2)
    .block_width(1)
    .block_height(1)
    .element_size(5)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_1_bw_1_is_2_os_2) {
  TransposeMicrokernelTester()
    .input_stride(2)
    .output_stride(2)
    .block_width(1)
    .block_height(1)
    .element_size(5)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_17_bw_19_ies_16) {
  TransposeMicrokernelTester()
    .input_stride(19)
    .output_stride(17)
    .block_width(19)
    .block_height(17)
    .element_size(5)
    .input_element_stride(16)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_3_bw_5_oes_16) {
  TransposeMicrokernelTester()
    .input_stride(5)
    .output_stride(3)
    .block_width(5)
    .block_height(3)
    .element_size(5)
    .output_element_stride(16)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}

TEST(XX_TRANSPOSEV__1X1_SCALAR_MEMCPY_5, bh_7_bw_23_ies_22_oes_18) {
  TransposeMicrokernelTester()
    .input_stride(28)
    .output_stride(13)
    .block_width(23)
    .block_height(7)
    .element_size(5)
    .input_element_stride(22)
    .output_element_stride(18)
    .iterations(1)
    .Test(xnn_xx_transposev_ukernel__1x1_scalar_memcpy);
}