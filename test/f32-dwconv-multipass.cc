// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-dwconv-multipass.yaml
//   Generator: tools/generate-dwconv-multipass-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/dwconv.h>
#include "dwconv-microkernel-tester.h"


TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(3)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(4)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, c_eq_1_multipass) {
  for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      for (size_t step = 2; step <= 2; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
      }
    }
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(3)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(4)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_multipass) {
  for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      for (size_t step = 2; step <= 2; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
      }
    }
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_2F2M2L1C1S1R__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_eq_4_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(4)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(3)
    .channels(4)
    .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_eq_4_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(4)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(4)
    .channels(4)
    .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_eq_4_multipass) {
  for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_div_4_first_pass_plus_one) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_div_4_first_pass_and_last_pass) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_div_4_multipass) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_gt_4_first_pass_plus_one) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_gt_4_first_pass_and_last_pass) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_gt_4_multipass) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_eq_4_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_eq_4_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, c_eq_4_multipass_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      for (size_t step = 2; step <= 2; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
      }
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR, input_offset) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(4)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(3)
    .channels(4)
    .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(2)
    .middle_pass_tile(2)
    .last_pass_tile(2)
    .channel_tile(4)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(4)
    .channels(4)
    .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_multipass) {
  for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_div_4_first_pass_plus_one) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_div_4_first_pass_and_last_pass) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_div_4_multipass) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_gt_4_first_pass_plus_one) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_gt_4_first_pass_and_last_pass) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_gt_4_multipass) {
  for (uint32_t channels = 5; channels < 8; channels++) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(3)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    DWConvMicrokernelTester()
      .first_pass_tile(2)
      .middle_pass_tile(2)
      .last_pass_tile(2)
      .channel_tile(4)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(4)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, c_eq_4_multipass_multipixel) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      for (size_t step = 2; step <= 2; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(2)
          .middle_pass_tile(2)
          .last_pass_tile(2)
          .channel_tile(4)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
      }
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 20; channels += 3) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(23)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_2F2M2L4C1S1R__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 8; channels < 64; channels += 12) {
    for (uint32_t kernel_size = 6; kernel_size < 8; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(2)
        .middle_pass_tile(2)
        .last_pass_tile(2)
        .channel_tile(4)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(112)
        .Test(xnn_f32_dwconv_ukernel_2f2m2l4c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(5)
    .middle_pass_tile(5)
    .last_pass_tile(5)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(6)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(5)
    .middle_pass_tile(5)
    .last_pass_tile(5)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(10)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, c_eq_1_multipass) {
  for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(6)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(10)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(6)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(10)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      for (size_t step = 2; step <= 5; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
      }
    }
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(5)
    .middle_pass_tile(5)
    .last_pass_tile(5)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(6)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(5)
    .middle_pass_tile(5)
    .last_pass_tile(5)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(10)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_multipass) {
  for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(6)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(10)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(6)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(10)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      for (size_t step = 2; step <= 5; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
      }
    }
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_5F5M5L1C1S1R__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(6)
    .middle_pass_tile(6)
    .last_pass_tile(7)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(7)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(6)
    .middle_pass_tile(6)
    .last_pass_tile(7)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(13)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, c_eq_1_multipass) {
  for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(7)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(13)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(7)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(13)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      for (size_t step = 2; step <= 6; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
      }
    }
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(6)
    .middle_pass_tile(6)
    .last_pass_tile(7)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(7)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(6)
    .middle_pass_tile(6)
    .last_pass_tile(7)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(13)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_multipass) {
  for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(7)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(13)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(7)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(6)
      .middle_pass_tile(6)
      .last_pass_tile(7)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(13)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      for (size_t step = 2; step <= 6; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(6)
          .middle_pass_tile(6)
          .last_pass_tile(7)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
      }
    }
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_6F6M7L1C1S1R__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 19; kernel_size < 25; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(6)
        .middle_pass_tile(6)
        .last_pass_tile(7)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_ukernel_6f6m7l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(8)
    .middle_pass_tile(8)
    .last_pass_tile(9)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(9)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(8)
    .middle_pass_tile(8)
    .last_pass_tile(9)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(17)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, c_eq_1_multipass) {
  for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(17)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(17)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      for (size_t step = 2; step <= 8; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
      }
    }
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar);
    }
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one) {
  DWConvMicrokernelTester()
    .first_pass_tile(8)
    .middle_pass_tile(8)
    .last_pass_tile(9)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(9)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass) {
  DWConvMicrokernelTester()
    .first_pass_tile(8)
    .middle_pass_tile(8)
    .last_pass_tile(9)
    .channel_tile(1)
    .channel_subtile(1)
    .channel_round(1)
    .kernel_size(17)
    .channels(1)
    .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_multipass) {
  for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(kernel_size)
      .channels(1)
      .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_plus_one) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(9)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, c_gt_1_first_pass_and_last_pass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(17)
      .channels(channels)
      .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, c_gt_1_multipass) {
  for (uint32_t channels = 2; channels < 10; channels++) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_plus_one_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(9)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_first_pass_and_last_pass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    DWConvMicrokernelTester()
      .first_pass_tile(8)
      .middle_pass_tile(8)
      .last_pass_tile(9)
      .channel_tile(1)
      .channel_subtile(1)
      .channel_round(1)
      .kernel_size(17)
      .channels(channels)
      .width(3)
      .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, c_eq_1_multipass_multipixel) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, multipixel_with_step) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      for (size_t step = 2; step <= 8; step++) {
        DWConvMicrokernelTester()
          .first_pass_tile(8)
          .middle_pass_tile(8)
          .last_pass_tile(9)
          .channel_tile(1)
          .channel_subtile(1)
          .channel_round(1)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .step(step)
          .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
      }
    }
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, multipixel_with_output_stride) {
  for (size_t channels = 1; channels <= 5; channels += 1) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .width(5)
        .output_stride(7)
        .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
    }
  }
}

TEST(F32_DWCONV_8F8M9L1C1S1R__SCALAR_ACC2, input_offset) {
  for (uint32_t channels = 2; channels < 16; channels += 3) {
    for (uint32_t kernel_size = 25; kernel_size < 33; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(8)
        .middle_pass_tile(8)
        .last_pass_tile(9)
        .channel_tile(1)
        .channel_subtile(1)
        .channel_round(1)
        .kernel_size(kernel_size)
        .channels(channels)
        .input_offset(48)
        .Test(xnn_f32_dwconv_ukernel_8f8m9l1c1s1r__scalar_acc2);
    }
  }
}

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
        }
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
        }
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMSIMD_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmsimd_acc2);
      }
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
        }
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_first_pass_plus_one) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(6)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_first_pass_and_last_pass) {
    DWConvMicrokernelTester()
      .first_pass_tile(5)
      .middle_pass_tile(5)
      .last_pass_tile(5)
      .channel_tile(4)
      .channel_subtile(4)
      .channel_round(4)
      .kernel_size(10)
      .channels(4)
      .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_multipass) {
    for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(kernel_size)
        .channels(4)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_div_4_first_pass_plus_one) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_div_4_first_pass_and_last_pass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_div_4_multipass) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_gt_4_first_pass_plus_one) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_gt_4_first_pass_and_last_pass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_gt_4_multipass) {
    for (uint32_t channels = 5; channels < 8; channels++) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_first_pass_plus_one_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(6)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_first_pass_and_last_pass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      DWConvMicrokernelTester()
        .first_pass_tile(5)
        .middle_pass_tile(5)
        .last_pass_tile(5)
        .channel_tile(4)
        .channel_subtile(4)
        .channel_round(4)
        .kernel_size(10)
        .channels(channels)
        .width(3)
        .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, c_eq_4_multipass_multipixel) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(3)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, multipixel_with_step) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        for (size_t step = 2; step <= 5; step++) {
          DWConvMicrokernelTester()
            .first_pass_tile(5)
            .middle_pass_tile(5)
            .last_pass_tile(5)
            .channel_tile(4)
            .channel_subtile(4)
            .channel_round(4)
            .kernel_size(kernel_size)
            .channels(channels)
            .width(3)
            .step(step)
            .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
        }
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, multipixel_with_output_stride) {
    for (size_t channels = 1; channels <= 20; channels += 3) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .width(5)
          .output_stride(23)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
      }
    }
  }

  TEST(F32_DWCONV_5F5M5L4C4S4R__WASMRELAXEDSIMD_FMA_ACC2, input_offset) {
    for (uint32_t channels = 8; channels < 64; channels += 12) {
      for (uint32_t kernel_size = 15; kernel_size < 20; kernel_size++) {
        DWConvMicrokernelTester()
          .first_pass_tile(5)
          .middle_pass_tile(5)
          .last_pass_tile(5)
          .channel_tile(4)
          .channel_subtile(4)
          .channel_round(4)
          .kernel_size(kernel_size)
          .channels(channels)
          .input_offset(112)
          .Test(xnn_f32_dwconv_ukernel_5f5m5l4c4s4r__wasmrelaxedsimd_fma_acc2);
      }
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD
