// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qs16-vmul.yaml
//   Generator: tools/generate-vbinary-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"
#include "vbinary-microkernel-tester.h"


TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, batch_eq_1) {
  VBinaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, batch_gt_1) {
  for (size_t batch_size = 2; batch_size < 10; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, a_zero_point) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .a_zero_point(a_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, b_zero_point) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .b_zero_point(b_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, y_zero_point) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .y_zero_point(y_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, a_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .a_scale(a_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, b_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .b_scale(b_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, y_scale) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .y_scale(y_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, qmin) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U1, qmax) {
  for (size_t batch_size = 1; batch_size <= 5; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u1, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, batch_eq_2) {
  VBinaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, batch_div_2) {
  for (size_t batch_size = 4; batch_size < 20; batch_size += 2) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, batch_lt_2) {
  for (size_t batch_size = 1; batch_size < 2; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, batch_gt_2) {
  for (size_t batch_size = 3; batch_size < 4; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, a_zero_point) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .a_zero_point(a_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, b_zero_point) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .b_zero_point(b_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, y_zero_point) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .y_zero_point(y_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, a_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .a_scale(a_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, b_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .b_scale(b_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, y_scale) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .y_scale(y_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, qmin) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U2, qmax) {
  for (size_t batch_size = 1; batch_size <= 10; batch_size += 1) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u2, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, batch_eq_4) {
  VBinaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, batch_div_4) {
  for (size_t batch_size = 8; batch_size < 40; batch_size += 4) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, batch_lt_4) {
  for (size_t batch_size = 1; batch_size < 4; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, batch_gt_4) {
  for (size_t batch_size = 5; batch_size < 8; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, a_zero_point) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .a_zero_point(a_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, b_zero_point) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .b_zero_point(b_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, y_zero_point) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .y_zero_point(y_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, a_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .a_scale(a_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, b_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .b_scale(b_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, y_scale) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .y_scale(y_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, qmin) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U4, qmax) {
  for (size_t batch_size = 1; batch_size <= 20; batch_size += 3) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u4, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, batch_eq_8) {
  VBinaryMicrokernelTester()
    .batch_size(8)
    .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, batch_div_8) {
  for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, batch_lt_8) {
  for (size_t batch_size = 1; batch_size < 8; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, batch_gt_8) {
  for (size_t batch_size = 9; batch_size < 16; batch_size++) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, inplace_a) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, inplace_b) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_b(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, inplace_a_and_b) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace_a(true)
      .inplace_b(true)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, a_zero_point) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .a_zero_point(a_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, b_zero_point) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .b_zero_point(b_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, y_zero_point) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .y_zero_point(y_zero_point)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, a_scale) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .a_scale(a_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, b_scale) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .b_scale(b_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, y_scale) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .y_scale(y_scale)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, qmin) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmin(128)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

TEST(QS16_VMUL_MINMAX_FP32__SCALAR_U8, qmax) {
  for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
    VBinaryMicrokernelTester()
      .batch_size(batch_size)
      .qmax(128)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__scalar_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE41;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, inplace_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, inplace_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U8, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE41;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, inplace_a) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, inplace_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, inplace_a_and_b) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, a_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, b_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, y_zero_point) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, a_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, b_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, y_scale) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, qmin) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__SSE41_U16, qmax) {
    TEST_REQUIRES_X86_SSE41;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__sse41_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, inplace_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, inplace_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, a_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, b_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, y_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, a_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, b_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, y_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U16, qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VBinaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, inplace_a) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, inplace_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, a_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, b_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, y_zero_point) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, a_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, b_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, y_scale) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, qmin) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX2_U32, qmax) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx2_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VBinaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 64; batch_size < 320; batch_size += 32) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 33; batch_size < 64; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, inplace_a) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, inplace_b) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, a_zero_point) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, b_zero_point) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, y_zero_point) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, a_scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, b_scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, y_scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U32, qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 160; batch_size += 31) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u32, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VBinaryMicrokernelTester()
      .batch_size(64)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 128; batch_size < 640; batch_size += 64) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size < 64; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 65; batch_size < 128; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, inplace_a) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, inplace_b) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, inplace_a_and_b) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, a_zero_point) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, b_zero_point) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, y_zero_point) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, a_scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, b_scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, y_scale) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, qmin) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__AVX512BW_U64, qmax) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t batch_size = 1; batch_size <= 320; batch_size += 63) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__avx512bw_u64, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, batch_eq_8) {
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, batch_div_8) {
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, batch_lt_8) {
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, batch_gt_8) {
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, a_zero_point) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, b_zero_point) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, y_zero_point) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, a_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, b_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, y_scale) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, qmin) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U8, qmax) {
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, batch_eq_16) {
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, batch_div_16) {
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, batch_lt_16) {
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, batch_gt_16) {
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, inplace_a) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, inplace_b) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, inplace_a_and_b) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, a_zero_point) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, b_zero_point) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, y_zero_point) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, a_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, b_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, y_scale) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, qmin) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__WASMSIMD_U16, qmax) {
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__wasmsimd_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 16; batch_size < 80; batch_size += 8) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 8; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 9; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U8, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 40; batch_size += 7) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u8, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VBinaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 32; batch_size < 160; batch_size += 16) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size < 16; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 17; batch_size < 32; batch_size++) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, inplace_a) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, inplace_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, inplace_a_and_b) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace_a(true)
        .inplace_b(true)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, a_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t a_zero_point = -128; a_zero_point <= 127; a_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_zero_point(a_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, b_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t b_zero_point = -128; b_zero_point <= 127; b_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_zero_point(b_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, y_zero_point) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (int32_t y_zero_point = -128; y_zero_point <= 127; y_zero_point += 51) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_zero_point(y_zero_point)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, a_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float a_scale = 0.1f; a_scale <= 10.0f; a_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .a_scale(a_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, b_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float b_scale = 0.1f; b_scale <= 10.0f; b_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .b_scale(b_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, y_scale) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      for (float y_scale = 0.1f; y_scale <= 10.0f; y_scale *= 3.14f) {
        VBinaryMicrokernelTester()
          .batch_size(batch_size)
          .y_scale(y_scale)
          .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
      }
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, qmin) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmin(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }

  TEST(QS16_VMUL_MINMAX_FP32__NEON_U16, qmax) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t batch_size = 1; batch_size <= 80; batch_size += 15) {
      VBinaryMicrokernelTester()
        .batch_size(batch_size)
        .qmax(128)
        .Test(xnn_qs16_vmul_minmax_fp32_ukernel__neon_u16, VBinaryMicrokernelTester::OpType::Mul, xnn_init_qs16_mul_minmax_params, xnn_qs16_requantize_fp32);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
