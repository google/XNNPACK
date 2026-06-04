// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"

namespace {

TEST(HardwareConfigResetTest, HardwareConfigResetAndInitialization) {
  xnn_hardware_config mock_supported{};
#if XNN_ARCH_ARM64
  mock_supported.arch_flags |= xnn_arch_arm_neon_fp16_arith;
#elif XNN_ARCH_X86_64
  mock_supported.arch_flags |= xnn_arch_x86_sse2;
  mock_supported.arch_flags |= xnn_arch_x86_avx;
  mock_supported.arch_flags |= xnn_arch_x86_f16c;
#else
  GTEST_SKIP();
#endif

  xnn_set_hardware_config(&mock_supported);

  const struct xnn_unary_elementwise_config* supported_abs =
      xnn_init_f16_abs_config();
  ASSERT_NE(supported_abs, nullptr);
  EXPECT_NE(supported_abs->ukernel, nullptr);

  xnn_hardware_config mock_no_support{};
  xnn_set_hardware_config(&mock_no_support);

  const struct xnn_unary_elementwise_config* no_support_abs =
      xnn_init_f16_abs_config();
  EXPECT_EQ(no_support_abs, nullptr);

  xnn_set_hardware_config(&mock_supported);

  const struct xnn_unary_elementwise_config* restored_abs =
      xnn_init_f16_abs_config();
  EXPECT_EQ(restored_abs, supported_abs);

  xnn_reset_hardware_config();
}

}  // namespace
