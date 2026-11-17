// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/raddstoreexpminusmax.h"

namespace {

TEST(HardwareConfigResetTest, HardwareConfigResetAndInitialization) {
  xnn_hardware_config mock_supported{};
#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR
  mock_supported.arch_flags |= xnn_arch_arm_neon_fp16_arith;
#elif XNN_ARCH_X86_64 && XNN_ENABLE_SSE2
  mock_supported.arch_flags |= xnn_arch_x86_sse2;
#else
  GTEST_SKIP();
#endif

  xnn_set_hardware_config(&mock_supported);

  const struct xnn_unary_elementwise_config* supported_abs =
      xnn_init_f16_abs_config();
  ASSERT_NE(supported_abs, nullptr);
  ASSERT_NE(supported_abs->ukernel, nullptr);
  const xnn_vunary_ukernel_fn supported_ukernel = supported_abs->ukernel;

  xnn_hardware_config mock_no_support{};
  xnn_set_hardware_config(&mock_no_support);

  const struct xnn_unary_elementwise_config* no_support_abs =
      xnn_init_f16_abs_config();
  ASSERT_NE(no_support_abs, nullptr);
  EXPECT_NE(no_support_abs->ukernel, nullptr);
#if (XNN_ARCH_ARM64 && XNN_ENABLE_ARM_FP16_VECTOR) || \
    (XNN_ARCH_X86_64 && XNN_ENABLE_SSE2)
  EXPECT_NE(no_support_abs->ukernel, supported_ukernel);
#endif

  xnn_set_hardware_config(&mock_supported);

  const struct xnn_unary_elementwise_config* restored_abs =
      xnn_init_f16_abs_config();
  ASSERT_NE(restored_abs, nullptr);
  EXPECT_EQ(restored_abs, supported_abs);
  EXPECT_EQ(restored_abs->ukernel, supported_ukernel);

  xnn_reset_hardware_config();
}

// NEON-FMA microkernels must not be selected on ARMv7 CPUs that have NEON
// but not VFPv4 (e.g. Cortex-A8, Cortex-A9), as they would SIGILL.
TEST(HardwareConfigResetTest, F32RaddstoreexpminusmaxNeonWithoutFma) {
#if XNN_ARCH_ARM
  xnn_hardware_config mock_neon{};
  mock_neon.arch_flags |= xnn_arch_arm_neon;
  xnn_set_hardware_config(&mock_neon);

  const struct xnn_raddstoreexpminusmax_config* config =
      xnn_init_f32_raddstoreexpminusmax_config(/*flags=*/0);
  ASSERT_NE(config, nullptr);
  ASSERT_NE(config->ukernel, nullptr);
  EXPECT_NE(config->ukernel,
            (xnn_raddstoreexpminusmax_ukernel_fn)
                xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2);

  xnn_reset_hardware_config();
#else
  GTEST_SKIP();
#endif
}

// Same check with the detected hardware config, e.g. when run under
// `qemu-arm -cpu cortex-a9`.
TEST(HardwareConfigResetTest, F32RaddstoreexpminusmaxDetectedHardware) {
#if XNN_ARCH_ARM
  const struct xnn_hardware_config* hardware_config =
      xnn_init_hardware_config();
  ASSERT_NE(hardware_config, nullptr);
  const struct xnn_raddstoreexpminusmax_config* config =
      xnn_init_f32_raddstoreexpminusmax_config(/*flags=*/0);
  ASSERT_NE(config, nullptr);
  if (!(hardware_config->arch_flags & xnn_arch_arm_neon_fma)) {
    EXPECT_NE(config->ukernel,
              (xnn_raddstoreexpminusmax_ukernel_fn)
                  xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2);
  }
#else
  GTEST_SKIP();
#endif
}

}  // namespace
