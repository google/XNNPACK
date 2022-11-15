// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "gemm-microkernel-tester.h"

#include <gtest/gtest.h>

#include <vector>

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/igemm.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/params.h>
#include <xnnpack/post-operation.h>



#if XNN_ARCH_ARM && XNN_ENABLE_JIT

TEST(XNN_GENERATE_F32_IGEMM_UKERNEL_4X8__AARCH32_NEON_CORTEX_A53, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
  GemmMicrokernelTester()
    .mr(4)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(4)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a53,
        fused_operators);
}

TEST(XNN_GENERATE_F32_IGEMM_UKERNEL_4X8__AARCH32_NEON_PRFM_CORTEX_A53, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
  GemmMicrokernelTester()
    .mr(4)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(4)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_prfm_cortex_a53,
        fused_operators);
}

TEST(XNN_GENERATE_F32_IGEMM_UKERNEL_4X8__AARCH32_NEON_CORTEX_A75, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
  GemmMicrokernelTester()
    .mr(4)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(4)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75,
        fused_operators);
}

#endif  // XNN_ARCH_ARM && XNN_ENABLE_JIT

#if XNN_ARCH_ARM64 && XNN_ENABLE_JIT

TEST(XNN_GENERATE_F32_IGEMM_UKERNEL_1X8__AARCH64_NEONFMA_CORTEX_A75, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
  GemmMicrokernelTester()
    .mr(1)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(1)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        fused_operators);
}

TEST(XNN_GENERATE_F32_IGEMM_UKERNEL_1X8__AARCH64_NEONFMA_PRFM_CORTEX_A75, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
  GemmMicrokernelTester()
    .mr(1)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(1)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        fused_operators);
}

TEST(XNN_GENERATE_F32_IGEMM_UKERNEL_4X8__AARCH64_NEONFMA_CORTEX_A75, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
  GemmMicrokernelTester()
    .mr(4)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(4)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75,
        fused_operators);
}

TEST(XNN_GENERATE_F32_IGEMM_UKERNEL_4X8__AARCH64_NEONFMA_PRFM_CORTEX_A75, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
  GemmMicrokernelTester()
    .mr(4)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(4)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75,
        fused_operators);
}

TEST(XNN_GENERATE_F32_IGEMM_UKERNEL_6X8__AARCH64_NEONFMA_CORTEX_A75, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
  GemmMicrokernelTester()
    .mr(6)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(6)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        fused_operators);
}

TEST(XNN_GENERATE_F32_IGEMM_UKERNEL_6X8__AARCH64_NEONFMA_PRFM_CORTEX_A75, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
  GemmMicrokernelTester()
    .mr(6)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(6)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        fused_operators);
}

TEST(XNN_GENERATE_F32_IGEMM_UKERNEL_6X8__AARCH64_NEONFMA_LD128, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_post_operation> fused_operators = { {xnn_post_operation_type_hardswish} };
  GemmMicrokernelTester()
    .mr(6)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(6)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128,
        fused_operators);
}

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_JIT
