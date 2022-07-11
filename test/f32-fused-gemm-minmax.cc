#include "gemm-microkernel-tester.h"

#include <gtest/gtest.h>

#include <vector>

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/params.h>

#if XNN_PLATFORM_JIT

TEST(GENERATE_F32_GEMM_UPTO6X8__AARCH64_NEONFMA_PRFM_CORTEX_A75, negate) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_fused_operator> fused_operators = { {xnn_fused_operator_type_negate} };
  GemmMicrokernelTester()
    .mr(6)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(6)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_gemm_ukernel_upto6x8__aarch64_neonfma_prfm_cortex_a75,
        fused_operators);
}

TEST(GENERATE_F32_GEMM_UPTO6X8__AARCH64_NEONFMA_PRFM_CORTEX_A75, hardswish) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_fused_operator> fused_operators = { {xnn_fused_operator_type_hardswish} };
  GemmMicrokernelTester()
    .mr(6)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(6)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_gemm_ukernel_upto6x8__aarch64_neonfma_prfm_cortex_a75,
        fused_operators);
}

TEST(GENERATE_F32_GEMM_UPTO6X8__AARCH64_NEONFMA_PRFM_CORTEX_A75, hardswish_negate_abs) {
  TEST_REQUIRES_ARM_NEON_FMA;
  std::vector<xnn_fused_operator> fused_operators = {
    {xnn_fused_operator_type_hardswish}, {xnn_fused_operator_type_negate}, {xnn_fused_operator_type_abs}};
  GemmMicrokernelTester()
    .mr(6)
    .nr(8)
    .kr(1)
    .sr(1)
    .m(6)
    .n(8)
    .k(8)
    .Test(
        xnn_generate_f32_gemm_ukernel_upto6x8__aarch64_neonfma_prfm_cortex_a75,
        fused_operators);
}

#endif  // XNN_PLATFORM_JIT
