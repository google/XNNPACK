// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packw.h"
#include "packw-microkernel-tester.h"

struct XnnTestParam {
  const char *name;
  bool (*isa_check)();
  xnn_x32_packw_gemm_goi_ukernel_fn fn;
  size_t nr, kr, sr, kblock, nr_scale;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {
};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

const XnnTestParam xnn_test_params[] = {
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  { "X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_U2", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X2__NEON_LD2LANE_U2_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x2__neon_ld2lane_u2_prfm, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U4_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__NEON_LD4LANE_U8_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8__neon_ld4lane_u8_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_U4", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_U4_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_U8", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__NEON_LD4LANE_U8_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__neon_ld4lane_u8_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_U4", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4, /*NR=*/12, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_U4_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u4_prfm, /*NR=*/12, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_U8", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8, /*NR=*/12, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X12__NEON_LD4LANE_U8_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x12__neon_ld4lane_u8_prfm, /*NR=*/12, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U4_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__NEON_LD4LANE_U8_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__neon_ld4lane_u8_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  { "X32_PACKW_GEMM_GOI_X2C4__SSE2_U4", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4, /*NR=*/2, /*KR=*/4, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X2C4__SSE2_U4_PRFM", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x2c4__sse2_u4_prfm, /*NR=*/2, /*KR=*/4, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__SSE2_U4", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__SSE2_U4_PRFM", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__SSE2_U8", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u8, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__SSE2_U8_PRFM", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8__sse2_u8_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__SSE2_U4", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__SSE2_U4_PRFM", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__SSE2_U8", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u8, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__SSE2_U8_PRFM", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__sse2_u8_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__SSE2_U4", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__SSE2_U4_PRFM", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__SSE2_U8", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__SSE2_U8_PRFM", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__sse2_u8_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16S4__SSE2_U4", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16S4__SSE2_U4_PRFM", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16S4__SSE2_U8", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u8, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16S4__SSE2_U8_PRFM", []() { return TEST_REQUIRES_X86_SSE2_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16s4__sse2_u8_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/8, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__AVX_U4", []() { return TEST_REQUIRES_X86_AVX_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__AVX_U4_PRFM", []() { return TEST_REQUIRES_X86_AVX_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8__avx_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__AVX_U4", []() { return TEST_REQUIRES_X86_AVX_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_u4, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__AVX_U4_PRFM", []() { return TEST_REQUIRES_X86_AVX_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__avx_u4_prfm, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__AVX_U4", []() { return TEST_REQUIRES_X86_AVX_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__AVX_U4_PRFM", []() { return TEST_REQUIRES_X86_AVX_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__avx_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16S4__AVX_U4", []() { return TEST_REQUIRES_X86_AVX_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16S4__AVX_U4_PRFM", []() { return TEST_REQUIRES_X86_AVX_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16s4__avx_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__AVX512F_U4", []() { return TEST_REQUIRES_X86_AVX512F_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__AVX512F_U4_PRFM", []() { return TEST_REQUIRES_X86_AVX512F_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x16__avx512f_u4_prfm, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  { "X32_PACKW_GEMM_GOI_X2C4__WASMSIMD_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x2c4__wasmsimd_u4, /*NR=*/2, /*KR=*/4, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__WASMSIMD_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x8__wasmsimd_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8S4__WASMSIMD_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x8s4__wasmsimd_u4, /*NR=*/8, /*KR=*/1, /*SR=*/4, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  { "X32_PACKW_GEMM_GOI_X2__SCALAR_FLOAT_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float_u4, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X2__SCALAR_INT_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int_u4, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X3__SCALAR_FLOAT_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x3__scalar_float_u4, /*NR=*/3, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X3__SCALAR_INT_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x3__scalar_int_u4, /*NR=*/3, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X4__SCALAR_FLOAT_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float_u4, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X4__SCALAR_INT_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int_u4, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__SCALAR_FLOAT_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x8__scalar_float_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X8__SCALAR_INT_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x8__scalar_int_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__SCALAR_FLOAT_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x16__scalar_float_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X32_PACKW_GEMM_GOI_X16__SCALAR_INT_U4", []() { return true; }, xnn_x32_packw_gemm_goi_ukernel_x16__scalar_int_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  { "X32_PACKW_GEMM_GOI_X1V__RVV_U2", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u2, /*NR=*/1, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X1V__RVV_U4", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u4, /*NR=*/1, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X1V__RVV_U8", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x1v__rvv_u8, /*NR=*/1, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X2V__RVV_U2", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u2, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X2V__RVV_U4", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u4, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X2V__RVV_U8", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x2v__rvv_u8, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X4V__RVV_U2", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u2, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X4V__RVV_U4", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u4, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X4V__RVV_U8", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x4v__rvv_u8, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X8V__RVV_U2", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u2, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X8V__RVV_U4", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
  { "X32_PACKW_GEMM_GOI_X8V__RVV_U8", []() { return TEST_REQUIRES_RISCV_VECTOR_VALUE; }, xnn_x32_packw_gemm_goi_ukernel_x8v__rvv_u8, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/8, /*NR_SCALE=*/xnn_init_hardware_config()->vlenb / sizeof(uint32_t) },
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
};

TEST_P(XnnTest, k_eq_kblock) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  PackWMicrokernelTester()
    .n(GetParam().nr * GetParam().nr_scale)
    .k(GetParam().kblock)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .Test(GetParam().fn);
}

TEST_P(XnnTest, k_div_kblock) {
  if (GetParam().kblock <= 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  PackWMicrokernelTester()
    .n(GetParam().nr * GetParam().nr_scale)
    .k(GetParam().kblock * 5)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .Test(GetParam().fn);
}

TEST_P(XnnTest, k_lt_kblock) {
  if (GetParam().kblock <= 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < GetParam().kblock; k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, k_gt_kblock) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = GetParam().kblock + 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, n_eq_nr) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, n_div_nr) {
  if (GetParam().nr <= 1 || GetParam().nr_scale != 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * 2 * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, n_lt_nr) {
  if (GetParam().nr <= 1 || GetParam().nr_scale != 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    for (size_t n = 1; n < GetParam().nr * GetParam().nr_scale; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(GetParam().nr * GetParam().nr_scale)
        .kr(GetParam().kr)
        .sr(GetParam().sr)
        .Test(GetParam().fn);
    }
  }
}

TEST_P(XnnTest, n_gt_nr) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    if (GetParam().nr_scale == 1) {
      for (size_t n = GetParam().nr + 1; n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2); n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(GetParam().nr)
          .kr(GetParam().kr)
          .sr(GetParam().sr)
          .Test(GetParam().fn);
      }
    } else {
      for (size_t n = (GetParam().nr + 1) * GetParam().nr_scale;
                  n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2) * GetParam().nr_scale;
                  n += 1 * GetParam().nr_scale) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(GetParam().nr * GetParam().nr_scale)
          .kr(GetParam().kr)
          .sr(GetParam().sr)
          .Test(GetParam().fn);
      }
    }
  }
}

TEST_P(XnnTest, g_gt_1) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
      if (GetParam().nr_scale == 1) {
        for (size_t n = GetParam().nr + 1; n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2); n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(GetParam().nr)
            .kr(GetParam().kr)
            .sr(GetParam().sr)
            .Test(GetParam().fn);
        }
      } else {
        for (size_t n = (GetParam().nr + 1) * GetParam().nr_scale;
                    n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2) * GetParam().nr_scale;
                    n += 1 * GetParam().nr_scale) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(GetParam().nr * GetParam().nr_scale)
            .kr(GetParam().kr)
            .sr(GetParam().sr)
            .Test(GetParam().fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, null_bias) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
      if (GetParam().nr_scale == 1) {
        for (size_t n = GetParam().nr + 1; n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2); n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(GetParam().nr)
            .kr(GetParam().kr)
            .sr(GetParam().sr)
            .Test(GetParam().fn);
        }
      } else {
        for (size_t n = (GetParam().nr + 1) * GetParam().nr_scale;
                    n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2) * GetParam().nr_scale;
                    n += 1 * GetParam().nr_scale) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(GetParam().nr * GetParam().nr_scale)
            .kr(GetParam().kr)
            .sr(GetParam().sr)
            .Test(GetParam().fn);
        }
      }
    }
  }
}
INSTANTIATE_TEST_SUITE_P(x32_packw,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);

