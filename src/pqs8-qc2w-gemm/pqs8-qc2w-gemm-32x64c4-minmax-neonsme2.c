// Copyright Google LLC
//
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/kai_matmul.h"
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_32x64c4__neonsme2_get_mr(void) {
#if XNN_ENABLE_KLEIDIAI
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa();
  return api.get_step(NULL).m;
#else
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_32x64c4__neonsme2_get_nr(void) {
#if XNN_ENABLE_KLEIDIAI
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa();
  return api.get_step(NULL).n;
#else
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

void xnn_pqs8_qc2w_gemm_minmax_fp32_ukernel_32x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
#if XNN_ENABLE_KLEIDIAI
  assert(dst_stride_col == sizeof(int8_t));

  const int32_t output_zero_point =
      minmax_params->fp32_scalar.output_zero_point;
  const int32_t output_min = minmax_params->fp32_scalar.output_min;
  const int32_t output_max = minmax_params->fp32_scalar.output_max;
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa();
  const struct kai_matmul_uker_lhs_dim_args lhs_shape = {
      .m = m,
      .k = k / sizeof(int8_t),
  };
  const struct kai_matmul_uker_rhs_dim_args rhs_shape = {
      .n = n,
      .k = k / sizeof(int8_t),
  };
  struct kai_matmul_uker_args args = {0};
  args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
  args.shape.m = m;
  args.shape.n = n;
  args.shape.k = k / sizeof(int8_t);
  args.operand.lhs.ptr = lhs_packed;
  args.operand.lhs.stride = api.get_lhs_stride(NULL, &lhs_shape);
  args.operand.rhs.ptr = rhs_packed;
  args.operand.rhs.stride = api.get_rhs_stride(NULL, &rhs_shape);
  args.operand.dst.ptr = dst;
  args.operand.dst.stride.m = dst_stride_row;
  args.operand.bias.scale_bias_global.ptr = &output_zero_point;
  args.activation.clamp.min_ptr = &output_min;
  args.activation.clamp.max_ptr = &output_max;

  api.run(NULL, &args);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}
