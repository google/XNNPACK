//
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/kai_matmul.h"
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_mr(void) {
#if XNN_ENABLE_KLEIDIAI
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot();
  return api.get_step(NULL).m;
#else
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}
size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_nr(void) {
#if XNN_ENABLE_KLEIDIAI
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot();
  return api.get_step(NULL).n;
#else
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

void xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
#if XNN_ENABLE_KLEIDIAI
  assert(m == 1);
  assert(dst_stride_col == sizeof(int8_t));

  const int32_t output_zero_point =
      minmax_params->fp32_scalar.output_zero_point;
  const int32_t output_min = minmax_params->fp32_scalar.output_min;
  const int32_t output_max = minmax_params->fp32_scalar.output_max;
  struct kai_matmul_uker_args args = {0};
  args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
  args.shape.m = m;
  args.shape.n = n;
  args.shape.k = k / sizeof(int8_t);
  args.operand.lhs.ptr = lhs_packed;
  args.operand.rhs.ptr = rhs_packed;
  args.operand.dst.ptr = dst;
  args.operand.dst.stride.m = dst_stride_row;
  args.operand.bias.scale_bias_global.ptr = &output_zero_point;
  args.activation.clamp.min_ptr = &output_min;
  args.activation.clamp.max_ptr = &output_max;

  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot();
  api.run(NULL, &args);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}
