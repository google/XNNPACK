// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/kai_matmul.h"
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x64c4__neonsme_get_mr(void) {
#if XNN_ENABLE_KLEIDIAI
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot();
  return api.get_step(NULL).m;
#else
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x64c4__neonsme_get_nr(void) {
#if XNN_ENABLE_KLEIDIAI
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot();
  return api.get_step(NULL).n;
#else
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

// Wraps the `kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot`
// GEMV microkernel with a name that is compatible with our tooling.
void xnn_qp8_f32_qb4w_gemm_minmax_ukernel_1x64c4__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
#if XNN_ENABLE_KLEIDIAI
  assert(dst_stride_col == sizeof(float));
  (void)dst_stride_col;

  const struct xnn_f32_qb4w_minmax_params* minmax_params = params;
  const float min = minmax_params->scalar.min;
  const float max = minmax_params->scalar.max;

  struct kai_matmul_uker_config config = {0};
  config.format.bl = minmax_params->scalar.blocksize;

  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot();

  // The microkernel reads the LHS and RHS row strides from the arguments, so
  // query them from the microkernel itself rather than recomputing them here.
  const struct kai_matmul_uker_lhs_dim_args lhs_shape = {.m = m, .k = k};
  const struct kai_matmul_uker_rhs_dim_args rhs_shape = {.n = n, .k = k};

  struct kai_matmul_uker_args args = {0};
  args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
  args.shape.m = m;
  args.shape.n = n;
  args.shape.k = k;
  args.operand.lhs.ptr = lhs_packed;
  args.operand.lhs.stride = api.get_lhs_stride(&config, &lhs_shape);
  args.operand.rhs.ptr = rhs_packed;
  args.operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape);
  args.operand.dst.ptr = dst;
  args.operand.dst.stride.m = dst_stride_row;
  args.activation.clamp.min_ptr = &min;
  args.activation.clamp.max_ptr = &max;

  api.run(&config, &args);
#else
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}
