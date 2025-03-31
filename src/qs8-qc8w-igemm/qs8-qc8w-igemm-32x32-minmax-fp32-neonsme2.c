#include <stddef.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/math.h"
#include <arm_neon.h>

size_t xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_32x32__neonsme2_get_mr(void)
{
  return kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
}

size_t xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_32x32__neonsme2_get_nr(void)
{
  return kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
}

void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_32x32__neonsme2(
  size_t mr,
  size_t nc,
  size_t kc,
  size_t ks,
  const int8_t** restrict a,
  const void* restrict w,
  int8_t* restrict c,
  size_t cm_stride,
  size_t cn_stride,
  size_t a_offset,
  const int8_t* zero,
  const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const size_t kai_mr = kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
  const size_t k_chunk_count = ks / sizeof(void*) / kai_mr;
  const size_t k_chunk_length = kc;
  const size_t kai_kr = 4;
  const size_t k = k_chunk_count * round_up(k_chunk_length, kai_kr);

  // Packs LHS.
  const size_t packed_lhs_size =
    kai_get_lhs_packed_size_lhs_imatmul_pack_x8p2vlx4_x8p_sme(mr, k_chunk_count, k_chunk_length);
  void* packed_lhs = malloc(packed_lhs_size);

  kai_run_lhs_imatmul_pack_x8p2vlx4_x8p_sme(mr, k_chunk_count, k_chunk_length, a, a_offset, zero, packed_lhs);

  // GEMM.
  struct kai_matmul_requantize32_params kai_params;
  kai_params.output_zero_point = params->fp32_neonv8.output_zero_point;
  kai_params.min_value = (int8_t) params->fp32_neonv8.output_min;
  kai_params.max_value = (int8_t) params->fp32_neonv8.output_max;

  kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
    mr, nc, k, packed_lhs, w, c, cm_stride, sizeof(int8_t), &kai_params);

  free(packed_lhs);
}
