#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.h"
#endif  // XNN_ENABLE_KLEIDIAI

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2_get_mr(void) {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
#else
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

size_t xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2_get_nr(void) {
#if XNN_ENABLE_KLEIDIAI
  return kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
#else
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
#endif  // XNN_ENABLE_KLEIDIAI
}

void xnn_pf16_f16_igemm_minmax_fp16_ukernel_32x32c2__neonsme2(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, float* restrict c, size_t cm_stride,
    const struct xnn_f16_minmax_params* params) {

#if XNN_ENABLE_KLEIDIAI
  const size_t kai_kr = 2;
  const size_t k = ks * round_up(kc, kai_kr);

  kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(
      mr, nc, k, packed_lhs, w, c,
      cm_stride, sizeof(xnn_float16),
      xnn_float16_to_float(params->scalar.min),
      xnn_float16_to_float(params->scalar.max));
#else
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
#endif  // XNN_ENABLE_KLEIDIAI
}