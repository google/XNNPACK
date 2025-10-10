// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "include/experimental.h"
#include "src/operators/fingerprint_id.h"

#define XNNPACK_FINGERPRINT_FC_OP(...) \
  case XNN_EXPAND_TYPES(xnn_fingerprint_id_fully_connected_nc, __VA_ARGS__):    \
    XNN_CONCAT_TYPES(xnn_fingerprint_fully_connected_nc, __VA_ARGS__)(); \
    break;

static void compute_fingerprint(const enum xnn_fingerprint_id fingerprint_id) {
  // LINT.IfChange(fingerprint_compute)
  switch (fingerprint_id) {
    XNNPACK_FINGERPRINT_FC_OP(f16);
    XNNPACK_FINGERPRINT_FC_OP(pf16);
    XNNPACK_FINGERPRINT_FC_OP(qd8,f16,qc4w);
    XNNPACK_FINGERPRINT_FC_OP(qdu8,f16,qc4w);
    XNNPACK_FINGERPRINT_FC_OP(qd8,f16,qb4w);
    XNNPACK_FINGERPRINT_FC_OP(qd8,f32,qc4w);
    XNNPACK_FINGERPRINT_FC_OP(qdu8,f32,qc4w);
    XNNPACK_FINGERPRINT_FC_OP(qp8,f32,qc4w);
    XNNPACK_FINGERPRINT_FC_OP(qp8,f32,qc8w);
    XNNPACK_FINGERPRINT_FC_OP(qp8,f32,qb4w);
    XNNPACK_FINGERPRINT_FC_OP(qd8,f32,qb4w);
    XNNPACK_FINGERPRINT_FC_OP(qdu8,f32,qb4w);
    XNNPACK_FINGERPRINT_FC_OP(qd8,f32,qc8w);
    XNNPACK_FINGERPRINT_FC_OP(qdu8,f32,qc8w);
    XNNPACK_FINGERPRINT_FC_OP(qd8,f16,qc8w);
    XNNPACK_FINGERPRINT_FC_OP(qdu8,f16,qc8w);
    XNNPACK_FINGERPRINT_FC_OP(bf16,f32);
    XNNPACK_FINGERPRINT_FC_OP(f32);
    XNNPACK_FINGERPRINT_FC_OP(pf32);
    XNNPACK_FINGERPRINT_FC_OP(f32,qc4w);
    XNNPACK_FINGERPRINT_FC_OP(f32,qc8w);
    XNNPACK_FINGERPRINT_FC_OP(qs8);
    XNNPACK_FINGERPRINT_FC_OP(qs8,qc4w);
    XNNPACK_FINGERPRINT_FC_OP(qs8,qc8w);
    XNNPACK_FINGERPRINT_FC_OP(pqs8,qc8w);
    XNNPACK_FINGERPRINT_FC_OP(qu8);
    case xnn_fingerprint_id_fully_connected_nc_f32_f32_f16:
      xnn_fingerprint_fully_connected_nc_f32();
      break;
    case xnn_fingerprint_id_fully_connected_nc_f32_f32_f32_nr2:
      xnn_fingerprint_fully_connected_nc_f32();
      break;
    case xnn_fingerprint_id_convolution2d_nchw_f16_f16_f16_conv2d_hwc2chw:
    case xnn_fingerprint_id_convolution2d_nchw_f16_f16_f16_conv2d_hwc2chw_fp32_static_weights:
    case xnn_fingerprint_id_convolution2d_nchw_f16_f16_f16_dwconv:
    case xnn_fingerprint_id_convolution2d_nchw_f16_f16_f16_dwconv_fp32_static_weights:
    case xnn_fingerprint_id_convolution2d_nchw_f32_f32_f32_dwconv:
    case xnn_fingerprint_id_convolution2d_nchw_f32_f32_f32_conv2d_hwc2chw:
      xnn_fingerprint_convolution2d_nchw(fingerprint_id);
    break;
    case xnn_fingerprint_id_unknown:
    case xnn_fingerprint_id_test:
    case xnn_fingerprint_id_test_f16_f32_qc8w_nr2:
    case xnn_fingerprint_id_no_fingerprint:
      break;
  }
  // LINT.ThenChange(../operators/fingerprint_id.h.inc:fingerprint_id)
}

bool xnn_check_fingerprint(const struct xnn_fingerprint fingerprint) {
  compute_fingerprint(fingerprint.id);
  const struct xnn_fingerprint* reference_fingerprint =
      xnn_get_fingerprint(fingerprint.id);
  return reference_fingerprint &&
         reference_fingerprint->value == fingerprint.value;
}
