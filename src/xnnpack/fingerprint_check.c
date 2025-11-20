// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "include/experimental.h"
#include "include/xnnpack.h"
#include "src/operators/fingerprint_id.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/log.h"

#define XNNPACK_FINGERPRINT_FC_OP(...)                                       \
  case XNN_EXPAND_TYPES(xnn_fingerprint_id_fully_connected_nc, __VA_ARGS__): \
    return XNN_CONCAT_TYPES(xnn_fingerprint_fully_connected_nc, __VA_ARGS__)()

static enum xnn_status compute_fingerprint(
    const enum xnn_fingerprint_id fingerprint_id) {
  // LINT.IfChange(fingerprint_compute)
  switch (fingerprint_id) {
    case xnn_fingerprint_id_test_f16_f32_qc8w_example_flag:
      xnn_set_fingerprint(
          (struct xnn_fingerprint){.id = fingerprint_id, .value = 0xbadbeef});
      return xnn_status_success;
    case xnn_fingerprint_id_unknown:
    case xnn_fingerprint_id_no_fingerprint:
      return xnn_status_unsupported_parameter;
  }
  // LINT.ThenChange(../operators/fingerprint_id.h.inc:fingerprint_id)
  xnn_log_error(
      "Fingerprint id (%x) is unknown, cannot check its validity.",
      fingerprint_id);
  return xnn_status_unsupported_parameter;
}

enum xnn_status xnn_check_fingerprint(
    const struct xnn_fingerprint fingerprint) {
  const struct xnn_fingerprint* reference_fingerprint =
      xnn_get_fingerprint(fingerprint.id);
  if (!reference_fingerprint) {
    XNN_RETURN_IF_ERROR(compute_fingerprint(fingerprint.id));
    reference_fingerprint = xnn_get_fingerprint(fingerprint.id);
  }
  if (reference_fingerprint &&
      reference_fingerprint->value == fingerprint.value) {
    return xnn_status_success;
  }
  return xnn_status_invalid_parameter;
}
