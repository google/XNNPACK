// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/operators/fingerprint_id.h"

#include <stdarg.h>
#include <stdint.h>

enum xnn_fingerprint_id xnn_compute_fingerprint_id_value(
    enum xnn_fingerprint_id_helper op, enum xnn_fingerprint_id_helper in,
    enum xnn_fingerprint_id_helper out, uint32_t weights, ...) {
  uint32_t id = op << XNN_FINGERPRINT_ID_OP_OFFSET |
                in << XNN_FINGERPRINT_ID_IN_OFFSET |
                out << XNN_FINGERPRINT_ID_OUT_OFFSET |
                weights << XNN_FINGERPRINT_ID_WEIGHTS_OFFSET;
  va_list args;
  va_start(args, weights);
  uint32_t flag;
  while ((flag = va_arg(args, uint32_t)) != 0) {
    id |= flag;
  }
  return (enum xnn_fingerprint_id)id;
}

const char* xnn_fingerprint_id_to_string(
    const uint32_t fingerprint_id) {
  switch ((enum xnn_fingerprint_id)fingerprint_id) {
#define XNN_STRINGIFY__(x) #x
#define XNN_STRINGIFY_(x) XNN_EXPAND(XNN_STRINGIFY__(x))
#define XNN_STRINGIFY(x) XNN_STRINGIFY_(x)
#define XNN_FINGERPRINT_ID(operator, in, out, ...)              \
  case XNN_FINGERPRINT_ID_NAME(operator, in, out, __VA_ARGS__): \
    return XNN_STRINGIFY(                                       \
        XNN_FINGERPRINT_ID_NAME(operator, in, out, __VA_ARGS__));
#include "fingerprint_id.h.inc"
    case xnn_fingerprint_id_test_f16_f32_qc8w_nr2:
      return "xnn_fingerprint_id_test_f16_f32_qc8w_nr2";
    case xnn_fingerprint_id_unknown:
      return "xnn_fingerprint_id_unknown";
    case xnn_fingerprint_id_no_fingerprint:
      return "xnn_fingerprint_id_no_fingerprint";
#undef XNN_FINGERPRINT_ID
  }
  return "<unhandled fingerprint id>";
}
