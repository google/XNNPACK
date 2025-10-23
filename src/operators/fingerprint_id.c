// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/operators/fingerprint_id.h"

#include <stdarg.h>

enum xnn_fingerprint_id xnn_compute_fingerprint_id_value(
    enum xnn_fingerprint_id_helper op, enum xnn_fingerprint_id_helper in,
    enum xnn_fingerprint_id_helper out, enum xnn_fingerprint_id_helper weights,
    ...) {
  int id = op << XNN_FINGERPRINT_ID_OP_OFFSET |
           in << XNN_FINGERPRINT_ID_IN_OFFSET |
           out << XNN_FINGERPRINT_ID_OUT_OFFSET |
           weights << XNN_FINGERPRINT_ID_WEIGHTS_OFFSET;
  va_list args;
  va_start(args, weights);
  enum xnn_fingerprint_id_helper flag;
  while ((flag = va_arg(args, enum xnn_fingerprint_id_helper)) != 0) {
    id |= flag;
  }
  return id;
}
