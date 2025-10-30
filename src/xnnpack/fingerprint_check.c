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

static void compute_fingerprint(const enum xnn_fingerprint_id fingerprint_id) {
  // LINT.IfChange(fingerprint_compute)
  switch (fingerprint_id) {
    case xnn_fingerprint_id_unknown:
    case xnn_fingerprint_id_test:
    case xnn_fingerprint_id_test_f16_f32_qc8w_example_flag:
    case xnn_fingerprint_id_no_fingerprint:
      return;
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

