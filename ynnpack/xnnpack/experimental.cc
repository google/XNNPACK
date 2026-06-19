// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include "include/xnnpack.h"

extern "C" {

const struct xnn_fingerprint* xnn_get_fingerprint(uint32_t id) {
  return nullptr;
}

const struct xnn_fingerprint* xnn_get_fingerprint_by_idx(uint32_t idx) {
  return nullptr;
}

struct xnn_fingerprint {
  uint32_t id;
  uint32_t value;
};

enum xnn_status xnn_check_fingerprint(struct xnn_fingerprint fingerprint) {
  return xnn_status_success;
}

}  // extern "C"
