// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_OPERATORS_FINGERPRINT_ID_H_
#define XNNPACK_SRC_OPERATORS_FINGERPRINT_ID_H_

// Identifies a fingerprint.
//
// This should be retrievable from a fingerprint and allows identifying what
// that fingerprint identifies.
//
// Because fingerprints may be stored to be compared to past/future versions of
// XNNPack, **these values MUST NOT CHANGE**.
enum xnn_fingerprint_id {
  // LINT.IfChange(fingerprint_id)
  xnn_fingerprint_id_unknown = 0,
  xnn_fingerprint_id_test = 1,
  // LINT.ThenChange(../xnnpack/fingerprint_check.c:fingerprint_compute)
};

#endif  // XNNPACK_SRC_OPERATORS_FINGERPRINT_ID_H_
