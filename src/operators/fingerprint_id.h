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
  xnn_fingerprint_id_fully_connected_nc_f16 = 2,
  xnn_fingerprint_id_fully_connected_nc_pf16 = 3,
  xnn_fingerprint_id_fully_connected_nc_qd8_f16_qc4w = 4,
  xnn_fingerprint_id_fully_connected_nc_qdu8_f16_qc4w = 5,
  xnn_fingerprint_id_fully_connected_nc_qd8_f16_qb4w = 6,
  xnn_fingerprint_id_fully_connected_nc_qd8_f32_qc4w = 7,
  xnn_fingerprint_id_fully_connected_nc_qdu8_f32_qc4w = 8,
  xnn_fingerprint_id_fully_connected_nc_qp8_f32_qc4w = 9,
  xnn_fingerprint_id_fully_connected_nc_qp8_f32_qc8w = 10,
  xnn_fingerprint_id_fully_connected_nc_qp8_f32_qb4w = 11,
  xnn_fingerprint_id_fully_connected_nc_qd8_f32_qb4w = 12,
  xnn_fingerprint_id_fully_connected_nc_qdu8_f32_qb4w = 13,
  xnn_fingerprint_id_fully_connected_nc_qd8_f32_qc8w = 14,
  xnn_fingerprint_id_fully_connected_nc_qdu8_f32_qc8w = 15,
  xnn_fingerprint_id_fully_connected_nc_qd8_f16_qc8w = 16,
  xnn_fingerprint_id_fully_connected_nc_qdu8_f16_qc8w = 17,
  xnn_fingerprint_id_fully_connected_nc_f32_f16 = 18,
  xnn_fingerprint_id_fully_connected_nc_bf16_f32 = 19,
  xnn_fingerprint_id_fully_connected_nc_f32 = 20,
  xnn_fingerprint_id_fully_connected_nc_f32_nr2 = 21,
  xnn_fingerprint_id_fully_connected_nc_pf32 = 22,
  xnn_fingerprint_id_fully_connected_nc_f32_qc4w = 23,
  xnn_fingerprint_id_fully_connected_nc_f32_qc8w = 24,
  xnn_fingerprint_id_fully_connected_nc_qs8 = 25,
  xnn_fingerprint_id_fully_connected_nc_qs8_qc4w = 26,
  xnn_fingerprint_id_fully_connected_nc_qs8_qc8w = 27,
  xnn_fingerprint_id_fully_connected_nc_pqs8_qc8w = 28,
  xnn_fingerprint_id_fully_connected_nc_qu8 = 29,
  // LINT.ThenChange(../xnnpack/fingerprint_check.c:fingerprint_compute)
};

#endif  // XNNPACK_SRC_OPERATORS_FINGERPRINT_ID_H_
