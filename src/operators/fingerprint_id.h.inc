// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// This file has purposefully no include guard to be reused in places where we
// want to get the whole list of fingerprint IDs.
//
// This is the list of fingerprints identifiers. It can be included to get the
// list of fingerprints.
//
// `XNN_FINGERPRINT_ID` should be a macro that takes the following armuments:
//
// - operator: name of the operator.
// - in: type of the operator inputs (available types in fingerprint_id.h).
// - out: type of the operator inputs (available types in fingerprint_id.h).
// - weights: type of the operator inputs (available types in fingerprint_id.h).
// - flags...: optional list of extra qualifications.
//
// See fingerprint_id.h for an example.

// LINT.IfChange(fingerprint_id)

XNN_FINGERPRINT_ID(test, f16, f32, qc8w, example_flag)

// LINT.ThenChange(../xnnpack/fingerprint_check.c:fingerprint_compute)
