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
XNN_FINGERPRINT_ID(fully_connected_nc, f16, f16, f16)
XNN_FINGERPRINT_ID(fully_connected_nc, pf16, pf16, pf16)
XNN_FINGERPRINT_ID(fully_connected_nc, qd8, f16, qc4w)
XNN_FINGERPRINT_ID(fully_connected_nc, qdu8, f16, qc4w)
XNN_FINGERPRINT_ID(fully_connected_nc, qd8, f16, qb4w)
XNN_FINGERPRINT_ID(fully_connected_nc, qd8, f32, qc2w)
XNN_FINGERPRINT_ID(fully_connected_nc, qd8, f32, qc4w)
XNN_FINGERPRINT_ID(fully_connected_nc, qdu8, f32, qc4w)
XNN_FINGERPRINT_ID(fully_connected_nc, qp8, f32, qc4w)
XNN_FINGERPRINT_ID(fully_connected_nc, qp8, f32, qc8w)
XNN_FINGERPRINT_ID(fully_connected_nc, qp8, f32, qb4w)
XNN_FINGERPRINT_ID(fully_connected_nc, qd8, f32, qb4w)
XNN_FINGERPRINT_ID(fully_connected_nc, qdu8, f32, qb4w)
XNN_FINGERPRINT_ID(fully_connected_nc, qd8, f32, qc8w)
XNN_FINGERPRINT_ID(fully_connected_nc, qdu8, f32, qc8w)
XNN_FINGERPRINT_ID(fully_connected_nc, qd8, f16, qc8w)
XNN_FINGERPRINT_ID(fully_connected_nc, qdu8, f16, qc8w)
XNN_FINGERPRINT_ID(fully_connected_nc, bf16, bf16, f32)
XNN_FINGERPRINT_ID(fully_connected_nc, f32, f32, f32)
XNN_FINGERPRINT_ID(fully_connected_nc, f32, f32, f32, nr2)
XNN_FINGERPRINT_ID(fully_connected_nc, pf32, pf32, pf32)
XNN_FINGERPRINT_ID(fully_connected_nc, f32, f32, qc4w)
XNN_FINGERPRINT_ID(fully_connected_nc, f32, f32, qc8w)
XNN_FINGERPRINT_ID(fully_connected_nc, qs8, qs8, qs8)
XNN_FINGERPRINT_ID(fully_connected_nc, qs8, qs8, qc4w)
XNN_FINGERPRINT_ID(fully_connected_nc, qs8, qs8, qc8w)
XNN_FINGERPRINT_ID(fully_connected_nc, pqs8, pqs8, qc8w)
XNN_FINGERPRINT_ID(fully_connected_nc, qu8, qu8, qu8)
XNN_FINGERPRINT_ID(convolution2d_nchw, f16, f16, f16, conv2d_hwc2chw)
XNN_FINGERPRINT_ID(convolution2d_nchw, f16, f16, f16, conv2d_hwc2chw, fp32_static_weights)
XNN_FINGERPRINT_ID(convolution2d_nchw, f16, f16, f16, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nchw, f16, f16, f16, dwconv, fp32_static_weights)
XNN_FINGERPRINT_ID(convolution2d_nchw, f32, f32, f32, conv2d_hwc2chw)
XNN_FINGERPRINT_ID(convolution2d_nchw, f32, f32, f32, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, f16, f16, f16, vmulcaddc, fp32_static_weights)
XNN_FINGERPRINT_ID(convolution2d_nhwc, f16, f16, f16, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, f16, f16, f16, dwconv, fp32_static_weights)
XNN_FINGERPRINT_ID(convolution2d_nhwc, f16, f16, f16, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, f16, f16, f16, fp32_static_weights)
XNN_FINGERPRINT_ID(convolution2d_nhwc, f16, f16, f16)
XNN_FINGERPRINT_ID(convolution2d_nhwc, pf16, pf16, pf16, vmulcaddc, fp32_static_weights)
XNN_FINGERPRINT_ID(convolution2d_nhwc, pf16, pf16, pf16, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, pf16, pf16, pf16, dwconv, fp32_static_weights)
XNN_FINGERPRINT_ID(convolution2d_nhwc, pf16, pf16, pf16, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, pf16, pf16, pf16, fp32_static_weights)
XNN_FINGERPRINT_ID(convolution2d_nhwc, pf16, pf16, pf16)
XNN_FINGERPRINT_ID(convolution2d_nhwc, f32, f32, f32, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, f32, f32, f32, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, f32, f32, f32)
XNN_FINGERPRINT_ID(convolution2d_nhwc, pf32, pf32, pf32)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qd8, f16, qc8w, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qd8, f16, qc8w, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qd8, f16, qc8w)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qdu8, f16, qc8w, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qdu8, f16, qc8w, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qdu8, f16, qc8w)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qd8, f32, qc8w, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qd8, f32, qc8w, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qd8, f32, qc8w)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qdu8, f32, qc8w, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qdu8, f32, qc8w, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qdu8, f32, qc8w)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qu8, qu8, qu8, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qu8, qu8, qu8, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qu8, qu8, qu8)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qs8, qs8, qs8, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qs8, qs8, qs8, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qs8, qs8, qs8)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qc8, qc8, qc8, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qc8, qc8, qc8, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, qc8, qc8, qc8)
XNN_FINGERPRINT_ID(convolution2d_nhwc, pqs8, qs8, qc8w, vmulcaddc)
XNN_FINGERPRINT_ID(convolution2d_nhwc, pqs8, qs8, qc8w, dwconv)
XNN_FINGERPRINT_ID(convolution2d_nhwc, pqs8, qs8, qc8w)

// LINT.ThenChange(../xnnpack/fingerprint_check.c:fingerprint_compute)
