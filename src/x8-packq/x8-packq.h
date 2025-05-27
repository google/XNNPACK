// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// arch_flags, ukernel, unroll

XNN_UKERNEL(0, xnn_x8_packq_f32qp8_ukernel__scalar_u1, 1)

#if XNN_ENABLE_KLEIDIAI
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2, 2)
#endif  // XNN_ENABLE_KLEIDIAI


