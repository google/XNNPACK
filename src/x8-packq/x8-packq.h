// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, unroll, params_type, init_params) \
    XNN_UKERNEL(arch_flags, ukernel, unroll)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, unroll) \
    XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, unroll, void, /*init_params=*/nullptr)
#define XNN_DEFINED_UKERNEL
#endif

// arch_flags, ukernel, unroll

XNN_UKERNEL(0, xnn_x8_packq_f32qp8_ukernel__scalar_u1, 1)

#if XNN_ENABLE_KLEIDIAI
XNN_UKERNEL(xnn_arch_arm_neon, xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2, 2)
#endif  // XNN_ENABLE_KLEIDIAI


#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif

