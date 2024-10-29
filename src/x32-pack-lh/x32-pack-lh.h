// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_UKERNEL_WITH_PARAMS
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, unroll, params_type, \
                                init_params)                              \
  XNN_UKERNEL(arch_flags, ukernel, unroll)
#define XNN_DEFINED_UKERNEL_WITH_PARAMS
#endif

#ifndef XNN_UKERNEL
#define XNN_UKERNEL(arch_flags, ukernel, unroll) \
  XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, unroll)
#define XNN_DEFINED_UKERNEL
#endif

// arch_flags, ukernel, unroll

#if XNN_ENABLE_KLEIDIAI
XNN_UKERNEL(xnn_arch_arm_sme, xnn_x32_pack_lh_ukernel__neonsme2,
            xnn_x32_pack_lh_size__neonsme2)
#endif  // XNN_ENABLE_KLEIDIAI

#ifdef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_DEFINED_UKERNEL_WITH_PARAMS
#undef XNN_UKERNEL_WITH_PARAMS
#endif

#ifdef XNN_DEFINED_UKERNEL
#undef XNN_DEFINED_UKERNEL
#undef XNN_UKERNEL
#endif
