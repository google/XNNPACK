// clang-format off
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


// arch_flags, ukernel, size_fn, packed_offset_fn

#if XNN_ENABLE_KLEIDIAI
XNN_UKERNEL(xnn_arch_arm_sme, xnn_x16_pack_lh_ukernel__neonsme2,
            xnn_x16_pack_lh_size__neonsme2, xnn_x16_pack_lh_offset__neonsme2)
#endif  // XNN_ENABLE_KLEIDIAI
