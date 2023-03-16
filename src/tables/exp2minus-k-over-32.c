// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>

#include <xnnpack/common.h>


// Table of exp2(k / 32) values decremented (as integer) by (k << 18), k = 0..31
XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_32[32] = {
  0x3F800000, 0x3F7ECD87, 0x3F7DAAC3, 0x3F7C980F, 0x3F7B95C2, 0x3F7AA43A, 0x3F79C3D3, 0x3F78F4F0,
  0x3F7837F0, 0x3F778D3A, 0x3F76F532, 0x3F767043, 0x3F75FED7, 0x3F75A15B, 0x3F75583F, 0x3F7523F6,
  0x3F7504F3, 0x3F74FBAF, 0x3F7508A4, 0x3F752C4D, 0x3F75672A, 0x3F75B9BE, 0x3F76248C, 0x3F76A81E,
  0x3F7744FD, 0x3F77FBB8, 0x3F78CCDF, 0x3F79B907, 0x3F7AC0C7, 0x3F7BE4BA, 0x3F7D257D, 0x3F7E83B3,
};
