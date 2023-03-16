// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>

#include <xnnpack/common.h>


// Table of exp2(k / 16) values decremented (as integer) by (k << 19), k = 0..15
XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_16[16] = {
  0x3F800000, 0x3F7DAAC3, 0x3F7B95C2, 0x3F79C3D3, 0x3F7837F0, 0x3F76F532, 0x3F75FED7, 0x3F75583F,
  0x3F7504F3, 0x3F7508A4, 0x3F75672A, 0x3F76248C, 0x3F7744FD, 0x3F78CCDF, 0x3F7AC0C7, 0x3F7D257D,
};
