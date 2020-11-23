// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/common.h>


// Table of exp2(k / 16) values decremented (as integer) by (k << 19), k = 0..15
XNN_INTERNAL const float xnn_table_exp2minus_k_over_16[16] = {
  0x1.000000p+0f, 0x1.FB5586p-1f, 0x1.F72B84p-1f, 0x1.F387A6p-1f,
  0x1.F06FE0p-1f, 0x1.EDEA64p-1f, 0x1.EBFDAEp-1f, 0x1.EAB07Ep-1f,
  0x1.EA09E6p-1f, 0x1.EA1148p-1f, 0x1.EACE54p-1f, 0x1.EC4918p-1f,
  0x1.EE89FAp-1f, 0x1.F199BEp-1f, 0x1.F5818Ep-1f, 0x1.FA4AFAp-1f,
};
