// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/common.h>


// Table of exp2(k / 32) values decremented (as integer) by (k << 18), k = 0..31
XNN_INTERNAL const float xnn_table_exp2minus_k_over_32[32] = {
  0x1.000000p+0f, 0x1.FD9B0Ep-1f, 0x1.FB5586p-1f, 0x1.F9301Ep-1f,
  0x1.F72B84p-1f, 0x1.F54874p-1f, 0x1.F387A6p-1f, 0x1.F1E9E0p-1f,
  0x1.F06FE0p-1f, 0x1.EF1A74p-1f, 0x1.EDEA64p-1f, 0x1.ECE086p-1f,
  0x1.EBFDAEp-1f, 0x1.EB42B6p-1f, 0x1.EAB07Ep-1f, 0x1.EA47ECp-1f,
  0x1.EA09E6p-1f, 0x1.E9F75Ep-1f, 0x1.EA1148p-1f, 0x1.EA589Ap-1f,
  0x1.EACE54p-1f, 0x1.EB737Cp-1f, 0x1.EC4918p-1f, 0x1.ED503Cp-1f,
  0x1.EE89FAp-1f, 0x1.EFF770p-1f, 0x1.F199BEp-1f, 0x1.F3720Ep-1f,
  0x1.F5818Ep-1f, 0x1.F7C974p-1f, 0x1.FA4AFAp-1f, 0x1.FD0766p-1f,
};
