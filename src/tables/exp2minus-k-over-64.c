// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/common.h>


// Table of exp2(k / 64) values decremented (as integer) by (k << 17), k = 0..63
XNN_INTERNAL const float xnn_table_exp2minus_k_over_64[64] = {
  0x1.000000p+0f, 0x1.FEC9A4p-1f, 0x1.FD9B0Ep-1f, 0x1.FC7452p-1f,
  0x1.FB5586p-1f, 0x1.FA3EC4p-1f, 0x1.F9301Ep-1f, 0x1.F829AAp-1f,
  0x1.F72B84p-1f, 0x1.F635BEp-1f, 0x1.F54874p-1f, 0x1.F463B8p-1f,
  0x1.F387A6p-1f, 0x1.F2B456p-1f, 0x1.F1E9E0p-1f, 0x1.F1285Ap-1f,
  0x1.F06FE0p-1f, 0x1.EFC08Cp-1f, 0x1.EF1A74p-1f, 0x1.EE7DB4p-1f,
  0x1.EDEA64p-1f, 0x1.ED60A2p-1f, 0x1.ECE086p-1f, 0x1.EC6A2Cp-1f,
  0x1.EBFDAEp-1f, 0x1.EB9B28p-1f, 0x1.EB42B6p-1f, 0x1.EAF474p-1f,
  0x1.EAB07Ep-1f, 0x1.EA76F2p-1f, 0x1.EA47ECp-1f, 0x1.EA2388p-1f,
  0x1.EA09E6p-1f, 0x1.E9FB24p-1f, 0x1.E9F75Ep-1f, 0x1.E9FEB6p-1f,
  0x1.EA1148p-1f, 0x1.EA2F34p-1f, 0x1.EA589Ap-1f, 0x1.EA8D9Ap-1f,
  0x1.EACE54p-1f, 0x1.EB1AEAp-1f, 0x1.EB737Cp-1f, 0x1.EBD82Ap-1f,
  0x1.EC4918p-1f, 0x1.ECC668p-1f, 0x1.ED503Cp-1f, 0x1.EDE6B6p-1f, 
  0x1.EE89FAp-1f, 0x1.EF3A2Cp-1f, 0x1.EFF770p-1f, 0x1.F0C1EAp-1f, 
  0x1.F199BEp-1f, 0x1.F27F12p-1f, 0x1.F3720Ep-1f, 0x1.F472D4p-1f, 
  0x1.F5818Ep-1f, 0x1.F69E60p-1f, 0x1.F7C974p-1f, 0x1.F902EEp-1f, 
  0x1.FA4AFAp-1f, 0x1.FBA1BEp-1f, 0x1.FD0766p-1f, 0x1.FE7C18p-1f, 
};
