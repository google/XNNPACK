// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/common.h>


// Table of exp2(k / 64) values, k = 0..63
XNN_INTERNAL const float xnn_table_exp2_k_over_64[64] = {
  0x1.000000p+0f, 0x1.02C9A4p+0f, 0x1.059B0Ep+0f, 0x1.087452p+0f,
  0x1.0B5586p+0f, 0x1.0E3EC4p+0f, 0x1.11301Ep+0f, 0x1.1429AAp+0f,
  0x1.172B84p+0f, 0x1.1A35BEp+0f, 0x1.1D4874p+0f, 0x1.2063B8p+0f,
  0x1.2387A6p+0f, 0x1.26B456p+0f, 0x1.29E9E0p+0f, 0x1.2D285Ap+0f,
  0x1.306FE0p+0f, 0x1.33C08Cp+0f, 0x1.371A74p+0f, 0x1.3A7DB4p+0f,
  0x1.3DEA64p+0f, 0x1.4160A2p+0f, 0x1.44E086p+0f, 0x1.486A2Cp+0f,
  0x1.4BFDAEp+0f, 0x1.4F9B28p+0f, 0x1.5342B6p+0f, 0x1.56F474p+0f,
  0x1.5AB07Ep+0f, 0x1.5E76F2p+0f, 0x1.6247ECp+0f, 0x1.662388p+0f,
  0x1.6A09E6p+0f, 0x1.6DFB24p+0f, 0x1.71F75Ep+0f, 0x1.75FEB6p+0f,
  0x1.7A1148p+0f, 0x1.7E2F34p+0f, 0x1.82589Ap+0f, 0x1.868D9Ap+0f,
  0x1.8ACE54p+0f, 0x1.8F1AEAp+0f, 0x1.93737Cp+0f, 0x1.97D82Ap+0f,
  0x1.9C4918p+0f, 0x1.A0C668p+0f, 0x1.A5503Cp+0f, 0x1.A9E6B6p+0f,
  0x1.AE89FAp+0f, 0x1.B33A2Cp+0f, 0x1.B7F770p+0f, 0x1.BCC1EAp+0f,
  0x1.C199BEp+0f, 0x1.C67F12p+0f, 0x1.CB720Ep+0f, 0x1.D072D4p+0f,
  0x1.D5818Ep+0f, 0x1.DA9E60p+0f, 0x1.DFC974p+0f, 0x1.E502EEp+0f,
  0x1.EA4AFAp+0f, 0x1.EFA1BEp+0f, 0x1.F50766p+0f, 0x1.FA7C18p+0f,
};
