// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/common.h>


// Table of exp2(k / 4) values decremented (as integer) by (k << 21), k = 0..3
XNN_INTERNAL const float xnn_table_exp2minus_k_over_4[4] = {
  0x1.000000p+0f, 0x1.F06FE0p-1f, 0x1.EA09E6p-1f, 0x1.EE89FAp-1f,
};
