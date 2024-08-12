// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"


void xnn_u8_vclamp_ukernel__scalar_u4(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t voutput_max = params->scalar.max;
  const uint32_t voutput_min = params->scalar.min;

  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    uint32_t vt0 = (uint32_t) input[0];
    uint32_t vt1 = (uint32_t) input[1];
    uint32_t vt2 = (uint32_t) input[2];
    uint32_t vt3 = (uint32_t) input[3];
    input += 4;

    vt0 = math_max_u32(vt0, voutput_min);
    vt1 = math_max_u32(vt1, voutput_min);
    vt2 = math_max_u32(vt2, voutput_min);
    vt3 = math_max_u32(vt3, voutput_min);

    vt0 = math_min_u32(vt0, voutput_max);
    vt1 = math_min_u32(vt1, voutput_max);
    vt2 = math_min_u32(vt2, voutput_max);
    vt3 = math_min_u32(vt3, voutput_max);

    output[0] = (uint8_t) vt0;
    output[1] = (uint8_t) vt1;
    output[2] = (uint8_t) vt2;
    output[3] = (uint8_t) vt3;
    output += 4;
  }

  if XNN_UNLIKELY(batch != 0) {
    do {
      uint32_t vt = (uint32_t) *input++;
      vt = math_max_u32(vt, voutput_min);
      vt = math_min_u32(vt, voutput_max);
      *output++ = (uint8_t) vt;

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}
