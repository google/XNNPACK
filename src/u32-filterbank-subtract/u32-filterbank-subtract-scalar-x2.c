// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/math.h"
#include "xnnpack/filterbank.h"


void xnn_u32_filterbank_subtract_ukernel__scalar_x2(
    size_t batch_size,
    const uint32_t* input,
    uint32_t smoothing,
    uint32_t alternate_smoothing,
    uint32_t one_minus_smoothing,
    uint32_t alternate_one_minus_smoothing,
    uint32_t min_signal_remaining,
    uint32_t smoothing_bits,  /* 0 in FE */
    uint32_t spectral_subtraction_bits,  /* 14 in FE */
    uint32_t* noise_estimate,
    uint32_t* output) {

  assert(batch_size != 0);
  assert(batch_size % 2 == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(noise_estimate != NULL);

  batch_size >>= 1;  /* 48 in FE */

  do {
    const uint32_t vinput0 = input[0];
    const uint32_t vinput1 = input[1];
    input += 2;

    uint32_t vnoise_estimate0 = noise_estimate[0];
    uint32_t vnoise_estimate1 = noise_estimate[1];

    // Scale up signal for smoothing filter computation.
    const uint32_t vsignal_scaled_up0 = vinput0 << smoothing_bits;
    const uint32_t vsignal_scaled_up1 = vinput1 << smoothing_bits;

    vnoise_estimate0 = (uint32_t) ((math_mulext_u32(vsignal_scaled_up0, smoothing) +
                                    math_mulext_u32(vnoise_estimate0,   one_minus_smoothing)) >> spectral_subtraction_bits);
    vnoise_estimate1 = (uint32_t) ((math_mulext_u32(vsignal_scaled_up1, alternate_smoothing) +
                                    math_mulext_u32(vnoise_estimate1,   alternate_one_minus_smoothing)) >> spectral_subtraction_bits);

    noise_estimate[0] = vnoise_estimate0;
    noise_estimate[1] = vnoise_estimate1;
    noise_estimate += 2;

    const uint32_t vfloor0 = (uint32_t) (math_mulext_u32(vinput0, min_signal_remaining) >> spectral_subtraction_bits);
    const uint32_t vfloor1 = (uint32_t) (math_mulext_u32(vinput1, min_signal_remaining) >> spectral_subtraction_bits);
    const uint32_t vsubtracted0 = math_doz_u32(vsignal_scaled_up0, vnoise_estimate0) >> smoothing_bits;
    const uint32_t vsubtracted1 = math_doz_u32(vsignal_scaled_up1, vnoise_estimate1) >> smoothing_bits;
    const uint32_t vout0 = math_max_u32(vsubtracted0, vfloor0);
    const uint32_t vout1 = math_max_u32(vsubtracted1, vfloor1);

    output[0] = vout0;
    output[1] = vout1;
    output += 2;

  } while (--batch_size != 0);
}
