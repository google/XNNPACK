// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/reduce.h"

void xnn_f32_rwsum_ukernel__scalar_u2_acc2(
    const size_t batch,
    const float* input,
    const float init_value,
    const int64_t* padding, 
    const int64_t base_dilation, 
    const int64_t window_dilations,
    const int64_t window_dimensions, 
    const int64_t window_strides,
    float* output,
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input != NULL);
    assert(output != NULL);

    const int64_t size = batch / sizeof(float);

    const int64_t padded_size = size + (size - 1) * (base_dilation - 1) + padding[0] + padding[1];
    const int64_t output_size = (padded_size < (window_dimensions - 1) * window_dilations + 1) ? 
                        0 : (padded_size - (window_dimensions - 1) * window_dilations - 1) / window_strides + 1;
        
    const int64_t lcm_value = math_lcm_s64(window_dilations, base_dilation) / base_dilation;
    const int64_t scaled_lcm_u2 = 2 * lcm_value;

    for (int64_t i = 0; i < output_size; i++) {
        float sum0 = init_value * (window_dimensions + 1);
        float sum1 = 0.0f;
        const int64_t window_start = i * window_strides;
        const int64_t pad_high_boundary = math_divide_round_up_s64((padding[0] - window_start), window_dilations);
        const int64_t pad_low_boundary = math_divide_round_up_s64((padded_size - padding[1] - window_start), window_dilations);

        int64_t adjusted_pad_high_boundary = math_min_s64(pad_high_boundary, window_dimensions);
        adjusted_pad_high_boundary = math_max_s64(adjusted_pad_high_boundary, 0);

        const int64_t offset = window_start - padding[0];
        const int64_t adjusted_pad_low_boundary = math_min_s64(pad_low_boundary, window_dimensions);
        const int64_t win_boundary = math_divide_round_up_s64((offset + (adjusted_pad_low_boundary) * window_dilations), base_dilation);
        int64_t counter = 0;

        for (int64_t curr_win_idx = adjusted_pad_high_boundary; curr_win_idx < adjusted_pad_low_boundary; curr_win_idx++) {
            int64_t window_row0 = offset + curr_win_idx * window_dilations;

            if (window_row0 % base_dilation == 0) {
                window_row0 /= base_dilation;
                int64_t window_row1 = window_row0 + lcm_value;
                while (window_row1 < win_boundary) { 
                    sum0 += input[window_row0];
                    sum1 += input[window_row1];
                    counter++;
                    window_row0 += scaled_lcm_u2;
                    window_row1 += scaled_lcm_u2;
                }
                counter *= 2;
                sum0 += sum1;
                
                if (window_row0 < win_boundary){
                  sum0 +=  input[window_row0]; 
                  counter += 1;
                }
                break;
            }
        }
        sum0 -= counter * init_value;
        output[i] = sum0;
    }
}