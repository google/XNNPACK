// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/reduce.h"
 
void xnn_f32_rwdsum_ukernel_1p1x__scalar_c4(
    const size_t rows_val,
    const size_t channels_val,
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
    assert(rows_val != 0);
    assert(channels_val != 0);
    assert(input != NULL);
    assert(output != NULL);
    
    const int64_t rows = (int64_t)rows_val;
    const int64_t channels = (int64_t)channels_val;

    const int64_t padded_size = rows + (rows - 1) * (base_dilation - 1) + padding[0] + padding[1];
    const int64_t output_size = (padded_size < (window_dimensions - 1) * window_dilations + 1) ? 
                        0 : (padded_size - (window_dimensions - 1) * window_dilations - 1) / window_strides + 1;

    const int64_t lcm_value = math_lcm_s64(window_dilations, base_dilation) / base_dilation;
    const int64_t scaled_lcm = lcm_value * channels;

    for (int64_t i = 0; i < output_size; i++) {
        float sum = init_value * (window_dimensions + 1);
        const int64_t window_start = i * window_strides;
        const int64_t pad_high_boundary = math_divide_round_up_s64((padding[0] - window_start), window_dilations);
        const int64_t pad_low_boundary = math_divide_round_up_s64((padded_size - padding[1] - window_start), window_dilations);
        int64_t curr_win_size = 0;

        int64_t adjusted_pad_high_boundary = math_min_s64(pad_high_boundary, window_dimensions);
        adjusted_pad_high_boundary = math_max_s64(adjusted_pad_high_boundary, 0);
        curr_win_size += adjusted_pad_high_boundary;

        const int64_t offset = window_start - padding[0];
        const int64_t adjusted_pad_low_boundary = math_min_s64(pad_low_boundary, window_dimensions);
        const int64_t win_boundary = math_divide_round_up_s64((offset + (adjusted_pad_low_boundary) * window_dilations), base_dilation);
        int64_t j = 3;
        for (; j < channels; j += 4) {
            float curr_sum1 = sum;
            float curr_sum2 = sum;
            float curr_sum3 = sum;
            float curr_sum4 = sum;
            int64_t counter = 0;
            for (int64_t channel_win_size = curr_win_size; channel_win_size < adjusted_pad_low_boundary; channel_win_size++) {
                int64_t window_row = offset + channel_win_size * window_dilations;
                if (window_row % base_dilation == 0) {
                    window_row = window_row / base_dilation;
                    int64_t base_index4 = window_row * channels + j;
                    int64_t base_index3 = base_index4 - 1;
                    int64_t base_index2 = base_index4 - 2;
                    int64_t base_index1 = base_index4 - 3;
                    while(window_row < win_boundary){
                        curr_sum1 += input[base_index1];
                        curr_sum2 += input[base_index2];
                        curr_sum3 += input[base_index3];
                        curr_sum4 += input[base_index4];
                        counter++;
                        window_row += lcm_value;
                        base_index1 += scaled_lcm;
                        base_index2 += scaled_lcm;
                        base_index3 += scaled_lcm;
                        base_index4 += scaled_lcm;
                    }
                    break;
                }
            }
            curr_sum1 -= counter * init_value;
            curr_sum2 -= counter * init_value;
            curr_sum3 -= counter * init_value;
            curr_sum4 -= counter * init_value;
            output[i * channels + j] = curr_sum4;
            output[i * channels + j - 1] = curr_sum3;
            output[i * channels + j - 2] = curr_sum2;
            output[i * channels + j - 3] = curr_sum1;
        }
        j -= 3;
        for (; j < channels; j++) {
            float curr_sum = sum;
            int64_t counter = 0;
            for (int64_t channel_win_size = curr_win_size; channel_win_size < adjusted_pad_low_boundary; channel_win_size++) {
                int64_t window_row = offset + channel_win_size * window_dilations;
                if (window_row % base_dilation == 0) {
                    window_row = window_row / base_dilation;
                    int64_t base_index = window_row * channels + j;
                    while (window_row < win_boundary){
                        curr_sum += input[base_index];
                        counter++;
                        window_row += lcm_value;
                        base_index += scaled_lcm;
                    }
                    break;
                }
            }
            curr_sum -= counter * init_value;
            output[i * channels + j] = curr_sum;
        }
    }
}