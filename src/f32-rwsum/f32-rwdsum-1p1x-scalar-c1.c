// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/reduce.h"

// reduce window for vertical reduction 
void xnn_f32_rwdsum_ukernel_1p1x__scalar_c1(
    size_t rows,
    size_t channels,
    const float* input,
    float init_value,
    int64_t* padding, 
    int64_t base_dilation, 
    int64_t window_dilations,
    int64_t window_dimensions, 
    int64_t window_strides,
    float* output,
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  int64_t padded_size = rows + (rows - 1) * (base_dilation - 1) + padding[0] + padding[1];
  int64_t output_size = (padded_size < (window_dimensions - 1) * window_dilations + 1) ? 
                    0 : (padded_size - (window_dimensions - 1) * window_dilations - 1) / window_strides + 1;


  int64_t lcm_value = math_lcm_s64(window_dilations, base_dilation);
  lcm_value = lcm_value / base_dilation;
  int64_t scaled_lcm = lcm_value * channels;

  for (int64_t i = 0; i < output_size; i++) {
        float sum = init_value * (window_dimensions + 1);
        int64_t window_start = i * window_strides;
        int64_t pad_high_boundary = math_divide_round_up_s64((padding[0] - window_start) , window_dilations);
        int64_t pad_low_boundary = math_divide_round_up_s64((padded_size - padding[1] - window_start) , window_dilations);
        int64_t curr_win_size = 0;

        int64_t adjusted_pad_high_boundary = math_min_s64(pad_high_boundary, window_dimensions);
        adjusted_pad_high_boundary = math_max_s64(adjusted_pad_high_boundary , 0);
        curr_win_size += adjusted_pad_high_boundary;

        int64_t offset = window_start - padding[0];
        int64_t adjusted_pad_low_boundary = math_min_s64(pad_low_boundary, window_dimensions);
        int64_t win_boundary = math_divide_round_up_s64((offset + (adjusted_pad_low_boundary) * window_dilations), base_dilation);

        for (int64_t j = 0; j < channels; j++) {
            float curr_sum = sum;
            int64_t channel_win_size = curr_win_size;
            int64_t counter = 0;
            for (; channel_win_size < adjusted_pad_low_boundary; channel_win_size++) {
                int64_t window_row = offset + channel_win_size * window_dilations;
                if (window_row % base_dilation == 0) {
                    window_row = window_row / base_dilation;
                    int64_t base_index = window_row * channels + j;
                    while(window_row < win_boundary){
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