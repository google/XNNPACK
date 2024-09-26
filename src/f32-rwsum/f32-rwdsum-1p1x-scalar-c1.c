// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"

#define CEILING_POS(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
#define CEILING_NEG(X) (int)(X)
#define CEIL(X) ( ((X) > 0) ? CEILING_POS(X) : CEILING_NEG(X) )
#define FLOORING_POS(X) (int)(X)
#define FLOORING_NEG(X) ((X-(int)(X)) > 0 ? (int)(X-1) : (int)(X))
#define FLOOR(X) ( ((X) > 0) ? FLOORING_POS(X) : FLOORING_NEG(X) )
#define MAX(X,Y) (X > Y ? X : Y)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

void xnn_f32_rwdsum_ukernel_1p1x__scalar_c1(
    size_t rows,
    size_t channels,
    const float* input,
    float init_value,
    int* padding, 
    int base_dilation, 
    int window_dilations,
    int window_dimensions, 
    int window_strides,
    float* output,
    const union xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  int padded_size = rows + MAX((rows - 1),0) * (base_dilation - 1) + padding[0] + padding[1];
  int output_size = (padded_size < (window_dimensions - 1) * window_dilations + 1) ? 
                    0 : FLOOR((padded_size - (window_dimensions - 1) * window_dilations - 1) / (float)window_strides) + 1;

  int64_t inverse_base_dilation = (1LL << 32) / base_dilation;
  int64_t inverse_win_dilation = (1LL << 32) / window_dilations;

    for (int i = 0; i < output_size; i++) {
        float sum = init_value;
        int window_start = i * window_strides;
        int loop_end1 = CEIL((((padding[0] - window_start) * inverse_win_dilation) >> 32));
        int loop_end2 = CEIL((((padded_size - padding[1] - window_start) * inverse_win_dilation) >> 32));
        int curr_win_size = 0;

        int loop1 = MIN(loop_end1, window_dimensions);
        loop1 = MAX(loop1 , 0);
        sum += init_value * loop1;
        curr_win_size += loop1;

        int offset = window_start - padding[0];
        loop_end2 = MIN(loop_end2, window_dimensions);

        for (int j = 0; j < channels; j++) {
            float curr_sum = sum;
            int channel_win_size = curr_win_size;
            for (; channel_win_size < loop_end2; channel_win_size++) {
                int window_row = offset + channel_win_size * window_dilations;
                if (((window_row * inverse_base_dilation) >> 32) * base_dilation != window_row) {
                    curr_sum += init_value;
                } else {
                    window_row = (window_row * inverse_base_dilation) >> 32;;
                    curr_sum += input[window_row * channels + j];
                }
            }

            curr_sum += init_value * MAX((window_dimensions - channel_win_size), 0);
            output[i * channels + j] = curr_sum;
        }
    }
}
