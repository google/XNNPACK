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

// reduce window for horizontal reduction 
void xnn_f32_rwsum_ukernel__scalar_u1(
    size_t batch,
    const float* input,
    float init_value,
    int* padding, 
    int base_dilation, 
    int window_dilations,
    int window_dimensions, 
    int window_strides,
    float* output,
    const struct xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  int size = batch / sizeof(float);

  int padded_size = size + MAX((size - 1),0) * (base_dilation - 1) + padding[0] + padding[1];
  int output_size = (padded_size < (window_dimensions - 1) * window_dilations + 1) ? 
                    0 : FLOOR((padded_size - (window_dimensions - 1) * window_dilations - 1) / (float)window_strides) + 1;

  // replaced modulo and division by multiplicative scaled inverse
  int64_t inverse_base_dilation = (1LL << 32) / base_dilation;
  int64_t inverse_win_dilation = (1LL << 32) / window_dilations;

  for (int i = 0; i < output_size; i++) {

        float sum = init_value;
        int window_start = i * window_strides;
        int pad_high_boundary = CEIL((((padding[0] - window_start) * inverse_win_dilation) >> 32));
        int pad_low_boundary = CEIL((((padded_size - padding[1] - window_start) * inverse_win_dilation) >> 32));

        int curr_win_size = 0;
        int adjusted_pad_high_boundary = MIN(pad_high_boundary, window_dimensions);
        adjusted_pad_high_boundary = MAX(adjusted_pad_high_boundary , 0);

        sum += init_value * adjusted_pad_high_boundary;
        curr_win_size += adjusted_pad_high_boundary;
        int offset = window_start - padding[0];
        int adjusted_pad_low_boundary = MIN(pad_low_boundary, window_dimensions);

        for (; curr_win_size < adjusted_pad_low_boundary; curr_win_size++) {
            int window_row = offset + curr_win_size * window_dilations;
            if (((window_row * inverse_base_dilation) >> 32) * base_dilation != window_row) {
                sum += init_value;
            } else {
                window_row = (window_row * inverse_base_dilation) >> 32;;
                sum += input[window_row];
            }
        }

        sum += init_value * MAX((window_dimensions - curr_win_size), 0);
        output[i] = sum;
  }
}