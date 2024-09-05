// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"

#define MAX(X,Y) (X > Y ? X : Y)
#define CEILING_POS(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
#define CEILING_NEG(X) (int)(X)
#define CEIL(X) ( ((X) > 0) ? CEILING_POS(X) : CEILING_NEG(X) )

#define FLOORING_POS(X) (int)(X)
#define FLOORING_NEG(X) ((X-(int)(X)) > 0 ? (int)(X-1) : (int)(X))
#define FLOOR(X) ( ((X) > 0) ? FLOORING_POS(X) : FLOORING_NEG(X) )

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
    const union xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  int size = batch / sizeof(float);

  int padded_size = size + MAX((size - 1),0) * (base_dilation - 1) + padding[0] + padding[1];
  int output_size = (padded_size < (window_dimensions - 1) * window_dilations + 1) ? 
                    0 : FLOOR((padded_size - (window_dimensions - 1) * window_dilations - 1) / (float)window_strides) + 1;

    for (int i = 0; i < output_size; i++) {
            float sum = init_value;

            for (int k = 0; k < window_dimensions; k++) {
                int window_row = i * window_strides + k * window_dilations;
                if (window_row < padding[0] || 
                    window_row >= padded_size - padding[1] || 
                    (window_row - padding[0]) % base_dilation != 0) {
                    sum += init_value;
                    continue;
                }
                window_row = (window_row - padding[0]) / base_dilation;
                sum += input[window_row]; 
            }
            output[i] = sum;   
    }
}
