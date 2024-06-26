# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_vrelu_ukernel__wasm32_shr_u4(
#     size_t batch,             0
#     const float* input,       1
#     float* output,             2
#     const union params)   3 unused

# locals
#     float value0          4
#     float value1          5
#     float value2          6
#     float value3          7
#     float mask0           8
#     float mask1           9
#     float mask2           10
#     float mask3           11

BEGIN_FUNCTION  xnn_f32_vrelu_ukernel__wasm32_shr_u4
    .functype   xnn_f32_vrelu_ukernel__wasm32_shr_u4 (i32, i32, i32, i32) -> ()
    .local      i32, i32, i32, i32, i32, i32, i32, i32

    local.get    0
    i32.const    16      # count >= 16
    i32.ge_s
    if
      loop
        local.get    1
        i32.load     0        # load 4 floats from src
        local.set    4
        local.get    1
        i32.load     4
        local.set    5
        local.get    1
        i32.load     8
        local.set    6
        local.get    1
        i32.load     12
        local.set    7

        local.get    4        # (v >> 31) - 1) & v
        i32.const    31
        i32.shr_u
        local.set    8
        local.get    5
        i32.const    31
        i32.shr_u
        local.set    9
        local.get    6
        i32.const    31
        i32.shr_u
        local.set    10
        local.get    7
        i32.const    31
        i32.shr_u
        local.set    11

        local.get    8
        i32.const    -1
        i32.add
        local.set    8
        local.get    9
        i32.const    -1
        i32.add
        local.set    9
        local.get    10
        i32.const    -1
        i32.add
        local.set    10
        local.get    11
        i32.const    -1
        i32.add
        local.set    11

        local.get    4
        local.get    8
        i32.and
        local.set    4
        local.get    5
        local.get    9
        i32.and
        local.set    5
        local.get    6
        local.get    10
        i32.and
        local.set    6
        local.get    7
        local.get    11
        i32.and
        local.set    7

        local.get    2
        local.get    4
        i32.store    0        # store 4 floats
        local.get    2
        local.get    5
        i32.store    4
        local.get    2
        local.get    6
        i32.store    8
        local.get    2
        local.get    7
        i32.store    12

        local.get    2        # dst += 16
        i32.const    16
        i32.add
        local.set    2

        local.get    1        # src += 16
        i32.const    16
        i32.add
        local.set    1

        local.get    0
        i32.const    -16
        i32.add              # count -= 16
        local.set    0

        local.get    0
        i32.const    16      # count >= 16
        i32.ge_s
        br_if        0       # loop
      end_loop
    end_if

    local.get    0
    i32.const    4       # if count >= 4
    i32.ge_s
    if
      loop
        local.get    1        # src
        i32.load     0        # load float from src
        local.set    4

        local.get    1        # src += 4
        i32.const    4
        i32.add
        local.set    1

        local.get    4        # (v >> 31) - 1) & v
        i32.const    31
        i32.shr_u
        local.set    5

        local.get    5
        i32.const    -1
        i32.add
        local.set    5

        local.get    4
        local.get    5
        i32.and
        local.set    4

        local.get    2        # dst
        local.get    4
        i32.store    0        # store float

        local.get    2        # dst += 4
        i32.const    4
        i32.add
        local.set    2

        local.get    0
        i32.const    -4
        i32.add              # count -= 4
        local.set    0

        local.get    0
        i32.const    4       # count >= 4
        i32.ge_s
        br_if        0       # loop
      end_loop
    end_if
END_FUNCTION xnn_f32_vrelu_ukernel__wasm32_shr_u4
