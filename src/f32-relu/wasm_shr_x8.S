# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_relu_ukernel__wasm32_shr_x8(
#     size_t n,             0
#     const float* x,       1
#     float* y,             2
#     const union params)   3 unused

# locals
#     float value0          4
#     float value1          5
#     float value2          6
#     float value3          7

#     float value4          8
#     float value5          9
#     float value6          10
#     float value7          11

#     float mask0           12
#     float mask1           13
#     float mask2           14
#     float mask3           15

BEGIN_FUNCTION  xnn_f32_relu_ukernel__wasm32_shr_x8
    .functype   xnn_f32_relu_ukernel__wasm32_shr_x8 (i32, i32, i32, i32) -> ()
    .local      i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32

    local.get    0
    i32.const    32      # count >= 32
    i32.ge_s

    if
      loop
        # Mainloop - 8 floats per loop

        local.get    1
        i32.load     0        # load 8 floats from src into locals 4-11
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
        local.get    1
        i32.load     16
        local.set    8
        local.get    1
        i32.load     20
        local.set    9
        local.get    1
        i32.load     24
        local.set    10
        local.get    1
        i32.load     28
        local.set    11

        local.get    4        # (v >> 31) - 1) & v
        i32.const    31
        i32.shr_u
        local.set    12
        local.get    5
        i32.const    31
        i32.shr_u
        local.set    13
        local.get    6
        i32.const    31
        i32.shr_u
        local.set    14
        local.get    7
        i32.const    31
        i32.shr_u
        local.set    15

        local.get    12
        i32.const    -1
        i32.add
        local.set    12
        local.get    13
        i32.const    -1
        i32.add
        local.set    13
        local.get    14
        i32.const    -1
        i32.add
        local.set    14
        local.get    15
        i32.const    -1
        i32.add
        local.set    15

        local.get    4
        local.get    12
        i32.and
        local.set    4
        local.get    5
        local.get    13
        i32.and
        local.set    5
        local.get    6
        local.get    14
        i32.and
        local.set    6
        local.get    7
        local.get    15
        i32.and
        local.set    7

        local.get    8        # (v >> 31) - 1) & v
        i32.const    31
        i32.shr_u
        local.set    12
        local.get    9
        i32.const    31
        i32.shr_u
        local.set    13
        local.get    10
        i32.const    31
        i32.shr_u
        local.set    14
        local.get    11
        i32.const    31
        i32.shr_u
        local.set    15

        local.get    12
        i32.const    -1
        i32.add
        local.set    12
        local.get    13
        i32.const    -1
        i32.add
        local.set    13
        local.get    14
        i32.const    -1
        i32.add
        local.set    14
        local.get    15
        i32.const    -1
        i32.add
        local.set    15

        local.get    8
        local.get    12
        i32.and
        local.set    8
        local.get    9
        local.get    13
        i32.and
        local.set    9
        local.get    10
        local.get    14
        i32.and
        local.set    10
        local.get    11
        local.get    15
        i32.and
        local.set    11

        local.get    2
        local.get    4
        i32.store    0        # store 8 floats
        local.get    2
        local.get    5
        i32.store    4
        local.get    2
        local.get    6
        i32.store    8
        local.get    2
        local.get    7
        i32.store    12
        local.get    2
        local.get    8
        i32.store    16
        local.get    2
        local.get    9
        i32.store    20
        local.get    2
        local.get    10
        i32.store    24
        local.get    2
        local.get    11
        i32.store    28

        local.get    2        # dst += 32
        i32.const    32
        i32.add
        local.set    2

        local.get    1        # src += 32
        i32.const    32
        i32.add
        local.set    1

        local.get    0
        i32.const    -32
        i32.add              # count -= 32
        local.set    0

        local.get    0
        i32.const    32      # count >= 32
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
END_FUNCTION xnn_f32_relu_ukernel__wasm32_shr_x8
