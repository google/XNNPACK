# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

# void xnn_f32_vrelu_ukernel__wasm32_shr_u2(
#     size_t batch,             0
#     const float* input,       1
#     float* output,             2
#     const union params)   3 unused

# locals
#     float value0          4
#     float value1          5
#     float mask0           6
#     float mask1           7


BEGIN_FUNCTION  xnn_f32_vrelu_ukernel__wasm32_shr_u2
    .functype   xnn_f32_vrelu_ukernel__wasm32_shr_u2 (i32, i32, i32, i32) -> ()
    .local      i32, i32, i32, i32

    local.get    0
    i32.const    8       # count >= 8
    i32.ge_s
    if
      loop
        local.get    1        # src
        i32.load     0        # load float from src
        local.set    4
        local.get    1        # src
        i32.load     4        # load 2nd float from src + 4
        local.set    5

        local.get    1        # src += 8
        i32.const    8
        i32.add
        local.set    1

        local.get    4        # (v >> 31) - 1) & v
        i32.const    31
        i32.shr_u
        local.set    6
        local.get    5        # 2nd mask
        i32.const    31
        i32.shr_u
        local.set    7

        local.get    6
        i32.const    -1
        i32.add
        local.set    6
        local.get    7
        i32.const    -1
        i32.add
        local.set    7

        local.get    4
        local.get    6
        i32.and
        local.set    4
        local.get    5
        local.get    7
        i32.and
        local.set    5

        local.get    2        # dst
        local.get    4
        i32.store    0        # store float
        local.get    2        # dst
        local.get    5
        i32.store    4        # store 2nd float

        local.get    2        # dst += 8
        i32.const    8
        i32.add
        local.set    2

        local.get    0
        i32.const    -8
        i32.add              # count -= 8
        local.set    0

        local.get    0
        i32.const    8       # count >= 8
        i32.ge_s
        br_if        0       # loop
      end_loop
    end_if

    local.get    0
    i32.const    4       # if count >= 4
    i32.ge_s
    if
      local.get    1        # src
      i32.load     0        # load float from src
      local.set    4

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
    end_if
END_FUNCTION xnn_f32_vrelu_ukernel__wasm32_shr_u2
