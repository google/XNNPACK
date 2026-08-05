# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Binary kernel generators."""

# pylint: disable=undefined-variable

from collections.abc import Sequence
import sys

from ynnpack.kernels.binary.kernels import *  # pylint: disable=wildcard-import
from ynnpack.kernels.elementwise.generator import generate_elementwise_kernels


def main(argv: Sequence[str]) -> None:
  output_src = argv[1]
  output_inc = argv[2]
  target = argv[3]

  kernels = {
      "hexagon_hvx": [
          # go/keep-sorted start
          (add_fp32, (32, 1)),
          (subtract_fp32, (32, 1)),
          # go/keep-sorted end
      ],
      "x86_sse2": [
          # go/keep-sorted start
          (add_fp32, (8, 1)),
          (add_fp64, (4, 1)),
          (copysign_fp32, (8, 1)),
          (copysign_fp64, (4, 1)),
          (divide_fp32, (8, 1)),
          (divide_fp64, (4, 1)),
          (exp_subtract_fp32, (16, 1)),
          (max_fp32, (8, 1)),
          (max_fp64, (4, 1)),
          (min_fp32, (8, 1)),
          (min_fp64, (4, 1)),
          (multiply_fp32, (8, 1)),
          (multiply_fp64, (4, 1)),
          (multiply_int32_fp32, (8, 1)),
          (squared_difference_fp32, (8, 1)),
          (squared_difference_fp64, (4, 1)),
          (subtract_fp32, (8, 1)),
          (subtract_fp64, (4, 1)),
          # go/keep-sorted end
      ],
      "x86_avx": [
          # go/keep-sorted start
          (add_fp32, (16, 1)),
          (add_fp64, (8, 1)),
          (copysign_fp32, (16, 1)),
          (copysign_fp64, (8, 1)),
          (divide_fp32, (16, 1)),
          (divide_fp64, (8, 1)),
          (exp_subtract_fp32, (16, 1)),
          (max_fp32, (16, 1)),
          (max_fp64, (8, 1)),
          (min_fp32, (16, 1)),
          (min_fp64, (8, 1)),
          (multiply_fp32, (16, 1)),
          (multiply_fp64, (8, 1)),
          (squared_difference_fp32, (16, 1)),
          (squared_difference_fp64, (8, 1)),
          (subtract_fp32, (16, 1)),
          (subtract_fp64, (8, 1)),
          # go/keep-sorted end
      ],
      "x86_fma3": [
          # go/keep-sorted start
          (exp_subtract_fp32, (16, 1)),
          # go/keep-sorted end
      ],
      "x86_avx2": [
          # go/keep-sorted start
          (divide_bf16_fp32, (32, 1)),
          (divide_bf16_fp32_bf16, (32, 1)),
          (exp_subtract_fp32, (16, 1)),
          (multiply_bf16_fp32, (64, 1)),
          (multiply_bf16_fp32_bf16, (32, 1)),
          (multiply_int32_fp32, (16, 1)),
          (subtract_bf16_fp32, (32, 1)),
          (subtract_fp32_bf16_bf16, (32, 1)),
          # go/keep-sorted end
      ],
      "x86_avx2_fma3": [
          # go/keep-sorted start
          (exp_subtract_fp32, (16, 1)),
          # go/keep-sorted end
      ],
      "x86_avx512": [
          # go/keep-sorted start
          (add_fp32, (32, 1)),
          (add_fp64, (16, 1)),
          (divide_bf16_fp32, (64, 1)),
          (divide_bf16_fp32_bf16, (64, 1)),
          (divide_fp32, (32, 1)),
          (divide_fp64, (16, 1)),
          (exp_subtract_fp32, (32, 1)),
          (max_fp32, (32, 1)),
          (max_fp64, (16, 1)),
          (min_fp32, (32, 1)),
          (min_fp64, (16, 1)),
          (multiply_bf16_fp32, (64, 1)),
          (multiply_bf16_fp32_bf16, (64, 1)),
          (multiply_fp32, (32, 1)),
          (multiply_fp64, (16, 1)),
          (multiply_int32_fp32, (32, 1)),
          (squared_difference_fp32, (32, 1)),
          (squared_difference_fp64, (16, 1)),
          (subtract_bf16_fp32, (64, 1)),
          (subtract_fp32, (32, 1)),
          (subtract_fp32_bf16_bf16, (64, 1)),
          (subtract_fp64, (16, 1)),
          # go/keep-sorted end
      ],
      "x86_avx512bf16": [
          # go/keep-sorted start
          (divide_bf16_fp32_bf16, (64, 1)),
          (multiply_bf16_fp32_bf16, (64, 1)),
          (subtract_fp32_bf16_bf16, (64, 1)),
          # go/keep-sorted end
      ],
      "arm_neon": [
          # go/keep-sorted start
          (add_fp32, (8, 1)),
          (copysign_fp32, (8, 1)),
          (divide_bf16_fp32, (32, 1)),
          (divide_bf16_fp32_bf16, (16, 1)),
          (divide_fp32, (8, 1)),
          (exp_subtract_fp32, (16, 1)),
          (max_fp32, (8, 1)),
          (min_fp32, (8, 1)),
          (multiply_bf16_fp32, (32, 1)),
          (multiply_bf16_fp32_bf16, (16, 1)),
          (multiply_fp32, (8, 1)),
          (multiply_int32_fp32, (8, 1)),
          (squared_difference_fp32, (32, 1)),
          (subtract_bf16_fp32, (16, 1)),
          (subtract_fp32, (8, 1)),
          # go/keep-sorted end
      ],
      "arm_neonfma": [
          # go/keep-sorted start
          (exp_subtract_fp32, (16, 1)),
          # go/keep-sorted end
      ],
      "arm64_neon": [
          # go/keep-sorted start
          (add_fp64, (4, 1)),
          (copysign_fp64, (4, 1)),
          (divide_fp64, (4, 1)),
          (max_fp64, (4, 1)),
          (min_fp64, (4, 1)),
          (multiply_fp64, (4, 1)),
          (squared_difference_fp64, (16, 1)),
          (subtract_fp64, (4, 1)),
          # go/keep-sorted end
      ],
      "arm_neonbf16": [
          # go/keep-sorted start
          (divide_bf16_fp32_bf16, (16, 1)),
          (multiply_bf16_fp32_bf16, (64, 1)),
          # go/keep-sorted end
      ],
      "wasm_simd128": [
          # go/keep-sorted start
          (add_fp32, (8, 1)),
          (copysign_fp32, (8, 1)),
          (divide_fp32, (8, 1)),
          (exp_subtract_fp32, (8, 1)),
          (max_fp32, (8, 1)),
          (min_fp32, (8, 1)),
          (multiply_fp32, (8, 1)),
          (multiply_int32_fp32, (8, 1)),
          (squared_difference_fp32, (8, 1)),
          (subtract_fp32, (8, 1)),
          # go/keep-sorted end
      ],
  }[target]

  generate_elementwise_kernels(output_src, output_inc, target, kernels)


if __name__ == "__main__":
  main(sys.argv)
