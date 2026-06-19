# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory directory of this source tree.

"""Ternary kernel generators."""

# pylint: disable=undefined-variable

from collections.abc import Sequence
import sys

from ynnpack.kernels.elementwise.generator import generate_elementwise_kernels
from ynnpack.kernels.ternary.convert import *  # pylint: disable=wildcard-import
from ynnpack.kernels.ternary.kernels import *  # pylint: disable=wildcard-import


def main(argv: Sequence[str]) -> None:
  output_src = argv[1]
  output_inc = argv[2]
  target = argv[3]

  kernels = {
      "x86_sse2": [
          (quantize_fp32_to_int8, (16, 1)),
          (multiply_int32_fp32_fp32, (8, 1)),
          (multiply_add_fp32_fp32_fp32, (8, 1)),
          (multiply_add_fp64_fp64_fp64, (4, 1)),
          (clamp_fp32_fp32_fp32, (8, 1)),
          (clamp_fp64_fp64_fp64, (4, 1)),
      ],
      "x86_sse41": [
          (subtract_multiply_int32_int32_int32, (8, 1)),
          (quantize_fp32_to_uint8, (16, 1)),
      ],
      "x86_avx": [
          (multiply_add_fp32_fp32_fp32, (16, 1)),
          (multiply_add_fp64_fp64_fp64, (8, 1)),
          (clamp_fp32_fp32_fp32, (16, 1)),
          (clamp_fp64_fp64_fp64, (8, 1)),
      ],
      "x86_avx2": [
          (quantize_fp32_to_int8, (32, 1)),
          (quantize_fp32_to_uint8, (32, 1)),
          (multiply_int32_fp32_fp32, (16, 1)),
          (subtract_multiply_int32_int32_int32, (16, 1)),
      ],
      "x86_avx512": [
          (multiply_int32_fp32_fp32, (32, 1)),
          (subtract_multiply_int32_int32_int32, (32, 1)),
          (multiply_add_fp32_fp32_fp32, (32, 1)),
          (multiply_add_fp64_fp64_fp64, (16, 1)),
          (clamp_fp32_fp32_fp32, (32, 1)),
          (clamp_fp64_fp64_fp64, (16, 1)),
          (quantize_fp32_to_int8, (64, 1)),
          (quantize_fp32_to_uint8, (64, 1)),
      ],
      "arm_neon": [
          (quantize_fp32_to_int8, (16, 1)),
          (multiply_int32_fp32_fp32, (64, 1)),
          (subtract_multiply_int32_int32_int32, (8, 1)),
          (multiply_add_fp32_fp32_fp32, (32, 1)),
          (clamp_fp32_fp32_fp32, (32, 1)),
      ],
      "arm64_neon": [
          (multiply_add_fp64_fp64_fp64, (16, 1)),
          (clamp_fp64_fp64_fp64, (16, 1)),
      ],
      "wasm_simd128": [
          (quantize_fp32_to_int8, (16, 1)),
          (multiply_int32_fp32_fp32, (8, 1)),
          (subtract_multiply_int32_int32_int32, (8, 1)),
          (multiply_add_fp32_fp32_fp32, (8, 1)),
          (clamp_fp32_fp32_fp32, (8, 1)),
      ],
  }[target]

  generate_elementwise_kernels(output_src, output_inc, target, kernels)


if __name__ == "__main__":
  main(sys.argv)
