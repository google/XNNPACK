# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unary kernel generators."""

# pylint: disable=undefined-variable

from collections.abc import Sequence
import sys

from ynnpack.kernels.elementwise.generator import generate_elementwise_kernels
from ynnpack.kernels.unary.convert import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.exp import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.kernels import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.sigmoid import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.sine_cosine import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.tanh import *  # pylint: disable=wildcard-import


def main(argv: Sequence[str]) -> None:
  output_src = argv[1]
  output_inc = argv[2]
  target = argv[3]

  kernels = {
      "x86_sse2": [
          (abs_fp32, (8, 1)),
          (erf_fp32, (8, 1)),
          (exp_fp32, (8, 1)),
          (log_fp32, (4, 1)),
          (negate_fp32, (8, 1)),
          (poly3_fp32, (8, 1)),
          (reciprocal_square_root_fp32, (8, 1)),
          (square_root_fp32, (8, 1)),
          (sigmoid_fp32, (8, 1)),
          (tanh_fp32, (8, 1)),
      ],
      "x86_sse41": [
          (ceil_fp32, (8, 1)),
          (cosine_fp32, (32, 1)),
          (floor_fp32, (8, 1)),
          (round_fp32, (8, 1)),
          (sine_fp32, (32, 1)),
      ],
      "x86_avx": [
          (abs_fp32, (16, 1)),
          (ceil_fp32, (16, 1)),
          (cosine_fp32, (16, 1)),
          (erf_fp32, (16, 1)),
          (floor_fp32, (16, 1)),
          (negate_fp32, (16, 1)),
          (poly3_fp32, (16, 1)),
          (reciprocal_square_root_fp32, (16, 1)),
          (round_fp32, (16, 1)),
          (sine_fp32, (16, 1)),
          (square_root_fp32, (16, 1)),
          (tanh_fp32, (16, 1)),
      ],
      "x86_avx2": [
          (convert_bf16_to_fp32, (16, 1)),
          (convert_fp32_to_bf16, (16, 1)),
          (exp_fp32, (16, 1)),
          (log_fp32, (8, 1)),
          (poly3_fp32, (16, 1)),
          (sigmoid_fp32, (16, 1)),
      ],
      "x86_fma3": [
          (cosine_fp32, (16, 1)),
          (erf_fp32, (16, 1)),
          (poly3_fp32, (16, 1)),
          (reciprocal_square_root_fp32, (16, 1)),
          (sine_fp32, (16, 1)),
      ],
      "x86_avx2_fma3": [
          (log_fp32, (8, 1)),
      ],
      "x86_f16c": [
          (convert_fp16_to_fp32, (16, 1)),
          (convert_fp32_to_fp16, (16, 1)),
      ],
      "x86_avx512": [
          (ceil_fp32, (32, 1)),
          (convert_bf16_to_fp32, (64, 1)),
          (convert_fp32_to_bf16, (64, 1)),
          (cosine_fp32, (32, 1)),
          (erf_fp32, (32, 1)),
          (exp_fp32, (32, 1)),
          (log_fp32, (32, 1)),
          (floor_fp32, (32, 1)),
          (negate_fp32, (32, 1)),
          (poly3_fp32, (32, 1)),
          (reciprocal_square_root_fp32, (32, 1)),
          (round_fp32, (32, 1)),
          (sine_fp32, (32, 1)),
          (square_root_fp32, (32, 1)),
          (sigmoid_fp32, (32, 1)),
          (tanh_fp32, (32, 1)),
      ],
      "x86_avx512bf16": [
          (convert_fp32_to_bf16, (32, 1)),
      ],
      "arm_neon": [
          (abs_fp32, (8, 1)),
          (ceil_fp32, (8, 1)),
          (convert_bf16_to_fp32, (16, 1)),
          (convert_fp32_to_bf16, (16, 1)),
          (cosine_fp32, (8, 1)),
          (erf_fp32, (8, 1)),
          (exp_fp32, (8, 1)),
          (log_fp32, (8, 1)),
          (floor_fp32, (8, 1)),
          (negate_fp32, (8, 1)),
          (poly3_fp32, (8, 1)),
          (reciprocal_square_root_fp32, (8, 1)),
          (round_fp32, (8, 1)),
          (sine_fp32, (8, 1)),
          (square_root_fp32, (8, 1)),
          (sigmoid_fp32, (8, 1)),
          (tanh_fp32, (8, 1)),
      ],
      "arm_neonbf16": [
          (convert_fp32_to_bf16, (16, 1)),
      ],
      "arm_neonfp16": [
          (convert_fp16_to_fp32, (16, 1)),
          (convert_fp32_to_fp16, (16, 1)),
      ],
      "wasm_simd128": [
          (abs_fp32, (8, 1)),
          (ceil_fp32, (8, 1)),
          (cosine_fp32, (8, 1)),
          (erf_fp32, (8, 1)),
          (exp_fp32, (8, 1)),
          (floor_fp32, (8, 1)),
          (negate_fp32, (8, 1)),
          (poly3_fp32, (8, 1)),
          (reciprocal_square_root_fp32, (8, 1)),
          (round_fp32, (8, 1)),
          (sine_fp32, (8, 1)),
          (square_root_fp32, (8, 1)),
          (sigmoid_fp32, (8, 1)),
          (tanh_fp32, (8, 1)),
      ],
  }[target]

  generate_elementwise_kernels(output_src, output_inc, target, kernels)


if __name__ == "__main__":
  main(sys.argv)
