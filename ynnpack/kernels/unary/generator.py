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
from ynnpack.kernels.unary.erf import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.exp import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.kernels import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.log import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.sigmoid import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.sine_cosine import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.tanh import *  # pylint: disable=wildcard-import


def main(argv: Sequence[str]) -> None:
  output_src = argv[1]
  output_inc = argv[2]
  target = argv[3]

  # We want to flag kernels that are numerically consistent with all other
  # kernels of the same operator. Generally we can say the kernels are
  # consistent if:
  # 1. The kernel has zero tolerance for error (abs, round, etc.)
  # 2. For fp32 kernels that use fma, we consider kernels using fma to be
  # consistent. We provide emulated FMA kernels when FMA is not available.
  # These kernels are slow, but if consistent arithmetic is requested, we
  # prioritize that over performance.
  # 3. For fp64 kernels, we can't emulate fma, so we disable fma based kernels.
  consistent = "unary_flag::consistent_arithmetic"

  kernels = {
      "x86_sse2": [
          (abs_fp32, (8, 1), consistent),
          (abs_fp64, (4, 1), consistent),
          (convert_fp32_to_fp64, (8, 1), consistent),
          (convert_fp64_to_fp32, (8, 1), consistent),
          (erf_fp32, (16, 1)),
          (exp_fp32, (16, 1)),
          (exp_fp64, (8, 1), consistent),
          (expm1_fp32, (16, 1)),
          (expm1_fp64, (8, 1), consistent),
          (log_fp32, (16, 1)),
          (log_fp64, (4, 1), consistent),
          (log1p_fp32, (16, 1)),
          (log1p_fp64, (4, 1), consistent),
          (negate_fp32, (8, 1), consistent),
          (negate_fp64, (4, 1), consistent),
          (poly3_fp32, (16, 1)),
          (poly3_fp64, (8, 1), consistent),
          (reciprocal_square_root_fp32, (8, 1)),
          (reciprocal_square_root_fp64, (4, 1), consistent),
          (square_fp32, (8, 1), consistent),
          (square_fp64, (4, 1), consistent),
          (square_root_fp32, (8, 1)),
          (square_root_fp64, (4, 1), consistent),
          (sigmoid_fp32, (32, 1)),
          (sigmoid_fp64, (8, 1), consistent),
          (tanh_fp32, (16, 1)),
          (tanh_fp64, (8, 1), consistent),
      ],
      "x86_sse2_fma": [
          (erf_fp32, (16, 1), consistent),
          (expm1_fp32, (16, 1), consistent),
          (log_fp32, (16, 1), consistent),
          (log1p_fp32, (16, 1), consistent),
          (poly3_fp32, (16, 1), consistent),
          (reciprocal_square_root_fp32, (8, 1), consistent),
          (square_root_fp32, (8, 1), consistent),
          (sigmoid_fp32, (32, 1), consistent),
          (tanh_fp32, (16, 1), consistent),
      ],
      "x86_sse41": [
          (ceil_fp32, (8, 1), consistent),
          (ceil_fp64, (4, 1), consistent),
          (cosine_fp32, (32, 1)),
          (floor_fp32, (8, 1), consistent),
          (floor_fp64, (4, 1), consistent),
          (round_fp32, (8, 1), consistent),
          (round_fp64, (4, 1), consistent),
          (sine_fp32, (32, 1)),
      ],
      "x86_avx": [
          (abs_fp32, (16, 1), consistent),
          (abs_fp64, (8, 1), consistent),
          (ceil_fp32, (16, 1), consistent),
          (ceil_fp64, (8, 1), consistent),
          (convert_fp32_to_fp64, (16, 1), consistent),
          (convert_fp64_to_fp32, (16, 1), consistent),
          (cosine_fp32, (64, 1)),
          (erf_fp32, (64, 1)),
          (floor_fp32, (16, 1), consistent),
          (floor_fp64, (8, 1), consistent),
          (negate_fp32, (16, 1), consistent),
          (negate_fp64, (8, 1), consistent),
          (poly3_fp32, (16, 1)),
          (poly3_fp64, (8, 1), consistent),
          (reciprocal_square_root_fp32, (16, 1)),
          (reciprocal_square_root_fp64, (8, 1), consistent),
          (round_fp32, (16, 1), consistent),
          (round_fp64, (8, 1), consistent),
          (sine_fp32, (64, 1)),
          (square_fp32, (16, 1), consistent),
          (square_fp64, (8, 1), consistent),
          (square_root_fp32, (16, 1)),
          (square_root_fp64, (8, 1), consistent),
          (tanh_fp32, (32, 1)),
          (tanh_fp64, (16, 1), consistent),
      ],
      "x86_avx2": [
          (convert_bf16_to_fp32, (16, 1), consistent),
          (convert_fp32_to_bf16, (16, 1), consistent),
          (convert_int2_to_int8, (32, 1), consistent),
          (convert_int4_to_int8, (32, 1), consistent),
          (exp_fp32, (16, 1)),
          (exp_fp64, (16, 1), consistent),
          (expm1_fp32, (16, 1)),
          (expm1_fp64, (16, 1), consistent),
          (log_fp32, (32, 1)),
          (log_fp64, (16, 1), consistent),
          (log1p_fp32, (32, 1)),
          (log1p_fp64, (16, 1), consistent),
          (round_to_bf16_fp32, (16, 1), consistent),
          (sigmoid_fp32, (16, 1)),
          (sigmoid_fp64, (8, 1), consistent),
      ],
      "x86_fma3": [
          (cosine_fp32, (16, 1), consistent),
          (erf_fp32, (16, 1), consistent),
          (poly3_fp32, (16, 1), consistent),
          (poly3_fp64, (16, 1)),
          (sine_fp32, (16, 1), consistent),
      ],
      "x86_avx2_fma3": [
          (exp_fp32, (32, 1), consistent),
          (exp_fp64, (16, 1)),
          (expm1_fp32, (32, 1), consistent),
          (expm1_fp64, (16, 1)),
          (log_fp32, (32, 1), consistent),
          (log_fp64, (16, 1)),
          (log1p_fp32, (32, 1), consistent),
          (log1p_fp64, (16, 1)),
      ],
      "x86_f16c": [
          (convert_fp16_to_fp32, (16, 1), consistent),
          (convert_fp32_to_fp16, (16, 1), consistent),
      ],
      "x86_avx512": [
          (ceil_fp32, (32, 1), consistent),
          (ceil_fp64, (16, 1), consistent),
          (convert_bf16_to_fp32, (64, 1), consistent),
          (convert_fp32_to_bf16, (64, 1), consistent),
          (convert_fp32_to_fp64, (64, 1), consistent),
          (convert_fp64_to_fp32, (64, 1), consistent),
          (cosine_fp32, (32, 1), consistent),
          (erf_fp32, (32, 1), consistent),
          (exp_fp32, (32, 1), consistent),
          (exp_fp64, (16, 1)),
          (expm1_fp32, (32, 1), consistent),
          (expm1_fp64, (16, 1)),
          (log_fp32, (32, 1), consistent),
          (log_fp64, (16, 1)),
          (log1p_fp32, (32, 1), consistent),
          (log1p_fp64, (16, 1)),
          (floor_fp32, (32, 1), consistent),
          (floor_fp64, (16, 1), consistent),
          (negate_fp32, (32, 1), consistent),
          (negate_fp64, (16, 1), consistent),
          (poly3_fp32, (32, 1), consistent),
          (poly3_fp64, (32, 1)),
          (reciprocal_square_root_fp32, (32, 1), consistent),
          (reciprocal_square_root_fp64, (32, 1)),
          (round_fp32, (32, 1), consistent),
          (round_fp64, (16, 1), consistent),
          (round_to_bf16_fp32, (32, 1), consistent),
          (sine_fp32, (32, 1), consistent),
          (square_fp32, (32, 1), consistent),
          (square_fp64, (16, 1), consistent),
          (square_root_fp32, (32, 1), consistent),
          (square_root_fp64, (16, 1)),
          (sigmoid_fp32, (32, 1), consistent),
          (sigmoid_fp64, (16, 1)),
          (tanh_fp32, (32, 1), consistent),
          (tanh_fp64, (16, 1)),
          (convert_int2_to_int8, (64, 1), consistent),
          (convert_int4_to_int8, (64, 1), consistent),
      ],
      "x86_avx512bf16": [
          (convert_fp32_to_bf16, (32, 1), consistent),
          (round_to_bf16_fp32, (32, 1), consistent),
      ],
      "arm_neon": [
          (abs_fp32, (8, 1), consistent),
          (ceil_fp32, (8, 1), consistent),
          (convert_bf16_to_fp32, (16, 1), consistent),
          (convert_fp32_to_bf16, (16, 1), consistent),
          (cosine_fp32, (32, 1), consistent),
          (erf_fp32, (16, 1), consistent),
          (exp_fp32, (16, 1), consistent),
          (expm1_fp32, (16, 1), consistent),
          (log_fp32, (16, 1), consistent),
          (floor_fp32, (8, 1), consistent),
          (negate_fp32, (8, 1), consistent),
          (poly3_fp32, (32, 1), consistent),
          (reciprocal_square_root_fp32, (8, 1), consistent),
          (round_fp32, (8, 1), consistent),
          (round_to_bf16_fp32, (16, 1), consistent),
          (sine_fp32, (32, 1), consistent),
          (square_fp32, (8, 1), consistent),
          (square_root_fp32, (8, 1), consistent),
          (sigmoid_fp32, (8, 1), consistent),
          (tanh_fp32, (16, 1), consistent),
      ],
      "arm64_neon": [
          (abs_fp64, (4, 1), consistent),
          (ceil_fp64, (4, 1), consistent),
          (convert_fp32_to_fp64, (16, 1), consistent),
          (convert_fp64_to_fp32, (16, 1), consistent),
          (exp_fp64, (8, 1), consistent),
          (expm1_fp64, (8, 1), consistent),
          (log_fp64, (8, 1), consistent),
          (floor_fp64, (4, 1), consistent),
          (negate_fp64, (4, 1), consistent),
          (poly3_fp64, (16, 1), consistent),
          (reciprocal_square_root_fp64, (4, 1), consistent),
          (round_fp64, (4, 1), consistent),
          (sigmoid_fp64, (4, 1), consistent),
          (square_fp64, (4, 1), consistent),
          (square_root_fp64, (4, 1), consistent),
          (tanh_fp64, (8, 1), consistent),
      ],
      "arm_neonbf16": [
          (convert_fp32_to_bf16, (64, 1), consistent),
          (round_to_bf16_fp32, (64, 1), consistent),
      ],
      "arm_neonfp16": [
          (convert_fp16_to_fp32, (16, 1), consistent),
          (convert_fp32_to_fp16, (16, 1), consistent),
      ],
      "wasm_simd128": [
          (abs_fp32, (8, 1)),
          (ceil_fp32, (8, 1)),
          (cosine_fp32, (8, 1)),
          (erf_fp32, (8, 1)),
          (exp_fp32, (8, 1)),
          (expm1_fp32, (8, 1)),
          (floor_fp32, (8, 1)),
          (negate_fp32, (8, 1)),
          (poly3_fp32, (8, 1)),
          (reciprocal_square_root_fp32, (8, 1)),
          (round_fp32, (8, 1)),
          (sine_fp32, (8, 1)),
          (square_fp32, (8, 1)),
          (square_root_fp32, (8, 1)),
          (sigmoid_fp32, (8, 1)),
          (tanh_fp32, (8, 1)),
      ],
  }[target]

  generate_elementwise_kernels(output_src, output_inc, target, kernels)


if __name__ == "__main__":
  main(sys.argv)
