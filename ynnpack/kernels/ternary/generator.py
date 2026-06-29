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
          # go/keep-sorted start
          (clamp_fp32_fp32_fp32, (8, 1)),
          (clamp_fp64_fp64_fp64, (4, 1)),
          (dequantize_int8_to_fp32, (16, 1)),
          (multiply_add_fp32_fp32_fp32, (8, 1)),
          (multiply_add_fp64_fp64_fp64, (4, 1)),
          (multiply_int32_fp32_fp32, (8, 1)),
          (quantize_fp32_to_int8, (16, 1)),
          # go/keep-sorted end
      ],
      "x86_sse41": [
          # go/keep-sorted start
          (dequantize_uint8_to_fp32, (16, 1)),
          (quantize_fp32_to_uint8, (16, 1)),
          (subtract_multiply_int32_int32_int32, (8, 1)),
          # go/keep-sorted end
      ],
      "x86_avx": [
          # go/keep-sorted start
          (clamp_fp32_fp32_fp32, (16, 1)),
          (clamp_fp64_fp64_fp64, (8, 1)),
          (multiply_add_fp32_fp32_fp32, (16, 1)),
          (multiply_add_fp64_fp64_fp64, (8, 1)),
          # go/keep-sorted end
      ],
      "x86_avx2": [
          # go/keep-sorted start
          (dequantize_int8_to_bf16, (32, 1)),
          (dequantize_int8_to_fp32, (32, 1)),
          (dequantize_uint8_to_bf16, (32, 1)),
          (dequantize_uint8_to_fp32, (32, 1)),
          (multiply_int32_fp32_fp32, (16, 1)),
          (quantize_bf16_to_int8, (32, 1)),
          (quantize_bf16_to_uint8, (32, 1)),
          (quantize_fp32_to_int8, (32, 1)),
          (quantize_fp32_to_uint8, (32, 1)),
          (subtract_multiply_int32_int32_int32, (16, 1)),
          # go/keep-sorted end
      ],
      "x86_avx512": [
          # go/keep-sorted start
          (clamp_fp32_fp32_fp32, (32, 1)),
          (clamp_fp64_fp64_fp64, (16, 1)),
          (dequantize_int8_to_bf16, (64, 1)),
          (dequantize_int8_to_fp16, (64, 1)),
          (dequantize_int8_to_fp32, (64, 1)),
          (dequantize_uint8_to_bf16, (64, 1)),
          (dequantize_uint8_to_fp16, (64, 1)),
          (dequantize_uint8_to_fp32, (64, 1)),
          (multiply_add_fp32_fp32_fp32, (32, 1)),
          (multiply_add_fp64_fp64_fp64, (16, 1)),
          (multiply_int32_fp32_fp32, (32, 1)),
          (quantize_bf16_to_int8, (64, 1)),
          (quantize_bf16_to_uint8, (64, 1)),
          (quantize_fp16_to_int8, (64, 1)),
          (quantize_fp16_to_uint8, (64, 1)),
          (quantize_fp32_to_int8, (64, 1)),
          (quantize_fp32_to_uint8, (64, 1)),
          (subtract_multiply_int32_int32_int32, (32, 1)),
          # go/keep-sorted end
      ],
      "x86_avx512bf16": [
          # go/keep-sorted start
          (dequantize_int8_to_bf16, (64, 1)),
          (dequantize_uint8_to_bf16, (64, 1)),
          # go/keep-sorted end
      ],
      "x86_f16c": [
          # go/keep-sorted start
          (dequantize_int8_to_fp16, (32, 1)),
          (dequantize_uint8_to_fp16, (32, 1)),
          (quantize_fp16_to_int8, (32, 1)),
          (quantize_fp16_to_uint8, (32, 1)),
          # go/keep-sorted end
      ],
      "arm_neon": [
          # go/keep-sorted start
          (clamp_fp32_fp32_fp32, (32, 1)),
          (dequantize_int8_to_bf16, (16, 1)),
          (dequantize_int8_to_fp32, (16, 1)),
          (dequantize_uint8_to_bf16, (16, 1)),
          (dequantize_uint8_to_fp32, (16, 1)),
          (multiply_add_fp32_fp32_fp32, (32, 1)),
          (multiply_int32_fp32_fp32, (64, 1)),
          (quantize_bf16_to_int8, (16, 1)),
          (quantize_bf16_to_uint8, (16, 1)),
          (quantize_fp32_to_int8, (16, 1)),
          (subtract_multiply_int32_int32_int32, (8, 1)),
          # go/keep-sorted end
      ],
      "arm_neonbf16": [
          # go/keep-sorted start
          (dequantize_int8_to_bf16, (16, 1)),
          (dequantize_uint8_to_bf16, (16, 1)),
          # go/keep-sorted end
      ],
      "arm_neonfp16": [
          # go/keep-sorted start
          (dequantize_int8_to_fp16, (16, 1)),
          (dequantize_uint8_to_fp16, (16, 1)),
          (quantize_fp16_to_int8, (16, 1)),
          (quantize_fp16_to_uint8, (16, 1)),
          # go/keep-sorted end
      ],
      "arm64_neon": [
          # go/keep-sorted start
          (clamp_fp64_fp64_fp64, (16, 1)),
          (multiply_add_fp64_fp64_fp64, (16, 1)),
          # go/keep-sorted end
      ],
      "wasm_simd128": [
          # go/keep-sorted start
          (clamp_fp32_fp32_fp32, (8, 1)),
          (dequantize_int8_to_fp32, (16, 1)),
          (dequantize_uint8_to_fp32, (16, 1)),
          (multiply_add_fp32_fp32_fp32, (8, 1)),
          (multiply_int32_fp32_fp32, (8, 1)),
          (quantize_fp32_to_int8, (16, 1)),
          (subtract_multiply_int32_int32_int32, (8, 1)),
          # go/keep-sorted end
      ],
  }[target]

  generate_elementwise_kernels(output_src, output_inc, target, kernels)


if __name__ == "__main__":
  main(sys.argv)
