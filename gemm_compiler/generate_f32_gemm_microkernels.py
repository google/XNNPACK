#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from gemm_compiler import avx512f_template
from gemm_compiler import fma3_template
from gemm_compiler import generate
from gemm_compiler import neonfma_template

"""Generates f32 assembly gemm microkernels."""


output_base = 'src/f32-gemm/gen/'


def generate_f32_gemm_microkernels():
  if '/bazel-out/' in os.getcwd():
    os.chdir(os.environ['BUILD_WORKING_DIRECTORY'])

  for nr in range(16, 33, 16):
    for mr in range(1, 12):
      generate.generate_gemm_microkernel(
          M=mr,
          N=nr,
          isa=avx512f_template.Avx512F(),
          output_file=os.path.join(
              output_base,
              f'f32-gemm-{mr}x{nr}-minmax-asm-amd64-avx512f-broadcast.S',
          ),
      )

  # not enough SIMD registers to go above 5x64
  for mr in range(1, 6):
    generate.generate_gemm_microkernel(
        M=mr,
        N=64,
        isa=avx512f_template.Avx512F(),
        output_file=os.path.join(
            output_base,
            f'f32-gemm-{mr}x64-minmax-asm-amd64-avx512f-broadcast.S',
        ),
    )

  for unroll in {1, 2, 4}:
    decrement = 32 * unroll
    for mr in range(1, 6):
      generate.generate_gemm_microkernel(
          M=mr,
          N=16,
          isa=neonfma_template.NeonFma(unroll),
          output_file=os.path.join(
              output_base,
              f'f32-gemm-{mr}x16-minmax-asm-aarch64-neonfma-ld{decrement}.S',
          ),
      )

  for unroll in {1, 2, 4}:
    decrement = 32 * unroll
    for mr in range(1, 9):
      generate.generate_gemm_microkernel(
          M=mr,
          N=8,
          isa=neonfma_template.NeonFma(unroll),
          output_file=os.path.join(
              output_base,
              f'f32-gemm-{mr}x8-minmax-asm-aarch64-neonfma-ld{decrement}-2.S',
          ),
      )

  # Generate C2 variants.
  for mr in range(1, 12):
    generate.generate_gemm_microkernel(
        M=mr,
        N=16,
        isa=avx512f_template.Avx512FC(c=2),
        output_file=os.path.join(
            output_base,
            f'f32-gemm-{mr}x16c2-minmax-asm-amd64-avx512f-broadcast.S',
        ),
    )

  # not enough SIMD registers to go above 5x32
  for mr in range(1, 6):
    generate.generate_gemm_microkernel(
        M=mr,
        N=32,
        isa=avx512f_template.Avx512FC(c=2),
        output_file=os.path.join(
            output_base,
            f'f32-gemm-{mr}x32c2-minmax-asm-amd64-avx512f-broadcast.S',
        ),
    )

