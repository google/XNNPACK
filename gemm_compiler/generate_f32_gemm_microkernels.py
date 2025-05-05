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


output_base = 'src/f32-gemm/gen/'


def generate_f32_gemm_microkernels():
  """Generates f32 assembly gemm microkernels."""
  if '/bazel-out/' in os.getcwd():
    os.chdir(os.environ['BUILD_WORKING_DIRECTORY'])

  nr = 8
  for mr in range(1, 10):
    generate.generate_gemm_microkernel(
        isa=fma3_template.Fma3(mr, nr, c=1),
        output_file=os.path.join(
            output_base,
            f'f32-gemm-{mr}x{nr}-minmax-asm-amd64-fma3-broadcast.S',
        ),
    )

  nr = 16
  for mr in range(1, 5):
    generate.generate_gemm_microkernel(
        isa=fma3_template.Fma3(mr, nr, c=1),
        output_file=os.path.join(
            output_base,
            f'f32-gemm-{mr}x{nr}-minmax-asm-amd64-fma3-broadcast.S',
        ),
    )

  for nr in range(16, 33, 16):
    for mr in range(1, 12):
      generate.generate_gemm_microkernel(
          isa=avx512f_template.Avx512F(mr, nr, c=1),
          output_file=os.path.join(
              output_base,
              f'f32-gemm-{mr}x{nr}-minmax-asm-amd64-avx512f-broadcast.S',
          ),
      )

  # not enough SIMD registers to go above 5x64
  for mr in range(1, 6):
    generate.generate_gemm_microkernel(
        isa=avx512f_template.Avx512F(mr, n=64, c=1),
        output_file=os.path.join(
            output_base,
            f'f32-gemm-{mr}x64-minmax-asm-amd64-avx512f-broadcast.S',
        ),
    )

  for unroll in [1, 2, 4]:
    decrement = 32 * unroll
    for mr in range(1, 6):
      generate.generate_gemm_microkernel(
          isa=neonfma_template.NeonFma(mr, n=16, unroll_factor=unroll),
          output_file=os.path.join(
              output_base,
              f'f32-gemm-{mr}x16-minmax-asm-aarch64-neonfma-ld{decrement}.S',
          ),
      )

  for unroll in {1, 2, 4}:
    decrement = 32 * unroll
    for mr in range(1, 9):
      generate.generate_gemm_microkernel(
          isa=neonfma_template.NeonFma(mr, n=8, unroll_factor=unroll),
          output_file=os.path.join(
              output_base,
              f'f32-gemm-{mr}x8-minmax-asm-aarch64-neonfma-ld{decrement}-2.S',
          ),
      )

  # Generate C2 variants.
  for mr in range(1, 12):
    generate.generate_gemm_microkernel(
        isa=avx512f_template.Avx512FC(mr, n=16, c=2),
        output_file=os.path.join(
            output_base,
            f'f32-gemm-{mr}x16c2-minmax-asm-amd64-avx512f-broadcast.S',
        ),
    )

  # not enough SIMD registers to go above 5x32
  for mr in range(1, 6):
    generate.generate_gemm_microkernel(
        isa=avx512f_template.Avx512FC(mr, n=32, c=2),
        output_file=os.path.join(
            output_base,
            f'f32-gemm-{mr}x32c2-minmax-asm-amd64-avx512f-broadcast.S',
        ),
    )
