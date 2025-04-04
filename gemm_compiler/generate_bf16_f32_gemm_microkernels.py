#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from gemm_compiler import avx512bf16_template
from gemm_compiler import generate


output_base = 'src/bf16-f32-gemm/gen/'


def generate_bf16_f32_gemm_microkernels():
  """Generates bf16-f32 assembly gemm microkernels."""
  if '/bazel-out/' in os.getcwd():
    os.chdir(os.environ['BUILD_WORKING_DIRECTORY'])

  for nr in range(16, 33, 16):
    for mr in range(1, 12):
      generate.generate_gemm_microkernel(
          isa=avx512bf16_template.Avx512Bf16(mr, nr),
          output_file=os.path.join(
              output_base,
              f'bf16-f32-gemm-{mr}x{nr}c2-minmax-asm-amd64-avx512bf16-broadcast.S',
          ),
      )

  # not enough SIMD registers to go above 5x64
  for mr in range(1, 6):
    generate.generate_gemm_microkernel(
        isa=avx512bf16_template.Avx512Bf16(mr, n=64),
        output_file=os.path.join(
            output_base,
            f'bf16-f32-gemm-{mr}x64c2-minmax-asm-amd64-avx512bf16-broadcast.S',
        ),
    )
