#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from gemm_compiler import avx512vnni_template
from gemm_compiler import generate
from gemm_compiler import neondot_template


output_base = 'src/qd8-f32-qc4w-gemm/gen/'


def generate_qd8_f32_qc4w_gemm_microkernels():
  """Generates qd8-f32-qc4w assembly gemm microkernels."""
  if '/bazel-out/' in os.getcwd():
    os.chdir(os.environ['BUILD_WORKING_DIRECTORY'])

  for unroll in {1, 2, 4}:
    decrement = 32 * unroll
    for nr in range(8, 17, 8):
      for mr in range(1, 5):
        generate.generate_gemm_microkernel(
            isa=neondot_template.NeonDotQC4W(mr, nr, unroll),
            output_file=os.path.join(
                output_base,
                f'qd8-f32-qc4w-gemm-{mr}x{nr}-minmax-asm-aarch64-neondot-ld{decrement}.S',
            ),
        )

  for mr in range(1, 12):
    generate.generate_gemm_microkernel(
        isa=avx512vnni_template.Avx512VnniQc4w(m=mr, n=32, c=4),
        output_file=os.path.join(
            output_base,
            f'qd8-f32-qc4w-gemm-{mr}x32-minmax-asm-amd64-avx512vnni.S',
        ),
    )

  # not enough SIMD registers to go above 5x64
  for mr in range(1, 6):
    generate.generate_gemm_microkernel(
        isa=avx512vnni_template.Avx512VnniQc4w(m=mr, n=64, c=4),
        output_file=os.path.join(
            output_base,
            f'qd8-f32-qc4w-gemm-{mr}x64-minmax-asm-amd64-avx512vnni.S',
        ),
    )

  for mr in range(1, 6):
    generate.generate_gemm_microkernel(
        isa=avx512vnni_template.Avx512VnniQc4w(m=mr, n=32, c=8),
        output_file=os.path.join(
            output_base,
            f'qd8-f32-qc4w-gemm-{mr}x32c8-minmax-asm-amd64-avx512vnni.S',
        ),
    )

  for mr in range(1, 12):
    generate.generate_gemm_microkernel(
        isa=avx512vnni_template.Avx512VnniQc4w(m=mr, n=16, c=8),
        output_file=os.path.join(
            output_base,
            f'qd8-f32-qc4w-gemm-{mr}x16c8-minmax-asm-amd64-avx512vnni.S',
        ),
    )
