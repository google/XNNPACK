#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from gemm_compiler import avx512vnni_template
from gemm_compiler import neondot_template
from gemm_compiler import generate


output_base = 'src/qs8-qc4w-gemm/gen/'


def generate_qs8_qc4w_gemm_microkernels():
  """Generates qs8-qc4w assembly gemm microkernels."""
  if '/bazel-out/' in os.getcwd():
    os.chdir(os.environ['BUILD_WORKING_DIRECTORY'])

  for unroll in {1, 2, 4}:
    decrement = 32 * unroll
    nr = 16
    for mr in range(1, 6):
      generate.generate_gemm_microkernel(
          isa=neondot_template.NeonDotQS8QC4W(mr, nr, unroll),
          output_file=os.path.join(
              output_base,
              f'qs8-qc4w-gemm-{mr}x{nr}-minmax-fp32-asm-aarch64-neondot-ld{decrement}.S',
          ),
      )

  nr = 16
  for mr in range(1, 10):
    generate.generate_gemm_microkernel(
        isa=avx512vnni_template.Avx512VnniQS8QC4W(mr, nr, 8),
        output_file=os.path.join(
            output_base,
            f'qs8-qc4w-gemm-{mr}x{nr}c8-minmax-fp32-asm-amd64-avx512vnni.S',
        ),
    )
