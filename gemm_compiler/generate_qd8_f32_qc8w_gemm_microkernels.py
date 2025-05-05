#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

from gemm_compiler import avx512vnni_template
from gemm_compiler import generate
from gemm_compiler import neondot_template
from gemm_compiler import neonmlal_aarch32_template


output_base = 'src/qd8-f32-qc8w-gemm/gen/'


def generate_qd8_f32_qc8w_gemm_microkernels():
  """Generates qd8-f32-qc8w assembly gemm microkernels."""
  if '/bazel-out/' in os.getcwd():
    os.chdir(os.environ['BUILD_WORKING_DIRECTORY'])

  for nr in range(16, 33, 16):
    for mr in range(1, 12):
      generate.generate_gemm_microkernel(
          isa=avx512vnni_template.Avx512Vnni(m=mr, n=nr, c=4),
          output_file=os.path.join(
              output_base,
              f'qd8-f32-qc8w-gemm-{mr}x{nr}-minmax-asm-amd64-avx512vnni.S',
          ),
      )

  # not enough SIMD registers to go above 5x64
  for mr in range(1, 6):
    generate.generate_gemm_microkernel(
        isa=avx512vnni_template.Avx512Vnni(m=mr, n=64, c=4),
        output_file=os.path.join(
            output_base,
            f'qd8-f32-qc8w-gemm-{mr}x64-minmax-asm-amd64-avx512vnni.S',
        ),
    )

  for unroll in {1, 2, 4}:
    decrement = 32 * unroll
    for nr in range(8, 17, 8):
      for mr in range(1, 5):
        generate.generate_gemm_microkernel(
            isa=neondot_template.NeonDot(mr, nr, unroll),
            output_file=os.path.join(
                output_base,
                f'qd8-f32-qc8w-gemm-{mr}x{nr}-minmax-asm-aarch64-neondot-ld{decrement}.S',
            ),
        )

  nr = 8
  unroll = 2
  decrement = 32 * unroll
  for mr in range(1, 5):
    generate.generate_gemm_microkernel(
        isa=neonmlal_aarch32_template.NeonMlal(mr, nr, unroll),
        output_file=os.path.join(
            output_base,
            f'qd8-f32-qc8w-gemm-{mr}x{nr}-minmax-asm-aarch32-neonmlal-ld{decrement}.S',
        ),
    )

  for mr in range(1, 5):
    generate.generate_gemm_microkernel(
        isa=neonmlal_aarch32_template.NeonMlalF16(mr, nr, unroll),
        output_file=os.path.join(
            output_base,
            f'qd8-f16-qc8w-gemm-{mr}x{nr}-minmax-asm-aarch32-neonfp16arith-ld{decrement}.S',
        ),
    )

  # Generate C8 variants.
  for mr in range(1, 12):
    generate.generate_gemm_microkernel(
        isa=avx512vnni_template.Avx512Vnni(m=mr, n=16, c=8),
        output_file=os.path.join(
            output_base,
            f'qd8-f32-qc8w-gemm-{mr}x16c8-minmax-asm-amd64-avx512vnni.S',
        ),
    )

  for mr in range(1, 6):
    generate.generate_gemm_microkernel(
        isa=avx512vnni_template.Avx512Vnni(m=mr, n=32, c=8),
        output_file=os.path.join(
            output_base,
            f'qd8-f32-qc8w-gemm-{mr}x32c8-minmax-asm-amd64-avx512vnni.S',
        ),
    )
