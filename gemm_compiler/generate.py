#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from gemm_compiler import base_architecture

"""Shared logic for assembly gemm microkernel generation."""


def generate_gemm_microkernel(
    M: int, N: int, isa: base_architecture.BaseArchitecture, output_file: str
):
  num_horizontal_registers = int(N / isa.n_step())
  asm_string = isa.header(M, N, isa.prefix(), isa.isa())

  k_register = isa.k_register()
  acc_registers = isa.acc_registers()
  w_ptr_reg = isa.w_ptr_register()

  # adjust inner loop
  asm_string += isa.adjust_kc()

  # setup a{1}->a{M-1} & c{1]->c{M-1}registers
  if isa.isa() == 'avx512fp16':
    res = isa.input_output_register_setup(
        M=M,
    )
  asm_string += isa.input_output_register_setup(
      M=M,
  )

  ## Pre outer loop preparation
  asm_string += isa.outer_loop_prepare(M=M, N=num_horizontal_registers)

  ## the outer loop label
  asm_string += '\nouter_loop:\n'
  asm_string += '# Initialize k counter.\n'
  asm_string += isa.initialize_k_register(k_register)

  ## Read a registers from the stack if required
  asm_string += isa.read_a_registers(M=M)

  ## Initialize accumulators
  asm_string += isa.init_accumulators(
      M=M,
      N=num_horizontal_registers,
  )

  ## inner loop
  asm_string += isa.inner_loop(M, N)

  asm_string += 'inner_loop_end:\n'
  asm_string += isa.dequantize(M=M, N=num_horizontal_registers, W=w_ptr_reg)

  ## min/max clamping
  asm_string += '# Min/max clamping.\n'
  asm_string += isa.clamp(M, N)

  ## store
  asm_string += isa.store(
      M=M,
      N=N,
  )

  asm_string += isa.epilogue(M, N, isa)

  # Correctly indent the generated assembly.
  lines = asm_string.splitlines()
  stripped_lines = [line.lstrip() for line in lines]
  # Indent all lines that are not labels.
  stripped_lines = [
      '      ' + line
      if not (line.endswith(':') or 'FUNCTION' in line or 'include' in line)
      else line
      for line in stripped_lines
  ]
  # Strip indentation from empty lines.
  stripped_lines = ['' if line.isspace() else line for line in stripped_lines]
  # Strip indentation from header.
  stripped_lines = [
      line.lstrip() if line.startswith('      //') else line
      for line in stripped_lines
  ]
  asm_string = '\n'.join(stripped_lines)

  with open(output_file, 'w') as f:
    f.write(asm_string)
