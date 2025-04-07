#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from gemm_compiler import base_architecture


def generate_gemm_microkernel(
    isa: base_architecture.BaseArchitecture, output_file: str
):
  """Shared logic for assembly gemm microkernel generation."""
  asm_string = isa.header()

  # adjust inner loop
  asm_string += isa.adjust_kc()

  # setup a{1}->a{M-1} & c{1]->c{M-1}registers
  asm_string += isa.input_output_register_setup()

  ## Pre outer loop preparation
  asm_string += isa.outer_loop_prepare()

  ## the outer loop label
  asm_string += '\n.Louter_loop:\n'
  asm_string += '# Initialize k counter.\n'
  asm_string += isa.initialize_k_register()

  ## Read a registers from the stack if required
  asm_string += isa.read_a_registers()

  ## Initialize accumulators
  asm_string += isa.init_accumulators()

  ## inner loop
  asm_string += isa.inner_loop()

  asm_string += '.Linner_loop_end:\n'
  asm_string += isa.convert_to_output_type()

  ## min/max clamping
  asm_string += '# Min/max clamping.\n'
  asm_string += isa.clamp()

  ## store
  asm_string += isa.store()

  asm_string += isa.epilogue()

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
