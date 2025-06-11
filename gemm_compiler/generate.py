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
  isa.header()

  # adjust inner loop
  isa.adjust_kc()

  # setup a{1}->a{M-1} & c{1]->c{M-1}registers
  isa.input_output_register_setup()

  ## Pre outer loop preparation
  isa.outer_loop_prepare()

  ## the outer loop label
  isa.label('outer_loop')
  isa.comment('Initialize k counter.')
  isa.initialize_k_register()

  ## Read a registers from the stack if required
  isa.read_a_registers()

  ## Initialize accumulators
  isa.init_accumulators()

  ## inner loop
  isa.inner_loop()

  isa.label('inner_loop_end')
  isa.convert_to_output_type()

  ## min/max clamping
  isa.comment('Min/max clamping.')
  isa.clamp()

  ## store
  isa.store()

  isa.epilogue()

  asm_string = isa.get_string()
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
