import abc

#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import base_architecture


class Arm(base_architecture.BaseArchitecture):
  """All common features for aarch32 and aarch64."""

  @abc.abstractmethod
  def clamp_inputs_and_outputs(self, labels, input_registers, output_registers):
    raise NotImplementedError

  def input_output_strides(self, registers, stride):
    INPUT_OUTPUT_REGISTER_SETUP = """add {aM}, {aM_1}, {STRIDE}\n"""
    ret = ''
    for mr in range(1, self.m):
      ret += INPUT_OUTPUT_REGISTER_SETUP.format(
          M=mr,
          M_1=mr - 1,
          aM=registers[mr],
          aM_1=registers[mr - 1],
          STRIDE=stride,
      )
    return ret

  def input_output_register_setup(self):
    # setup a{0}->a{M-1} registers
    if self.m == 1:
      return ''
    asm_string = '# Setup and alias a & c pointers.\n'
    asm_string += self.load_strides()
    asm_string += self.input_output_strides(
        registers=self.am_registers(), stride=self.astride_register()
    )

    # setup c{0}->c{M-1} registers
    asm_string += self.input_output_strides(
        registers=self.cm_registers(), stride=self.cm_stride_register()
    )

    # if mr < MR
    clamp_string, _ = self.clamp_inputs_and_outputs(
        self.labels(), self.am_registers(), self.cm_registers()
    )
    asm_string += clamp_string
    return asm_string

  # Load a & cm_strides parameters.
  def load_strides(self):
    return ''

  def outer_loop_prepare(self):
    return ''

  def increment_ptr(self, ptr, step):
    return f'add {ptr}, {ptr}, {step}\n'

  def cmp_k_and_jump_if_less(self, label, decrement, cond):
    k_register = self.k_register()
    return f"""subs {k_register}, {k_register}, {decrement}
      {cond} {label}\n"""

  def mask_register(self):
    return ''

  def tmp_w_register(self):
    return ''

  def dequantize(self):
    return ''

  def adjust_kc(self):
    return ''

  def quantization_params(self):
    return ''
