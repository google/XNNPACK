#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import base_architecture as base_architecture


class Arm(base_architecture.BaseArchitecture):
  """All common features for aarch32 and aarch64."""

  def input_output_register_setup(self, M):
    registers = self.am_registers()
    a_stride = self.astride_register()
    c_stride = self.cm_stride_register()
    a_base_ptr = self.a_ptr_register()
    c_base_ptr = self.c_ptr_register()
    # setup a{0}->a{M-1} registers
    if M == 1:
      return ''
    asm_string = '# Setup and alias a & c pointers.\n'
    asm_string += self.load_strides()
    asm_string += self.input_output_strides(
        M=M, registers=self.am_registers(), stride=self.astride_register()
    )

    # setup c{0}->c{M-1} registers
    asm_string += self.input_output_strides(
        M=M, registers=self.cm_registers(), stride=self.cm_stride_register()
    )

    # if mr < MR
    clamp_string, outer = self.clamp_inputs_and_outputs(
        M, self.labels(), self.am_registers(), self.cm_registers()
    )
    asm_string += clamp_string
    return asm_string

  def input_output_strides(self, M, registers, stride):
    INPUT_OUTPUT_REGISTER_SETUP = """add {aM}, {aM_1}, {STRIDE}\n"""
    ret = ''
    for mr in range(1, M):
      ret += INPUT_OUTPUT_REGISTER_SETUP.format(
          M=mr,
          M_1=mr - 1,
          aM=registers[mr],
          aM_1=registers[mr - 1],
          STRIDE=stride,
      )
    return ret

  # Load a & cm_strides parameters.
  def load_strides(self):
    return ''

  def outer_loop_prepare(self, M, N):
    return ''

  def increment_ptr(self, ptr, step):
    return f'add {ptr}, {ptr}, {step}\n'

  def cmp_k_and_jump_if_less(self, label, decrement, cond):
    kc_register = self.kc_register()
    k_register = self.k_register()
    return f"""subs {k_register}, {k_register}, {decrement}
      {cond} {label}\n"""

  def mask_register(self):
    return ''

  def tmp_w_register(self):
    return ''

  def dequantize(self, M, N, W):
    return ''

  def adjust_kc(self):
    return ''

  def quantization_params(self, M):
    return ''
