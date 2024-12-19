#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import base_architecture as base_architecture

"""All non SIMD features for aarch64."""


class Aarch64(base_architecture.BaseArchitecture):

  def astride_register(self):
    return 'x4'

  def kc_register(self):
    return 'x2'

  def k_register(self):
    return 'x20'

  def cm_stride_register(self):
    return 'x7'

  def am_registers(self):
    return [self.a_ptr_register()] + ['x9', 'x10', 'x11', 'x12', 'x21', 'x22']

  def a_ptr_register(self):
    return 'x3'

  def c_ptr_register(self):
    return 'x6'

  def cm_registers(self):
    return [self.c_ptr_register()] + ['x13', 'x14', 'x15', 'x19', 'x23', 'x24']

  def w_ptr_register(self):
    return 'x5'

  def min_register(self):
    return 'v0'

  def max_register(self):
    return 'v1'

  def nc_register(self):
    return 'x1'

  def mr_register(self):
    return 'x0'

  def tmp_gp_registers(self):
    return ['x22', 'x23']

  def dequantize(self, M, N, W):
    return ''

  def adjust_kc(self):
    return ''

  def register_map_byte(self, reg):
    map = {
        'x0': 'x0',
        'x1': 'x1',
        'x2': 'x2',
        'x3': 'x3',
        'x4': 'x4',
        'x5': 'x5',
        'x6': 'x6',
        'x7': 'x7',
        'x8': 'x8',
        'x9': 'x9',
        'x10': 'x10',
        'x11': 'x11',
        'x12': 'x12',
        'x13': 'x13',
        'x14': 'x10',
        'x15': 'x15',
    }
    return map[reg]

  def register_map_dword(self, reg):
    map = {
        'x0': 'q0',
        'x1': 'q1',
        'x2': 'q2',
        'x3': 'q3',
        'x4': 'q4',
        'x5': 'q5',
        'x6': 'q6',
        'x7': 'q7',
        'x8': 'q8',
        'x9': 'q9',
        'x10': 'q10',
        'x11': 'q11',
        'x12': 'q12',
        'x13': 'q13',
        'x14': 'q10',
        'x15': 'q15',
    }
    return map[reg]

  def function_name(self, M, N, isa):
    return f'xnn_f32_gemm_minmax_ukernel_{M}x{N}__asm_aarch64_{isa}_lane\n'

  def quantization_params(self):
    return ''

  def header(self, M, N, prefix, isa):
    HEADER = '#include "xnnpack/assembly.h"\n\n'

    HEADER += 'BEGIN_FUNCTION ' + self.function_name(M, N, isa)
    HEADER += """
      # Free up GP registers.
      stp x19, x20, [sp, -48]
      stp x21, x22, [sp, -32]
      stp x23, x24, [sp, -16]

      # Preserve callee saved q8-q15 registers.
      stp q8, q9, [sp, -176]
      stp q10, q11, [sp, -144]
      stp q12, q13, [sp, -112]
      stp q14, q15, [sp, -80]

      # Load params.
      ldr x13, [sp, 8]

      # Load min/max values.
      ld2r {v0.4s, v1.4s}, [x13]\n"""
    HEADER += self.quantization_params()
    return HEADER

  def jump_to_label(self, label):
    return f'b {label}\n'

  def read_a_registers(self, M):
    return ''

  def inner_loop(self, M, N):
    N_COUNT = N // self.n_step()
    asm_string = '\ninner_loop:\n'
    if 'before' in self.input_asm():
      asm_string += self.input_asm()['before']
    for mr in range(0, M):
      for l in self.input_asm()['loop']:
        asm_string += l.format(
            AM_ptr=self.am_registers()[mr],
            AM=self.a_registers(mr),
            a_offset=self.k_register(),
        )
    if 'after' in self.input_asm():
      asm_string += self.input_asm()['after']

    # weights
    if 'before' in self.weights_asm():
      asm_string += self.weights_asm()['before']
    for l in self.weights_asm()['loop_2']:
      for nr in range(0, N_COUNT, 2):
        asm_string += l.format(
            W_ptr=self.w_ptr_register(),
            W=self.w_registers()[nr],
            W_1=self.w_registers()[nr + 1],
            offset=self.register_bytes() * nr,
            w_step=self.register_bytes() * N_COUNT,
        )
    for l in self.weights_asm()['loop']:
      if N_COUNT % 2 != 0:
        asm_string += l.format(
            W_ptr=self.w_ptr_register(),
            W=self.w_registers()[nr],
            offset=self.register_bytes() * nr,
            w_step=self.register_bytes() * N_COUNT,
        )
    if 'after' in self.weights_asm():
      asm_string += self.weights_asm()['after'].format(
          W=self.w_ptr_register(), w_step=self.register_bytes() * N_COUNT
      )

    for l in self.compute_asm()['loop']:
      for nr in range(0, N_COUNT):
        for mr in range(0, M):
          asm_string += l.format(
              W=self.w_registers()[nr],
              A=self.a_registers(mr),
              ACC=self.acc_registers()[M * nr + mr],
          )
    return asm_string

  def outer_loop_prepare(self, M, N):
    return ''

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
    asm_string += self.input_output_strides(
        M=M, registers=self.am_registers(), stride=self.astride_register()
    )

    # setup c{0}->c{M-1} registers
    asm_string += self.input_output_strides(
        M=M, registers=self.cm_registers(), stride=self.cm_stride_register()
    )

    # Pre outer loop preparation
    # asm_string += isa.outer_loop_prepare(M=M, N=N_COUNT, W=w_ptr_reg, accumulators=acc_registers)

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

  def clamp_inputs_and_outputs(
      self, M, labels, input_registers, output_registers
  ):
    clamping = {
        'clamp': """
      cmp {mr_reg}, {M}
      csel  {AM_1}, {AM_0}, {AM_1}, LO
      csel  {CM_1}, {CM_0}, {CM_1}, LO
      csel  {AM_2}, {AM_1}, {AM_2}, LS
      csel  {CM_2}, {CM_1}, {CM_2}, LS\n""",
    }
    ret = ''
    outer = M
    # clamp a & c
    end_index = M if (M % 2 == 1) else M - 1
    for mr in range(2, end_index, 2):
      ret += clamping['clamp'].format(
          mr_reg=self.mr_register(),
          AM_0=input_registers[mr - 2],
          AM_1=input_registers[mr - 1],
          AM_2=input_registers[mr],
          CM_0=output_registers[mr - 2],
          CM_1=output_registers[mr - 1],
          CM_2=output_registers[mr],
          M=mr,
      )
    if end_index != M:
      ret += """
      cmp {mr_reg}, {M}
      csel  {AM_1}, {AM_0}, {AM_1}, LO
      csel  {CM_1}, {CM_0}, {CM_1}, LO\n""".format(
          mr_reg=self.mr_register(),
          AM_0=input_registers[end_index - 1],
          AM_1=input_registers[end_index],
          CM_0=output_registers[end_index - 1],
          CM_1=output_registers[end_index],
          M=end_index + 1,
      )

    return ret, outer

  def increment_ptr(self, ptr, step):
    return f'add {ptr}, {ptr}, {step}\n'

  def zero_gp_register(self, reg):
    return f'eor {reg}, {reg}, {reg}\n'

  def cmp_k_and_jump_if_less(self, label):
    kc_register = self.kc_register()
    k_register = self.k_register()
    return """add {k_register}, {k_register}, 4
      cmp {kc_register}, {k_register}
      bne {label}\n""".format(
        label=label, k_register=k_register, kc_register=kc_register
    )

  def epilogue(self, M, N, isa):
    restore_stack = """
return:
      # Restore the callee saved GP registers.
      ldp x19, x20, [sp, -48]
      ldp x21, x22, [sp, -32]
      ldp x23, x24, [sp, -16]

      # Restore callee saved q8-q15 registers.
      ldp q8, q9, [sp, -176]
      ldp q10, q11, [sp, -144]
      ldp q12, q13, [sp, -112]
      ldp q14, q15, [sp, -80]
      ret
END_FUNCTION {function_name}""".format(
        M=M, N=N, function_name=isa.function_name(M, N, isa.isa())
    )
    return restore_stack
