#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import arm_template as arm


class Aarch64(arm.Arm):
  """All non SIMD features for aarch64."""

  def __init__(self):
    self.decrement = 4
    self.unroll_factor = 1

  def astride_register(self):
    return 'x4'

  def kc_register(self):
    return 'x2'

  def k_register(self):
    return 'x20'

  def cm_stride_register(self):
    return 'x7'

  def am_registers(self):
    return [self.a_ptr_register()] + [
        'x9',
        'x10',
        'x11',
        'x12',
        'x21',
        'x22',
        'x25',
    ]

  def a_ptr_register(self):
    return 'x3'

  def c_ptr_register(self):
    return 'x6'

  def cm_registers(self):
    return [self.c_ptr_register()] + [
        'x13',
        'x14',
        'x15',
        'x19',
        'x23',
        'x24',
        'x26',
    ]

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

  def register_map_byte(self, reg):
    return reg.replace('x', 'w')

  def register_map_dword(self, reg):
    return reg.replace('x', 'q')

  def function_name(self, M, N, isa):
    LD = self.unroll_factor * 32
    return f'xnn_f32_gemm_minmax_ukernel_{M}x{N}__asm_aarch64_{isa}_ld{LD}_2'

  def header(self, M, N, prefix, isa):
    HEADER = """// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION {function_name}

      # Free up GP registers.
      stp x19, x20, [sp, -64]
      stp x21, x22, [sp, -48]
      stp x23, x24, [sp, -32]
      stp x25, x26, [sp, -16]

      # Preserve callee saved q8-q15 registers.
      stp d8, d9, [sp, -128]
      stp d10, d11, [sp, -112]
      stp d12, d13, [sp, -96]
      stp d14, d15, [sp, -80]

      # Load params.
      ldr x13, [sp, 8]

      # Load min/max values.
      ld2r {{v0.4s, v1.4s}}, [x13]
""".format(
        function_name=self.function_name(M, N, isa)
    )
    HEADER += self.quantization_params(M)
    return HEADER

  def jump_to_label(self, label):
    return f'b {label}\n'

  def read_a_registers(self, M):
    return ''

  def do_loop(self, M, N, pos):
    N_COUNT = N // self.n_step()
    asm_string = ''
    for l in self.weights_asm()['loop_2']:
      for nr in range(0, N_COUNT - 1, 2):
        asm_string += l.format(
            W_ptr=self.w_ptr_register(),
            W=self.w_registers()[nr],
            W_1=self.w_registers()[nr + 1],
            offset=self.register_bytes() * nr,
            w_step=self.register_bytes() * N_COUNT,
            mask=self.mask_register(),
            tmp_W=self.tmp_w_register(),
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
              POS=pos,
          )
    return asm_string

  def inner_loop(self, M, N):
    asm_string = ''
    if self.unroll_factor > 1:
      DECREMENT = self.unroll_factor * 4
      k_register = self.k_register()
      asm_string += f'\n# Are there at least {DECREMENT} bytes?\n'
      asm_string += f'cmp {k_register}, {DECREMENT}\n'
      asm_string += f'blt inner_loop_tail\n'
      asm_string += f'sub {k_register}, {k_register}, {DECREMENT}\n'

    asm_string += '\ninner_loop:\n'
    decrement = 4 * self.unroll_factor
    if 'before' in self.input_asm():
      asm_string += self.input_asm()['before']
    if self.unroll_factor > 1:
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
    inner_loop_label = 'inner_loop'
    if self.unroll_factor > 1:
      for u in range(self.unroll_factor):
        asm_string += self.do_loop(M, N, u)
      # loop counter
      asm_string += self.cmp_k_and_jump_if_less(
          label=inner_loop_label, decrement=decrement, cond='bhs'
      )

      asm_string += f"""
      add x20, x20, {decrement}
      cmp x20, 4
      blt inner_loop_end
      \ninner_loop_tail:\n"""
      inner_loop_label = 'inner_loop_tail'

    for mr in range(0, M):
      for l in self.base_input_asm()['loop']:
        asm_string += l.format(
            AM_ptr=self.am_registers()[mr],
            AM=self.a_registers(mr),
            a_offset=self.k_register(),
        )
    asm_string += self.do_loop(M, N, 0)
    # loop counter
    asm_string += self.cmp_k_and_jump_if_less(
        label=inner_loop_label, decrement=4, cond='bne'
    )
    asm_string += '\n'

    return asm_string

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

  def initialize_k_register(self, reg):
    kc_register = self.kc_register()
    return f'mov {reg}, {kc_register}\n'

  def epilogue(self, M, N, isa):
    restore_stack = """
return:
      # Restore the callee saved GP registers.
      ldp x19, x20, [sp, -64]
      ldp x21, x22, [sp, -48]
      ldp x23, x24, [sp, -32]
      ldp x25, x26, [sp, -16]

      # Restore callee saved q8-q15 registers.
      ldp d8, d9, [sp, -128]
      ldp d10, d11, [sp, -112]
      ldp d12, d13, [sp, -96]
      ldp d14, d15, [sp, -80]
      ret
END_FUNCTION {function_name}
""".format(
        M=M, N=N, function_name=isa.function_name(M, N, isa.isa())
    )
    return restore_stack
