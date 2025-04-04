#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc

from gemm_compiler import arm_template


class Aarch32(arm_template.Arm):
  """All non SIMD features for aarch32."""

  def __init__(self, m: int, n: int):
    super().__init__(m, n)
    self.decrement = 4

  def register_bytes(self):
    return 16

  def n_step(self):
    return 4

  def prefix(self):
    return 'v'

  def astride_register(self):
    return 'r4'

  def kc_register(self):
    return 'r2'

  def k_register(self):
    return 'r0'

  def cm_stride_register(self):
    return 'r12'

  def am_registers(self):
    return [self.a_ptr_register()] + [
        'r7',
        'r9',
        'r10',
        'r11',
    ]

  def a_ptr_register(self):
    return 'r3'

  def c_ptr_register(self):
    return 'r6'

  def cm_registers(self):
    return [self.c_ptr_register()] + [
        'r4',
        'r8',
        'r14',
    ]

  def w_ptr_register(self):
    return 'r5'

  def min_register(self):
    return 'q8'

  def max_register(self):
    return 'q9'

  def nc_register(self):
    return 'r1'

  def mr_register(self):
    return 'r0'

  def register_map_byte(self, reg):
    return reg.replace('x', 'w')

  def header(self):
    header = """// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"\n\n"""

    header += 'BEGIN_FUNCTION ' + self.function_name()
    header += """
      # Free up GP registers. Decrement sp by 36.
      push {r4, r5, r6, r7, r8, r9, r10, r11, r14}

      # Preserve callee saved q4-q7 registers. Decrement sp by 64.
      vpush {d8-d15}

      # Load weight's ptr.
      ldr r5, [sp, #104]

      # Load c ptr.
      ldr r6, [sp, #108]

      # Load params.
      ldr r4, [sp, #124]

      # Load min/max values.
      vld1.8 {q8, q9},  [r4]\n"""
    header += self.quantization_params()
    return header

  def jump_to_label(self, label):
    return f'b {label}\n'

  def read_a_registers(self):
    return ''

  def do_loop(self, k: int, iterations: int, no_interleaved_instructions: bool):
    """One unrolled iteration of the inner loop.

    Args:
      k: iteration number.
      iterations: total number of iterations.
      no_interleaved_instructions: Do not interleaved instructions for the next
        loop iteration. Usually used for the last loop iteration.

    Returns:
       A string containing the assembly instructions for one unrolled iteration
       of the inner loop.
    """
    second_half = k + 1 > iterations - self.m
    last_iter = k == iterations - 1
    asm_string = ''
    for l in self.weights_asm()['loop']:
      if last_iter and no_interleaved_instructions:
        continue
      asm_string += l.format(
          W_ptr=self.w_ptr_register(),
          W=str(int(self.w_registers()[(k + 1) % 2]) * 2),
      )
    for l in self.compute_asm()['loop']:
      for nr in range(0, self.n):
        if nr == self.n // 2:
          if 'after' in self.weights_asm() and not last_iter:
            for r in self.weights_asm()['after']:
              asm_string += r.format(
                  W=str(int(self.w_registers()[(k + 1) % 2]) * 2),
                  W_2=self.w_registers()[(k + 1) % 2],
              )
          if second_half and not no_interleaved_instructions:
            for r in self.input_asm()['loop']:
              pos = k - (iterations - self.m)
              asm_string += r.format(
                  AM_ptr=self.am_registers()[pos],
                  AM=str(int(self.a_registers(pos)) * 2),
              )

        for mr in range(0, self.m):
          asm_string += l.format(
              ACC=self.acc_registers()[mr * 2 + nr],
              W=str(int(self.w_registers()[k % 2]) * 2 + nr),
              A=str(int(self.a_registers(mr)) * 2 + (k // 4)),
              POS=k % 4,
          )

    return asm_string

  def extend_inputs(self):
    asm_string = ''
    if 'after' in self.input_asm():
      for mr in range(0, self.m):
        for l in self.input_asm()['after']:
          asm_string += l.format(
              AM=str(int(self.a_registers(mr)) * 2), AM_2=self.a_registers(mr)
          )
    return asm_string

  def extend_inputs_and_weights(self):
    asm_string = ''
    if 'after' in self.weights_asm():
      for l in self.weights_asm()['after']:
        asm_string += l.format(
            W=str(int(self.w_registers()[0]) * 2),
            W_2=self.w_registers()[0],
        )
    asm_string += self.extend_inputs()
    return asm_string

  @abc.abstractmethod
  def k_unroll(self) -> int:
    raise NotImplementedError

  def inner_loop_in_order(self):
    """An inner loop for in-order cores.

    Returns:
       A string containing the assembly instructions for the entire inner loop.
    """
    asm_string = f"""\n# Are there at least {self.k_unroll()} bytes?
      subs r0, r0, #8
      blo .Lfinal_iteration\n"""

    asm_string += '\n.Linner_loop:\n'

    # weights
    if 'before' in self.weights_asm():
      asm_string += self.weights_asm()['before']
    inner_loop_label = '.Linner_loop'
    asm_string += self.extend_inputs_and_weights()

    # The main loop which interleaves loads of the A and B data for the next
    # iteration with the loop.
    for k in range(self.k_unroll()):
      asm_string += self.do_loop(k, self.k_unroll(), False)
    # loop counter
    asm_string += self.cmp_k_and_jump_if_less(
        label=inner_loop_label, decrement=self.k_unroll(), cond='bhs'
    )
    # The final iteration, which is identical except that it does not load the
    # As and Bs for the next iteration.
    asm_string += '\n.Lfinal_iteration:\n'
    asm_string += self.extend_inputs_and_weights()
    for k in range(self.k_unroll()):
      asm_string += self.do_loop(k, self.k_unroll(), True)
    asm_string += 'adds r0, r0, #8\n'
    asm_string += 'bne .Lepilogue\n'

    asm_string += '\n'

    return asm_string

  def load_strides(self):
    """Load a & cm_strides parameters.

    The offsets come from the aarch32 calling convention.

    Returns:
       A string containing the assembly instructions to load a & cm_stride into
       general purpose registers.
    """
    asm_string = '# Load a and cm stride registers.\n'
    a_stride_reg = self.astride_register()
    cm_stride_reg = self.cm_stride_register()
    asm_string += f'ldr {a_stride_reg}, [sp, #100]\n'
    asm_string += f'ldr {cm_stride_reg}, [sp, #112]\n'
    return asm_string

  def clamp_inputs_and_outputs(self, labels, input_registers, output_registers):
    clamping = {
        'clamp': """
      cmp {mr_reg}, #{M}
      movlo  {AM_1}, {AM_0}
      movlo  {CM_1}, {CM_0}
      movls  {AM_2}, {AM_1}
      movls  {CM_2}, {CM_1}\n""",
    }
    ret = ''
    outer = self.m
    # clamp a & c
    end_index = self.m if (self.m % 2 == 1) else self.m - 1
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
    if end_index != self.m:
      ret += """
      cmp {mr_reg}, #{M}
      movlo  {AM_1}, {AM_0}
      movlo  {CM_1}, {CM_0}\n""".format(
          mr_reg=self.mr_register(),
          AM_0=input_registers[end_index - 1],
          AM_1=input_registers[end_index],
          CM_0=output_registers[end_index - 1],
          CM_1=output_registers[end_index],
          M=end_index + 1,
      )

    return ret, outer

  def initialize_k_register(self):
    return f'subs {self.k_register()}, {self.kc_register()}, #8\n'

  def epilogue(self) -> str:
    asm_string = """
.Lreturn:
      # Restore callee saved q4-q7 registers.
      vpop       {d8-d15}

      # Restore the callee saved GP registers.
      pop {r4, r5, r6, r7, r8, r9, r10, r11, r14}

      bx lr
"""
    asm_string += self.inner_loop_epilogue()
    asm_string += """
END_FUNCTION {function_name}""".format(function_name=self.function_name())
    return asm_string

  def inner_loop_epilogue(self) -> str:
    """Remainder handling for the inner loop.

    Process up to k_unroll - 1 channels. This is placed at the very end of the
    function for control flow reasons. Instructions for the following channel
    may not be interleaved as we do not know how many channels must be
    processed. The epilogue iteration is identical except that it does not load
    the As and Bs for the next iteration and checks k after each mlal batch.

    Returns:
       A string containing the assembly instructions for the inner loop
       epilogue.
    """
    k_register = self.k_register()
    asm_string = '\n.Lepilogue:\n'
    asm_string += f'and {k_register}, {k_register}, #7\n'
    asm_string += f'\n# Load {self.m} As and B0\n'
    # change this to input & weight's asm.
    for mr in range(self.m):
      am_ptr = self.am_registers()[mr]
      am = str(int(self.a_registers(mr)) * 2)
      asm_string += f'vld1.8 d{am}, [{am_ptr}]\n'
      asm_string += f'add {am_ptr}, {k_register}\n'
    asm_string += self.extend_inputs()
    k_unroll = self.k_unroll() - 1
    for k in range(k_unroll):
      asm_string += self.epilogue_iteration(k, k_unroll)
    asm_string += 'b .Linner_loop_end\n'
    return asm_string

  def epilogue_iteration(self, k: int, iterations: int) -> str:
    """Process one channel of the loop epilogue.

    Args:
      k: iteration number.
      iterations: total number of iterations.

    Returns:
       A string containing the assembly instructions for one iteration of the
       inner loop epilogue.
    """
    asm_string = ''
    for l in self.weights_asm()['loop']:
      asm_string += l.format(
          W_ptr=self.w_ptr_register(),
          W=str(int(self.w_registers()[0]) * 2),
      )
      if 'after' in self.weights_asm():
        for r in self.weights_asm()['after']:
          asm_string += r.format(
              W=str(int(self.w_registers()[0]) * 2),
              W_2=self.w_registers()[0],
          )
    for l in self.compute_asm()['loop']:
      for nr in range(0, self.n):
        for mr in range(0, self.m):
          asm_string += l.format(
              ACC=self.acc_registers()[mr * 2 + nr],
              W=str(int(self.w_registers()[0]) * 2 + nr),
              A=str(int(self.a_registers(mr)) * 2 + (k // 4)),
              POS=k % 4,
          )
    if (k % 2) == 0:
      if k != iterations - 1:
        k_register = self.k_register()
        k_val = k + 2
        asm_string += f'cmp {k_register}, #{k_val}\n'
        asm_string += 'blo .Linner_loop_end\n'
    else:
      asm_string += 'beq .Linner_loop_end\n'

    return asm_string

  def cmp_k_and_jump_if_less(self, label, decrement, cond):
    k_register = self.k_register()
    return f"""subs {k_register}, {k_register}, #{decrement}
      {cond} {label}\n"""
