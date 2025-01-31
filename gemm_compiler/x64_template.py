#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import base_architecture as base_architecture

"""All non SIMD features for x64."""


class X64(base_architecture.BaseArchitecture):

  def astride_register(self):
    return 'r8'

  def kc_register(self):
    return 'rdx'

  def k_register(self):
    return 'r11'

  def cm_stride_register(self):
    return 'r11'

  def am_registers(self):
    return [self.a_ptr_register()] + [
        'rax',
        'r15',
        'r14',
        'r12',
        'r10',
        'r13',
        'rbx',
        'rbp',
        'r8',
        'rdi',
    ]

  def a_ptr_register(self):
    return 'rcx'

  def c_ptr_register(self):
    return 'rcx'

  def cm_registers(self):
    return self.am_registers()

  def acc_registers(self):
    return [
        'mm7',
        'mm8',
        'mm9',
        'mm14',
        'mm15',
        'mm16',
        'mm17',
        'mm18',
        'mm19',
        'mm20',
        'mm21',
        'mm22',
        'mm23',
        'mm24',
        'mm25',
        'mm26',
        'mm27',
        'mm28',
        'mm29',
        'mm30',
        'mm12',
        'mm13',
    ]

  def w_ptr_register(self):
    return 'r9'

  def min_register(self):
    return 'mm0'

  def max_register(self):
    return 'mm1'

  def nc_register(self):
    return 'rsi'

  def mr_register(self):
    return 'rdi'

  def tmp_gp_registers(self):
    return ['rdi', 'r11']

  def register_map_byte(self, reg):
    """Maps 64 bit register names to their low 8 bits."""
    map = {
        'rax': 'al',
        'rcx': 'cl',
        'rdx': 'dl',
        'rbx': 'bl',
        'rsi': 'sil',
        'rdi': 'dil',
        'rsp': 'spl',
        'rbp': 'bpl',
        'r8': 'r8b',
        'r9': 'r9b',
        'r10': 'r10b',
        'r11': 'r11b',
        'r12': 'r12b',
        'r13': 'r13b',
        'r14': 'r14b',
        'r15': 'r15b',
    }
    return map[reg]

  def register_map_dword(self, reg):
    """Maps 64 bit register names to their low 32 bits."""
    map = {
        'rax': 'eax',
        'rcx': 'ecx',
        'rdx': 'edx',
        'rbx': 'ebx',
        'rsi': 'esi',
        'rdi': 'edi',
        'rsp': 'esp',
        'rbp': 'ebp',
        'r8': 'r8d',
        'r9': 'r9d',
        'r10': 'r10d',
        'r11': 'r11d',
        'r12': 'r12d',
        'r13': 'r13d',
        'r14': 'r14d',
        'r15': 'r15d',
    }
    return map[reg]

  def jump_to_label(self, label):
    return f'jmp {label}'

  def function_name(self, M, N, isa):
    return f'xnn_f32_gemm_minmax_ukernel_{M}x{N}__asm_amd64_{isa}_broadcast'

  def params_offset(self):
    return 80

  def header(self, M, N, prefix, isa):
    HEADER = """// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION {function_name}

      .intel_syntax noprefix

      # Free up GP registers.
      push rbx
      push rbp
      push r15
      push r14
      push r13
      push r12

      # load params to free up a GP registers
      mov r13, [rsp + {params_offset}] # params
      vbroadcastss {prefix}mm0, DWORD PTR [r13]
      vbroadcastss {prefix}mm1, DWORD PTR [r13 + 4]

      # Load c pointer.
      mov r10, [rsp + 56]
      # Load cm_stride.
      mov r11, [rsp + 64]
""".format(
        function_name=self.function_name(M, N, isa),
        M=M, N=N, prefix=prefix, isa=isa, params_offset=self.params_offset()
    )
    return HEADER

  def input_output_register_setup(self, M):
    registers = self.am_registers()
    a_stride = self.astride_register()
    c_stride = self.cm_stride_register()
    INPUT_OUTPUT_REGISTER_SETUP = """
      # Clamp a & c pointers if mr <= {M}
      mov {aM}, {aM_1}
      add {aM}, {A_STRIDE}
      mov {cM}, {cM_1}
      add {cM}, {C_STRIDE}
      cmp {mr_reg}, {M}
      cmovle {aM}, {aM_1}
      cmovle {cM}, {cM_1}\n"""
    INPUT_OUTPUT_REGISTER_PUSH = """
      mov [rsp + {a_rsp_offset}], {aM}
      mov [rsp + {c_rsp_offset}], {cM}\n"""
    ret = ''
    if self.stack_size(M) != 0:
      ret += '\n# Allocate some space on the stack.\n'
      ret += """sub rsp, {stack_size}\n""".format(
          stack_size=self.stack_size(M)
      )
    # Write rsi & r10 if required to the stack.
    if M > self.max_M_before_spilling():
      ret += (
          '# Write rsi (a pointer) to the stack as we need the register.\n'
      )
      ret += 'mov [rsp], rcx\n'
      ret += (
          '# Write r10 (c pointer) to the stack as we need the register.\n'
      )
      ret += 'mov [rsp + 8], r10\n'
    for mr in range(1, M):
      # cycle size of 2 if required
      if M > self.max_M_before_spilling():
        a_pos = mr % 2
        c_pos = (mr % 2) + self.max_M_before_spilling()
        a_pos_1 = (mr + 1) % 2
        c_pos_1 = ((mr + 1) % 2) + self.max_M_before_spilling()
      else:
        a_pos = mr
        c_pos = mr + self.max_M_before_spilling()
        a_pos_1 = a_pos - 1
        c_pos_1 = c_pos - 1
      a_rsp_offset = 16 + (mr - 1) * 16
      ret += INPUT_OUTPUT_REGISTER_SETUP.format(
          M=mr,
          aM=registers[a_pos],
          aM_1=registers[a_pos_1],
          cM=registers[c_pos],
          cM_1=registers[c_pos_1],
          A_STRIDE=a_stride,
          C_STRIDE=c_stride,
          mr_reg=self.mr_register(),
          a_rsp_offset=a_rsp_offset,
          c_rsp_offset=a_rsp_offset + 8,
      )
      if M > self.max_M_before_spilling():
        ret += INPUT_OUTPUT_REGISTER_PUSH.format(
            M=mr,
            aM=registers[a_pos],
            aM_1=registers[a_pos_1],
            cM=registers[c_pos],
            cM_1=registers[c_pos_1],
            A_STRIDE=a_stride,
            C_STRIDE=c_stride,
            mr_reg=self.mr_register(),
            a_rsp_offset=a_rsp_offset,
            c_rsp_offset=a_rsp_offset + 8,
        )

    return ret

  def max_M_before_spilling(self):
    return 5

  def read_a_registers(self, M):
    registers = self.am_registers()
    if M <= self.max_M_before_spilling():
      return ''
    ret = '# Read a pointers from stack into GP registers.\n'
    POP_A = 'mov {aM}, [rsp + {a_rsp_offset}]\n'
    for mr in range(0, M):
      a_rsp_offset = mr * 16
      ret += POP_A.format(aM=registers[mr], a_rsp_offset=a_rsp_offset)
    ret += '\n'
    return ret

  def increment_ptr(self, ptr, step):
    return f'add {ptr}, {step}\n'

  def initialize_k_register(self, reg):
    return f'mov {reg}, 0\n'

  def inner_loop_increment(self):
    return 4

  def cmp_k_and_jump_if_less(self, label):
    kc_register = self.kc_register()
    k_register = self.k_register()
    increment_bytes = self.inner_loop_increment()
    return f"""
      add {k_register}, {increment_bytes}
      cmp {kc_register}, {k_register}
      jne {label}\n"""

  def load_from_stack(self, reg, offset):
    """Load 8 bytes from the given offset from the stack pointer to reg."""
    return f'mov {reg}, [rsp + {offset}]\n'

  def epilogue(self, M, N, isa):
    restore_stack = '\nreturn:\n'
    if isa.stack_size(M) != 0:
      restore_stack += 'add rsp, {stack_ptr_sub}\n'.format(
          stack_ptr_sub=isa.stack_size(M)
      )
    restore_stack += """
      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      ret
END_FUNCTION {function_name}

#ifdef __has_feature
#if __has_feature(dataflow_sanitizer)
BEGIN_FUNCTION {function_name}.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION {function_name}.dfsan
#endif
#endif
""".format(
        M=M, N=N, function_name=isa.function_name(M, N, isa.isa())
    )
    return restore_stack

  def stack_size(self, M):
    """Returns the required stack storage space."""
    return 0

  def inner_loop_tail(self, M, N):
    return ''

  def inner_loop(self, M, N):
    asm_string = '\ninner_loop:\n'
    if M > self.max_M_before_spilling():
      asm_string += self.inner_loop_spill_gp(M, N)
    else:
      asm_string += self.inner_loop_small_M_N(M, N)
    # loop counter
    asm_string += self.cmp_k_and_jump_if_less(label='inner_loop')
    asm_string += self.inner_loop_tail(M, N)

    return asm_string
