#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from gemm_compiler import base_architecture


class X64(base_architecture.BaseArchitecture):
  """All non SIMD features for x64."""

  @property
  def c(self) -> int:
    return 1

  def element_size(self):
    return 4

  def mask(self):
    return ''

  def outer_loop_prepare(self):
    return ''

  def astride_register(self):
    return 'r8'

  def kc_register(self):
    return 'rdx'

  def k_register(self):
    return 'r11'

  def cm_stride_register(self):
    return 'r11'

  def mask_register(self):
    return 'mm13'

  def am_registers(self):
    return [self.a_ptr_register()] + [
        'rax',
        'r15',
        'r14',
        'r10',
        'r12',
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

  def bias_registers(self):
    return self.acc_registers()

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
    byte_from_64_bit_reg = {
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
    return byte_from_64_bit_reg[reg]

  def register_map_dword(self, reg):
    """Maps 64 bit register names to their low 32 bits."""
    int_from_64_bit_reg = {
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
    return int_from_64_bit_reg[reg]

  def jump_to_label(self, label):
    return f'jmp {label}'

  def function_name(self):
    return (
        f'xnn_f32_gemm_minmax_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'__asm_amd64_{self.isa()}_broadcast'
    )

  def params_offset(self):
    return 96

  def pre_header(self):
    return

  def copyright(self):
    self.asm_string += """// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"
"""

  def header(self):
    self.copyright()
    self.pre_header()
    self.asm_string += """
BEGIN_FUNCTION {function_name}

      .intel_syntax noprefix
      # Free up GP registers.
      # Save register arguments for tail call to msan annotation helper.
      push rdi
      push rsi
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
      mov r10, [rsp + 72]
      # Load cm_stride.
      mov r11, [rsp + 80]
""".format(
        function_name=self.function_name(),
        prefix=self.prefix(),
        params_offset=self.params_offset(),
    )

  # Quantization parameters are pushed to the stack at this offset.
  def quantization_params_offset(self):
    return 0

  def a_ptr_stack_offset(self):
    return 16

  def c_ptr_stack_offset(self):
    return self.a_ptr_stack_offset() + 8

  def input_output_register_setup(self):
    registers = self.am_registers()
    a_stride = self.astride_register()
    c_stride = self.cm_stride_register()
    setup_string = """
      # Clamp a & c pointers if mr <= {M}
      mov {aM}, {aM_1}
      add {aM}, {A_STRIDE}
      mov {cM}, {cM_1}
      add {cM}, {C_STRIDE}
      cmp {mr_reg}, {M}
      cmovle {aM}, {aM_1}
      cmovle {cM}, {cM_1}\n"""
    push_string = """
      mov [rsp + {a_rsp_offset}], {aM}
      mov [rsp + {c_rsp_offset}], {cM}\n"""
    if self.quantization_params_offset() != 0:
      self.asm_string += (
          '\n# Move stack parameters which have not yet been loaded\n'
      )
      self.asm_string += 'mov r12, [rsp + {stack_params_offset}]\n'.format(
          stack_params_offset=self.params_offset() + 8
      )
    self.asm_string += '\n# Align the stack pointer.\n'
    self.asm_string += 'mov r13, rsp\n'
    self.asm_string += 'sub rsp, 64\n'
    self.asm_string += 'and rsp, 0xFFFFFFFFFFFFFFC0\n'
    self.asm_string += (
        '# Store the old stack pointer containing the return address\n'
    )
    self.asm_string += 'mov [rsp], r13\n'
    if self.quantization_params_offset() != 0:
      self.asm_string += '# Push additional stack parameters to the new stack\n'
      offset = self.quantization_params_offset()
      self.asm_string += f'mov [rsp + {offset}], r12\n'
    if self.stack_size() != 0:
      self.asm_string += '\n# Allocate some space on the stack.\n'
      self.asm_string += """sub rsp, {stack_size}\n""".format(
          stack_size=self.stack_size()
      )
    # Write rsi & r10 if required to the stack.
    if self.m > self.max_m_before_spilling():
      offset = self.a_ptr_stack_offset()
      self.asm_string += (
          '# Write rsi (a pointer) to the stack as we need the register.\n'
      )
      self.asm_string += f'mov [rsp + {offset}], rcx\n'
      offset = self.c_ptr_stack_offset()
      self.asm_string += (
          '# Write r10 (c pointer) to the stack as we need the register.\n'
      )
      self.asm_string += f'mov [rsp + {offset}], r10\n'
    for mr in range(1, self.m):
      # cycle size of 2 if required
      if self.m > self.max_m_before_spilling():
        a_pos = mr % 2
        c_pos = (mr % 2) + self.max_m_before_spilling()
        a_pos_1 = (mr + 1) % 2
        c_pos_1 = ((mr + 1) % 2) + self.max_m_before_spilling()
      else:
        a_pos = mr
        c_pos = mr + self.max_m_before_spilling()
        a_pos_1 = a_pos - 1
        c_pos_1 = c_pos - 1
      a_rsp_offset = 32 + (mr - 1) * 16
      self.asm_string += setup_string.format(
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
      if self.m > self.max_m_before_spilling():
        self.asm_string += push_string.format(
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

  def read_a_registers(self):
    registers = self.am_registers()
    if self.m <= self.max_m_before_spilling():
      return
    self.asm_string += '# Read a pointers from stack into GP registers.\n'
    pop_a = 'mov {aM}, [rsp + {a_rsp_offset}]\n'
    for mr in range(0, self.m):
      a_rsp_offset = mr * 16 + self.a_ptr_stack_offset()
      self.asm_string += pop_a.format(
          aM=registers[mr], a_rsp_offset=a_rsp_offset
      )
    self.asm_string += '\n'

  def increment_ptr(self, ptr, step):
    self.asm_string += f'add {ptr}, {step}\n'

  def initialize_k_register(self):
    self.asm_string += f'mov {self.k_register()}, 0\n'

  def inner_loop_increment(self):
    return self._c * self.element_size()

  def cmp_k_and_jump_if_less(self, label):
    kc_register = self.kc_register()
    k_register = self.k_register()
    increment_bytes = self.inner_loop_increment()
    self.asm_string += f"""
      add {k_register}, {increment_bytes}
      cmp {kc_register}, {k_register}
      jne {label}\n"""

  def load_from_stack(self, reg, offset):
    """Load 8 bytes from the given offset from the stack pointer to reg."""
    return f'mov {reg}, [rsp + {offset}]\n'

  def epilogue(self):
    restore_stack = '\n.Lreturn:\n'
    if self.stack_size() != 0:
      restore_stack += 'add rsp, {stack_ptr_sub}\n'.format(
          stack_ptr_sub=self.stack_size()
      )
    restore_stack += """mov r13, [rsp]
    mov rsp, r13"""
    restore_stack += """
      # Restore the callee saved registers.
      pop r12
      pop r13
      pop r14
      pop r15
      pop rbp
      pop rbx
      pop rsi
      pop rdi
#if XNN_HAS_FEATURE(memory_sanitizer)
      jmp xnn_gemm_ukernel_msan_sizeof_c_{sizeof_c}
#else
      ret
#endif
END_FUNCTION {function_name}

#if XNN_HAS_FEATURE(dataflow_sanitizer)
BEGIN_FUNCTION {function_name}.dfsan
      .intel_syntax noprefix
      # We could implement this by calling a function that implements the dfsan instrumentation.
      # For now, just break, so if someone tries to use this, they'll know where the problem is.
      int 3
      ret
END_FUNCTION {function_name}.dfsan
#endif

#ifdef __ELF__
.section .note.GNU-stack, "", @progbits
#endif  // __ELF__
""".format(function_name=self.function_name(), sizeof_c=4)
    self.asm_string += restore_stack

  def stack_size(self):
    """Returns the required stack storage space."""
    return 0

  def inner_loop_tail(self):
    return

  @abc.abstractmethod
  def inner_loop_spill_gp(self, tail=False):
    raise NotImplementedError

  @abc.abstractmethod
  def inner_loop_small_M_N(self, tail=False):
    raise NotImplementedError

  def inner_loop(self):
    self.label('inner_loop')
    if self.m > self.max_m_before_spilling():
      self.inner_loop_spill_gp()
    else:
      self.inner_loop_small_M_N()
    # loop counter
    self.cmp_k_and_jump_if_less(label='.Linner_loop')
    self.inner_loop_tail()

  def _inner_loop_small_M_N(self, n: int, tail: bool = False) -> str:
    # input
    if 'before' in self.input_asm():
      self.asm_string += self.input_asm()['before']
    if 'after' in self.input_asm():
      self.asm_string += self.input_asm()['after']

    # weights
    if 'before' in self.weights_asm():
      self.asm_string += self.weights_asm()['before']
    if 'loop_2' in self.weights_asm():
      for l in self.weights_asm()['loop_2']:
        for nr in range(0, n, 2):
          self.asm_string += l.format(
              W_ptr=self.w_ptr_register(),
              W=self.w_registers()[nr],
              W_1=self.w_registers()[nr + 1],
              offset=self.register_bytes() * nr // 2,
              w_step=self.register_bytes() * self.n,
              mask=self.mask_register(),
          )
    for l in self.weights_asm()['loop']:
      for nr in range(0, n):
        self.asm_string += l.format(
            W_ptr=self.w_ptr_register(),
            W=self.w_registers()[nr],
            offset=self.register_bytes() * nr,
            w_step=self.register_bytes() * n,
            mask=self.mask_register(),
        )
    if 'after' in self.weights_asm():
      for l in self.weights_asm()['after']:
        self.asm_string += l.format(
            W=self.w_ptr_register(), w_step=self.w_register_bytes() * n
        )

    loop = 'loop_tail' if tail else 'loop'
    for mr in range(0, self.m):
      for l in self.input_asm()['loop']:
        self.asm_string += l.format(
            AM_ptr=self.am_registers()[mr],
            AM=self.a_registers(mr),
            a_offset=self.k_register(),
            A=self.a_registers(mr),
        )
      for m in self.compute_asm()[loop]:
        for nr in range(0, n):
          self.asm_string += m.format(
              W=self.w_registers()[nr],
              A=self.a_registers(mr),
              ACC=self.acc_registers()[self.m * nr + mr],
              mask=self.mask(),
          )

  def _inner_loop_spill_gp(self, n: int, tail: bool = False) -> str:
    # weights
    if 'before' in self.weights_asm():
      self.asm_string += self.weights_asm()['before']
    if 'loop_2' in self.weights_asm():
      for l in self.weights_asm()['loop_2']:
        for nr in range(0, n, 2):
          self.asm_string += l.format(
              W_ptr=self.w_ptr_register(),
              W=self.w_registers()[nr],
              W_1=self.w_registers()[nr + 1],
              offset=self.register_bytes() * nr // 2,
              w_step=self.register_bytes() * self.n,
              mask=self.mask_register(),
          )
    for l in self.weights_asm()['loop']:
      for nr in range(0, n):
        self.asm_string += l.format(
            W_ptr=self.w_ptr_register(),
            W=self.w_registers()[nr],
            offset=self.register_bytes() * nr,
            w_step=self.register_bytes() * n,
            mask=self.mask_register(),
        )

    # input
    if 'before' in self.input_asm():
      self.asm_string += self.input_asm()['before']
    if 'after' in self.input_asm():
      self.asm_string += self.input_asm()['after']
    if 'after' in self.weights_asm():
      for l in self.weights_asm()['after']:
        self.asm_string += l.format(
            W=self.w_ptr_register(), w_step=self.w_register_bytes() * n
        )

    for mr in range(0, self.m):
      for l in self.input_asm()['loop']:
        self.asm_string += l.format(
            AM_ptr=self.am_registers()[mr],
            AM=self.a_registers(0),
            a_offset=self.k_register(),
            A=self.a_registers(0),
        )
        loop = 'loop_tail' if tail else 'loop'
        for m in self.compute_asm()[loop]:
          for nr in range(0, n):
            self.asm_string += m.format(
                W=self.w_registers()[nr],
                A=self.a_registers(0),
                ACC=self.acc_registers()[self.m * nr + mr],
                mask=self.mask(),
            )
