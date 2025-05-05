#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

from gemm_compiler import x64_template


class Fma3(x64_template.X64):
  """All SIMD features for fma3."""

  def __init__(self, m: int, n: int, c: int):
    super().__init__(m, n)
    self._c = c

  @property
  def c(self) -> int:
    return self._c

  def isa(self):
    return 'fma3'

  def adjust_kc(self):
    return ''

  def init_accumulators(self):
    self.comment('Initialize accumulators with the biases.')
    w_reg = self.w_ptr_register()
    accumulators = self.acc_registers()
    bias = 'vmovaps  {prefix}{ACC}, [{W} + {offset}]\n'
    for nr in range(0, self.n):
      self.asm_string += bias.format(
          prefix=self.prefix(),
          W=w_reg,
          ACC=accumulators[nr * self.m],
          offset=self.register_bytes() * nr,
      )
    for nr in range(0, self.n):
      for mr in range(1, self.m):
        self.copy_simd_register(
            prefix=self.prefix(),
            src=accumulators[self.m * nr],
            dst=accumulators[self.m * nr + mr],
        )
    self.increment_ptr(ptr=w_reg, step=self.register_bytes() * self.n)

  def inner_loop_spill_gp(self, tail: bool = False) -> str:
    return self._inner_loop_spill_gp(self.n, tail)

  def inner_loop_small_M_N(self, tail: bool = False) -> str:
    return self._inner_loop_small_M_N(self.n, tail)

  def register_bytes(self):
    return 32

  def w_register_bytes(self):
    return self.register_bytes()

  def max_m_before_spilling(self):
    return 4

  def convert_to_output_type(self):
    return ''

  def prefix(self):
    return 'y'

  def n_step(self) -> int:
    return 8

  def a_registers(self, idx):
    registers = ['ymm2', 'ymm3', 'ymm4', 'ymm5']
    assert idx < len(registers)
    return registers[idx]

  def w_registers(self):
    return ['ymm14', 'ymm15']

  def input_asm(self):
    in_asm = {
        'loop': [
            'vbroadcastss {AM}, DWORD PTR [{AM_ptr} + {a_offset}]\n',
        ]
    }
    return in_asm

  def weights_asm(self) -> dict[str, list[str]]:
    w_asm = {
        'loop': [
            'vmovaps  {W}, [{W_ptr} + {offset}]\n',
        ],
        'after': ['add {W}, {w_step}\n'],
    }
    return w_asm

  def compute_asm(self):
    c_asm = {
        'loop': ['vfmadd231ps  y{ACC}, {A}, {W}\n'],
    }
    return c_asm

  def load_bias(self):
    return 'vmovaps  y{ACC}, [{W} + {offset}]\n'

  def copy_simd_register(self, prefix, src, dst):
    self.asm_string += f'vmovaps {prefix}{dst}, {prefix}{src}\n'

  def clamp_min(self, reg, prefix):
    max_reg = self.max_register()
    self.asm_string += (
        f'vminps  {prefix}{reg}, {prefix}{max_reg}, {prefix}{reg}\n'
    )

  def clamp_max(self, reg, prefix):
    min_reg = self.min_register()
    self.asm_string += (
        f'vmaxps  {prefix}{reg}, {prefix}{min_reg}, {prefix}{reg}\n'
    )

  def acc_registers(self):
    return [
        'mm6',
        'mm7',
        'mm8',
        'mm9',
        'mm10',
        'mm11',
        'mm12',
        'mm13',
        'mm15',
    ]

  def store(self):
    accumulators = self.acc_registers()
    cm_registers = self.cm_registers()
    nc_reg = self.nc_register()
    nc_lo = self.register_map_byte(nc_reg)
    pop_c = self.m > self.max_m_before_spilling()
    c_reg_offset = self.max_m_before_spilling()
    if pop_c:
      self.asm_string += '\n' + '# Pop output pointers from the stack.\n'
      c_reg_offset = 0
      pop_c_str = 'mov {C_REG}, [rsp + {offset}]\n'
      for mr in range(0, self.m):
        sp_offset = (mr) * 16 + self.c_ptr_stack_offset()
        self.asm_string += pop_c_str.format(
            C_REG=cm_registers[mr], offset=sp_offset
        )
    self.asm_string += """
    # Check whether full or partial store.
    cmp {nc}, {n_step}
    jl .Ltail_{N_2}
    """.format(
        n_step=self.n * self.n_step(),
        N_2=(self.n * self.n_step()) // 2,
        nc=nc_reg,
    )
    for mr in range(0, self.m):
      self.asm_string += 'vmovups  [{c_reg}], y{ACC}\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
      )
      for nr in range(1, self.n):
        self.asm_string += 'vmovups  [{c_reg} + {offset}], y{ACC}\n'.format(
            ACC=accumulators[self.m * nr + mr],
            c_reg=cm_registers[mr + c_reg_offset],
            offset=self.register_bytes() * nr,
        )
    for mr in range(0, self.m):
      self.asm_string += 'add {cm}, {cn_stride}\n'.format(
          cn_stride=self.n * self.register_bytes(),
          cm=cm_registers[mr + c_reg_offset],
      )
    if pop_c:
      self.asm_string += '\n' + '# Write output pointers to the stack.\n'
      pop_c_str = 'mov [rsp + {offset}], {C_REG}\n'
      for mr in range(0, self.m):
        sp_offset = (mr) * 16 + self.c_ptr_stack_offset()
        self.asm_string += pop_c_str.format(
            C_REG=cm_registers[mr + c_reg_offset], offset=sp_offset
        )
    check = """
    sub {nc}, {n_step}
    jne .Louter_loop
    jmp .Lreturn""".format(
        n_step=(self.n * self.n_step()), nc=nc_reg, OUTER=self.labels()[self.m]
    )
    self.asm_string += check
    if self.n > 1:
      self.asm_string += """
      .Ltail_8:
      test {nc_lo}, 8
      jz .Ltail_4\n""".format(
          N=self.n * self.n_step(),
          N_2=(self.n * self.n_step()) // 2,
          nc_lo=nc_lo,
      )
      for mr in range(0, self.m):
        self.asm_string += 'vmovups  [{c_reg}], y{ACC}\n'.format(
            ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
        )
      for nr in range(1, self.n):
        for mr in range(0, self.m):
          self.asm_string += 'vmovaps  y{ACC0}, y{ACC1}\n'.format(
              ACC0=accumulators[mr], ACC1=accumulators[mr + self.m * nr]
          )
      for mr in range(0, self.m):
        self.asm_string += 'add {cm}, 32\n'.format(
            cm=cm_registers[mr + c_reg_offset]
        )
    self.asm_string += """
\n.Ltail_4:
    test {nc_lo}, 4
    jz .Ltail_2\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.asm_string += 'vmovups  [{c_reg}], x{ACC}\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
      )
    for mr in range(0, self.m):
      self.asm_string += 'add  {c_reg}, 16\n'.format(
          c_reg=cm_registers[mr + c_reg_offset]
      )
    for mr in range(0, self.m):
      self.asm_string += 'vextractf128 x{ACC}, y{ACC}, 1\n'.format(
          ACC=accumulators[mr]
      )
    self.asm_string += """
\n.Ltail_2:
    test {nc_lo}, 2
    jz .Ltail_1\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.asm_string += 'vmovlps  QWORD PTR [{c_reg}], x{ACC}\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
      )
    for mr in range(0, self.m):
      self.asm_string += 'add {c_reg}, 8\n'.format(
          c_reg=cm_registers[mr + c_reg_offset]
      )
    for mr in range(0, self.m):
      self.asm_string += 'vmovhlps x{ACC}, x{ACC}, x{ACC}\n'.format(
          ACC=accumulators[mr]
      )
    self.asm_string += """
\n.Ltail_1:
    test {nc_lo}, 1
    jz .Lreturn\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.asm_string += 'vmovss  DWORD PTR [{c_reg}], x{ACC}\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
      )

  def stack_size(self):
    # Increase the stack size to allow for storing the original stack pointer,
    # nc, odd bits of k and other registers as required.
    size = self.m * 16 + 64
    # round up to multiple of 64.
    return math.ceil(size / 64) * 64
