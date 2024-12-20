#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import x64_template as arch

"""All SIMD features for fma3."""


class Fma3(arch.X64):

  def __init__(self):
    pass  # Empty constructor

  def isa(self):
    return 'fma3'

  def register_bytes(self):
    return 32

  def prefix(self):
    return 'y'

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

  def weights_asm(self):
    w_asm = {
        'loop': [
            'vmovaps  {W}, [{W_ptr} + {offset}]\n',
        ],
        'after': 'add {W}, {w_step}\n',
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
    return f'vmovaps {prefix}{dst}, {prefix}{src}\n'

  def clamp_min(self, reg, prefix):
    max_reg = self.max_register()
    return f'vminps  {prefix}{reg}, {prefix}{max_reg}, {prefix}{reg}\n'

  def clamp_max(self, reg, prefix):
    min_reg = self.min_register()
    return f'vmaxps  {prefix}{reg}, {prefix}{min_reg}, {prefix}{reg}\n'

  def store(
      self,
      M,
      N,
  ):
    accumulators = self.acc_registers()
    cm_registers = self.cm_registers()
    nc_reg = self.nc_register()
    nc_lo = self.register_map_byte(nc_reg)
    N_STEP = 8
    N_COUNT = N // N_STEP
    asm_string = """
    cmp {nc}, {n_step}
    jl tail_{N_2}
    """.format(n_step=N, N_2=N // 2, nc=nc_reg)
    for mr in range(0, M):
      asm_string += 'vmovups  [{c_reg}], y{ACC}\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr]
      )
      for nr in range(1, N_COUNT):
        asm_string += 'vmovups  [{c_reg} + {offset}], y{ACC}\n'.format(
            ACC=accumulators[M * nr + mr],
            c_reg=cm_registers[mr],
            offset=isa.register_bytes() * nr,
        )
    for mr in range(0, M):
      asm_string += 'add {cm}, {cn_stride}\n'.format(
          cn_stride=cn_stride_reg, cm=cm_registers[mr]
      )
    CHECK = """
    sub {nc}, {n_step}
    jne {OUTER}
    jmp return""".format(n_step=N, nc=nc_reg, OUTER=labels[M])
    asm_string += CHECK
    N = N // 2
    if N * 2 > N_STEP:
      asm_string += """
      tail_{N}:
      test {nc_lo}, {N}
      jz tail_{N_2}\n""".format(N=N, N_2=N // 2, nc_lo=nc_lo)
      for mr in range(0, M):
        asm_string += 'vmovups  [{c_reg}], y{ACC}\n'.format(
            ACC=accumulators[mr], c_reg=cm_registers[mr]
        )
      N_COUNT = N // N_STEP
      for nr in range(1, N_COUNT):
        for mr in range(0, M):
          asm_string += 'vmovups  [{c_reg} + {offset}], y{ACC}\n'.format(
              ACC=accumulators[M * nr + mr],
              c_reg=cm_registers[mr],
              offset=isa.register_bytes() * nr,
          )
      for mr in range(0, M):
        asm_string += 'vmovaps  y{ACC0}, y{ACC1}\n'.format(
            ACC0=accumulators[mr], ACC1=accumulators[mr + M * nr]
        )
      for mr in range(0, M):
        asm_string += 'add {cm}, 32\n'.format(
            cn_stride=cn_stride_reg, cm=cm_registers[mr]
        )
    asm_string += """
tail_4:
    test {nc_lo}, 4
    jz tail_2\n""".format(nc_lo=nc_lo)
    for mr in range(0, M):
      asm_string += 'vmovups  [{c_reg}], x{ACC}\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr]
      )
    for mr in range(0, M):
      asm_string += 'add  {c_reg}, 16\n'.format(c_reg=cm_registers[mr])
    for mr in range(0, M):
      asm_string += 'vextractf128 x{ACC}, y{ACC}, 1\n'.format(
          ACC=accumulators[mr]
      )
    asm_string += """
tail_2:
    test {nc_lo}, 2
    jz tail_1\n""".format(nc_lo=nc_lo)
    for mr in range(0, M):
      asm_string += 'vmovlps  QWORD PTR [{c_reg}], x{ACC}\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr]
      )
    for mr in range(0, M):
      asm_string += 'add {c_reg}, 8\n'.format(c_reg=cm_registers[mr])
    for mr in range(0, M):
      asm_string += 'vmovhlps x{ACC}, x{ACC}, x{ACC}\n'.format(
          ACC=accumulators[mr]
      )
    asm_string += """
tail_1:
    test {nc_lo}, 1
    jz return\n""".format(nc_lo=nc_lo)
    for mr in range(0, M):
      asm_string += 'vmovss  DWORD PTR [{c_reg}], x{ACC}\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr]
      )

    return asm_string
