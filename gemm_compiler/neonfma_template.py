#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

from gemm_compiler import aarch64_template as arch

"""All SIMD features for Aarch64 neondot."""


class NeonFma(arch.Aarch64):

  def __init__(self, unroll_factor):
    self.unroll_factor = unroll_factor
    self.decrement = 4 * unroll_factor

  def n_step(self):
    return 4

  def isa(self):
    return 'neonfma'

  def register_bytes(self):
    return 16

  def weights_register_bytes(self):
    return self.register_bytes()

  def prefix(self):
    return 'v'

  def acc_registers(self):
    return [
        '11',
        '12',
        '13',
        '14',
        '15',
        '16',
        '17',
        '18',
        '19',
        '20',
        '21',
        '22',
        '23',
        '24',
        '25',
        '26',
        '27',
        '28',
        '29',
        '30',
        '10',
    ]

  def a_registers(self, idx):
    registers = ['2', '3', '4', '5', '6', '31', '29', '30']
    assert idx < len(registers)
    return registers[idx]

  def w_registers(self):
    return ['7', '8', '9', '10']

  def input_asm(self):
    match self.unroll_factor:
      case 1:
        return {
            'loop': [
                'ldr s{AM}, [{AM_ptr}], 4\n',
            ]
        }
      case 2:
        return {
            'loop': [
                'ldr d{AM}, [{AM_ptr}], 8\n',
            ]
        }
      case 4:
        return {
            'loop': [
                'ldr q{AM}, [{AM_ptr}], 16\n',
            ]
        }
      case _:
        raise NotImplementedError

  def base_input_asm(self):
    return {
        'loop': [
            'ldr s{AM}, [{AM_ptr}], 4\n',
        ]
    }

  def weights_asm(self):
    w_asm = {
        'loop__': [
            'ldr q{W}, [{W_ptr}], {w_step}\n',
        ],
        'loop': [
            'ldp q{W}, q{W_1}, [{W_ptr}], 16\n',
        ],
        'loop_2': [
            'ldp q{W}, q{W_1}, [{W_ptr}], 32\n',
        ],
    }
    return w_asm

  def compute_asm(self):
    c_asm = {
        'loop': ['fmla  v{ACC}.4s, v{W}.4s, v{A}.s[{POS}]\n'],
    }
    return c_asm

  def init_accumulators(self, M, N):
    ret = '\n# Initialize accumulators with the biases.\n'
    accumulators = self.acc_registers()
    W = self.w_ptr_register()
    single_bias = 'ldr q{ACC}, [{W}, {offset}]\n'
    pair_bias = 'ldp q{ACC}, q{ACC_1}, [{W}, {offset}]\n'

    for nr in range(0, N - 1, 2):
      ret += pair_bias.format(
          W=W,
          ACC=accumulators[nr * M],
          ACC_1=accumulators[nr * M + M],
          offset=self.register_bytes() * nr,
      )
    if N % 2 != 0:
      ret += single_bias.format(
          W=W,
          ACC=accumulators[(N - 1) * M],
          offset=self.register_bytes() * (N - 1),
      )
    for nr in range(0, N):
      for mr in range(1, M):
        ret += self.copy_simd_register(
            prefix=self.prefix(),
            src=accumulators[M * nr],
            dst=accumulators[M * nr + mr],
        )

    num_horizontal_registers = int(N / self.n_step())
    ret += self.increment_ptr(ptr=W, step=self.register_bytes() * N)
    return ret

  def copy_simd_register(self, prefix, src, dst):
    return f'mov {prefix}{dst}.16b, {prefix}{src}.16b\n'

  def clamp_min(self, reg, prefix):
    max_reg = self.max_register()
    return f'fmin  {prefix}{reg}.4s, {max_reg}.4s, {prefix}{reg}.4s\n'

  def clamp_max(self, reg, prefix):
    min_reg = self.min_register()
    return f'fmax  {prefix}{reg}.4s, {min_reg}.4s, {prefix}{reg}.4s\n'

  def store(
      self,
      M,
      N,
  ):
    accumulators = self.acc_registers()
    cm_registers = self.cm_registers()
    nc_reg = self.nc_register()
    nc_lo = self.register_map_byte(nc_reg)
    N_COUNT = N // self.n_step()
    asm_string = """
      # Check whether full or partial store.
      cmp {nc}, {n_step}
      b.lo tail_{N_2}\n""".format(n_step=N, N_2=N // 2, nc=nc_reg)
    for mr in range(0, M):
      asm_string += 'stp  q{ACC}, q{ACC_1}, [{c_reg}], 32\n'.format(
          ACC=accumulators[mr],
          ACC_1=accumulators[M + mr],
          c_reg=cm_registers[mr],
      )
      for nr in range(2, N_COUNT - 1, 2):
        asm_string += 'stp  q{ACC}, q{ACC_1}, [{c_reg}], 32\n'.format(
            ACC=accumulators[M * 2 + mr],
            ACC_1=accumulators[M * 3 + mr],
            c_reg=cm_registers[mr],
        )
      if N_COUNT % 2 != 0:
        asm_string += 'str  q{ACC}, [{c_reg}], 16\n'.format(
            ACC=accumulators[M * 2 + mr],
            c_reg=cm_registers[mr],
        )

    for mr in range(0, M):
      AM_PTR = self.am_registers()[mr]
      kc_register = self.kc_register()
      asm_string += f'sub {AM_PTR}, {AM_PTR}, {kc_register}\n'
    CHECK = """
      sub {nc}, {nc}, {n_step}
      b.ne outer_loop
      b return""".format(n_step=N, nc=nc_reg)
    asm_string += CHECK
    N = N // 2
    if N * 2 > self.n_step():
      if N == 8:
        asm_string += """
\ntail_8:
      tbz {nc_lo}, 3, tail_4\n""".format(nc_lo=nc_lo)
        for mr in range(0, M):
          asm_string += 'stp  q{ACC}, q{ACC_1}, [{c_reg}], 32\n'.format(
              ACC=accumulators[mr],
              ACC_1=accumulators[mr + M],
              c_reg=cm_registers[mr],
          )
        for mr in range(0, M):
          asm_string += 'mov  v{ACC0}.16b, v{ACC1}.16b\n'.format(
              ACC0=accumulators[mr], ACC1=accumulators[mr + 2 * M]
          )
          asm_string += 'mov  v{ACC0}.16b, v{ACC1}.16b\n'.format(
              ACC0=accumulators[mr + M], ACC1=accumulators[mr + 3 * M]
          )
      asm_string += """
\ntail_4:
      tbz {nc_lo}, 2, tail_2\n""".format(nc_lo=nc_lo)
      for mr in range(0, M):
        asm_string += 'str  q{ACC}, [{c_reg}], 16\n'.format(
            ACC=accumulators[mr], c_reg=cm_registers[mr]
        )
      for mr in range(0, M):
        asm_string += 'mov  v{ACC0}.16b, v{ACC1}.16b\n'.format(
            ACC0=accumulators[mr], ACC1=accumulators[mr + M]
        )
    asm_string += """
\ntail_2:
      tbz {nc_lo}, 1, tail_1\n""".format(nc_lo=nc_lo)
    for mr in range(0, M):
      asm_string += 'str  d{ACC}, [{c_reg}], 8\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr]
      )
    for mr in range(0, M):
      asm_string += 'dup d{ACC}, v{ACC}.d[1]\n'.format(ACC=accumulators[mr])
    asm_string += """
\ntail_1:
      tbz {nc_lo}, 0, return\n""".format(nc_lo=nc_lo)
    for mr in range(0, M):
      asm_string += 'str  s{ACC}, [{c_reg}]\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr]
      )

    return asm_string
