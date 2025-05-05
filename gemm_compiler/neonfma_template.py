#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import aarch64_template


class NeonFma(aarch64_template.Aarch64):
  """All SIMD features for Aarch64 neondot."""

  def __init__(self, m: int, n: int, unroll_factor: int):
    super().__init__(m, n)
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
        'loop': ['fmla v{ACC}.4s, v{W}.4s, v{A}.s[{POS}]\n'],
    }
    return c_asm

  def init_accumulators(self):
    self.asm_string += '\n# Initialize accumulators with the biases.\n'
    accumulators = self.acc_registers()
    w_ptr = self.w_ptr_register()

    for nr in range(0, self.n - 1, 2):
      self.load_simd_register_pair(
          q0=accumulators[nr],
          q1=accumulators[nr + 1],
          ptr=w_ptr,
          offset=self.register_bytes() * nr,
      )

    for nr in range(0, self.n):
      for mr in range(1, self.m):
        self.copy_simd_register(
            prefix=self.prefix(),
            src=accumulators[nr],
            dst=accumulators[nr + mr * self.n],
        )

    self.increment_ptr(ptr=w_ptr, step=self.register_bytes() * self.n)

  def copy_simd_register(self, prefix, src, dst):
    self.asm_string += f'mov {prefix}{dst}.16b, {prefix}{src}.16b\n'

  def clamp_min(self, reg, prefix):
    max_reg = self.max_register()
    self.asm_string += (
        f'fmin {prefix}{reg}.4s, {max_reg}.4s, {prefix}{reg}.4s\n'
    )

  def clamp_max(self, reg, prefix):
    min_reg = self.min_register()
    self.asm_string += (
        f'fmax {prefix}{reg}.4s, {min_reg}.4s, {prefix}{reg}.4s\n'
    )

  def store(self):
    accumulators = self.acc_registers()
    cm_registers = self.cm_registers()
    nc_reg = self.nc_register()
    nc_lo = self.register_map_byte(nc_reg)
    nc = self.n * self.n_step()
    self.asm_string += """
      # Check whether full or partial store.
      cmp {nc}, {n_step}
      b.lo .Ltail_{N_2}\n""".format(n_step=nc, N_2=nc // 2, nc=nc_reg)
    for mr in range(0, self.m):
      self.store_simd_register_pair(
          q0=accumulators[mr * self.n],
          q1=accumulators[mr * self.n + 1],
          ptr=cm_registers[mr],
          post_increment=32,
      )
      for nr in range(2, self.n - 1, 2):
        self.store_simd_register_pair(
            q0=accumulators[mr * self.n + nr],
            q1=accumulators[mr * self.n + nr + 1],
            ptr=cm_registers[mr],
            post_increment=32,
        )

    for mr in range(0, self.m):
      am_ptr = self.am_registers()[mr]
      kc_register = self.kc_register()
      self.asm_string += f'sub {am_ptr}, {am_ptr}, {kc_register}\n'
    check = """
      sub {nc}, {nc}, {n_step}
      b.ne .Louter_loop
      b .Lreturn""".format(n_step=nc, nc=nc_reg)
    self.asm_string += check
    nc = nc // 2
    if nc * 2 > self.n_step():
      if nc == 8:
        self.asm_string += """
\n.Ltail_8:
      tbz {nc_lo}, 3, .Ltail_4\n""".format(nc_lo=nc_lo)
        for mr in range(0, self.m):
          self.store_simd_register_pair(
              q0=accumulators[mr * self.n],
              q1=accumulators[mr * self.n + 1],
              ptr=cm_registers[mr],
              post_increment=32,
          )
        for mr in range(0, self.m):
          self.asm_string += 'mov v{ACC0}.16b, v{ACC1}.16b\n'.format(
              ACC0=accumulators[mr * self.n],
              ACC1=accumulators[mr * self.n + 2],
          )
          self.asm_string += 'mov v{ACC0}.16b, v{ACC1}.16b\n'.format(
              ACC0=accumulators[mr * self.n + 1],
              ACC1=accumulators[mr * self.n + 3],
          )
      self.asm_string += """
\n.Ltail_4:
      tbz {nc_lo}, 2, .Ltail_2\n""".format(nc_lo=nc_lo)
      for mr in range(0, self.m):
        self.store_simd_register(
            r=accumulators[mr * self.n],
            prefix='q',
            ptr=cm_registers[mr],
            post_increment=16,
        )
      for mr in range(0, self.m):
        self.asm_string += 'mov v{ACC0}.16b, v{ACC1}.16b\n'.format(
            ACC0=accumulators[mr * self.n],
            ACC1=accumulators[mr * self.n + 1],
        )
    self.asm_string += """
\n.Ltail_2:
      tbz {nc_lo}, 1, .Ltail_1\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.store_simd_register(
          r=accumulators[mr * self.n],
          prefix='d',
          ptr=cm_registers[mr],
          post_increment=8,
      )
    for mr in range(0, self.m):
      self.asm_string += 'dup d{ACC}, v{ACC}.d[1]\n'.format(
          ACC=accumulators[mr * self.n]
      )
    self.asm_string += """
\n.Ltail_1:
      tbz {nc_lo}, 0, .Lreturn\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.store_simd_register(
          r=accumulators[mr * self.n],
          prefix='s',
          ptr=cm_registers[mr],
          post_increment=0,
      )
