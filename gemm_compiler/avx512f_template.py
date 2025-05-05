#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import fma3_template


class Avx512F(fma3_template.Fma3):
  """All SIMD features for avx512f."""

  def isa(self):
    return 'avx512f'

  def register_bytes(self):
    return 64

  def prefix(self):
    return 'z'

  def a_registers(self, idx):
    registers = ['zmm2', 'zmm3', 'zmm4', 'zmm5', 'zmm6']
    assert idx < len(registers)
    return registers[idx]

  def w_registers(self):
    return ['zmm7', 'zmm8', 'zmm9', 'zmm10']

  def n_step(self):
    return 16

  def compute_asm(self):
    c_asm = {
        'loop': ['vfmadd231ps  z{ACC}, {A}, {W}\n'],
        'loop_tail': ['vfmadd231ps  z{ACC}{{{mask}}}, {A}, {W}\n'],
    }
    return c_asm

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

  def copy_simd_register(self, prefix, src, dst):
    self.asm_string += f'vmovaps {prefix}{dst}, {prefix}{src}\n'

  def max_m_before_spilling(self):
    return 5

  def acc_registers(self):
    return [
        'mm11',
        'mm12',
        'mm13',
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
        'mm9',
        'mm10',
    ]

  def store(self):
    tmp_gp_regs = self.tmp_gp_registers()
    accumulators = self.acc_registers()
    cm_registers = self.cm_registers()
    nc_reg = self.nc_register()
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
      jl .Ltail\n""".format(n_step=self.n * self.n_step(), nc=nc_reg)
    for mr in range(0, self.m):
      self.asm_string += """
      vmovups  [{c_reg}], z{ACC}""".format(
          ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
      )
      for nr in range(1, self.n):
        self.asm_string += """
      vmovups  [{c_reg} + {offset}], z{ACC}""".format(
            ACC=accumulators[self.m * nr + mr],
            c_reg=cm_registers[mr + c_reg_offset],
            offset=self.register_bytes() * nr,
        )
    self.asm_string += '\n'
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
            C_REG=cm_registers[mr], offset=sp_offset
        )
    check = """
      sub {nc}, {n_step}
      jne .Louter_loop
      jmp .Lreturn\n""".format(n_step=self.n * self.n_step(), nc=nc_reg)
    self.asm_string += check

    self.asm_string += '\n.Ltail:'
    if self.n * self.n_step() == 64:
      self.asm_string += """
      mov {tmp1}, -1
      shlx {tmp1}, {tmp1}, {nc_reg}
      not {tmp1}
      kmovw k1, {tmp1_lo}
      shr {tmp1}, 16
      kmovw k2, {tmp1_lo}
      shr {tmp1}, 16
      kmovw k3, {tmp1_lo}
      shr {tmp1}, 16
      kmovw k4, {tmp1_lo}\n
      """.format(
          nc_reg=nc_reg,
          tmp1=tmp_gp_regs[1],
          tmp1_lo=self.register_map_dword(tmp_gp_regs[1]),
      )
      for mr in range(0, self.m):
        self.asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg}]{{k1}}, z{ACC}\n'.format(
                ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
            )
        )
        self.asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg} + 64]{{k2}}, z{ACC}\n'.format(
                ACC=accumulators[mr + self.m],
                c_reg=cm_registers[mr + c_reg_offset],
            )
        )
        self.asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg} + 128]{{k3}}, z{ACC}\n'.format(
                ACC=accumulators[mr + 2 * self.m],
                c_reg=cm_registers[mr + c_reg_offset],
            )
        )
        self.asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg} + 192]{{k4}}, z{ACC}\n'.format(
                ACC=accumulators[mr + 3 * self.m],
                c_reg=cm_registers[mr + c_reg_offset],
            )
        )
    elif self.n * self.n_step() == 32:
      self.asm_string += """
      mov {tmp1}, -1
      shlx {tmp1}, {tmp1}, {nc_reg}
      not {tmp1}
      kmovw k1, {tmp1_lo}
      shr {tmp1_lo}, 16
      kmovw k2, {tmp1_lo}\n""".format(
          nc_reg=nc_reg,
          tmp1_lo=self.register_map_dword(tmp_gp_regs[1]),
          tmp1=tmp_gp_regs[1],
      )
      for mr in range(0, self.m):
        self.asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg}]{{k1}}, z{ACC}\n'.format(
                ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
            )
        )
        self.asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg} + 64]{{k2}}, z{ACC}\n'.format(
                ACC=accumulators[mr + self.m],
                c_reg=cm_registers[mr + c_reg_offset],
            )
        )
    else:
      self.asm_string += """
      mov {tmp1}, -1
      shlx {tmp1}, {tmp1}, {nc_reg}
      not {tmp1}
      kmovw k1, {tmp1_lo}\n""".format(
          nc_reg=nc_reg,
          tmp1=tmp_gp_regs[1],
          tmp1_lo=self.register_map_dword(tmp_gp_regs[1]),
      )
      for mr in range(0, self.m):
        self.asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg}]{{k1}}, z{ACC}\n'.format(
                ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
            )
        )

class Avx512FC(Avx512F):
  """All SIMD features for avx512fc."""

  def input_asm(self):
    in_asm = {
        'loop': [
            'vbroadcastsd {AM}, QWORD PTR [{AM_ptr} + {a_offset}]\n',
        ]
    }
    return in_asm

  def pre_header(self):
    self.asm_string += """.PERMUTATION:
        .long   0
        .long   2
        .long   4
        .long   6
        .long   8
        .long   10
        .long   12
        .long   14
        .long   16
        .long   18
        .long   20
        .long   22
        .long   24
        .long   26
        .long   28
        .long   30
        """

  def function_name(self):
    return (
        f'xnn_f32_gemm_minmax_ukernel_{self.m}x{self.n * self.n_step()}c{self.c}'
        + f'__asm_amd64_{self.isa()}_broadcast'
    )

  def convert_to_output_type(self):
    shift_add = """vpsrlq {tmp}, z{acc}, 32
      vaddps z{acc}, z{acc}, {tmp}\n"""
    accumulators = self.acc_registers()
    for nr in range(0, self.n * 2):
      for mr in range(0, self.m):
        self.asm_string += shift_add.format(
            acc=accumulators[self.m * nr + mr], tmp=self.w_registers()[0]
        )
    perm_reg = self.w_registers()[0]
    self.asm_string += f'vmovups {perm_reg}, zmmword ptr [rip + .PERMUTATION]\n'
    perm = 'vpermt2ps z{acc0}, {perm_reg}, z{acc1}\n'
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += perm.format(
            perm_reg=perm_reg,
            acc0=accumulators[2 * self.m * nr + mr],
            acc1=accumulators[2 * self.m * nr + self.m + mr],
        )

  def k_mask(self):
    return '0xFFFFFFFFFFFFFFFB'

  def init_accumulators(self):
    w_ptr = self.w_ptr_register()
    accumulators = self.acc_registers()
    bias_registers = self.bias_registers()
    bias = 'vmovaps  {ACC}, [{W} + {offset}]\n'
    for nr in range(0, self.n):
      self.asm_string += bias.format(
          W=w_ptr, ACC=bias_registers[nr], offset=self.register_bytes() * nr
      )

    c = self.c * self.element_size()
    self.asm_string += '# Interleave with zeros.\n'
    unpack_lo = 'vpmovzxdq z{acc0}, y{acc1}\n'
    unpack_hi = """vextracti64x4 y{acc1}, z{acc1}, 1
    vpmovzxdq z{acc0}, y{acc1}
    """
    for nr in range(0, self.n):
      self.asm_string += unpack_lo.format(
          acc0=accumulators[2 * self.m * nr],
          acc1=bias_registers[nr][1:],
      )
      self.asm_string += unpack_hi.format(
          acc0=accumulators[2 * self.m * nr + self.m],
          acc1=bias_registers[nr][1:],
      )
    for nr in range(0, self.n * 2):
      for mr in range(1, self.m):
        self.copy_simd_register(
            prefix=self.prefix(),
            src=accumulators[self.m * nr],
            dst=accumulators[self.m * nr + mr],
        )
    self.increment_ptr(ptr=w_ptr, step=self.register_bytes() * self.n)
    self.asm_string += f"""
      # Are there at least {c} bytes?
      cmp rdx, {c}
      js .Linner_loop_tail\n"""

  def outer_loop_prepare(self):
    k_register = self.k_register()
    kc_register = self.kc_register()
    offset = self.m * 16 + self.c_ptr_stack_offset()
    element_size = self.element_size()
    k_mask = self.k_mask()
    mask = self.mask()
    self.asm_string += f"""
      # Copy k and flip bit.
      mov {k_register}, rdx
      and {k_register}, 0x{element_size}
      and {kc_register}, {k_mask}
      mov [rsp + {offset}], {k_register}
      mov r11, 0x5555
      kmovw {mask}, r11d\n"""

  def mask(self):
    return 'k3'

  def bias_registers(self):
    return self.w_registers()

  def clamp_min(self, reg, prefix, other_reg):
    min_reg = self.max_register()
    self.asm_string += (
        f'vminps  {prefix}{reg}, {prefix}{min_reg}, {prefix}{other_reg}\n'
    )

  def clamp(self):
    """Clamp output registers while handling rotation to match standard registers."""
    acc_registers = self.acc_registers()
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.clamp_min(
            reg=acc_registers[self.m * nr + mr],
            prefix=self.prefix(),
            other_reg=acc_registers[self.m * nr + mr + nr * self.m],
        )
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.clamp_max(
            reg=acc_registers[self.m * nr + mr], prefix=self.prefix()
        )

  def inner_loop_spill_gp(self, tail: bool = False) -> str:
    return self._inner_loop_spill_gp(self._c * self.n, tail)

  def inner_loop_small_M_N(self, tail: bool = False) -> str:
    return self._inner_loop_small_M_N(self._c * self.n, tail)

  def inner_loop_tail(self):
    nc_register = self.nc_register()
    offset = self.m * 16 + self.c_ptr_stack_offset()
    nc_offset = offset + 8
    self.asm_string += f"""
      # Store nc_register.
      mov [rsp + {nc_offset}], {nc_register}
      # Load odd k bit.
      mov {nc_register}, [rsp + {offset}]
      # Check if channels are odd.
      test {nc_register}, {nc_register}
      mov {nc_register}, [rsp + {nc_offset}]
      jz .Linner_loop_end

      .Linner_loop_tail:\n"""
    if self.m > self.max_m_before_spilling():
      self.inner_loop_spill_gp(tail=True)
    else:
      self.inner_loop_small_M_N(tail=True)
