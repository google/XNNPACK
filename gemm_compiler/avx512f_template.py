#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from gemm_compiler import fma3_template as isa
from gemm_compiler import x64_template as arch

"""All SIMD features for avx512f."""


class Avx512F(isa.Fma3):

  def __init__(self):
    self.c = 1

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

  def dequantize(self, M, N, W):
    return ''

  def adjust_kc(self):
    return ''

  def compute_asm(self):
    c_asm = {
        'loop': ['vfmadd231ps  z{ACC}, {A}, {W}\n'],
    }
    return c_asm

  def inner_loop_spill_gp(self, M, N, tail=False):
    N_COUNT = N // self.n_step()
    # weights
    asm_string = ''
    if 'before' in self.weights_asm():
      asm_string += self.weights_asm()['before']
    for l in self.weights_asm()['loop']:
      for nr in range(0, N_COUNT):
        asm_string += l.format(
            W_ptr=self.w_ptr_register(),
            W=self.w_registers()[nr],
            offset=self.register_bytes() * nr,
            w_step=self.register_bytes() * N_COUNT,
        )

    # input
    if 'before' in self.input_asm():
      asm_string += self.input_asm()['before']
    if 'after' in self.input_asm():
      asm_string += self.input_asm()['after']
    if 'after' in self.weights_asm():
      asm_string += self.weights_asm()['after'].format(
          W=self.w_ptr_register(), w_step=self.register_bytes() * N_COUNT
      )

    for mr in range(0, M):
      for l in self.input_asm()['loop']:
        asm_string += l.format(
            AM_ptr=self.am_registers()[mr],
            AM=self.a_registers(0),
            a_offset=self.k_register(),
            A=self.a_registers(0),
        )
        loop = 'loop_tail' if tail else 'loop'
        for m in self.compute_asm()[loop]:
          for nr in range(0, N_COUNT):
            asm_string += m.format(
                W=self.w_registers()[nr],
                A=self.a_registers(0),
                ACC=self.acc_registers()[M * nr + mr],
                mask=self.mask()
            )
    return asm_string

  def inner_loop_small_M_N(self, M, N, tail=False):
    N_COUNT = N // self.n_step()
    # input
    asm_string = ''
    if 'before' in self.input_asm():
      asm_string += self.input_asm()['before']
    if 'after' in self.input_asm():
      asm_string += self.input_asm()['after']

    # weights
    if 'before' in self.weights_asm():
      asm_string += self.weights_asm()['before']
    for l in self.weights_asm()['loop']:
      for nr in range(0, N_COUNT):
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

    loop = 'loop_tail' if tail else 'loop'
    for mr in range(0, M):
      for l in self.input_asm()['loop']:
        asm_string += l.format(
            AM_ptr=self.am_registers()[mr],
            AM=self.a_registers(mr),
            a_offset=self.k_register(),
            A=self.a_registers(mr),
        )
      for m in self.compute_asm()[loop]:
        for nr in range(0, N_COUNT):
          asm_string += m.format(
              W=self.w_registers()[nr],
              A=self.a_registers(mr),
              ACC=self.acc_registers()[M * nr + mr],
              mask=self.mask()
          )
    return asm_string

  def init_accumulators(self, M, N):
    ret = '# Initialize accumulators with the biases.\n'
    W = self.w_ptr_register()
    accumulators = self.acc_registers()
    bias = 'vmovaps  z{ACC}, [{W} + {offset}]\n'
    for nr in range(0, N):
      ret += bias.format(
          W=W, ACC=accumulators[nr * M], offset=self.register_bytes() * nr
      )
    for nr in range(0, N):
      for mr in range(1, M):
        ret += self.copy_simd_register(
            prefix=self.prefix(),
            src=accumulators[M * nr],
            dst=accumulators[M * nr + mr],
        )
    ret += self.increment_ptr(ptr=W, step=self.register_bytes() * N)
    return ret

  def copy_simd_register(self, prefix, src, dst):
    return f'vmovaps {prefix}{dst}, {prefix}{src}\n'

  def store(
      self,
      M,
      N,
  ):
    tmp_gp_regs = self.tmp_gp_registers()
    accumulators = self.acc_registers()
    cm_registers = self.cm_registers()
    nc_reg = self.nc_register()
    pop_c = M > self.max_M_before_spilling()
    N_COUNT = N // self.n_step()
    asm_string = ''
    c_reg_offset = self.max_M_before_spilling()
    if pop_c:
      asm_string += '\n' + '# Pop output pointers from the stack.\n'
      c_reg_offset = 0
      POP_C = 'mov {C_REG}, [rsp + {offset}]\n'
      for mr in range(0, M):
        sp_offset = (mr) * 16 + self.c_ptr_stack_offset()
        asm_string += POP_C.format(C_REG=cm_registers[mr], offset=sp_offset)
    asm_string += """
      # Check whether full or partial store.
      cmp {nc}, {n_step}
      jl tail\n""".format(n_step=N, N_2=N // 2, nc=nc_reg)
    for mr in range(0, M):
      asm_string += """
      vmovups  [{c_reg}], z{ACC}""".format(
          ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
      )
      for nr in range(1, N_COUNT):
        asm_string += """
      vmovups  [{c_reg} + {offset}], z{ACC}""".format(
            ACC=accumulators[M * nr + mr],
            c_reg=cm_registers[mr + c_reg_offset],
            offset=self.register_bytes() * nr,
        )
    asm_string += '\n'
    for mr in range(0, M):
      asm_string += 'add {cm}, {cn_stride}\n'.format(
          cn_stride=N_COUNT * 64, cm=cm_registers[mr + c_reg_offset]
      )
    if pop_c:
      asm_string += '\n' + '# Write output pointers to the stack.\n'
      POP_C = 'mov [rsp + {offset}], {C_REG}\n'
      for mr in range(0, M):
        sp_offset = (mr) * 16 + self.c_ptr_stack_offset()
        asm_string += POP_C.format(C_REG=cm_registers[mr], offset=sp_offset)
    CHECK = """
      sub {nc}, {n_step}
      jne outer_loop
      jmp return\n""".format(n_step=N, nc=nc_reg)
    asm_string += CHECK

    asm_string += '\ntail:'
    if N == 64:
      asm_string += """
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
          ACC=accumulators[0],
          c_reg=cm_registers[0],
      )
      for mr in range(0, M):
        asm_string += 'vmovups  ZMMWORD PTR [{c_reg}]{{k1}}, z{ACC}\n'.format(
            ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
        )
        asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg} + 64]{{k2}}, z{ACC}\n'.format(
                ACC=accumulators[mr + M], c_reg=cm_registers[mr + c_reg_offset]
            )
        )
        asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg} + 128]{{k3}}, z{ACC}\n'.format(
                ACC=accumulators[mr + 2 * M],
                c_reg=cm_registers[mr + c_reg_offset],
            )
        )
        asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg} + 192]{{k4}}, z{ACC}\n'.format(
                ACC=accumulators[mr + 3 * M],
                c_reg=cm_registers[mr + c_reg_offset],
            )
        )
    elif N == 32:
      asm_string += """
      mov {tmp1}, -1
      shlx {tmp1}, {tmp1}, {nc_reg}
      not {tmp1}
      kmovw k1, {tmp1_lo}
      shr {tmp1_lo}, 16
      kmovw k2, {tmp1_lo}\n""".format(
          nc_reg=nc_reg,
          tmp1_lo=self.register_map_dword(tmp_gp_regs[1]),
          tmp1=tmp_gp_regs[1],
          ACC=accumulators[0],
          c_reg=cm_registers[0],
      )
      for mr in range(0, M):
        asm_string += 'vmovups  ZMMWORD PTR [{c_reg}]{{k1}}, z{ACC}\n'.format(
            ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
        )
        asm_string += (
            'vmovups  ZMMWORD PTR [{c_reg} + 64]{{k2}}, z{ACC}\n'.format(
                ACC=accumulators[mr + M], c_reg=cm_registers[mr + c_reg_offset]
            )
        )
    else:
      asm_string += """
      mov {tmp1}, -1
      shlx {tmp1}, {tmp1}, {nc_reg}
      not {tmp1}
      kmovw k1, {tmp1_lo}\n""".format(
          nc_reg=nc_reg,
          tmp1=tmp_gp_regs[1],
          tmp1_lo=self.register_map_dword(tmp_gp_regs[1]),
          ACC=accumulators[0],
          c_reg=cm_registers[0 + c_reg_offset],
      )
      for mr in range(0, M):
        asm_string += 'vmovups  ZMMWORD PTR [{c_reg}]{{k1}}, z{ACC}\n'.format(
            ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
        )

    return asm_string

  def stack_size(self, M):
    # Increase the stack size to allow for storing the original stack pointer,
    # nc, odd bits of k and other registers as required.
    size = M * 16 + 64
    # round up to multiple of 64.
    return math.ceil(size / 64) * 64

class Avx512FC(Avx512F):
  def __init__(self, c):
    self.c = c

  def input_asm(self):
    in_asm = {
        'loop': [
            'vbroadcastsd {AM}, QWORD PTR [{AM_ptr} + {a_offset}]\n',
        ]
    }
    return in_asm

  def pre_header(self):
    return '''.PERMUTATION:
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
        '''

  def function_name(self, M, N, isa):
    c = self.c
    return f'xnn_f32_gemm_minmax_ukernel_{M}x{N}c{c}__asm_amd64_{isa}_broadcast'

  def dequantize(self, M, N, W):
    shift_add = '''vpsrlq {tmp}, z{acc}, 32
      vaddps z{acc}, z{acc}, {tmp}\n'''
    asm_string = ''
    accumulators = self.acc_registers()
    for nr in range(0, N * 2):
      for mr in range(0, M):
        asm_string += shift_add.format(acc=accumulators[M * nr + mr], tmp=self.w_registers()[0])
    perm_reg = self.w_registers()[0]
    asm_string += f'vmovups {perm_reg}, zmmword ptr [rip + .PERMUTATION]\n'
    perm = 'vpermt2ps z{acc0}, {perm_reg}, z{acc1}\n'
    for nr in range(0, N):
      for mr in range(0, M):
        asm_string += perm.format(perm_reg=perm_reg, acc0=accumulators[2 * M * nr + mr], acc1=accumulators[2 * M * nr + M + mr])
    return asm_string

  def inner_loop_increment(self):
    return self.c * self.element_size()

  def k_mask(self):
    return "0xFFFFFFFFFFFFFFFB"

  def init_accumulators(self, M, N):
    W = self.w_ptr_register()
    accumulators = self.acc_registers()
    bias_registers = self.bias_registers()
    bias = 'vmovaps  {ACC}, [{W} + {offset}]\n'
    asm_string = ''
    for nr in range(0, N):
      asm_string += bias.format(
          W=W, ACC=bias_registers[nr], offset=self.register_bytes() * nr
      )

    c = self.c * self.element_size()
    asm_string += '# Interleave with zeros.\n'
    unpack_lo = 'vpmovzxdq z{acc0}, y{acc1}\n'
    unpack_hi = '''vextracti64x4 y{acc1}, z{acc1}, 1
    vpmovzxdq z{acc0}, y{acc1}
    '''
    for nr in range(0, N):
      asm_string += unpack_lo.format(
          acc0=accumulators[2 * M * nr],
          acc1=bias_registers[nr][1:],
      )
      asm_string += unpack_hi.format(
          acc0=accumulators[2 * M * nr + M],
          acc1=bias_registers[nr][1:],
      )
    for nr in range(0, N * 2):
      for mr in range(1, M):
        asm_string += self.copy_simd_register(
            prefix=self.prefix(),
            src=accumulators[M * nr],
            dst=accumulators[M * nr + mr],
        )
    asm_string += self.increment_ptr(ptr=W, step=self.register_bytes() * N)
    asm_string += f"""
      # Are there at least {c} bytes?
      cmp rdx, {c}
      js inner_loop_tail\n"""

    return asm_string

  def inner_loop(self, M, N):
    return super().inner_loop(M, N * 2)

  def inner_loop_tail(self, M, N):
    k_register = self.k_register()
    nc_register = self.nc_register()
    offset = M * 16 + self.c_ptr_stack_offset()
    nc_offset = offset + 8
    asm_string = f"""
      # Store nc_register.
      mov [rsp + {nc_offset}], {nc_register}
      # Load odd k bit.
      mov {nc_register}, [rsp + {offset}]
      # Check if channels are odd.
      test {nc_register}, {nc_register}
      mov {nc_register}, [rsp + {nc_offset}]
      jz inner_loop_end

      inner_loop_tail:\n"""
    if M > self.max_M_before_spilling():
      asm_string += self.inner_loop_spill_gp(M=M, N=N, tail=True)
    else:
      asm_string += self.inner_loop_small_M_N(M=M, N=N, tail=True)
    return asm_string

  def compute_asm(self):
    c_asm = {
        'loop': ['vfmadd231ps  z{ACC}, {A}, {W}\n'],
        'loop_tail': ['vfmadd231ps  z{ACC}{{{mask}}}, {A}, {W}\n'],
    }
    return c_asm

  def outer_loop_prepare(self, M, N):
    k_register = self.k_register()
    kc_register = self.kc_register()
    offset = M * 16 + self.c_ptr_stack_offset()
    element_size = self.element_size()
    k_mask = self.k_mask()
    mask = self.mask()
    asm_string = f"""
      # Copy k and flip bit.
      mov {k_register}, rdx
      and {k_register}, 0x{element_size}
      and {kc_register}, {k_mask}
      mov [rsp + {offset}], {k_register}
      mov r11, 0x5555
      kmovw {mask}, r11d\n"""
    return asm_string

  def mask(self):
    return 'k3'

  def bias_registers(self):
    return self.w_registers()

  def clamp_min(self, reg, prefix, other_reg):
    min_reg = self.max_register()
    return f'vminps  {prefix}{reg}, {prefix}{min_reg}, {prefix}{other_reg}\n'

  def clamp(self, M, N):
    '''
    Clamp output registers while handling rotation to match standard registers.
    '''
    num_horizontal_registers = int(N / self.n_step())
    acc_registers = self.acc_registers()
    asm_string = ''
    for nr in range(0, num_horizontal_registers):
      for mr in range(0, M):
        asm_string += self.clamp_min(
            reg=acc_registers[M * nr + mr], prefix=self.prefix(), other_reg=acc_registers[M * nr + mr +  nr * M],
        )
    for nr in range(0, num_horizontal_registers):
      for mr in range(0, M):
        asm_string += self.clamp_max(
            reg=acc_registers[M * nr + mr], prefix=self.prefix()
        )
    return asm_string
