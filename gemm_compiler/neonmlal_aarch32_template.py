#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import aarch32_template as isa

"""All SIMD features for Aarch32 neonmlal."""


class NeonMlal(isa.Aarch32):

  def __init__(self, unroll_factor):
    self.unroll_factor = unroll_factor
    self.decrement = 4 * unroll_factor

  def isa(self):
    return 'neonmlal'

  def a_registers(self, idx):
    registers = ['0', '1', '2', '3']
    assert idx < len(registers)
    return registers[idx]

  def w_registers(self):
    return ['6', '7']

  def ksum_regs(self, idx):
    return self.w_registers()[idx]

  def acc_registers(self):
    return [
        '8',
        '9',
        '10',
        '11',
        '12',
        '13',
        '14',
        '15',
    ]

  def function_name(self, M, N, isa):
    LD = self.unroll_factor * 32
    return f'xnn_qd8_f32_qc8w_gemm_minmax_ukernel_{M}x{N}__asm_aarch32_{isa}_ld{LD}_2\n'

  def zp_scale(self, pos):
    regs = ['4', '5']
    return regs[pos]

  # kc = round_up_po2(kc, channels)
  def adjust_kc(self):
    return ''

  def minmax_reg(self):
    return 'r11'

  def quantization_params(self, M):
    ret = """\n# Load quantization params
  ldr {quantization_params_reg}, [sp, #124]\n""".format(
        quantization_params_reg=self.quantization_params_register()
    )
    zp_scale = 'vld1.32 {{q{lo}, q{hi}}}, [{quantization_params_reg}]\n'
    ret += '# Load minmax pointer.\n'
    minmax_reg = self.minmax_reg()
    ret += f'ldr {minmax_reg}, [sp, #120]\n'
    ret += '# Load dynamic quantization params.\n'
    for mr in range(0, M, 4):
      ret += zp_scale.format(
          quantization_params_reg=self.quantization_params_register(),
          lo=str(int(self.zp_scale(mr // 2))),
          hi=str(int(self.zp_scale(mr // 2)) + 1),
      )
      return ret

  def quantization_params_register(self):
    return 'r7'

  def compute_asm(self):
    c_asm = {
        'loop': ['vmlal.s16  q{ACC}, d{W}, d{A}[{POS}]\n'],
    }
    return c_asm

  def cvtf(self):
    return 'vcvt.f32.s32 q{ACC}, q{ACC}\n'

  def input_asm(self):
    return {
        'loop': [
            'vld1.8 d{AM}, [{AM_ptr}]!\n',
        ],
        'after': ['vmovl.s8 q{AM_2}, d{AM}\n'],
    }

  def base_input_asm(self):
    return {
        'loop': [
            'ldr s{AM}, [{AM_ptr}], 4\n',
        ]
    }

  def weights_asm(self):
    w_asm = {
        'loop': [
            'vld1.8 d{W}, [{W_ptr}]!\n',
        ],
        'after': ['vmovl.s8 q{W_2}, d{W}\n'],
    }
    return w_asm

  def load_min_max(self):
    minmax_reg = self.minmax_reg()
    qmin_0 = int(self.min_register()) * 2
    qmin_1 = int(self.min_register()) * 2 + 1
    qmax_0 = int(self.max_register()) * 2
    qmax_1 = int(self.max_register()) * 2 + 1
    ret = f'vld1.32 {{d{qmin_0}[], d{qmin_1}[]}}, [{minmax_reg}]!\n'
    ret += f'vld1.32 {{d{qmax_0}[], d{qmax_1}[]}}, [{minmax_reg}]\n'
    ret += f'sub {minmax_reg}, {minmax_reg}, #4\n'
    return ret

  def dequantize(self, M, N, W):
    accumulators = self.acc_registers()
    ret = '\n# Convert from int32 to float.\n'
    for nr in range(0, N * M):
      ret += self.cvtf().format(ACC=accumulators[nr])
    ret += '# Multiply by input scale.\n'
    for nr in range(N):
      for mr in range(M):
        ret += 'vmul.f32 q{ACC}, q{ACC}, d{zp_scale}[1]\n'.format(
            ACC=accumulators[mr * 2 + nr],
            zp_scale=str(int(self.zp_scale(mr // 2)) * 2 + (mr & 0x1)),
        )
    ret += '# Load weights scale.\n'
    output_scale = 'vld1.32 {{d{W_SCALE_0}, d{W_SCALE_1}}}, [{W}]!\n'
    # output scales
    for nr in range(N):
      ret += output_scale.format(
          W=W,
          offset=self.register_bytes() * nr,
          W_SCALE_0=int(self.a_registers(nr)) * 2,
          W_SCALE_1=int(self.a_registers(nr)) * 2 + 1,
      )
    # biases
    ret += '# Load biases.\n'
    for nr in range(N):
      ret += output_scale.format(
          W=W,
          offset=self.register_bytes() * nr,
          W_SCALE=self.w_registers()[nr],
          W_SCALE_0=int(self.w_registers()[nr]) * 2,
          W_SCALE_1=int(self.w_registers()[nr]) * 2 + 1,
      )
    ret += "# Multiply by weight's scale.\n"
    for nr in range(N):
      for mr in range(M):
        ret += 'vmul.f32 q{ACC}, q{ACC}, q{SCALE}\n'.format(
            ACC=accumulators[mr * 2 + nr], SCALE=self.a_registers(nr)
        )
    ret += '# Load min/max into registers.\n'
    ret += self.load_min_max()
    ret += '# Add bias.\n'
    for nr in range(N):
      for mr in range(M):
        ret += 'vadd.f32 q{ACC}, q{ACC}, q{BIAS}\n'.format(
            ACC=accumulators[mr * 2 + nr], BIAS=self.w_registers()[nr]
        )

    return ret

  # The number of k elements to unroll the inner loop by.
  # 8 is the only value which makes sense for mlal.
  def k_unroll(self):
    return 8

  def outer_loop_prepare(self, M, N):
    return ''

  def inner_loop(self, M, N):
    return self.inner_loop_in_order(M, N)

  def init_accumulators(self, M, N):
    accumulators = self.acc_registers()
    W = self.w_ptr_register()
    ksum_x8 = 'vld1.32 {{q{lo}, q{hi}}}, [{W}]!\n'
    vksum = 'vmul.s32 q{ACC}, q{KSUM}, d{zp_scale}[0]\n'

    ret = ''
    for nr in range(0, N - 1, 2):
      ret += ksum_x8.format(
          W=W,
          lo=self.ksum_regs(nr),
          hi=self.ksum_regs(nr + 1),
      )
    ret += '# Initialize accumulators with k_sum * input zero point.\n'
    for nr in range(0, N):
      for mr in range(0, M):
        ret += vksum.format(
            ACC=accumulators[mr * 2 + nr],
            KSUM=self.ksum_regs(nr),
            zp_scale=str(int(self.zp_scale(mr // 2)) * 2 + (mr & 0x1)),
        )

    ret += """
    # jump to epilogue if lower than 8
    blo epilogue
    """
    ret += f'\n# Load {M} As and B0\n'
    for l in self.weights_asm()['loop']:
      ret += l.format(
          W_ptr=self.w_ptr_register(), W=str(int(self.w_registers()[0]) * 2)
      )
    for l in self.input_asm()['loop']:
      for mr in range(M):
        ret += l.format(
            AM_ptr=self.am_registers()[mr],
            AM=str(int(self.a_registers(mr)) * 2),
        )

    num_horizontal_registers = int(N / self.n_step())
    return ret

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
      cmp {nc}, #{n_step}
      blo tail_{N_2}\n""".format(n_step=N, N_2=N // 2, nc=nc_reg)
    for mr in range(M):
      for nr in range(N_COUNT):
        asm_string += """vst1.32  {{d{ACC_0}, d{ACC_1}}}, [{c_reg}]!\n""".format(
            ACC_0=int(accumulators[mr * 2 + nr]) * 2,
            ACC_1=int(accumulators[mr * 2 + nr]) * 2 + 1,
            c_reg=cm_registers[mr],
        )

    for mr in range(0, M):
      AM_PTR = self.am_registers()[mr]
      kc_register = self.kc_register()
      asm_string += f'sub {AM_PTR}, {AM_PTR}, {kc_register}\n'
    CHECK = """
      sub {nc}, {nc}, #{n_step}
      bne outer_loop
      b return\n""".format(n_step=N, nc=nc_reg)
    asm_string += CHECK
    N = N // 2
    if N * 2 > self.n_step():
      asm_string += """
\ntail_4:
      tst {nc_lo}, #4
      beq tail_2\n""".format(nc_lo=nc_lo)
      for mr in range(0, M):
        asm_string += 'vst1.32  {{q{ACC}}}, [{c_reg}]!\n'.format(
            ACC=accumulators[mr * 2], c_reg=cm_registers[mr]
        )
      for mr in range(0, M):
        asm_string += 'vmov  q{ACC0}, q{ACC1}\n'.format(
            ACC0=accumulators[mr * 2], ACC1=accumulators[mr * 2 + 1]
        )
    asm_string += """
\ntail_2:
      tst {nc_lo}, #2
      beq tail_1\n""".format(nc_lo=nc_lo)
    for mr in range(0, M):
      asm_string += 'vst1.32  d{ACC}, [{c_reg}]!\n'.format(
          ACC=int(accumulators[mr * 2]) * 2, c_reg=cm_registers[mr]
      )
    for mr in range(0, M):
      asm_string += 'vmov d{ACC}, d{ACC_1}\n'.format(
          ACC=int(accumulators[mr * 2]) * 2,
          ACC_1=int(accumulators[mr * 2]) * 2 + 1,
      )
    asm_string += """
\ntail_1:
      tst {nc_lo}, #1
      beq return\n""".format(nc_lo=nc_lo)
    for mr in range(0, M):
      asm_string += 'vst1.32  {{d{ACC}[0]}}, [{c_reg}]\n'.format(
          ACC=int(accumulators[mr * 2]) * 2, c_reg=cm_registers[mr]
      )

    return asm_string

  def min_register(self):
    return '0'

  def max_register(self):
    return '1'

  def clamp_min(self, reg, prefix):
    max_reg = self.max_register()
    return f'vmin.f32 q{reg}, q{reg}, q{max_reg}\n'

  def clamp_max(self, reg, prefix):
    min_reg = self.min_register()
    return f'vmax.f32 q{reg}, q{reg}, q{min_reg}\n'

class NeonMlalF16(NeonMlal):
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
      cmp {nc}, #{n_step}
      blo tail_{N_2}\n""".format(n_step=N, N_2=N // 2, nc=nc_reg)
    for mr in range(M):
      for nr in range(N_COUNT):
        asm_string += """vst1.16  d{ACC_0}, [{c_reg}]!\n""".format(
            ACC_0=int(accumulators[mr * 2 + nr]) * 2,
            c_reg=cm_registers[mr],
        )

    for mr in range(0, M):
      AM_PTR = self.am_registers()[mr]
      kc_register = self.kc_register()
      asm_string += f'sub {AM_PTR}, {AM_PTR}, {kc_register}\n'
    CHECK = """
      sub {nc}, {nc}, #{n_step}
      bne outer_loop
      b return\n""".format(n_step=N, nc=nc_reg)
    asm_string += CHECK
    N = N // 2
    if N * 2 > self.n_step():
      asm_string += """
\ntail_4:
      tst {nc_lo}, #4
      beq tail_2\n""".format(nc_lo=nc_lo)
      for mr in range(0, M):
        asm_string += 'vst1.16  {{d{ACC}}}, [{c_reg}]!\n'.format(
            ACC=int(accumulators[mr * 2]) * 2, c_reg=cm_registers[mr]
        )
      for mr in range(0, M):
        asm_string += 'vmov  d{ACC0}, d{ACC1}\n'.format(
            ACC0=int(accumulators[mr * 2]) * 2, ACC1=int(accumulators[mr * 2 + 1]) * 2
        )
    asm_string += """
\ntail_2:
      tst {nc_lo}, #2
      beq tail_1\n""".format(nc_lo=nc_lo)
    for mr in range(0, M):
      asm_string += 'vst1.32  {{d{ACC}[0]}}, [{c_reg}]!\n'.format(
          ACC=int(accumulators[mr * 2]) * 2, c_reg=cm_registers[mr]
      )
    for mr in range(0, M):
      asm_string += ' vext.8 d{ACC}, d{ACC}, d{ACC_1}, #4\n'.format(
          ACC=int(accumulators[mr * 2]) * 2,
          ACC_1=int(accumulators[mr * 2]) * 2 + 1,
      )
    asm_string += """
\ntail_1:
      tst {nc_lo}, #1
      beq return\n""".format(nc_lo=nc_lo)
    for mr in range(0, M):
      asm_string += 'vst1.16  {{d{ACC}[0]}}, [{c_reg}]\n'.format(
          ACC=int(accumulators[mr * 2]) * 2, c_reg=cm_registers[mr]
      )

    return asm_string

  def isa(self):
    return 'neonfp16arith'

  def function_name(self, M, N, isa):
    LD = self.unroll_factor * 32
    return f'xnn_qd8_f16_qc8w_gemm_minmax_ukernel_{M}x{N}__asm_aarch32_{isa}_ld{LD}_2\n'

  def clamp_min(self, reg, prefix):
    max_reg = int(self.max_register()) * 2
    reg_0 = reg
    reg_1 = int(reg) * 2
    asm_string = f'vcvt.f16.f32 d{reg_1}, q{reg_0}\n'
    asm_string += f'vmin.f16 d{reg_1}, d{reg_1}, d{max_reg}\n'
    return asm_string

  def clamp_max(self, reg, prefix):
    min_reg = int(self.min_register()) * 2
    reg = int(reg) * 2
    return f'vmax.f16 d{reg}, d{reg}, d{min_reg}\n'

  def load_min_max(self):
    minmax_reg = self.minmax_reg()
    min_reg = int(self.min_register()) * 2
    max_reg = int(self.max_register()) * 2
    ret =  f'vld1.32 {{d{max_reg}[0]}}, [{minmax_reg}]\n'
    ret += f'vdup.16 d{min_reg}, d{max_reg}[0]\n'
    ret += f'vdup.16 d{max_reg}, d{max_reg}[1]\n'
    return ret

