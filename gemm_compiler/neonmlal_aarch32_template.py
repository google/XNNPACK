#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import aarch32_template


class NeonMlal(aarch32_template.Aarch32):
  """All SIMD features for Aarch32 neonmlal."""

  def __init__(self, m: int, n: int, unroll_factor: int):
    super().__init__(m, n)
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

  def function_name(self):
    ld = self.unroll_factor * 32
    return (
        f'xnn_qd8_f32_qc8w_gemm_minmax_ukernel_{self.m}x{self.n * self.n_step()}'
        f'__asm_aarch32_{self.isa()}_ld{ld}_2\n'
    )

  def zp_scale(self, pos):
    regs = ['4', '5']
    return regs[pos]

  # kc = round_up_po2(kc, channels)
  def adjust_kc(self):
    return

  def minmax_reg(self):
    return 'r11'

  def quantization_params(self):
    self.asm_string += """\n# Load quantization params
  ldr {quantization_params_reg}, [sp, #124]\n""".format(
        quantization_params_reg=self.quantization_params_register()
    )
    zp_scale = 'vld1.32 {{q{lo}, q{hi}}}, [{quantization_params_reg}]\n'
    self.asm_string += '# Load minmax pointer.\n'
    minmax_reg = self.minmax_reg()
    self.asm_string += f'ldr {minmax_reg}, [sp, #120]\n'
    self.asm_string += '# Load dynamic quantization params.\n'
    for mr in range(0, self.m, 4):
      self.asm_string += zp_scale.format(
          quantization_params_reg=self.quantization_params_register(),
          lo=str(int(self.zp_scale(mr // 2))),
          hi=str(int(self.zp_scale(mr // 2)) + 1),
      )

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
    self.asm_string += f'vld1.32 {{d{qmin_0}[], d{qmin_1}[]}}, [{minmax_reg}]!\n'
    self.asm_string += f'vld1.32 {{d{qmax_0}[], d{qmax_1}[]}}, [{minmax_reg}]\n'
    self.asm_string += f'sub {minmax_reg}, {minmax_reg}, #4\n'

  def convert_to_output_type(self):
    accumulators = self.acc_registers()
    self.asm_string += '\n# Convert from int32 to float.\n'
    for nr in range(0, self.n * self.m):
      self.asm_string += self.cvtf().format(ACC=accumulators[nr])
    self.asm_string += '# Multiply by input scale.\n'
    for nr in range(self.n):
      for mr in range(self.m):
        self.asm_string += 'vmul.f32 q{ACC}, q{ACC}, d{zp_scale}[1]\n'.format(
            ACC=accumulators[mr * 2 + nr],
            zp_scale=str(int(self.zp_scale(mr // 2)) * 2 + (mr & 0x1)),
        )
    self.asm_string += '# Load weights scale.\n'
    output_scale = 'vld1.32 {{d{W_SCALE_0}, d{W_SCALE_1}}}, [{W}]!\n'
    # output scales
    for nr in range(self.n):
      self.asm_string += output_scale.format(
          W=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
          W_SCALE_0=int(self.a_registers(nr)) * 2,
          W_SCALE_1=int(self.a_registers(nr)) * 2 + 1,
      )
    # biases
    self.asm_string += '# Load biases.\n'
    for nr in range(self.n):
      self.asm_string += output_scale.format(
          W=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
          W_SCALE=self.w_registers()[nr],
          W_SCALE_0=int(self.w_registers()[nr]) * 2,
          W_SCALE_1=int(self.w_registers()[nr]) * 2 + 1,
      )
    self.asm_string += "# Multiply by weight's scale.\n"
    for nr in range(self.n):
      for mr in range(self.m):
        self.asm_string += 'vmul.f32 q{ACC}, q{ACC}, q{SCALE}\n'.format(
            ACC=accumulators[mr * 2 + nr], SCALE=self.a_registers(nr)
        )
    self.asm_string += '# Load min/max into registers.\n'
    self.load_min_max()
    self.asm_string += '# Add bias.\n'
    for nr in range(self.n):
      for mr in range(self.m):
        self.asm_string += 'vadd.f32 q{ACC}, q{ACC}, q{BIAS}\n'.format(
            ACC=accumulators[mr * 2 + nr], BIAS=self.w_registers()[nr]
        )

  # The number of k elements to unroll the inner loop by.
  # 8 is the only value which makes sense for mlal.
  def k_unroll(self):
    return 8

  def inner_loop(self):
    return self.inner_loop_in_order()

  def init_accumulators(self):
    accumulators = self.acc_registers()
    w_ptr = self.w_ptr_register()
    ksum_x8 = 'vld1.32 {{q{lo}, q{hi}}}, [{W}]!\n'
    vksum = 'vmul.s32 q{ACC}, q{KSUM}, d{zp_scale}[0]\n'

    for nr in range(0, self.n - 1, 2):
      self.asm_string += ksum_x8.format(
          W=w_ptr,
          lo=self.ksum_regs(nr),
          hi=self.ksum_regs(nr + 1),
      )
    self.asm_string += '# Initialize accumulators with k_sum * input zero point.\n'
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += vksum.format(
            ACC=accumulators[mr * 2 + nr],
            KSUM=self.ksum_regs(nr),
            zp_scale=str(int(self.zp_scale(mr // 2)) * 2 + (mr & 0x1)),
        )

    self.asm_string += """
    # jump to epilogue if lower than 8
    blo .Lepilogue
    """
    self.asm_string += f'\n# Load {self.m} As and B0\n'
    for l in self.weights_asm()['loop']:
      self.asm_string += l.format(
          W_ptr=self.w_ptr_register(), W=str(int(self.w_registers()[0]) * 2)
      )
    for l in self.input_asm()['loop']:
      for mr in range(self.m):
        self.asm_string += l.format(
            AM_ptr=self.am_registers()[mr],
            AM=str(int(self.a_registers(mr)) * 2),
        )

  def store(self):
    accumulators = self.acc_registers()
    cm_registers = self.cm_registers()
    nc_reg = self.nc_register()
    nc_lo = self.register_map_byte(nc_reg)
    nc = self.n * self.n_step()
    self.asm_string += """
      # Check whether full or partial store.
      cmp {nc}, #{n_step}
      blo .Ltail_{N_2}\n""".format(n_step=nc, N_2=nc // 2, nc=nc_reg)
    for mr in range(self.m):
      for nr in range(self.n):
        self.asm_string += (
            """vst1.32  {{d{ACC_0}, d{ACC_1}}}, [{c_reg}]!\n""".format(
                ACC_0=int(accumulators[mr * 2 + nr]) * 2,
                ACC_1=int(accumulators[mr * 2 + nr]) * 2 + 1,
                c_reg=cm_registers[mr],
            )
        )

    for mr in range(0, self.m):
      am_ptr = self.am_registers()[mr]
      kc_register = self.kc_register()
      self.asm_string += f'sub {am_ptr}, {am_ptr}, {kc_register}\n'
    check = """
      sub {nc}, {nc}, #{n_step}
      bne .Louter_loop
      b .Lreturn\n""".format(n_step=nc, nc=nc_reg)
    self.asm_string += check
    nc = nc // 2
    if nc * 2 > self.n_step():
      self.asm_string += """
\n.Ltail_4:
      tst {nc_lo}, #4
      beq .Ltail_2\n""".format(nc_lo=nc_lo)
      for mr in range(0, self.m):
        self.asm_string += 'vst1.32  {{q{ACC}}}, [{c_reg}]!\n'.format(
            ACC=accumulators[mr * 2], c_reg=cm_registers[mr]
        )
      for mr in range(0, self.m):
        self.asm_string += 'vmov  q{ACC0}, q{ACC1}\n'.format(
            ACC0=accumulators[mr * 2], ACC1=accumulators[mr * 2 + 1]
        )
    self.asm_string += """
\n.Ltail_2:
      tst {nc_lo}, #2
      beq .Ltail_1\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.asm_string += 'vst1.32  d{ACC}, [{c_reg}]!\n'.format(
          ACC=int(accumulators[mr * 2]) * 2, c_reg=cm_registers[mr]
      )
    for mr in range(0, self.m):
      self.asm_string += 'vmov d{ACC}, d{ACC_1}\n'.format(
          ACC=int(accumulators[mr * 2]) * 2,
          ACC_1=int(accumulators[mr * 2]) * 2 + 1,
      )
    self.asm_string += """
\n.Ltail_1:
      tst {nc_lo}, #1
      beq .Lreturn\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.asm_string += 'vst1.32  {{d{ACC}[0]}}, [{c_reg}]\n'.format(
          ACC=int(accumulators[mr * 2]) * 2, c_reg=cm_registers[mr]
      )

  def min_register(self):
    return '0'

  def max_register(self):
    return '1'

  def clamp_min(self, reg, prefix):
    max_reg = self.max_register()
    self.asm_string += f'vmin.f32 q{reg}, q{reg}, q{max_reg}\n'

  def clamp_max(self, reg, prefix):
    min_reg = self.min_register()
    self.asm_string += f'vmax.f32 q{reg}, q{reg}, q{min_reg}\n'


class NeonMlalF16(NeonMlal):
  """All SIMD features for Aarch32 neonmlal for `fp16`."""

  def store(self):
    accumulators = self.acc_registers()
    cm_registers = self.cm_registers()
    nc_reg = self.nc_register()
    nc_lo = self.register_map_byte(nc_reg)
    nc = self.n * self.n_step()
    self.asm_string += """
      # Check whether full or partial store.
      cmp {nc}, #{n_step}
      blo .Ltail_{N_2}\n""".format(n_step=nc, N_2=nc // 2, nc=nc_reg)
    for mr in range(self.m):
      for nr in range(self.n):
        self.asm_string += """vst1.16  d{ACC_0}, [{c_reg}]!\n""".format(
            ACC_0=int(accumulators[mr * 2 + nr]) * 2,
            c_reg=cm_registers[mr],
        )

    for mr in range(0, self.m):
      am_ptr = self.am_registers()[mr]
      kc_register = self.kc_register()
      self.asm_string += f'sub {am_ptr}, {am_ptr}, {kc_register}\n'
    check = """
      sub {nc}, {nc}, #{n_step}
      bne .Louter_loop
      b .Lreturn\n""".format(n_step=nc, nc=nc_reg)
    self.asm_string += check
    nc = nc // 2
    if nc * 2 > self.n_step():
      self.asm_string += """
\n.Ltail_4:
      tst {nc_lo}, #4
      beq .Ltail_2\n""".format(nc_lo=nc_lo)
      for mr in range(0, self.m):
        self.asm_string += 'vst1.16  {{d{ACC}}}, [{c_reg}]!\n'.format(
            ACC=int(accumulators[mr * 2]) * 2, c_reg=cm_registers[mr]
        )
      for mr in range(0, self.m):
        self.asm_string += 'vmov  d{ACC0}, d{ACC1}\n'.format(
            ACC0=int(accumulators[mr * 2]) * 2,
            ACC1=int(accumulators[mr * 2 + 1]) * 2,
        )
    self.asm_string += """
\n.Ltail_2:
      tst {nc_lo}, #2
      beq .Ltail_1\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.asm_string += 'vst1.32  {{d{ACC}[0]}}, [{c_reg}]!\n'.format(
          ACC=int(accumulators[mr * 2]) * 2, c_reg=cm_registers[mr]
      )
    for mr in range(0, self.m):
      self.asm_string += ' vext.8 d{ACC}, d{ACC}, d{ACC_1}, #4\n'.format(
          ACC=int(accumulators[mr * 2]) * 2,
          ACC_1=int(accumulators[mr * 2]) * 2 + 1,
      )
    self.asm_string += """
\n.Ltail_1:
      tst {nc_lo}, #1
      beq .Lreturn\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.asm_string += 'vst1.16  {{d{ACC}[0]}}, [{c_reg}]\n'.format(
          ACC=int(accumulators[mr * 2]) * 2, c_reg=cm_registers[mr]
      )

  def isa(self):
    return 'neonfp16arith'

  def function_name(self):
    ld = self.unroll_factor * 32
    return (
        f'xnn_qd8_f16_qc8w_gemm_minmax_ukernel_{self.m}x{self.n * self.n_step()}'
        f'__asm_aarch32_{self.isa()}_ld{ld}_2\n'
    )

  def clamp_min(self, reg, prefix):
    max_reg = int(self.max_register()) * 2
    reg_0 = reg
    reg_1 = int(reg) * 2
    self.asm_string += f'vcvt.f16.f32 d{reg_1}, q{reg_0}\n'
    self.asm_string += f'vmin.f16 d{reg_1}, d{reg_1}, d{max_reg}\n'

  def clamp_max(self, reg, prefix):
    min_reg = int(self.min_register()) * 2
    reg = int(reg) * 2
    self.asm_string += f'vmax.f16 d{reg}, d{reg}, d{min_reg}\n'

  def load_min_max(self):
    minmax_reg = self.minmax_reg()
    min_reg = int(self.min_register()) * 2
    max_reg = int(self.max_register()) * 2
    self.asm_string += f'vld1.32 {{d{max_reg}[0]}}, [{minmax_reg}]\n'
    self.asm_string += f'vdup.16 d{min_reg}, d{max_reg}[0]\n'
    self.asm_string += f'vdup.16 d{max_reg}, d{max_reg}[1]\n'
