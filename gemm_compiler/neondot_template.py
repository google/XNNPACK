#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import neonfma_template


class NeonDot(neonfma_template.NeonFma):
  """All SIMD features for Aarch64 neondot."""

  def isa(self):
    return 'neondot'

  def a_registers(self, idx):
    registers = ['2', '3', '4', '5', '11']
    assert idx < len(registers)
    return registers[idx]

  def w_registers(self):
    return ['6', '7', '8', '9']

  def acc_registers(self):
    return [
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
        '31',
    ]

  def function_name(self):
    ld = self.unroll_factor * 32
    return (
        f'xnn_qd8_f32_qc8w_gemm_minmax_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c4__asm_aarch64_{self.isa()}_ld{ld}_2'
    )

  def zp_scale(self, pos):
    regs = ['30', '31']
    return regs[pos]

  # kc = round_up_po2(kc, channels)
  def adjust_kc(self):
    channels = 4
    m = pow(2, 64) - channels
    not_channels = f'0x{m:016X}'
    self.comment('Round kc up to channels.')
    self.asm_string += """add {kc_reg}, {kc_reg}, #{channels}
      and {kc_reg}, {kc_reg}, #{not_channels}\n\n""".format(
        kc_reg=self.kc_register(),
        channels=channels - 1,
        not_channels=not_channels,
    )

  def quantization_params(self):
    quantization_params_reg = self.quantization_params_register()
    self.asm_string += f'ldr {quantization_params_reg}, [sp, 272]\n'

  def quantization_params_register(self):
    return 'x24'

  def compute_asm(self):
    c_asm = {
        'loop': ['sdot  v{ACC}.4s, v{W}.16b, v{A}.4b[{POS}]\n'],
    }
    return c_asm

  def cvtf(self, acc):
    self.asm_string += f'scvtf v{acc}.4s, v{acc}.4s\n'

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

  def convert_to_output_type(self):
    accumulators = self.acc_registers()
    self.comment('Convert from int32 to float.')
    for nr in range(0, self.n * self.m):
      self.cvtf(accumulators[nr])
    self.comment('Multiply by input scale.')
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.fmul_lane(
            a=accumulators[mr * self.n + nr],
            b=accumulators[mr * self.n + nr],
            c=self.zp_scale(mr // 2),
            lane=int((mr % 2) * 2) + 1,
        )
    self.comment('Load weights scale.')
    # output scales
    for nr in range(0, self.n, 2):
      self.load_simd_register_pair(
          q0=self.a_registers(nr),
          q1=self.a_registers(nr + 1),
          ptr=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
      )

    self.increment_ptr(
        ptr=self.w_ptr_register(), step=self.register_bytes() * self.n
    )
    # biases
    self.comment('Load biases.')
    for nr in range(0, self.n, 2):
      self.load_simd_register_pair(
          q0=self.w_registers()[nr],
          q1=self.w_registers()[nr + 1],
          ptr=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
      )

    self.increment_ptr(
        ptr=self.w_ptr_register(), step=self.register_bytes() * self.n
    )
    # do mul + add here instead of fmla.
    # fmla accumulaltes into the additional term, in this case the bias. This
    # means that the bias must be copied before the fmla.
    # From the Cortex X1 optimization guide, fmov takes 1 cycle with a
    # throughput of 4 and fmla takes 4 cycles with a throughput of 4. This means
    # 5 cycles for four movs + fmla. fadd takes 2 cycles with a throughput of 4
    # and fmul takes 3 cycles with a throughput of 4, for a total of 5 cycles
    # for 4 results.
    self.comment("Multiply by weight's scale.")
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.fmul(
            a=accumulators[mr * self.n + nr],
            b=accumulators[mr * self.n + nr],
            c=self.a_registers(nr),
        )
    self.comment('Add bias.')
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.fadd(
            a=accumulators[mr * self.n + nr],
            b=accumulators[mr * self.n + nr],
            c=self.w_registers()[nr],
        )

  def init_accumulators(self):
    self.comment('Initialize accumulators with k_sum * input zero point.')
    accumulators = self.acc_registers()

    if self.m > 2:
      self.load_simd_register_pair(
          q0=self.zp_scale(0),
          q1=self.zp_scale(1),
          ptr=self.quantization_params_register(),
          offset=0,
      )
    else:
      self.load_simd_register(
          q=self.zp_scale(0), ptr=self.quantization_params_register(), offset=0
      )
    for nr in range(0, self.n - 1, 2):
      self.load_simd_register_pair(
          q0=self.a_registers(nr),
          q1=self.a_registers(nr + 1),
          ptr=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
      )
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.mul_lane(
            a=accumulators[mr * self.n + nr],
            b=self.a_registers(nr),
            c=self.zp_scale(mr // 2),
            lane=int((mr % 2) * 2),
        )

    self.increment_ptr(
        ptr=self.w_ptr_register(), step=self.register_bytes() * self.n
    )

  def mask_register(self):
    return '10'

  def tmp_w_register(self):
    w_registers = self.w_registers()
    w_registers_size = len(w_registers)
    return w_registers[w_registers_size - 1]


class NeonDotQC4W(NeonDot):
  """All SIMD features for Aarch64 neondot with 4-bit weights."""

  def weights_asm(self):
    w_asm = {
        'loop': ["""ldr q{tmp_W}, [{W_ptr}], 32
            shl v{W}.16b, v{tmp_W}.16b, #4
            and v{W_1}.16b, v{tmp_W}.16b, v{mask}.16b\n"""],
        'loop_2': ["""ldr q{tmp_W}, [{W_ptr}], 16
            shl v{W}.16b, v{tmp_W}.16b, #4
            and v{W_1}.16b, v{tmp_W}.16b, v{mask}.16b\n"""],
    }
    return w_asm

  def quantization_params(self):
    self.asm_string += """# Load 0xF0 for masking the weights
  ldr {quantization_params_reg}, [sp, 272]
  movi v{mask}.16b, #240
  """.format(
        quantization_params_reg=self.quantization_params_register(),
        mask=self.mask_register(),
    )

  def function_name(self):
    ld = self.unroll_factor * 32
    return (
        f'xnn_qd8_f32_qc4w_gemm_minmax_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c4__asm_aarch64_{self.isa()}_ld{ld}_2'
    )

  def cvtf(self, acc):
    self.asm_string += f'scvtf v{acc}.4s, v{acc}.4s, #4\n'


class NeonDotQS8QC8W(NeonDot):

  def function_name(self):
    ld = self.unroll_factor * 32
    return (
        f'xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c4__asm_aarch64_{self.isa()}_ld{ld}_2'
    )

  def load_min_max(self):
    params_register = self.params_register()
    min_reg = self.min_register()
    max_reg = self.max_register()
    self.asm_string += f"""
      # Load min/max values.
      add {params_register}, {params_register}, 2
      ld2r {{{min_reg}.16b, {max_reg}.16b}}, [{params_register}]
      sub {params_register}, {params_register}, 2\n"""

  def init_accumulators(self):
    return super(NeonDot, self).init_accumulators()

  def cvts(self, acc):
    self.asm_string += f'fcvtns v{acc}.4s, v{acc}.4s\n'

  def output_n(self) -> int:
    return self.n // 4

  def dup_s16(self, ptr, q):
    self.asm_string += f'ld1r {{v{q}.8h}}, [{ptr}]\n'

  def convert_to_output_type(self):
    accumulators = self.acc_registers()
    self.comment('Convert from int32 to float.')
    for nr in range(0, self.n * self.m):
      self.cvtf(accumulators[nr])

    self.comment('Load weights scale.')
    # output scales
    for nr in range(0, self.n, 2):
      self.load_simd_register_pair(
          q0=self.a_registers(nr),
          q1=self.a_registers(nr + 1),
          ptr=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
      )
    self.increment_ptr(
        ptr=self.w_ptr_register(), step=self.register_bytes() * self.n
    )
    self.comment("Multiply by weight's scale.")
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.fmul(
            a=accumulators[mr * self.n + nr],
            b=accumulators[mr * self.n + nr],
            c=self.a_registers(nr),
        )
    self.comment('Reconvert to int32.')
    for nr in range(0, self.n * self.m):
      self.cvts(accumulators[nr])

    self.comment('Convert to int16.')
    for nr in range(0, self.n, 2):
      for mr in range(0, self.m):
        self.sqxtn(
            a=accumulators[mr * self.n + nr],
            b=accumulators[mr * self.n + nr],
            atype='sint32',
        )
    for nr in range(1, self.n, 2):
      for mr in range(0, self.m):
        self.sqxtn2(
            a=accumulators[mr * self.n + nr - 1],
            b=accumulators[mr * self.n + nr],
            atype='sint32',
        )
    self.dup_s16(ptr=self.params_register(), q=self.tmp_w_register())
    self.comment('Add output zero point.')
    for nr in range(0, self.n, 2):
      for mr in range(0, self.m):
        self.sqadd(
            a=accumulators[mr * self.n + nr],
            b=accumulators[mr * self.n + nr],
            c=self.tmp_w_register(),
        )
    self.comment('Convert to int8.')
    for mr in range(0, self.m):
      self.sqxtn(
          a=accumulators[mr * self.n],
          b=accumulators[mr * self.n],
          atype='sint16',
      )
    for mr in range(0, self.m):
      self.sqxtn2(
          a=accumulators[mr * self.n],
          b=accumulators[mr * self.n + 2],
          atype='sint16',
      )

  def clamp_min(self, reg, prefix):
    max_reg = self.max_register()
    self.asm_string += (
        f'smin  {prefix}{reg}.16b, {max_reg}.16b, {prefix}{reg}.16b\n'
    )

  def clamp_max(self, reg, prefix):
    min_reg = self.min_register()
    self.asm_string += (
        f'smax  {prefix}{reg}.16b, {min_reg}.16b, {prefix}{reg}.16b\n'
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
      self.store_simd_register(
          r=accumulators[mr * self.n],
          prefix='q',
          ptr=cm_registers[mr],
          post_increment=16,
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
    self.asm_string += """
\n.Ltail_8:
      tbz {nc_lo}, 3, .Ltail_4\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.store_simd_register(
          r=accumulators[mr * self.n],
          prefix='d',
          ptr=cm_registers[mr],
          post_increment=8,
      )
    for mr in range(0, self.m):
      self.asm_string += 'ext v{ACC}.16b, v{ACC}.16b, v{ACC}.16b, 8\n'.format(
          ACC=accumulators[mr * self.n]
      )
    self.asm_string += """
\n.Ltail_4:
      tbz {nc_lo}, 2, .Ltail_2\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.store_simd_register_lane(
          r=accumulators[mr * self.n],
          prefix='s',
          ptr=cm_registers[mr],
          lane=0,
          post_increment=4,
      )
    for mr in range(0, self.m):
      self.asm_string += 'ext v{ACC}.16b, v{ACC}.16b, v{ACC}.16b, 4\n'.format(
          ACC=accumulators[mr * self.n]
      )

    self.asm_string += """
\n.Ltail_2:
      tbz {nc_lo}, 1, .Ltail_1\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.store_simd_register_lane(
          r=accumulators[mr * self.n],
          prefix='h',
          ptr=cm_registers[mr],
          lane=0,
          post_increment=2,
      )
    for mr in range(0, self.m):
      self.asm_string += 'ext v{ACC}.16b, v{ACC}.16b, v{ACC}.16b, 2\n'.format(
          ACC=accumulators[mr * self.n]
      )

    self.asm_string += """
\n.Ltail_1:
      tbz {nc_lo}, 0, .Lreturn\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      self.store_simd_register_lane(
          r=accumulators[mr * self.n],
          prefix='b',
          ptr=cm_registers[mr],
          lane=0,
          post_increment=0,
      )

class NeonDotQS8QC4W(NeonDotQC4W, NeonDotQS8QC8W):
  def function_name(self):
    ld = self.unroll_factor * 32
    return (
        f'xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c4__asm_aarch64_{self.isa()}_ld{ld}_2'
    )
