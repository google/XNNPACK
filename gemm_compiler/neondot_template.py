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
    regs = ['10', '11']
    return regs[pos]

  # kc = round_up_po2(kc, channels)
  def adjust_kc(self):
    channels = 4
    m = pow(2, 64) - channels
    not_channels = f'0x{m:016X}'
    ret = '# Round kc up to channels.\n'
    ret += """add {kc_reg}, {kc_reg}, #{channels}
      and {kc_reg}, {kc_reg}, #{not_channels}\n\n""".format(
        kc_reg=self.kc_register(),
        channels=channels - 1,
        not_channels=not_channels,
    )
    return ret

  def quantization_params(self):
    return """ldr {quantization_params_reg}, [sp, 272]\n""".format(
        quantization_params_reg=self.quantization_params_register()
    )

  def quantization_params_register(self):
    return 'x24'

  def compute_asm(self):
    c_asm = {
        'loop': ['sdot  v{ACC}.4s, v{W}.16b, v{A}.4b[{POS}]\n'],
    }
    return c_asm

  def cvtf(self):
    return 'scvtf v{ACC}.4s, v{ACC}.4s\n'

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
    ret = '\n# Convert from int32 to float.\n'
    for nr in range(0, self.n * self.m):
      ret += self.cvtf().format(ACC=accumulators[nr])
    ret += '# Multiply by input scale.\n'
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        ret += 'fmul v{ACC}.4s, v{ACC}.4s, v{zp_scale}.s[{pos}]\n'.format(
            ACC=accumulators[nr * self.m + mr],
            zp_scale=self.zp_scale(mr // 2),
            pos=int((mr % 2) * 2) + 1,
        )
    ret += '# Load weights scale.\n'
    output_scale_pair = 'ldp q{W_SCALE_0}, q{W_SCALE_1}, [{W}, {offset}]\n'
    # output scales
    for nr in range(0, self.n, 2):
      ret += output_scale_pair.format(
          W=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
          W_SCALE_0=self.a_registers(nr),
          W_SCALE_1=self.a_registers(nr + 1),
      )
    ret += self.increment_ptr(
        ptr=self.w_ptr_register(), step=self.register_bytes() * self.n
    )
    # biases
    ret += '# Load biases.\n'
    for nr in range(0, self.n, 2):
      ret += output_scale_pair.format(
          W=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
          W_SCALE_0=self.w_registers()[nr],
          W_SCALE_1=self.w_registers()[nr + 1],
      )
    ret += 'add {W}, {W}, {increment}\n'.format(
        W=self.w_ptr_register(), increment=self.register_bytes() * self.n
    )
    # do mul + add here instead of fmla.
    # fmla accumulaltes into the additional term, in this case the bias. This
    # means that the bias must be copied before the fmla.
    # From the Cortex X1 optimization guide, fmov takes 1 cycle with a
    # throughput of 4 and fmla takes 4 cycles with a throughput of 4. This means
    # 5 cycles for four movs + fmla. fadd takes 2 cycles with a throughput of 4
    # and fmul takes 3 cycles with a throughput of 4, for a total of 5 cycles
    # for 4 results.
    ret += "# Multiply by weight's scale.\n"
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        ret += 'fmul v{ACC}.4s, v{ACC}.4s, v{SCALE}.4s\n'.format(
            ACC=accumulators[nr * self.m + mr], SCALE=self.a_registers(nr)
        )
    ret += '# Add bias.\n'
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        ret += 'fadd v{ACC}.4s, v{ACC}.4s, v{BIAS}.4s\n'.format(
            ACC=accumulators[nr * self.m + mr], BIAS=self.w_registers()[nr]
        )

    return ret

  def init_accumulators(self):
    ret = '# Initialize accumulators with k_sum * input zero point.\n'
    accumulators = self.acc_registers()
    zp_scale_x2 = 'ldr q{zp_scale}, [{quantization_params_reg}]\n'
    zp_scale_x4 = (
        'ldp q{zp_scale_0}, q{zp_scale_1}, [{quantization_params_reg}]\n'
    )
    ksum_x8 = 'ldp  q{KSUM_0}, q{KSUM_1}, [{W}, {offset}]\n'
    vksum = 'mul v{ACC}.4s, v{KSUM}.4s, v{zp_scale}.s[{pos}]\n'

    mr = 0
    for mr in range(0, self.m - 1, 4):
      ret += zp_scale_x4.format(
          quantization_params_reg=self.quantization_params_register(),
          zp_scale_0=self.zp_scale(mr),
          zp_scale_1=self.zp_scale(mr + 1),
      )
    if self.m % 2 == 1:
      ret += zp_scale_x2.format(
          quantization_params_reg=self.quantization_params_register(),
          zp_scale=self.zp_scale(mr),
      )
    for nr in range(0, self.n - 1, 2):
      ret += ksum_x8.format(
          W=self.w_ptr_register(),
          KSUM_0=self.a_registers(nr),
          KSUM_1=self.a_registers(nr + 1),
          offset=self.register_bytes() * nr,
      )
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        ret += vksum.format(
            ACC=accumulators[nr * self.m + mr],
            KSUM=self.a_registers(nr),
            zp_scale=self.zp_scale(mr // 2),
            pos=int((mr % 2) * 2),
        )

    ret += self.increment_ptr(
        ptr=self.w_ptr_register(), step=self.register_bytes() * self.n
    )
    return ret


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

  def mask_register(self):
    return '28'

  def tmp_w_register(self):
    return '29'

  def quantization_params(self):
    return """# Load 0xF0 for masking the weights
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

  def cvtf(self):
    return 'scvtf v{ACC}.4s, v{ACC}.4s, #4\n'

class NeonDotQS8QC8W(NeonDot):

  def function_name(self):
    ld = self.unroll_factor * 32
    return (
        f'xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c4__asm_aarch64_{self.isa()}_ld{ld}_2'
    )

  def load_min_max(self):
      return '''
      # Load min/max values.
      ld1r {v10.8h}, [x13]
      add x13, x13, 2
      ld2r {v0.16b, v1.16b}, [x13]\n'''

  def init_accumulators(self):
    return super(NeonDot, self).init_accumulators()

  def cvts(self):
    return 'fcvtns v{ACC}.4s, v{ACC}.4s\n'

  def output_n(self) -> int:
    return self.n // 4

  def convert_to_output_type(self):
    accumulators = self.acc_registers()
    ret = '\n# Convert from int32 to float.\n'
    for nr in range(0, self.n * self.m):
      ret += self.cvtf().format(ACC=accumulators[nr])

    ret += '# Load weights scale.\n'
    output_scale_pair = 'ldp q{W_SCALE_0}, q{W_SCALE_1}, [{W}, {offset}]\n'
    # output scales
    for nr in range(0, self.n, 2):
      ret += output_scale_pair.format(
          W=self.w_ptr_register(),
          offset=self.register_bytes() * nr,
          W_SCALE_0=self.a_registers(nr),
          W_SCALE_1=self.a_registers(nr + 1),
      )
    ret += self.increment_ptr(
        ptr=self.w_ptr_register(), step=self.register_bytes() * self.n
    )
    ret += "# Multiply by weight's scale.\n"
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        ret += 'fmul v{ACC}.4s, v{ACC}.4s, v{SCALE}.4s\n'.format(
            ACC=accumulators[nr * self.m + mr], SCALE=self.a_registers(nr)
        )
    ret += "# Reconvert to int32.\n"
    for nr in range(0, self.n * self.m):
      ret += self.cvts().format(ACC=accumulators[nr])

    ret += '# Convert to int16.\n'
    for nr in range(0, self.n, 2):
      for mr in range(0, self.m):
        ret += 'sqxtn v{ACC}.4h, v{ACC}.4s\n'.format(
            ACC=accumulators[nr * self.m + mr])
    for nr in range(1, self.n, 2):
      for mr in range(0, self.m):
        ret += 'sqxtn2 v{ACC_UPPER}.8h, v{ACC}.4s\n'.format(
            ACC_UPPER=accumulators[(nr-1) * self.m + mr],
            ACC=accumulators[nr * self.m + mr])
    ret += '# Add output zero point.\n'
    for nr in range(0, self.n, 2):
      for mr in range(0, self.m):
        ret += 'sqadd v{ACC}.8h, v{ACC}.8h, v10.8h\n'.format(
            ACC=accumulators[nr * self.m + mr])
    ret += '# Convert to int8.\n'
    for nr in range(0, self.n // 2, 2):
      for mr in range(0, self.m):
        ret += 'sqxtn v{ACC}.8b, v{ACC}.8h\n'.format(
            ACC=accumulators[nr * self.m + mr])
    for nr in range(1, self.n // 2, 2):
      for mr in range(0, self.m):
        ret += 'sqxtn2 v{ACC_UPPER}.16b, v{ACC}.8h\n'.format(
            ACC_UPPER=accumulators[(nr-1) * self.m + mr],
            ACC=accumulators[nr * 2 * self.m + mr])

    return ret

  def clamp_min(self, reg, prefix):
    max_reg = self.max_register()
    return f'smin  {prefix}{reg}.16b, {max_reg}.16b, {prefix}{reg}.16b\n'

  def clamp_max(self, reg, prefix):
    min_reg = self.min_register()
    return f'smax  {prefix}{reg}.16b, {min_reg}.16b, {prefix}{reg}.16b\n'

  def store(self):
    accumulators = self.acc_registers()
    cm_registers = self.cm_registers()
    nc_reg = self.nc_register()
    nc_lo = self.register_map_byte(nc_reg)
    nc = self.n * self.n_step()
    asm_string = """
      # Check whether full or partial store.
      cmp {nc}, {n_step}
      b.lo .Ltail_{N_2}\n""".format(n_step=nc, N_2=nc // 2, nc=nc_reg)
    for mr in range(0, self.m):
      asm_string += 'str q{ACC}, [{c_reg}], 16\n'.format(
          ACC=accumulators[mr],
          ACC_1=accumulators[self.m + mr],
          c_reg=cm_registers[mr],
      )

    for mr in range(0, self.m):
      am_ptr = self.am_registers()[mr]
      kc_register = self.kc_register()
      asm_string += f'sub {am_ptr}, {am_ptr}, {kc_register}\n'
    check = """
      sub {nc}, {nc}, {n_step}
      b.ne .Louter_loop
      b .Lreturn""".format(n_step=nc, nc=nc_reg)
    asm_string += check
    nc = nc // 2
    asm_string += """
\n.Ltail_8:
      tbz {nc_lo}, 3, .Ltail_4\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      asm_string += 'str d{ACC}, [{c_reg}], 8\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr]
      )
    for mr in range(0, self.m):
      asm_string += 'ext v{ACC}.16b, v{ACC}.16b, v{ACC}.16b, 8\n'.format(ACC=accumulators[mr])
    asm_string += """
\n.Ltail_4:
      tbz {nc_lo}, 2, .Ltail_2\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      asm_string += 'st1 {{v{ACC}.s}}[0], [{c_reg}], 4\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr]
      )
    for mr in range(0, self.m):
      asm_string += 'ext v{ACC}.16b, v{ACC}.16b, v{ACC}.16b, 4\n'.format(ACC=accumulators[mr])

    asm_string += """
\n.Ltail_2:
      tbz {nc_lo}, 1, .Ltail_1\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      asm_string += 'st1 {{v{ACC}.h}}[0], [{c_reg}], 2\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr]
      )
    for mr in range(0, self.m):
      asm_string += 'ext v{ACC}.16b, v{ACC}.16b, v{ACC}.16b, 2\n'.format(ACC=accumulators[mr])

    asm_string += """
\n.Ltail_1:
      tbz {nc_lo}, 0, .Lreturn\n""".format(nc_lo=nc_lo)
    for mr in range(0, self.m):
      asm_string += 'st1 {{v{ACC}.b}}[0], [{c_reg}]\n'.format(
          ACC=accumulators[mr], c_reg=cm_registers[mr]
      )

    return asm_string
