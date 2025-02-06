#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import neonfma_template as isa

"""All SIMD features for Aarch64 neondot."""


class NeonDot(isa.NeonFma):

  def __init__(self, unroll_factor):
    self.unroll_factor = unroll_factor
    self.decrement = 4 * unroll_factor

  def isa(self):
    return 'neondot'

  def a_registers(self, idx):
    registers = ['2', '3', '4', '5']
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
    ]

  def function_name(self, M, N, isa):
    LD = self.unroll_factor * 32
    return f'xnn_qd8_f32_qc8w_gemm_minmax_ukernel_{M}x{N}c4__asm_aarch64_{isa}_ld{LD}_2'

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

  def quantization_params(self, M):
    return """ldr {quantization_params_reg}, [sp, 16]\n""".format(
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

  def dequantize(self, M, N, W):
    accumulators = self.acc_registers()
    ret = '\n# Convert from int32 to float.\n'
    for nr in range(0, N * M):
      ret += self.cvtf().format(ACC=accumulators[nr])
    ret += '# Multiply by input scale.\n'
    for nr in range(0, N):
      for mr in range(0, M):
        ret += 'fmul v{ACC}.4s, v{ACC}.4s, v{zp_scale}.s[{pos}]\n'.format(
            ACC=accumulators[nr * M + mr],
            zp_scale=self.zp_scale(mr // 2),
            pos=int((mr % 2) * 2) + 1,
        )
    ret += '# Load weights scale.\n'
    output_scale_pair = 'ldp q{W_SCALE_0}, q{W_SCALE_1}, [{W}, {offset}]\n'
    # output scales
    for nr in range(0, N, 2):
      ret += output_scale_pair.format(
          W=W,
          offset=self.register_bytes() * nr,
          W_SCALE_0=self.a_registers(nr),
          W_SCALE_1=self.a_registers(nr + 1),
      )
    ret += self.increment_ptr(ptr=W, step=self.register_bytes() * N)
    # biases
    ret += '# Load biases.\n'
    for nr in range(0, N, 2):
      ret += output_scale_pair.format(
          W=W,
          offset=self.register_bytes() * nr,
          W_SCALE_0=self.w_registers()[nr],
          W_SCALE_1=self.w_registers()[nr + 1],
      )
    ret += 'add {W}, {W}, {increment}\n'.format(
        W=W, increment=self.register_bytes() * N
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
    for nr in range(0, N):
      for mr in range(0, M):
        ret += 'fmul v{ACC}.4s, v{ACC}.4s, v{SCALE}.4s\n'.format(
            ACC=accumulators[nr * M + mr], SCALE=self.a_registers(nr)
        )
    ret += '# Add bias.\n'
    for nr in range(0, N):
      for mr in range(0, M):
        ret += 'fadd v{ACC}.4s, v{ACC}.4s, v{BIAS}.4s\n'.format(
            ACC=accumulators[nr * M + mr], BIAS=self.w_registers()[nr]
        )

    return ret

  def init_accumulators(self, M, N):
    ret = '# Initialize accumulators with k_sum * input zero point.\n'
    accumulators = self.acc_registers()
    W = self.w_ptr_register()
    zp_scale_x2 = 'ldr q{zp_scale}, [{quantization_params_reg}]\n'
    zp_scale_x4 = (
        'ldp q{zp_scale_0}, q{zp_scale_1}, [{quantization_params_reg}]\n'
    )
    ksum_x8 = 'ldp  q{KSUM_0}, q{KSUM_1}, [{W}, {offset}]\n'
    vksum = 'mul v{ACC}.4s, v{KSUM}.4s, v{zp_scale}.s[{pos}]\n'

    mr = 0
    for mr in range(0, M - 1, 4):
      ret += zp_scale_x4.format(
          quantization_params_reg=self.quantization_params_register(),
          zp_scale_0=self.zp_scale(mr),
          zp_scale_1=self.zp_scale(mr + 1),
      )
    if M % 2 == 1:
      ret += zp_scale_x2.format(
          quantization_params_reg=self.quantization_params_register(),
          zp_scale=self.zp_scale(mr),
      )
    for nr in range(0, N - 1, 2):
      ret += ksum_x8.format(
          W=W,
          KSUM_0=self.a_registers(nr),
          KSUM_1=self.a_registers(nr + 1),
          offset=self.register_bytes() * nr,
      )
    for nr in range(0, N):
      for mr in range(0, M):
        ret += vksum.format(
            ACC=accumulators[nr * M + mr],
            KSUM=self.a_registers(nr),
            zp_scale=self.zp_scale(mr // 2),
            pos=int((mr % 2) * 2),
        )

    num_horizontal_registers = int(N / self.n_step())
    ret += self.increment_ptr(ptr=W, step=self.register_bytes() * N)
    return ret


class NeonDotQC4W(NeonDot):

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

  def quantization_params(self, M):
    return """# Load 0xF0 for masking the weights
  ldr {quantization_params_reg}, [sp, 16]
  movi v{mask}.16b, #240
  """.format(
        quantization_params_reg=self.quantization_params_register(),
        mask=self.mask_register(),
    )

  def function_name(self, M, N, isa):
    LD = self.unroll_factor * 32
    return f'xnn_qd8_f32_qc4w_gemm_minmax_ukernel_{M}x{N}c4__asm_aarch64_{isa}_ld{LD}_2'

  def cvtf(self):
    return 'scvtf v{ACC}.4s, v{ACC}.4s, #4\n'
