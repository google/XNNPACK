#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from gemm_compiler import avx512f_template


class Avx512Vnni(avx512f_template.Avx512F):
  """All SIMD features for avx512vnni."""

  def __init__(self, m: int, n: int, c: int):
    super().__init__(m, n)
    self._c = c

  def isa(self):
    return 'avx512vnni'

  def element_size(self):
    return 1

  def a_registers(self, idx):
    return 'zmm2'

  def scale_registers(self):
    return ['zmm10', 'zmm11', 'zmm2', 'zmm3']

  def w_registers(self):
    return ['zmm6', 'zmm7', 'zmm8', 'zmm9']

  def acc_registers(self):
    return [
        'mm5',
        'mm12',
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
        'mm4',
        'mm8',
        'mm9',
    ]

  def function_name(self):
    c = self._c
    return (
        f'xnn_qd8_f32_qc8w_gemm_minmax_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c{c}__asm_amd64_{self.isa()}'
    )

  def zp_scale(self, pos):
    regs = ['10', '11']
    return regs[pos]

  def channels(self):
    return self._c

  # kc = round_up_po2(kc, channels)
  def adjust_kc(self):
    channels = self.channels()
    ret = """
      add {kc_reg}, {channels}
      and {kc_reg}, {neg_channels}\n""".format(
        kc_reg=self.kc_register(), channels=channels - 1, neg_channels=-channels
    )
    return ret

  def quantization_params_register(self):
    return self.k_register()

  def pre_header(self):
    header_c4 = ''
    header_c8 = """
        .PERMUTATION:
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
    match self._c:
      case 4:
        return header_c4
      case 8:
        return header_c8
      case _:
        raise NotImplementedError

  def input_asm(self):
    loop_c4 = 'vpbroadcastd {AM}, [{AM_ptr} + {a_offset}]\n'
    loop_c8 = 'vbroadcasti32x2 {AM}, QWORD PTR [{AM_ptr} + {a_offset}]\n'
    match self._c:
      case 4:
        loop = loop_c4
      case 8:
        loop = loop_c8
      case _:
        raise NotImplementedError
    in_asm = {
        'loop': [loop],
        'compute': ['vpdpbusd  z{ACC}, {A}, {W}\n'],
    }
    return in_asm

  def weights_asm(self):
    w_asm = {
        'loop': [
            'vmovaps  {W}, [{W_ptr} + {offset}]\n',
        ],
        'after': [
            'add {W}, {w_step}\n',
        ],
    }
    return w_asm

  def compute_asm(self):
    c_asm = {
        'loop': ['vpdpbusd  z{ACC}, {A}, {W}\n'],
    }
    return c_asm

  # Quantization parameters are pushed to the stack at this offset.
  def quantization_params_offset(self):
    return 8

  def accumulator_reg_offset(self, M, nr):
    match self._c:
      case 4:
        return 0
      case 8:
        return nr * M
      case _:
        raise NotImplementedError

  def convert_to_float(self):
    accumulators = self.acc_registers()
    ret = ''
    ret += '\n# Convert from int32 to float.\n'
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        reg_ofset = self.accumulator_reg_offset(self.m, nr)
        other_reg = accumulators[self.m * nr + mr + reg_ofset]
        ACC = accumulators[self.m * nr + mr]
        ret += f'vcvtdq2ps z{ACC}, z{other_reg}\n'
    return ret

  def dequantize(self):
    W = self.w_ptr_register()
    asm_string = ''
    if self._c == 8:
      shift_add = """vpsrlq {tmp}, z{acc}, 32
        vpaddd z{acc}, z{acc}, {tmp}\n"""
      asm_string = ''
      accumulators = self.acc_registers()
      for nr in range(0, self.n * 2):
        for mr in range(0, self.m):
          asm_string += shift_add.format(
              acc=accumulators[self.m * nr + mr], tmp=self.w_registers()[0]
          )
      perm_reg = self.w_registers()[0]
      asm_string += f'vmovups {perm_reg}, zmmword ptr [rip + .PERMUTATION]\n'
      perm = 'vpermt2ps z{acc0}, {perm_reg}, z{acc1}\n'
      for nr in range(0, self.n):
        for mr in range(0, self.m):
          asm_string += perm.format(
              perm_reg=perm_reg,
              acc0=accumulators[2 * self.m * nr + mr],
              acc1=accumulators[2 * self.m * nr + self.m + mr],
          )

    asm_string += self.convert_to_float()
    accumulators = self.acc_registers()
    asm_string += '# Load quantization_params pointer from stack\n'
    asm_string += 'mov {quantization_params_reg}, [rsp + {offset}]\n'.format(
        quantization_params_reg=self.quantization_params_register(),
        offset=self.stack_size() + self.quantization_params_offset(),
    )
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        asm_string += (
            'vmulps z{ACC}, z{ACC}, DWORD PTR [{quantization_params_reg} +'
            ' {offset}]{{1to16}}\n'.format(
                ACC=accumulators[nr * self.m + mr],
                offset=4 + mr * 8,
                quantization_params_reg=self.quantization_params_register(),
            )
        )
    output_scale = 'vmovaps {W_SCALE}, [{W} + {offset}]\n'
    # output scales
    for nr in range(0, self.n):
      asm_string += output_scale.format(
          W=W,
          offset=self.register_bytes() * nr,
          W_SCALE=self.scale_registers()[nr],
      )
    asm_string += self.increment_ptr(ptr=W, step=self.register_bytes() * self.n)
    # biases
    for nr in range(0, self.n):
      asm_string += output_scale.format(
          W=W, offset=self.register_bytes() * nr, W_SCALE=self.w_registers()[nr]
      )
    asm_string += self.increment_ptr(ptr=W, step=self.register_bytes() * self.n)
    # Intel gets points here for its fma instructions which can accumulate into
    # any of the registers. For once, Intel has saner instructions than Arm.
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        asm_string += 'vfmadd213ps z{ACC}, {SCALE}, {BIAS}\n'.format(
            ACC=accumulators[nr * self.m + mr],
            SCALE=self.scale_registers()[nr],
            BIAS=self.w_registers()[nr],
        )

    return asm_string

  def outer_loop_prepare(self):
    # outside the outer loop
    zp_scale_load_push = (
        """mov {tmp_reg}, [{quantization_params_reg} + {zp_offset}]
      vpbroadcastd {tmp_s_reg}, {tmp_reg}
      vmovaps zmmword ptr [rsp + {offset}], {tmp_s_reg}\n"""
    )
    ret = '\n# Load quantization_params pointer from stack\n'
    ret += 'mov {quantization_params_reg}, [rsp + {offset}]\n'.format(
        quantization_params_reg=self.quantization_params_register(),
        offset=self.stack_size() + self.quantization_params_offset(),
    )
    for mr in range(0, self.m, 1):
      ret += zp_scale_load_push.format(
          tmp_reg=self.register_map_dword(self.tmp_gp_registers()[0]),
          quantization_params_reg=self.quantization_params_register(),
          tmp_s_reg=self.w_registers()[0],
          offset=self.stupid_offset() + mr * 64,
          zp_offset=mr * 8,
      )
    return ret

  def init_accumulators(self):
    ret = '# Initialize accumulators with k_sum * input zero point.\n'
    accumulators = self.acc_registers()
    W = self.w_ptr_register()

    ksum_x16 = 'vmovaps  {KSUM}, [{W} + {offset}]\n'
    vksum = 'vpmulld z{ACC}, {KSUM}, ZMMWORD PTR [rsp + {offset}]\n'

    for nr in range(0, self.n):
      ret += ksum_x16.format(
          W=self.w_ptr_register(),
          KSUM=self.w_registers()[nr],
          offset=self.register_bytes() * nr,
      )
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        ret += vksum.format(
            ACC=accumulators[nr * self.m + mr],
            KSUM=self.w_registers()[nr],
            pos=int((mr % 2) * 2),
            offset=self.stupid_offset() + mr * 64,
        )

    ret += self.increment_ptr(ptr=W, step=self.register_bytes() * self.n)
    ret += self.interleave_zeros(self.m, self.n)
    return ret

  def stupid_offset(self):
    size = self.m * 16 + self.c_ptr_stack_offset()
    return math.ceil(size / 64) * 64

  def stack_size(self):
    size = self.stupid_offset() + self.m * 64
    # round up to multiple of 64.
    return math.ceil(size / 64) * 64

  def interleave_zeros(self, M, N):
    match self._c:
      case 4:
        return ''
      case 8:
        W = self.w_ptr_register()
        accumulators = self.acc_registers()
        bias_registers = self.bias_registers()

        c = self._c * self.element_size()
        asm_string = '# Interleave with zeros.\n'
        unpack_lo = 'vpmovzxdq z{acc0}, y{acc1}\n'
        unpack_hi = """vextracti64x4 y{acc0}, z{acc1}, 1
        vpmovzxdq z{acc0}, y{acc0}
        """
        for mr in range(0, M):
          for nr in reversed(range(0, N)):
            asm_string += unpack_hi.format(
                acc0=accumulators[mr + 2 * M * nr + M],
                acc1=accumulators[mr + nr * M],
            )
            asm_string += unpack_lo.format(
                acc0=accumulators[mr + 2 * M * nr],
                acc1=accumulators[mr + nr * M],
            )

        return asm_string
      case _:
        raise NotImplementedError

  def inner_loop_spill_gp(self, tail: bool = False) -> str:
    return self._inner_loop_spill_gp(self._c // 4 * self.n, tail)

  def inner_loop_small_M_N(self, tail: bool = False) -> str:
    return self._inner_loop_small_M_N(self._c // 4 * self.n, tail)


class Avx512VnniQc4w(Avx512Vnni):

  def function_name(self):
    c = self._c
    return (
        f'xnn_qd8_f32_qc4w_gemm_minmax_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c{c}__asm_amd64_{self.isa()}'
    )

  def weights_asm(self):
    w_asm = {
        'loop': [],
        'loop_2': ["""vmovaps {W_1}, [{W_ptr} + {offset}]
               vpslld {W}, {W_1}, 4
               vpandd {W}, {W}, z{mask}
               vpandd {W_1}, {W_1}, z{mask}\n"""],
        'after': ['add {W}, {w_step}\n'],
    }
    return w_asm

  def outer_loop_prepare(self):
    res = super().outer_loop_prepare()
    res += """
    mov {quantization_params_reg}, [rsp + 88]
    # Load 0xF0 for masking the weights
    vbroadcastsd  z{mask}, qword ptr [rip + .MASK]\n
    """.format(
        quantization_params_reg=self.quantization_params_register(),
        mask=self.mask_register(),
    )
    return res

  def convert_to_float(self):
    accumulators = self.acc_registers()
    ret = ''
    ret += '\n# Convert from int32 to float.\n'
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        reg_ofset = self.accumulator_reg_offset(self.m, nr)
        other_reg = accumulators[self.m * nr + mr + reg_ofset]
        ACC = accumulators[self.m * nr + mr]
        ret += f'vpsrad z{other_reg}, z{other_reg}, 4\n'
        ret += f'vcvtdq2ps z{ACC}, z{other_reg}\n'
    return ret

  def pre_header(self):
    asm_string = super().pre_header()
    return asm_string + """.MASK:
        .quad   -1085102592571150096\n"""

  def w_register_bytes(self):
    return self.register_bytes() // 2
