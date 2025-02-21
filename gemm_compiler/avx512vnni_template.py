#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from gemm_compiler import avx512f_template as isa

"""All SIMD features for avx512vnni."""


class Avx512Vnni(isa.Avx512F):

  def __init__(self):
    self.c = 4

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

  def function_name(self, M, N, isa):
    c = self.c
    return f'xnn_qd8_f32_qc8w_gemm_minmax_ukernel_{M}x{N}c{c}__asm_amd64_{isa}'

  def zp_scale(self, pos):
    regs = ['10', '11']
    return regs[pos]

  def channels(self):
    return self.c

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

  def input_asm(self):
    in_asm = {
        'loop': [
            'vpbroadcastd {AM}, [{AM_ptr} + {a_offset}]\n',
        ],
        'compute': ['vpdpbusd  z{ACC}, {A}, {W}\n'],
    }
    return in_asm

  def weights_asm(self):
    w_asm = {
        'loop': [
            'vmovaps  {W}, [{W_ptr} + {offset}]\n',
        ],
        'after': 'add {W}, {w_step}\n',
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

  def convert_to_float(self, M, N, W):
    accumulators = self.acc_registers()
    ret = ''
    ret += '\n# Convert from int32 to float.\n'
    for nr in range(0, N * M):
      ret += 'vcvtdq2ps z{ACC}, z{ACC}\n'.format(ACC=accumulators[nr])
    return ret

  def dequantize(self, M, N, W):
    ret = self.convert_to_float(M, N, W)
    accumulators = self.acc_registers()
    ret += '# Load quantization_params pointer from stack\n'
    ret += 'mov {quantization_params_reg}, [rsp + {offset}]\n'.format(
        quantization_params_reg=self.quantization_params_register(),
        offset=self.stack_size(M) + self.quantization_params_offset(),
    )
    for nr in range(0, N):
      for mr in range(0, M):
        ret += (
            'vmulps z{ACC}, z{ACC}, DWORD PTR [{quantization_params_reg} +'
            ' {offset}]{{1to16}}\n'.format(
                ACC=accumulators[nr * M + mr],
                offset=4 + mr * 8,
                quantization_params_reg=self.quantization_params_register(),
            )
        )
    output_scale = 'vmovaps {W_SCALE}, [{W} + {offset}]\n'
    # output scales
    for nr in range(0, N):
      ret += output_scale.format(
          W=W,
          offset=self.register_bytes() * nr,
          W_SCALE=self.scale_registers()[nr],
      )
    ret += self.increment_ptr(ptr=W, step=self.register_bytes() * N)
    # biases
    for nr in range(0, N):
      ret += output_scale.format(
          W=W, offset=self.register_bytes() * nr, W_SCALE=self.w_registers()[nr]
      )
    ret += self.increment_ptr(ptr=W, step=self.register_bytes() * N)
    # Intel gets points here for its fma instructions which can accumulate into
    # any of the registers. For once, Intel has saner instructions than Arm.
    for nr in range(0, N):
      for mr in range(0, M):
        ret += 'vfmadd213ps z{ACC}, {SCALE}, {BIAS}\n'.format(
            ACC=accumulators[nr * M + mr],
            SCALE=self.scale_registers()[nr],
            BIAS=self.w_registers()[nr],
        )

    return ret

  def outer_loop_prepare(self, M, N):
    W = self.w_ptr_register()
    accumulators = self.acc_registers()
    # outside the outer loop
    zp_scale_load_push = (
        """mov {tmp_reg}, [{quantization_params_reg} + {zp_offset}]
      vpbroadcastd {tmp_s_reg}, {tmp_reg}
      vmovaps zmmword ptr [rsp + {offset}], {tmp_s_reg}\n"""
    )
    ret = '\n# Load quantization_params pointer from stack\n'
    ret += 'mov {quantization_params_reg}, [rsp + {offset}]\n'.format(
        quantization_params_reg=self.quantization_params_register(),
        offset=self.stack_size(M) + self.quantization_params_offset(),
    )
    for mr in range(0, M, 1):
      ret += zp_scale_load_push.format(
          tmp_reg=self.register_map_dword(self.tmp_gp_registers()[0]),
          quantization_params_reg=self.quantization_params_register(),
          tmp_s_reg=self.w_registers()[0],
          offset=self.stupid_offset(M) + mr * 64,
          zp_offset=mr * 8,
      )
    return ret

  def init_accumulators(self, M, N):
    ret = '# Initialize accumulators with k_sum * input zero point.\n'
    accumulators = self.acc_registers()
    W = self.w_ptr_register()

    ksum_x16 = 'vmovaps  {KSUM}, [{W} + {offset}]\n'
    vksum = 'vpmulld z{ACC}, {KSUM}, ZMMWORD PTR [rsp + {offset}]\n'

    for nr in range(0, N):
      ret += ksum_x16.format(
          W=W, KSUM=self.w_registers()[nr], offset=self.register_bytes() * nr
      )
    for nr in range(0, N):
      for mr in range(0, M):
        ret += vksum.format(
            ACC=accumulators[nr * M + mr],
            KSUM=self.w_registers()[nr],
            pos=int((mr % 2) * 2),
            offset=self.stupid_offset(M) + mr * 64,
        )

    ret += self.increment_ptr(ptr=W, step=self.register_bytes() * N)
    return ret

  def stupid_offset(self, M):
    size = M * 16 + self.c_ptr_stack_offset()
    return math.ceil(size / 64) * 64

  def stack_size(self, M):
    size = self.stupid_offset(M) + M * 64
    # round up to multiple of 64.
    return math.ceil(size / 64) * 64


class Avx512VnniC(Avx512Vnni):

  def __init__(self, c):
    self.c = c

  def input_asm(self):
    in_asm = {
        'loop': [
            'vbroadcasti32x2 {AM}, QWORD PTR [{AM_ptr} + {a_offset}]\n',
        ]
    }
    return in_asm

  def pre_header(self):
    return """.PERMUTATION:
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

  def convert_to_float(self, M, N, W):
    accumulators = self.acc_registers()
    ret = ''
    ret += '\n# Convert from int32 to float.\n'
    for nr in range(0, N):
      for mr in range(0, M):
        other_reg = accumulators[M * nr + mr + nr * M]
        ACC = accumulators[M * nr + mr]
        ret += f'vcvtdq2ps z{ACC}, z{other_reg}\n'
    return ret

  def dequantize(self, M, N, W):
    shift_add = """vpsrlq {tmp}, z{acc}, 32
      vpaddd z{acc}, z{acc}, {tmp}\n"""
    asm_string = ''
    accumulators = self.acc_registers()
    for nr in range(0, N * 2):
      for mr in range(0, M):
        asm_string += shift_add.format(
            acc=accumulators[M * nr + mr], tmp=self.w_registers()[0]
        )
    perm_reg = self.w_registers()[0]
    asm_string += f'vmovups {perm_reg}, zmmword ptr [rip + .PERMUTATION]\n'
    perm = 'vpermt2ps z{acc0}, {perm_reg}, z{acc1}\n'
    for nr in range(0, N):
      for mr in range(0, M):
        asm_string += perm.format(
            perm_reg=perm_reg,
            acc0=accumulators[2 * M * nr + mr],
            acc1=accumulators[2 * M * nr + M + mr],
        )
    asm_string += super().dequantize(M, N, W)
    return asm_string

  def init_accumulators(self, M, N):
    asm_string = super().init_accumulators(M, N)
    W = self.w_ptr_register()
    accumulators = self.acc_registers()
    bias_registers = self.bias_registers()

    c = self.c * self.element_size()
    asm_string += '# Interleave with zeros.\n'
    unpack_lo = 'vpmovzxdq z{acc0}, y{acc1}\n'
    unpack_hi = """vextracti64x4 y{acc0}, z{acc1}, 1
    vpmovzxdq z{acc0}, y{acc0}
    """
    for nr in reversed(range(0, N)):
      asm_string += unpack_hi.format(
          acc0=accumulators[2 * M * nr + M],
          acc1=accumulators[nr * M],
      )
      asm_string += unpack_lo.format(
          acc0=accumulators[2 * M * nr],
          acc1=accumulators[nr * M],
      )
    for nr in range(0, N * 2):
      for mr in range(1, M):
        asm_string += self.copy_simd_register(
            prefix=self.prefix(),
            src=accumulators[M * nr],
            dst=accumulators[M * nr + mr],
        )

    return asm_string

  def inner_loop(self, M, N):
    return super().inner_loop(M, N * 2)

  def bias_registers(self):
    return self.w_registers()
