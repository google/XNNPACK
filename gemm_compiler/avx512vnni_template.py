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
    super().__init__(m, n, c)

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
    self.asm_string += """
      add {kc_reg}, {channels}
      and {kc_reg}, {neg_channels}\n""".format(
        kc_reg=self.kc_register(), channels=channels - 1, neg_channels=-channels
    )

  def quantization_params_register(self):
    return self.k_register()

  def pre_header(self):
    alignment = int(math.log2(self.register_bytes()))
    header_c4 = ''
    header_c8 = f"""
        .p2align {alignment}, 0x0
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
        self.asm_string += header_c4
      case 8:
        self.asm_string += header_c8
      case _:
        raise NotImplementedError

  def input_asm(self):
    loop_c4 = 'vpbroadcastd {AM}, [{AM_ptr} + {a_offset}]\n'
    loop_c8 = 'vbroadcasti32x2 {AM}, qword ptr [{AM_ptr} + {a_offset}]\n'
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
    self.asm_string += '\n# Convert from int32 to float.\n'
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        reg_ofset = self.accumulator_reg_offset(self.m, nr)
        other_reg = accumulators[self.m * nr + mr + reg_ofset]
        ACC = accumulators[self.m * nr + mr]
        self.asm_string += f'vcvtdq2ps z{ACC}, z{other_reg}\n'

  # TODO: This is a qd8 conversion, move it to a child class?
  def convert_to_output(self):
    W = self.w_ptr_register()
    if self._c == 8:
      shift_add = """vpsrlq {tmp}, z{acc}, 32
        vpaddd z{acc}, z{acc}, {tmp}\n"""
      accumulators = self.acc_registers()
      for nr in range(0, self.n * 2):
        for mr in range(0, self.m):
          self.asm_string += shift_add.format(
              acc=accumulators[self.m * nr + mr], tmp=self.w_registers()[0]
          )
      perm_reg = self.w_registers()[0]
      self.asm_string += (
          f'vmovaps {perm_reg}, zmmword ptr [rip + .PERMUTATION]\n'
      )
      perm = 'vpermt2ps z{acc0}, {perm_reg}, z{acc1}\n'
      for nr in range(0, self.n):
        for mr in range(0, self.m):
          self.asm_string += perm.format(
              perm_reg=perm_reg,
              acc0=accumulators[2 * self.m * nr + mr],
              acc1=accumulators[2 * self.m * nr + self.m + mr],
          )

    self.convert_to_float()
    accumulators = self.acc_registers()
    self.asm_string += '# Load quantization_params pointer from stack\n'
    self.asm_string += (
        'mov {quantization_params_reg}, [rsp + {offset}]\n'.format(
            quantization_params_reg=self.quantization_params_register(),
            offset=self.stack_size() + self.quantization_params_offset(),
        )
    )
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += (
            'vmulps z{ACC}, z{ACC}, dword ptr [{quantization_params_reg} +'
            ' {offset}]{{1to16}}\n'.format(
                ACC=accumulators[nr * self.m + mr],
                offset=4 + mr * 8,
                quantization_params_reg=self.quantization_params_register(),
            )
        )
    output_scale = 'vmovaps {W_SCALE}, [{W} + {offset}]\n'
    # output scales
    for nr in range(0, self.n):
      self.asm_string += output_scale.format(
          W=W,
          offset=self.register_bytes() * nr,
          W_SCALE=self.scale_registers()[nr],
      )
    self.increment_ptr(ptr=W, step=self.register_bytes() * self.n)
    # biases
    for nr in range(0, self.n):
      self.asm_string += output_scale.format(
          W=W, offset=self.register_bytes() * nr, W_SCALE=self.w_registers()[nr]
      )
    self.increment_ptr(ptr=W, step=self.register_bytes() * self.n)
    # Intel gets points here for its fma instructions which can accumulate into
    # any of the registers. For once, Intel has saner instructions than Arm.
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += 'vfmadd213ps z{ACC}, {SCALE}, {BIAS}\n'.format(
            ACC=accumulators[nr * self.m + mr],
            SCALE=self.scale_registers()[nr],
            BIAS=self.w_registers()[nr],
        )

    self.comment('Min/max clamping.')
    self.clamp()

    return self.asm_string

  # TODO: This is qd8 specific, move it to a child class?
  def outer_loop_prepare(self):
    # outside the outer loop
    zp_scale_load_push = (
        """mov {tmp_reg}, [{quantization_params_reg} + {zp_offset}]
      vpbroadcastd {tmp_s_reg}, {tmp_reg}
      vmovaps zmmword ptr [rsp + {offset}], {tmp_s_reg}\n"""
    )
    self.comment('Load quantization_params pointer from stack')
    self.asm_string += (
        'mov {quantization_params_reg}, [rsp + {offset}]\n'.format(
            quantization_params_reg=self.quantization_params_register(),
            offset=self.stack_size() + self.quantization_params_offset(),
        )
    )
    for mr in range(0, self.m, 1):
      self.asm_string += zp_scale_load_push.format(
          tmp_reg=self.register_map_dword(self.tmp_gp_registers()[0]),
          quantization_params_reg=self.quantization_params_register(),
          tmp_s_reg=self.w_registers()[0],
          offset=self.stupid_offset() + mr * 64,
          zp_offset=mr * 8,
      )

  def init_accumulators(self):
    self.comment('Initialize accumulators with k_sum * input zero point.')
    accumulators = self.acc_registers()
    W = self.w_ptr_register()

    ksum_x16 = 'vmovaps  {KSUM}, [{W} + {offset}]\n'
    vksum = 'vpmulld z{ACC}, {KSUM}, zmmword ptr [rsp + {offset}]\n'

    for nr in range(0, self.n):
      self.asm_string += ksum_x16.format(
          W=self.w_ptr_register(),
          KSUM=self.w_registers()[nr],
          offset=self.register_bytes() * nr,
      )
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += vksum.format(
            ACC=accumulators[nr * self.m + mr],
            KSUM=self.w_registers()[nr],
            pos=int((mr % 2) * 2),
            offset=self.stupid_offset() + mr * 64,
        )

    self.increment_ptr(ptr=W, step=self.register_bytes() * self.n)
    self.interleave_zeros(self.m, self.n)

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
        return
      case 8:
        W = self.w_ptr_register()
        accumulators = self.acc_registers()
        bias_registers = self.bias_registers()

        c = self._c * self.element_size()
        self.comment('Interleave with zeros.')
        unpack_lo = 'vpmovzxdq z{acc0}, y{acc1}\n'
        unpack_hi = """vextracti64x4 y{acc0}, z{acc1}, 1
        vpmovzxdq z{acc0}, y{acc0}
        """
        for mr in range(0, M):
          for nr in reversed(range(0, N)):
            self.asm_string += unpack_hi.format(
                acc0=accumulators[mr + 2 * M * nr + M],
                acc1=accumulators[mr + nr * M],
            )
            self.asm_string += unpack_lo.format(
                acc0=accumulators[mr + 2 * M * nr],
                acc1=accumulators[mr + nr * M],
            )
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
    super().outer_loop_prepare()
    self.asm_string += """
    mov {quantization_params_reg}, [rsp + 88]
    # Load 0xF0 for masking the weights
    vbroadcastsd  z{mask}, qword ptr [rip + .MASK]\n
    """.format(
        quantization_params_reg=self.quantization_params_register(),
        mask=self.mask_register(),
    )

  def convert_to_float(self):
    accumulators = self.acc_registers()
    self.asm_string += '\n# Convert from int32 to float.\n'
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        reg_ofset = self.accumulator_reg_offset(self.m, nr)
        other_reg = accumulators[self.m * nr + mr + reg_ofset]
        ACC = accumulators[self.m * nr + mr]
        self.asm_string += f'vpsrad z{other_reg}, z{other_reg}, 4\n'
        self.asm_string += f'vcvtdq2ps z{ACC}, z{other_reg}\n'

  def pre_header(self):
    super().pre_header()
    self.asm_string += """.MASK:
        .quad   -1085102592571150096\n"""

  def w_register_bytes(self):
    return self.register_bytes() // 2


class Avx512VnniQS8QC8W(Avx512Vnni):

  def function_name(self):
    c = self._c
    return (
        f'xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c{c}__asm_amd64_{self.isa()}'
    )

  def quantization_params_offset(self):
    # TODO: Reorganize class hierarchy so that this override is not needed, qd8
    # should be a child/sibling of this.
    return 0

  def sign_mask_register(self):
    # mask register is unused here, just use that one.
    return self.mask_register()

  def pre_header(self):
    super().pre_header()
    self.asm_string += """.SIGN_MASK:
        .quad   -9187201950435737472  # 0x8080808080808080\n"""

  def load_params(self, reg):
    return """
      movsx         eax, word ptr [{reg}]
      vpbroadcastd zmm31, eax

      vpbroadcastb x{min}, byte ptr [{reg} + 2]

      movsx         eax, word ptr [{reg} + 4]
      vpbroadcastd  z{max}, eax
      vpsubd        z{max}, z{max}, zmm31
      vcvtdq2ps     z{max}, z{max}
""".format(
        reg=reg,
        prefix=self.prefix(),
        min=self.min_register(),
        max=self.max_register(),
    )

  def init_accumulators(self):
    self.comment('Initialize accumulators with bias')
    accumulators = self.acc_registers()
    W = self.w_ptr_register()

    vksum = 'vmovaps z{ACC}, [{W} + {offset}]\n'

    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += vksum.format(
            ACC=accumulators[nr * self.m + mr],
            W=self.w_ptr_register(),
            offset=self.register_bytes() * nr,
        )

    self.increment_ptr(ptr=W, step=self.register_bytes() * self.n)
    self.interleave_zeros(self.m, self.n)

  def input_asm(self):
    loop_c4 = (
        'vpxord {AM}, z{sign_mask}, dword ptr [{AM_ptr} + {a_offset}]{{1to16}}\n'
    )
    loop_c8 = (
        'vpxorq {AM}, z{sign_mask}, qword ptr [{AM_ptr} + {a_offset}]{{1to8}}\n'
    )
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

  def convert_to_output(self):
    W = self.w_ptr_register()
    if self._c == 8:
      shift_add = """vpsrlq {tmp}, z{acc}, 32
        vpaddd z{acc}, z{acc}, {tmp}\n"""
      accumulators = self.acc_registers()
      for nr in range(0, self.n * 2):
        for mr in range(0, self.m):
          self.asm_string += shift_add.format(
              acc=accumulators[self.m * nr + mr], tmp=self.w_registers()[0]
          )
      perm_reg = self.w_registers()[0]
      self.asm_string += (
          f'vmovaps {perm_reg}, zmmword ptr [rip + .PERMUTATION]\n'
      )
      perm = 'vpermt2ps z{acc0}, {perm_reg}, z{acc1}\n'
      for nr in range(0, self.n):
        for mr in range(0, self.m):
          self.asm_string += perm.format(
              perm_reg=perm_reg,
              acc0=accumulators[2 * self.m * nr + mr],
              acc1=accumulators[2 * self.m * nr + self.m + mr],
          )

    self.convert_to_float()
    accumulators = self.acc_registers()

    output_scale = 'vmovaps {W_SCALE}, [{W} + {offset}]\n'
    # output scales
    for nr in range(0, self.n):
      self.asm_string += output_scale.format(
          W=W,
          offset=self.register_bytes() * nr,
          W_SCALE=self.scale_registers()[nr],
      )
    self.increment_ptr(ptr=W, step=self.register_bytes() * self.n)
    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += 'vmulps z{ACC}, z{ACC}, {SCALE}\n'.format(
            ACC=accumulators[nr * self.m + mr],
            SCALE=self.scale_registers()[nr],
        )

    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += 'vminps z{ACC}, z{ACC}, z{MAX}\n'.format(
            ACC=accumulators[nr * self.m + mr],
            MAX=self.max_register(),
        )

    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += 'vcvtps2dq z{ACC}, z{ACC}\n'.format(
            ACC=accumulators[nr * self.m + mr],
        )

    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += 'vpaddd z{ACC}, z{ACC}, zmm31\n'.format(
            ACC=accumulators[nr * self.m + mr],
        )

    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += 'vpmovsdb x{ACC}, z{ACC}\n'.format(
            ACC=accumulators[nr * self.m + mr],
        )

    for nr in range(0, self.n):
      for mr in range(0, self.m):
        self.asm_string += 'vpmaxsb x{ACC}, x{ACC}, x{MIN}\n'.format(
            ACC=accumulators[nr * self.m + mr],
            MIN=self.min_register(),
        )

    return self.asm_string

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
      vmovups  [{c_reg}], x{ACC}""".format(
          ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
      )
      for nr in range(1, self.n):
        self.asm_string += """
      vmovups  [{c_reg} + {offset}], x{ACC}""".format(
            ACC=accumulators[self.m * nr + mr],
            c_reg=cm_registers[mr + c_reg_offset],
            offset=self.register_bytes()
            * nr
            // 4,  # TODO: Find a better way to compute this stride than dividing by 4.
        )
    self.asm_string += '\n'
    for mr in range(0, self.m):
      self.asm_string += 'add {cm}, {cn_stride}\n'.format(
          cn_stride=self.n
          * self.register_bytes()
          // 4,  # TODO: Find a better way to compute this stride than dividing by 4.
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
            'vmovdqu8  xmmword ptr [{c_reg}]{{k1}}, z{ACC}\n'.format(
                ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
            )
        )
        self.asm_string += (
            'vmovdqu8  xmmword ptr [{c_reg} + 16]{{k2}}, z{ACC}\n'.format(
                ACC=accumulators[mr + self.m],
                c_reg=cm_registers[mr + c_reg_offset],
            )
        )
        self.asm_string += (
            'vmovdqu8  xmmword ptr [{c_reg} + 32]{{k3}}, z{ACC}\n'.format(
                ACC=accumulators[mr + 2 * self.m],
                c_reg=cm_registers[mr + c_reg_offset],
            )
        )
        self.asm_string += (
            'vmovdqu8  xmmword ptr [{c_reg} + 64]{{k4}}, z{ACC}\n'.format(
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
            'vmovdqu8  xmmword ptr [{c_reg}]{{k1}}, z{ACC}\n'.format(
                ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
            )
        )
        self.asm_string += (
            'vmovdqu8  xmmword ptr [{c_reg} + 16]{{k2}}, z{ACC}\n'.format(
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
            'vmovdqu8  xmmword ptr [{c_reg}]{{k1}}, x{ACC}\n'.format(
                ACC=accumulators[mr], c_reg=cm_registers[mr + c_reg_offset]
            )
        )

  def outer_loop_prepare(self):
    self.asm_string += """
    # Load 0x80 for xoring the weights
    vbroadcastsd  z{sign_mask}, qword ptr [rip + .SIGN_MASK]\n
    """.format(
        sign_mask=self.sign_mask_register(),
    )

class Avx512VnniQS8QC4W(Avx512VnniQc4w, Avx512VnniQS8QC8W):

  def function_name(self):
    c = self._c
    return (
        f'xnn_qs8_qc4w_gemm_minmax_fp32_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c{c}__asm_amd64_{self.isa()}'
    )

  def sign_mask_register(self):
    return "mm30"