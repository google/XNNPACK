#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import avx512f_template


class Avx512Bf16(avx512f_template.Avx512F):
  """All SIMD features for avx512f."""

  def __init__(self, m: int, n: int):
    super().__init__(m=m, n=n)
    self._c = 2

  @property
  def c(self) -> int:
    return 2

  def isa(self):
    return 'avx512bf16'

  def n_step(self):
    return 16

  def compute_asm(self):
    c_asm = {
        'loop': ['vdpbf16ps  z{ACC}, {A}, {W}\n'],
        'loop_tail': ["""vpslld {A}, {A}, 16
               vpsrld {A}, {A}, 16
               vdpbf16ps  z{ACC}, {A}, {W}\n
            """],
    }
    return c_asm

  def function_name(self):
    return (
        f'xnn_bf16_f32_gemm_minmax_ukernel_{self.m}x{self.n * self.n_step()}'
        + f'c2__asm_amd64_{self.isa()}_broadcast'
    )

  def init_accumulators(self):
    asm_string = super().init_accumulators()
    asm_string += """
      # Are there at least 4 bytes?
      cmp rdx, 4
      js .Linner_loop_tail\n"""

    return asm_string

  def outer_loop_prepare(self):
    k_register = self.k_register()
    kc_register = self.kc_register()
    offset = self.m * 16 + self.c_ptr_stack_offset()
    kmask = self.k_mask()
    asm_string = f"""
      # Copy k and flip bit.
      mov {k_register}, rdx
      and {k_register}, 0x2
      and {kc_register}, {kmask}
      mov [rsp + {offset}], {k_register}\n"""
    return asm_string

  def inner_loop_tail(self):
    nc_register = self.nc_register()
    offset = self.m * 16 + self.c_ptr_stack_offset()
    nc_offset = offset + 8
    asm_string = f"""
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
      asm_string += self.inner_loop_spill_gp(tail=True)
    else:
      asm_string += self.inner_loop_small_M_N(tail=True)
    return asm_string

  def element_size(self):
    return 2

  def k_mask(self):
    return '0xFFFFFFFFFFFFFFFD'
