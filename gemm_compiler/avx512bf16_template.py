#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from gemm_compiler import avx512f_template as isa

"""All SIMD features for avx512f."""


class Avx512Bf16(isa.Avx512F):

  def __init__(self):
    self.c = 2

  def isa(self):
    return 'avx512bf16'

  def n_step(self):
    return 16

  def compute_asm(self):
    c_asm = {
        'loop': ['vdpbf16ps  z{ACC}, {A}, {W}\n'],
        'loop_tail': ["""vpslld {A}, {A}, 16
               vpsrad {A}, {A}, 16
               vdpbf16ps  z{ACC}, {A}, {W}\n
            """],
    }
    return c_asm

  def function_name(self, M, N, isa):
    return f'xnn_bf16_f32_gemm_minmax_ukernel_{M}x{N}c2__asm_amd64_{isa}_broadcast'

  def init_accumulators(self, M, N):
    asm_string = super().init_accumulators(M, N)
    asm_string += """
      # Are there at least 4 bytes?
      cmp rdx, 4
      js inner_loop_tail\n"""

    return asm_string

  def outer_loop_prepare(self, M, N):
    k_register = self.k_register()
    kc_register = self.kc_register()
    offset = M * 16 + self.c_ptr_stack_offset()
    kmask = self.k_mask()
    asm_string = f"""
      # Copy k and flip bit.
      mov {k_register}, rdx
      and {k_register}, 0x2
      and {kc_register}, {kmask}
      mov [rsp + {offset}], {k_register}\n"""
    return asm_string

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

  def element_size(self):
    return 2

  def k_mask(self):
    return "0xFFFFFFFFFFFFFFFD"
