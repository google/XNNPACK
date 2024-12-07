#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from gemm_compiler import aarch64_template
from gemm_compiler import avx512f_template
from gemm_compiler import avx512vnni_template
from gemm_compiler import fma3_template
from gemm_compiler import neondot_template
from gemm_compiler import neonfma_template
from gemm_compiler import x64_template

"""Shared logic for assembly gemm microkernel generation."""


def generate_gemm_microkernel(M, N, arch, isa, output_file):
  N_STEP = isa.n_step()
  N_COUNT = int(N / N_STEP)
  ukernel = isa.header(M, N, isa.prefix(), isa.isa())

  astride_register = arch.astride_register()
  kc_register = arch.kc_register()
  k_register = arch.k_register()
  cm_stride_reg = arch.cm_stride_register()
  am_registers = arch.am_registers()
  cm_registers = arch.cm_registers()
  acc_registers = isa.acc_registers()
  w_ptr_reg = arch.w_ptr_reg()
  min_reg = arch.min_reg()
  max_reg = arch.max_reg()
  nc_reg = arch.nc_reg()
  mr_reg = arch.mr_reg()
  labels = arch.labels()
  tmp_gp_regs = arch.tmp_gp_regs()

  nc_byte = arch.register_map_byte(nc_reg)

  asm_string = ukernel

  # adjust inner loop
  asm_string += isa.adjust_kc(4)

  # setup a{1}->a{M-1} & c{1]->c{M-1}registers
  asm_string += arch.input_output_register_setup(
      M=M,
      registers=am_registers,
      a_stride=astride_register,
      c_stride=cm_stride_reg,
      isa=isa,
  )

  # Pre outer loop preparation
  asm_string += isa.outer_loop_prepare(
      M=M, N=N_COUNT, W=w_ptr_reg, accumulators=acc_registers
  )

  # if mr < MR
  clamp_string, outer = arch.clamp_inputs_and_outputs(
      M, labels, am_registers, cm_registers
  )

  # the outer loop label
  OUTER = labels[outer]
  asm_string += f'\n{OUTER}: # Outer loop\n'
  asm_string += arch.zero_gp_reg(k_register)

  # pop a registers from the stack if required
  asm_string += isa.pop_a_registers(M=M, registers=am_registers)

  # Initialize accumulators
  asm_string += isa.init_accumulators(
      M=M, N=N_COUNT, W=w_ptr_reg, accumulators=acc_registers
  )
  asm_string += arch.increment_ptr(
      ptr=w_ptr_reg, step=isa.register_bytes() * N_COUNT
  )

  # inner loop label
  asm_string += '\ninner_loop:\n'

  asm_string += isa.inner_loop(M, N)

  # loop counter
  asm_string += arch.cmp_and_jump(
      label='inner_loop', k_register=k_register, kc_register=kc_register
  )

  asm_string += isa.dequant(
      M=M, N=N_COUNT, W=w_ptr_reg, accumulators=acc_registers
  )

  # min/max clamping
  for nr in range(0, N_COUNT):
    for mr in range(0, M):
      asm_string += isa.clamp_min(
          max_reg=max_reg, reg=acc_registers[M * nr + mr], prefix=isa.prefix()
      )
  for nr in range(0, N_COUNT):
    for mr in range(0, M):
      asm_string += isa.clamp_max(
          min_reg=min_reg, reg=acc_registers[M * nr + mr], prefix=isa.prefix()
      )

  # store
  asm_string += isa.store(
      M=M,
      N=N,
      acc_registers=acc_registers,
      nc_reg=nc_reg,
      nc_lo=nc_byte,
      cm_registers=cm_registers,
      kc_register=kc_register,
      am_registers=am_registers,
      labels=labels,
      outer_label=outer,
      tmp_gp_regs=tmp_gp_regs,
  )

  asm_string += arch.epilogue(M, N, isa)
  with open(output_file, 'w') as f:
    f.write(asm_string)
