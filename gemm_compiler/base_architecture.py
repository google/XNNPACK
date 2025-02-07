#!/usr/bin/env python3
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

from gemm_compiler import base_architecture as base_architecture

"""Base architecture for GEMM microkernel generation"""


class BaseArchitecture:

  def __init__(self):
    pass  # Empty constructor

  def labels(self):
    return [
        'zero',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine',
        'ten',
        'eleven',
    ]

  @abstractmethod
  def astride_register(self):
    """Returns the register containing a_stride."""
    raise NotImplementedError

  @abstractmethod
  def kc_register(self):
    """Returns the register containing kc, the number of channels (the reduction dimensions)."""
    raise NotImplementedError

  @abstractmethod
  def k_register(self):
    """Returns the register containing k, the current channel being processed."""
    raise NotImplementedError

  @abstractmethod
  def cm_stride_register(self):
    """Returns the register containing cm_stride."""
    raise NotImplementedError

  @abstractmethod
  def am_registers(self):
    """Returns the registers containing the pointers to each row of A (LHS)."""
    raise NotImplementedError

  @abstractmethod
  def a_ptr_register(self):
    """Returns the register containing the A pointer."""
    raise NotImplementedError

  @abstractmethod
  def c_ptr_register(self):
    """Returns the register containing the C pointer."""
    raise NotImplementedError

  @abstractmethod
  def cm_registers(self):
    """Returns the registers containing the pointers to each row of C (Output)."""
    raise NotImplementedError

  @abstractmethod
  def acc_registers(self):
    """Returns the accumulator registers."""
    raise NotImplementedError

  @abstractmethod
  def w_ptr_register(self):
    """Returns the register containing the weight's pointer."""
    raise NotImplementedError

  @abstractmethod
  def min_register(self):
    """Returns the register containing the min value for clamping."""
    raise NotImplementedError

  @abstractmethod
  def max_register(self):
    """Returns the register containing the max value for clamping."""
    raise NotImplementedError

  @abstractmethod
  def nc_register(self):
    """Returns the register containing nc, the number of output rows processed per iteration."""
    raise NotImplementedError

  @abstractmethod
  def mr_register(self):
    """Returns the register containing mr, the number of input rows processed per kernel call."""
    raise NotImplementedError

  @abstractmethod
  def tmp_gp_registers(self):
    """Returns some general purpose registers which may be used for storing temporary data."""
    raise NotImplementedError

  @abstractmethod
  def jump_to_label(self, label):
    """Jump to the given label."""
    raise NotImplementedError

  @abstractmethod
  def function_name(self, M, N, isa):
    """Returns the microkernel name."""
    raise NotImplementedError

  @abstractmethod
  def header(self, M, N, prefix, isa):
    """Returns the assembly header."""
    raise NotImplementedError

  @abstractmethod
  def input_output_register_setup(self, M):
    """Setup the input (A) and output (C) registers."""
    raise NotImplementedError

  @abstractmethod
  def max_M_before_spilling(self):
    """How large can M be before spilling A and C registers to the stack."""
    raise NotImplementedError

  @abstractmethod
  def read_a_registers(self, M):
    """Read the A registers from the stack."""
    raise NotImplementedError

  @abstractmethod
  def increment_ptr(self, ptr, step):
    """Increment the given pointer by step bytes."""
    raise NotImplementedError

  @abstractmethod
  def initialize_k_register(self, reg):
    """Initialized the given general purpose register for inner loop control."""
    raise NotImplementedError

  @abstractmethod
  def cmp_k_and_jump_if_less(self, label):
    """If k is less than kc, then do another iteration of the inner loop."""
    raise NotImplementedError

  @abstractmethod
  def epilogue(self, M, N, isa):
    """Returns the function epilogue."""
    raise NotImplementedError

  @abstractmethod
  def inner_loop(self, M, N):
    """Returns the assemebly for the microkernel's inner loop."""
    raise NotImplementedError

  def clamp(self, M, N):
    num_horizontal_registers = int(N / self.n_step())
    acc_registers = self.acc_registers()
    asm_string = ''
    for nr in range(0, num_horizontal_registers):
      for mr in range(0, M):
        asm_string += self.clamp_min(
            reg=acc_registers[M * nr + mr], prefix=self.prefix()
        )
    for nr in range(0, num_horizontal_registers):
      for mr in range(0, M):
        asm_string += self.clamp_max(
            reg=acc_registers[M * nr + mr], prefix=self.prefix()
        )
    return asm_string
