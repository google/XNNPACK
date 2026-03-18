# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base class for dot kernel generators.

Provides the basic structure and shared logic for generating dot kernels."""

# pylint: disable=missing-function-docstring

from collections.abc import Sequence


def indent(text, prefix):
  return "\n".join(prefix + line if line else "" for line in text.splitlines())


class dot_base:
  """This is a code generator for `dot`` kernels.

  Dot kernels compute the following operation:

    C(i, j) += A(i, k3, k2, k1) * B(k3, k2, k1, j)

  The idea here is to break the output into `block`s of size `mr`x`nr`x`kr`, and
  then break blocks into `tiles` of size `ms`x`ns`x`ks`. If the code generator
  is C/C++, we expect a tile to correspond to a type (e.g. `__m128`,
  `float32x4_t`, etc.). If the code generator is assembly, we expect a tile to
  correspond to a register of some kind.

  Block shapes are:
  - A: `mr`x`kr`
  - B: `kr`x`nr`
  - C: `mr`x`nr`

  Tile shapes are:
  - A: `ms`x`ks`
  - B: `ks`x`ns`
  - C: `ms`x`ns`

  The code generator implementation is minimally responsible for generating
  operations to work on one tile, and that implementation will be unrolled to
  fill the block. However, the implementation may optionally override the
  implementation of a block if it can do this more efficiently.
  """

  def __init__(self, arch, kind):
    self.arch = arch
    self.kind = kind
    self.block_shape = (1, 1, 1)
    self.tile_shape = (1, 1, 1)
    self.a_type = ""
    self.b_type = ""
    self.c_type = ""
    self.b_chunk_n = 1
    self.min_tiles = 4
    self.flags = []

  def header(self):
    return """
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>

#if !defined(__has_attribute)
#define YNN_COMPILER_HAS_ATTRIBUTE(x) 0
#else
#define YNN_COMPILER_HAS_ATTRIBUTE(x) __has_attribute(x)
#endif

#if defined(__GNUC__)
#define YNN_ALWAYS_INLINE inline __attribute__((__always_inline__))
#elif defined(_MSC_VER)
#define YNN_ALWAYS_INLINE __forceinline
#else
#define YNN_ALWAYS_INLINE inline
#endif

#if YNN_COMPILER_HAS_ATTRIBUTE(unused)
#define YNN_UNUSED __attribute__((unused))
#else
#define YNN_UNUSED
#endif

#define YNN_INTRINSIC YNN_UNUSED YNN_ALWAYS_INLINE

namespace ynn {

namespace {

template <typename T>
YNN_INTRINSIC T* offset_bytes(T* ptr, std::ptrdiff_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr) + offset);
}

template <typename T>
YNN_INTRINSIC const T* offset_bytes(const T* ptr, std::ptrdiff_t offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(ptr) +
                                    offset);
}

YNN_INTRINSIC std::size_t min(std::size_t a, std::size_t b) {
  return a < b ? a : b;
}

YNN_INTRINSIC std::size_t sub_sat(std::size_t a, std::size_t b) {
  return a > b ? a - b : 0;
}

}  // namespace
"""

  def footer(self):
    return "}  // namespace ynn\n"

  def b_alignment_required(self):
    return self.tile_shape[1] * self.tile_shape[2]

  def begin_func(self, func_name):
    result = f"""
void {func_name}(
    std::size_t M, std::size_t N, std::size_t K3, std::size_t K2, std::size_t K1,
    std::size_t A_stride_m, std::size_t A_stride_k3, std::size_t A_stride_k2, const void* A,
    std::size_t B_stride_k3, std::size_t B_stride_k2, std::size_t B_stride_k1, const void* B,
    std::size_t C_in_stride_m, const void* C_in, std::size_t C_out_stride_m, void* C_out) {{
  assert(M > 0);
  assert(N > 0);
  assert(K3 > 0);
  assert(K2 > 0);
  assert(K1 > 0);
  assert(M <= {self.block_shape[0]});
"""

    if "dot_flag::unaligned_b" not in self.flags:
      result += f"""\
  assert(reinterpret_cast<uintptr_t>(B) % ({self.b_alignment_required()} * sizeof({self.b_type})) == 0);
  assert(B_stride_k1 % ({self.tile_shape[1]} * sizeof({self.b_type})) == 0 || K1 == 1);
"""

    if self.tile_shape[2] > 1:
      result += f"  assert(K1 % {self.tile_shape[2]} == 0);\n"

    return result

  def end_func(self):
    return "\n}\n"

  def a_ptr(self, i, k1, ty=None):
    """Get pointers to a(i, k1) within the current block."""
    ty = ty or self.a_type
    # When we clamp, we need to align down to the nearest tile.
    i = f"min({i}, (M - 1) & ~{self.tile_shape[0] - 1})" if i != 0 else i
    if "dot_flag::transpose_a" in self.flags:
      k1 //= self.tile_shape[2]
      i = f"{i} * {self.tile_shape[2]}"
      i, k1 = k1, i
    offset = f"({i} * A_stride_m) + ({k1} * sizeof({self.a_type}))"
    return f"reinterpret_cast<const {ty}*>(offset_bytes(A_k1, {offset}))"

  def b_ptr(self, k1, j, ty=None):
    """Get pointers to b(k1, j) within the current block."""
    ty = ty or self.b_type
    # Split j into `jo` and `ji`, where `jo` is which B pointer we use, and `ji`
    # is the offset from that B pointer we want.
    jo = (j // self.b_chunk_n) * self.b_chunk_n
    ji = j - jo
    return (
        f"reinterpret_cast<const {ty}*>(offset_bytes(B_k1_{jo}, ({k1} *"
        f" B_stride_k1) + ({ji * self.tile_shape[2]} * sizeof({self.b_type}))))"
    )

  def c_in_ptr(self, i, j, ty=None):
    """Get pointer to c(i, j) within the current block."""
    ty = ty or self.c_type
    i = f"min({i}, M - 1)" if i != 0 else i
    return (
        f"reinterpret_cast<const {ty}*>(offset_bytes(C_in, ({i} *"
        f" C_in_stride_m) + ({j} * sizeof({self.c_type}))))"
    )

  def c_out_ptr(self, i, j, ty=None):
    """Get pointer to c(i, j) within the current block."""
    ty = ty or self.c_type
    i = f"min({i}, M - 1)" if i != 0 else i
    return (
        f"reinterpret_cast<{ty}*>(offset_bytes(C_out, ({i} * C_out_stride_m) +"
        f" ({j} * sizeof({self.c_type}))))"
    )

  # Tile operations. An implementation of this class must implement all of the
  # tile operations.

  def init_c_tile(self, i, j):
    """Initialize accumulators for c(i, j)."""
    raise NotImplementedError()

  def finalize_c_tile(self, i, j):
    """Perform any necessary transformations of accumulators c(i, j) after accumulation but before adding and storing to the output."""
    return ""

  def load_a_tile(self, i, k):
    """Load one tile of a(i, k)."""
    raise NotImplementedError()

  # TODO: I think that we should just have `load_a_tile`, and the arguments
  # should be ranges. This is probably true of all of these functions.
  def load_a_tile_k_tail(self, i, k, nk):
    """Load `nk` values starting at a(i, k)."""
    # We assume that we can just use load_a_tile.
    return self.load_a_tile(i, k)

  def load_b_tile(self, k, j):
    """Load one tile of b(k, j)."""
    raise NotImplementedError()

  def product(self, i, j, k):
    """Compute c(i, j) += a(i, k) * b(k, j)."""
    raise NotImplementedError()

  def add_c_tile(self, i, j):
    """Add one tile of output from c_ptr(i, j) to c(i, j)."""
    raise NotImplementedError()

  def add_c_tile_tail(self, i, j, n):
    """Add `n` values of output from c_ptr(i, j) to c(i, j)."""
    raise NotImplementedError()

  def store_c_tile(self, i, j):
    """Write c(i, j) to c_ptr(i, j)."""
    raise NotImplementedError()

  def store_c_tile_tail(self, i, j, n):
    """Write `n` values of c(i, j) to c_ptr(i, j)."""
    raise NotImplementedError()

  # Block operations. An implementation may optionally override these, but the
  # default implementation uses the tile operations to implement blocks, and is
  # usually sufficient.

  def init_c_block(self):
    """Initialize accumulators for a block to use."""
    result = ""
    for i in range(0, self.block_shape[0], self.tile_shape[0]):
      for j in range(0, self.block_shape[1], self.tile_shape[1]):
        result += self.init_c_tile(i, j)
    return result

  def finalize_c_block(self):
    """Perform any necessary transformations of accumulators after accumulation but before adding and storing to the output."""
    result = ""
    for i in range(0, self.block_shape[0], self.tile_shape[0]):
      for j in range(0, self.block_shape[1], self.tile_shape[1]):
        result += self.finalize_c_tile(i, j)
    return result

  def release_c_block(self):
    """After all usage of the accumulators is completed, they can be released.

    If `init_c_block` allocates resources, release them here.
    """
    return ""

  def generate_block(self, nk):
    """Generate the loads and products to implement a `block_shape` of the dot product."""
    result = ""
    for i in range(0, self.block_shape[0], self.tile_shape[0]):
      for k in range(0, nk, self.tile_shape[2]):
        result += self.load_a_tile_k_tail(i, k, nk)
    result += "\n"

    # TODO: This is a hack, find a better way to decide this loop ordering.
    if "neon" in self.arch:
      for j in range(0, self.block_shape[1], self.tile_shape[1]):
        for k in range(0, nk, self.tile_shape[2]):
          result += self.load_b_tile(k, j)
        for i in range(0, self.block_shape[0], self.tile_shape[0]):
          for k in range(0, nk, self.tile_shape[2]):
            result += self.product(i, j, k)
    else:
      for k in range(0, nk, self.tile_shape[2]):
        for j in range(0, self.block_shape[1], self.tile_shape[1]):
          result += self.load_b_tile(k, j)

      for k in range(0, nk, self.tile_shape[2]):
        for i in range(0, self.block_shape[0], self.tile_shape[0]):
          for j in range(0, self.block_shape[1], self.tile_shape[1]):
            result += self.product(i, j, k)
    result += "\n"
    return result

  def store_block(self):
    result = ""
    for i in reversed(range(0, self.block_shape[0], self.tile_shape[0])):
      for j in range(0, self.block_shape[1], self.tile_shape[1]):
        result += self.store_c_tile(i, j)
    result += "\n"
    return result

  def store_block_tail(self):
    result = ""
    for i in reversed(range(0, self.block_shape[0], self.tile_shape[0])):
      for j in range(0, self.block_shape[1], self.tile_shape[1]):
        result += self.store_c_tile_tail(i, j, "N")
    result += "\n"
    return result

  def add_c_block(self):
    add_tiles = ""
    for i in range(0, self.block_shape[0], self.tile_shape[0]):
      for j in range(0, self.block_shape[1], self.tile_shape[1]):
        add_tiles += self.add_c_tile(i, j)
    result = "if (C_in) {\n"
    result += indent(add_tiles, "  ") + "\n"
    result += "}\n"

    result += self.store_block()
    return result

  def add_c_tiles(self, n):
    """Add accumulators to the output."""
    assert(n % self.tile_shape[1] == 0)
    add_tiles = ""
    for i in range(0, self.block_shape[0], self.tile_shape[0]):
      for j in range(0, n, self.tile_shape[1]):
        add_tiles += self.add_c_tile(i, j)
    result = "if (C_in) {\n"
    result += indent(add_tiles, "  ") + "\n"
    result += "}\n"

    for i in reversed(range(0, self.block_shape[0], self.tile_shape[0])):
      for j in range(0, n, self.tile_shape[1]):
        result += self.store_c_tile(i, j)

    return result

  def add_c_block_tail(self):
    add_tiles = ""
    for i in range(0, self.block_shape[0], self.tile_shape[0]):
      for j in range(0, self.block_shape[1], self.tile_shape[1]):
        add_tiles += self.add_c_tile_tail(i, j, "N")
    result = "if (C_in) {\n"
    result += indent(add_tiles, "  ") + "\n"
    result += "}\n"

    result += self.store_block_tail()
    return result

  def add_c(self, handle_tail):
    block = self.add_c_block()
    block += "\n"
    block += f"N -= {self.block_shape[1]};\n"
    block += f"B = offset_bytes(B, {self.block_shape[1] * self.tile_shape[2]} * sizeof({self.b_type}));\n"
    block += (
        f"C_in = C_in ? offset_bytes(C_in, {self.block_shape[1]} *"
        f" sizeof({self.c_type})) : nullptr;\n"
    )
    block += (
        f"C_out = offset_bytes(C_out, {self.block_shape[1]} *"
        f" sizeof({self.c_type}));\n"
    )

    if handle_tail:
      result = f"if (N >= {self.block_shape[1]}) {{\n"
      result += indent(block  , "  ")
      result += "\n} else {\n"
      block = self.add_c_block_tail()
      result += indent(block, "  ")
      result += "\n}"
      return result
    else:
      return block

  def loop_k1(self):
    result = ""
    for j in range(0, self.block_shape[1], self.b_chunk_n):
      result += f"const void* B_k1_{j} = B_k2_{j};\n"
    result += """const void* A_k1 = A_k2;
std::ptrdiff_t k1 = K1;
"""
    block_body = self.generate_block(self.block_shape[2])
    tile_body = self.generate_block(self.tile_shape[2])

    if block_body == tile_body:
      tile_body = None

    if tile_body:
      result += f"while (k1 >= {self.block_shape[2]}) {{\n"
    else:
      result += "do {\n"
    block_body += f"k1 -= {self.block_shape[2]};\n"
    for j in range(0, self.block_shape[1], self.b_chunk_n):
      block_body += (
          f"B_k1_{j} = offset_bytes(B_k1_{j}, {self.block_shape[2]} *"
          " B_stride_k1);\n"
      )
    if "dot_flag::transpose_a" in self.flags:
      a_step = f"{self.block_shape[2]//self.tile_shape[2]} * A_stride_m"
    else:
      a_step = f"{self.block_shape[2]} * sizeof({self.a_type})"
    block_body += f"A_k1 = offset_bytes(A_k1, {a_step});\n"
    result += indent(block_body, "  ") + "\n"
    if tile_body:
      result += "}\n"
    else:
      result += "} while (k1 > 0);\n"

    if tile_body:
      if self.tile_shape[2] * 2 == self.block_shape[2]:
        result += "if (k1 > 0) {\n"
      else:
        result += "while (k1 > 0) {\n"
      if self.tile_shape[2] * 2 != self.block_shape[2]:
        tile_body += f"k1 -= {self.tile_shape[2]};\n"
        for j in range(0, self.block_shape[1], self.b_chunk_n):
          tile_body += (
              f"B_k1_{j} = offset_bytes(B_k1_{j}, {self.tile_shape[2]} *"
              " B_stride_k1);\n"
          )
        tile_body += (
            f"A_k1 = offset_bytes(A_k1, {self.tile_shape[2]} *"
            f" sizeof({self.a_type}));\n"
        )
      result += indent(tile_body, "  ") + "\n"
      result += "}\n"

    return result

  def loop_k2(self):
    result = ""
    for j in range(0, self.block_shape[1], self.b_chunk_n):
      result += f"const void* B_k2_{j} = B_k3_{j};\n"
    result += """const void *A_k2 = A_k3;
std::size_t k2 = K2;
do {
"""
    body = self.loop_k1()
    body += "k2 -= 1;\n"
    for j in range(0, self.block_shape[1], self.b_chunk_n):
      body += f"B_k2_{j} = offset_bytes(B_k2_{j}, B_stride_k2);\n"
    body += "A_k2 = offset_bytes(A_k2, A_stride_k2);\n"

    result += indent(body, "  ")
    result += "\n} while (k2 > 0);\n"
    return result

  def loop_k3(self):
    result = "const void* B_k3_0 = B;\n"
    for j in range(self.b_chunk_n, self.block_shape[1], self.b_chunk_n):
      # Here, we need to implement the logic to avoid reading B out of bounds in
      # tail cases. `b_chunk_n` tells us how many values of B we can read at a
      # time without fear of going out of bounds. So we need to make a pointer
      # for each chunk, potentially clamping the address of each chunk. If the
      # chunk is out of bounds, we don't care what the data we read is as long
      # as it doesn't fault.
      result += (
          f"const void* B_k3_{j} = offset_bytes(B, N <= {j} ? 0 :"
          f" {j * self.tile_shape[2]} * sizeof({self.b_type}));\n"
      )
    result += """const void *A_k3 = A;
std::size_t k3 = K3;
do {
"""
    body = self.loop_k2()
    body += "k3 -= 1;\n"
    for j in range(0, self.block_shape[1], self.b_chunk_n):
      body += f"B_k3_{j} = offset_bytes(B_k3_{j}, B_stride_k3);\n"
    body += "A_k3 = offset_bytes(A_k3, A_stride_k3);\n"
    result += indent(body, "  ")
    result += "\n} while (k3 > 0);\n"
    return result

  def loop_j(self, n, k, handle_tail):
    """Generate the loop over N."""
    tiles_m = self.block_shape[0] // self.tile_shape[0]
    tiles_n = n // self.tile_shape[1]
    tiles = tiles_m * tiles_n

    # Ensure we have the minimum number of tiles.
    n = min(self.block_shape[1], n * max(1, self.min_tiles // tiles))

    # If we're unrolling the tail, we need to avoid reading B out of bounds.
    self.n = "N" if handle_tail else self.block_shape[1]
    self.b_chunk_n = self.tile_shape[1] if handle_tail else self.block_shape[1]

    self.block_shape = (self.block_shape[0], n, k)
    body = self.init_c_block()
    body += self.loop_k3()
    body += self.finalize_c_block()
    body += self.add_c(handle_tail)
    body += self.release_c_block()

    if handle_tail:
      loop = "while (N > 0) {\n"
    else:
      loop = f"while (N >= {self.block_shape[1]}) {{\n"
    loop += indent(body, "  ")
    loop += "\n}\n"

    return loop

  def generate_dot(self, m, n, k):
    """Return the code and declaration for a dot kernel with block shape (m, n, k)."""
    self.block_shape = (m, n, k)
    assert(self.block_shape[0] % self.tile_shape[0] == 0)
    assert(self.block_shape[1] % self.tile_shape[1] == 0)
    assert(self.block_shape[2] % self.tile_shape[2] == 0)

    func_name = (
        f"dot_{self.kind}_{m}x{n}x{k}_{'x'.join(str(s) for s in self.tile_shape)}_{self.arch}"
    )

    if self.flags:
      flags = " | ".join(self.flags)
    else:
      flags = "0"

    inc = ""
    inc += (
        f"YNN_DOT_KERNEL(arch_flag::{self.arch}, {func_name}, {m}, {n}, {k},"
        f" {', '.join(str(s) for s in self.tile_shape)}, /*flags=*/{flags},"
        f" {self.a_type}, {self.b_type}, {self.c_type})\n"
    )

    src = self.begin_func(func_name)
    # If we claim b is unaligned, we might use slow masked loads in the tail
    # case.
    if (n != self.tile_shape[1] or "dot_flag::unaligned_b" in self.flags):
      # The main loop (n = block_shape.n)
      body = self.loop_j(n, k, False)
      src += indent(body, "  ") + "\n"
    # The tail case (n = tile_shape.n))
    body = self.loop_j(self.tile_shape[1], k, True)
    src += indent(body, "  ") + "\n"
    src += self.end_func()

    return src, inc
