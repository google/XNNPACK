"""Generate optimized YNNPACK elementwise kernels."""

from collections.abc import Sequence
from typing import Any, Callable, Tuple

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.arm import *  # pylint: disable=wildcard-import
from ynnpack.kernels.elementwise.wasm import *  # pylint: disable=wildcard-import
from ynnpack.kernels.elementwise.x86 import *  # pylint: disable=wildcard-import

arch_to_target = {
    "x86_sse2": X86(["SSE2"]),
    "x86_sse41": X86(["SSE41"]),
    "x86_avx": X86(["AVX"]),
    "x86_avx2": X86(["AVX2"]),
    "x86_fma3": X86(["FMA3"]),
    "x86_f16c": X86(["F16C"]),
    "x86_avx2_fma3": X86(["AVX2", "FMA3"]),
    "x86_avx512": X86(["AVX512F", "AVX512BW"]),
    "x86_avx512f": X86(["AVX512F"]),
    "x86_avx512bw": X86(["AVX512BW"]),
    "x86_avx512bf16": X86(["AVX512BF16"]),
    "arm64_neon": ARM(["NEON"]),
    "arm_neon": ARM(["NEON"]),
    "arm_neonfp16": ARM(["NEONFP16"]),
    "arm_neonbf16": ARM(["NEONBF16"]),
    "arm_neon_fma": ARM(["NEON", "FMA"]),
    "wasm_simd128": WASM(["SIMD128"]),
}


def generate_elementwise_kernels(
    output_src: str,
    output_inc: str,
    target_name: str,
    kernels: Sequence[Tuple[Callable[..., Any], Tuple[int, int]]],
) -> None:
  """Generate `kernels` using the specified target."""

  target = arch_to_target[target_name]

  src = ""
  inc = ""

  src += target.header

  for fn, shape in kernels:
    vectorize, unroll = shape
    name = fn.__name__
    src_i, inc_i = target.compile_function(name, fn, [(unroll, vectorize)])

    src += src_i
    inc += inc_i

  with open(output_src, "w") as f:
    f.write(src)
  with open(output_inc, "w") as f:
    f.write(inc)
