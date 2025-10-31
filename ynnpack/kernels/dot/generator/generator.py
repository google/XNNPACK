"""Generate optimized YNNPACK `dot` kernels"""

from collections.abc import Sequence
import sys

from ynnpack.kernels.dot.generator.arm_bf16_bf16_fp32 import arm64_neon_bf16_bf16_fp32
from ynnpack.kernels.dot.generator.arm_fp32 import arm64_neon_fp32
from ynnpack.kernels.dot.generator.arm_int8_int8_int32 import arm_neon_int8_int8_int32
from ynnpack.kernels.dot.generator.arm_int8_int8_int32 import arm_neondot_int8_int8_int32
from ynnpack.kernels.dot.generator.arm_int8_int8_int32 import arm_neoni8mm_int8_int8_int32
from ynnpack.kernels.dot.generator.x86_bf16_bf16_fp32 import x86_avx2_fma3_bf16_bf16_fp32
from ynnpack.kernels.dot.generator.x86_bf16_bf16_fp32 import x86_avx512bf16_bf16_bf16_fp32
from ynnpack.kernels.dot.generator.x86_bf16_bf16_fp32 import x86_avx512f_bf16_bf16_fp32
from ynnpack.kernels.dot.generator.x86_fp32 import x86_avx512f_fp32
from ynnpack.kernels.dot.generator.x86_fp32 import x86_avx_fp32
from ynnpack.kernels.dot.generator.x86_fp32 import x86_fma3_fp32
from ynnpack.kernels.dot.generator.x86_fp32 import x86_sse2_fp32
from ynnpack.kernels.dot.generator.x86_fp32_k2 import x86_avx2_fma3_fp32_k2
from ynnpack.kernels.dot.generator.x86_fp32_k2 import x86_avx2_fp32_k2
from ynnpack.kernels.dot.generator.x86_fp32_k2 import x86_avx512f_fp32_k2
from ynnpack.kernels.dot.generator.x86_fp32_k4 import x86_avx512f_fp32_k4
from ynnpack.kernels.dot.generator.x86_int8_int8_int32 import x86_avx2_int8_int8_int32
from ynnpack.kernels.dot.generator.x86_int8_int8_int32 import x86_avx512bw_int8_int8_int32
from ynnpack.kernels.dot.generator.x86_int8_int8_int32_k16 import x86_avx512bw_int8_int8_int32_k16
from ynnpack.kernels.dot.generator.x86_uint8_int8_int32 import x86_avx512vnni_uint8_int8_int32
from ynnpack.kernels.dot.generator.x86_uint8_int8_int32_k16 import x86_avx512vnni_uint8_int8_int32_k16

arch_to_generator = {
    "x86_sse2_fp32": x86_sse2_fp32(),
    "x86_avx_fp32": x86_avx_fp32(),
    "x86_avx2_fp32_k2": x86_avx2_fp32_k2(),
    "x86_fma3_fp32": x86_fma3_fp32(),
    "x86_avx2_fma3_fp32_k2": x86_avx2_fma3_fp32_k2(),
    "x86_avx512f_fp32_k2": x86_avx512f_fp32_k2(),
    "x86_avx512f_fp32": x86_avx512f_fp32(),
    "x86_avx512f_fp32_k4": x86_avx512f_fp32_k4(),
    "x86_avx2_fma3_bf16_bf16_fp32": x86_avx2_fma3_bf16_bf16_fp32(),
    "x86_avx512f_bf16_bf16_fp32": x86_avx512f_bf16_bf16_fp32(),
    "x86_avx512bf16_bf16_bf16_fp32": x86_avx512bf16_bf16_bf16_fp32(),
    "x86_avx2_int8_int8_int32": x86_avx2_int8_int8_int32(),
    "x86_avx512bw_int8_int8_int32": x86_avx512bw_int8_int8_int32(),
    "x86_avx512bw_int8_int8_int32_k16": x86_avx512bw_int8_int8_int32_k16(),
    "x86_avx512vnni_uint8_int8_int32": x86_avx512vnni_uint8_int8_int32(),
    "x86_avx512vnni_uint8_int8_int32_k16": (
        x86_avx512vnni_uint8_int8_int32_k16()
    ),
    "arm_neon_int8_int8_int32": arm_neon_int8_int8_int32(),
    "arm_neondot_int8_int8_int32": arm_neondot_int8_int8_int32(),
    "arm_neoni8mm_int8_int8_int32": arm_neoni8mm_int8_int8_int32(),
    "arm64_neon_fp32": arm64_neon_fp32(),
    "arm64_neon_bf16_bf16_fp32": arm64_neon_bf16_bf16_fp32(),
}

def main(argv: Sequence[str]) -> None:

  output_src = argv[1]
  output_inc = argv[2]

  gen = arch_to_generator[argv[3]]
  src = gen.header()
  inc = ""


  for i in argv[4:]:
    args = i.split(",")
    kind = args[0]
    mr, nr, kr = args[1].split("x")
    if kind == "dot":
      src_i, inc_i = gen.generate_dot(int(mr), int(nr), int(kr))
    else:
      raise ValueError(f"Unknown kind: {kind}")

    src += src_i
    inc += inc_i

  src += gen.footer()

  with open(output_src, "w") as f:
    f.write(src)
  with open(output_inc, "w") as f:
    f.write(inc)

if __name__ == "__main__":
  main(sys.argv)
