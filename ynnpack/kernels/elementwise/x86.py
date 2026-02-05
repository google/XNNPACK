"""X86 target for elementwise kernels compiler."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.common_rules import add_saturating_cast_rules
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def make_x86_cast_patterns(vector_bits, prefix):
  """Adds x86 cast patterns."""

  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              cast(Float(32), i32_a),
              Op(Float(32), prefix + "cvtepi32_ps", [i32_a]),
          ),
          Rule(
              cast(Int(32), round(f32_a)),
              Op(Int(32), prefix + "cvtps_epi32", [f32_a]),
          ),
      ]
  ] + add_saturating_cast_rules(vector_bits)


def make_x86_bf16_patterns(vector_bits):
  """Adds x86 bfloat16 patterns.

  Args:
    vector_bits: The number of vector bits.

  Returns:
    A list of rules for bfloat16 patterns.
  """
  rules = []
  if vector_bits == 256:
    rules.append(
        Rule(
            cast(
                BFloat(16, vector_bits // 16),
                combine_vectors([
                    f32_a.with_lanes(vector_bits // 32),
                    f32_b.with_lanes(vector_bits // 32),
                ]),
            ),
            Op(
                BFloat(16, vector_bits // 16),
                "convert_fp32_to_bf16_avx2",
                [
                    f32_a.with_lanes(vector_bits // 32),
                    f32_b.with_lanes(vector_bits // 32),
                ],
            ),
            features=["AVX2"],
        )
    )
    rules.append(
        Rule(
            cast(
                Float(32, vector_bits // 32),
                bf16_a.with_lanes(vector_bits // 32),
            ),
            Op(
                Float(32, vector_bits // 32),
                "convert_bf16_to_fp32_avx2",
                [bf16_a.with_lanes(vector_bits // 32)],
            ),
            features=["AVX2"],
        )
    )
  if vector_bits == 512:
    rules.append(
        Rule(
            cast(
                BFloat(16, vector_bits // 16),
                combine_vectors([
                    f32_a.with_lanes(vector_bits // 32),
                    f32_b.with_lanes(vector_bits // 32),
                ]),
            ),
            Op(
                BFloat(16, vector_bits // 16),
                "convert_fp32_to_bf16_avx512",
                [
                    f32_a.with_lanes(vector_bits // 32),
                    f32_b.with_lanes(vector_bits // 32),
                ],
            ),
            features=["AVX512BF16"],
        )
    )
    rules.append(
        Rule(
            cast(
                Float(32, vector_bits // 32),
                bf16_a.with_lanes(vector_bits // 32),
            ),
            Op(
                Float(32, vector_bits // 32),
                "convert_bf16_to_fp32_avx512",
                [bf16_a.with_lanes(vector_bits // 32)],
            ),
            features=["AVX512F"],
        )
    )
  return rules


def make_x86_reinterpret_cast_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              Op(Float(32), "reinterpret_cast", [i32_a]),
              Op(
                  Float(32),
                  prefix + "castsi" + str(vector_bits) + "_ps",
                  [i32_a],
              ),
          ),
          Rule(
              Op(Int(32), "reinterpret_cast", [f32_a]),
              Op(Int(32), prefix + "castps_si" + str(vector_bits), [f32_a]),
          ),
          Rule(
              Op(Float(32), "reinterpret_cast", [u32_a]),
              Op(
                  Float(32),
                  prefix + "castsi" + str(vector_bits) + "_ps",
                  [u32_a],
              ),
          ),
          Rule(
              Op(UInt(32), "reinterpret_cast", [f32_a]),
              Op(UInt(32), prefix + "castps_si" + str(vector_bits), [f32_a]),
          ),
      ]
  ]


def make_x86_integer_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(u8_a + u8_b, Op(UInt(8), prefix + "add_epi8", [u8_a, u8_b])),
          Rule(i8_a + i8_b, Op(Int(8), prefix + "add_epi8", [i8_a, i8_b])),
          Rule(
              u16_a + u16_b, Op(UInt(16), prefix + "add_epi16", [u16_a, u16_b])
          ),
          Rule(
              i16_a + i16_b, Op(Int(16), prefix + "add_epi16", [i16_a, i16_b])
          ),
          Rule(
              u32_a + u32_b, Op(UInt(32), prefix + "add_epi32", [u32_a, u32_b])
          ),
          Rule(
              i32_a + i32_b, Op(Int(32), prefix + "add_epi32", [i32_a, i32_b])
          ),
          Rule(u8_a - u8_b, Op(UInt(8), prefix + "sub_epi8", [u8_a, u8_b])),
          Rule(i8_a - i8_b, Op(Int(8), prefix + "sub_epi8", [i8_a, i8_b])),
          Rule(
              u16_a - u16_b, Op(UInt(16), prefix + "sub_epi16", [u16_a, u16_b])
          ),
          Rule(
              i16_a - i16_b, Op(Int(16), prefix + "sub_epi16", [i16_a, i16_b])
          ),
          Rule(
              u32_a - u32_b, Op(UInt(32), prefix + "sub_epi32", [u32_a, u32_b])
          ),
          Rule(
              i32_a - i32_b, Op(Int(32), prefix + "sub_epi32", [i32_a, i32_b])
          ),
          Rule(
              i32_a * i32_b, Op(Int(32), prefix + "mullo_epi32", [i32_a, i32_b])
          ),
          Rule(min(u8_a, u8_b), Op(UInt(8), prefix + "min_epu8", [u8_a, u8_b])),
          Rule(max(u8_a, u8_b), Op(UInt(8), prefix + "max_epu8", [u8_a, u8_b])),
          Rule(
              min(i16_a, i16_b),
              Op(Int(16), prefix + "min_epi16", [i16_a, i16_b]),
          ),
          Rule(
              max(i16_a, i16_b),
              Op(Int(16), prefix + "max_epi16", [i16_a, i16_b]),
          ),
          Rule(
              u32_a & u32_b,
              Op(
                  UInt(32),
                  prefix + "and_si" + str(vector_bits),
                  [u32_a, u32_b],
              ),
          ),
          Rule(
              i32_a & i32_b,
              Op(Int(32), prefix + "and_si" + str(vector_bits), [i32_a, i32_b]),
          ),
          Rule(
              u32_a | u32_b,
              Op(UInt(32), prefix + "or_si" + str(vector_bits), [u32_a, u32_b]),
          ),
          Rule(
              i32_a | i32_b,
              Op(Int(32), prefix + "or_si" + str(vector_bits), [i32_a, i32_b]),
          ),
          Rule(
              u32_a ^ u32_b,
              Op(
                  UInt(32), prefix + "xor_si" + str(vector_bits), [u32_a, u32_b]
              ),
          ),
          Rule(
              i32_a ^ i32_b,
              Op(Int(32), prefix + "xor_si" + str(vector_bits), [i32_a, i32_b]),
          ),
          Rule(
              saturating_add(u8_a, u8_b),
              Op(UInt(8), prefix + "adds_epu8", [u8_a, u8_b]),
          ),
          Rule(
              saturating_add(i8_a, i8_b),
              Op(Int(8), prefix + "adds_epi8", [i8_a, i8_b]),
          ),
          Rule(
              saturating_add(u16_a, u16_b),
              Op(UInt(16), prefix + "adds_epu16", [u16_a, u16_b]),
          ),
          Rule(
              saturating_add(i16_a, i16_b),
              Op(Int(16), prefix + "adds_epi16", [i16_a, i16_b]),
          ),
          Rule(
              saturating_sub(u8_a, u8_b),
              Op(UInt(8), prefix + "subs_epu8", [u8_a, u8_b]),
          ),
          Rule(
              saturating_sub(i8_a, i8_b),
              Op(Int(8), prefix + "subs_epi8", [i8_a, i8_b]),
          ),
          Rule(
              saturating_sub(u16_a, u16_b),
              Op(UInt(16), prefix + "subs_epu16", [u16_a, u16_b]),
          ),
          Rule(
              saturating_sub(i16_a, i16_b),
              Op(Int(16), prefix + "subs_epi16", [i16_a, i16_b]),
          ),
      ]
  ] + [
      Rule(
          logical_shift_left(
              i16_a.with_lanes(vector_bits // 16),
              broadcast(i16_b, vector_bits // 16),
          ),
          Op(Int(16, vector_bits // 16), prefix + "slli_epi16", [i16_a, i16_b]),
      ),
      Rule(
          logical_shift_left(
              i32_a.with_lanes(vector_bits // 32),
              broadcast(i32_b, vector_bits // 32),
          ),
          Op(
              Int(32, vector_bits // 32),
              prefix + "slli_epi32",
              [i32_a.with_lanes(vector_bits // 32), i32_b],
          ),
      ),
      Rule(
          u32_a.with_lanes(vector_bits // 32)
          >> broadcast(u32_b, vector_bits // 32),
          Op(
              UInt(32, vector_bits // 32),
              prefix + "srli_epi32",
              [u32_a.with_lanes(vector_bits // 32), u32_b],
          ),
      ),
  ]


def make_x86_integer_min_max_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              min(i8_a, i8_b),
              Op(Int(8), prefix + "min_epi8", [i8_a, i8_b]),
          ),
          Rule(
              max(i8_a, i8_b),
              Op(Int(8), prefix + "max_epi8", [i8_a, i8_b]),
          ),
          Rule(
              min(u16_a, u16_b),
              Op(UInt(16), prefix + "min_epu16", [u16_a, u16_b]),
          ),
          Rule(
              max(u16_a, u16_b),
              Op(UInt(16), prefix + "max_epu16", [u16_a, u16_b]),
          ),
          Rule(
              min(i32_a, i32_b),
              Op(Int(32), prefix + "min_epi32", [i32_a, i32_b]),
          ),
          Rule(
              max(i32_a, i32_b),
              Op(Int(32), prefix + "max_epi32", [i32_a, i32_b]),
          ),
          Rule(
              min(u32_a, u32_b),
              Op(UInt(32), prefix + "min_epu32", [u32_a, u32_b]),
          ),
          Rule(
              max(u32_a, u32_b),
              Op(UInt(32), prefix + "max_epu32", [u32_a, u32_b]),
          ),
      ]
  ]


# TODO(vksnk): These are only correct for SSE2
def make_x86_float_comparison_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              equal(f32_a, f32_b),
              Op(Float(32), prefix + "cmpeq_ps", [f32_a, f32_b]),
          ),
          Rule(
              not_equal(f32_a, f32_b),
              Op(Float(32), prefix + "cmpneq_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a > f32_b,
              Op(Float(32), prefix + "cmpgt_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a < f32_b,
              Op(Float(32), prefix + "cmplt_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a >= f32_b,
              Op(Float(32), prefix + "cmpge_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a <= f32_b,
              Op(Float(32), prefix + "cmple_ps", [f32_a, f32_b]),
          ),
      ]
  ]


# TODO(vksnk): These are only correct for SSE2
def make_x86_integer_comparison_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              equal(i8_a, i8_b), Op(Int(8), prefix + "cmpeq_epi8", [i8_a, i8_b])
          ),
          Rule(
              equal(i16_a, i16_b),
              Op(Int(16), prefix + "cmpeq_epi16", [i16_a, i16_b]),
          ),
          Rule(
              equal(i32_a, i32_b),
              Op(Int(32), prefix + "cmpeq_epi32", [i32_a, i32_b]),
          ),
          Rule(
              i8_a > i8_b,
              Op(Int(8), prefix + "cmpgt_epi8", [i8_a, i8_b]),
          ),
          Rule(
              i16_a > i16_b,
              Op(Int(16), prefix + "cmpgt_epi16", [i16_a, i16_b]),
          ),
          Rule(
              i32_a > i32_b,
              Op(Int(32), prefix + "cmpgt_epi32", [i32_a, i32_b]),
          ),
          Rule(
              i8_a < i8_b,
              Op(Int(8), prefix + "cmpgt_epi8", [i8_b, i8_a]),
          ),
          Rule(
              i16_a < i16_b,
              Op(Int(16), prefix + "cmpgt_epi16", [i16_b, i16_a]),
          ),
          Rule(
              i32_a < i32_b,
              Op(Int(32), prefix + "cmpgt_epi32", [i32_b, i32_a]),
          ),
      ]
  ]


def make_x86_broadcast_patterns(vector_bits, prefix):
  return [
      Rule(
          broadcast(i8_a, vector_bits // 8),
          Op(Int(8, vector_bits // 8), prefix + "set1_epi8", [i8_a]),
      ),
      Rule(
          broadcast(u8_a, vector_bits // 8),
          Op(UInt(8, vector_bits // 8), prefix + "set1_epi8", [u8_a]),
      ),
      Rule(
          broadcast(i16_a, vector_bits // 16),
          Op(Int(16, vector_bits // 16), prefix + "set1_epi16", [i16_a]),
      ),
      Rule(
          broadcast(u16_a, vector_bits // 16),
          Op(UInt(16, vector_bits // 16), prefix + "set1_epi16", [u16_a]),
      ),
      Rule(
          broadcast(i32_a, vector_bits // 32),
          Op(Int(32, vector_bits // 32), prefix + "set1_epi32", [i32_a]),
      ),
      Rule(
          broadcast(u32_a, vector_bits // 32),
          Op(UInt(32, vector_bits // 32), prefix + "set1_epi32", [u32_a]),
      ),
      Rule(
          broadcast(f32_a, vector_bits // 32),
          Op(Float(32, vector_bits // 32), prefix + "set1_ps", [f32_a]),
      ),
      Rule(
          broadcast(bf16_a, vector_bits // 16),
          Op(BFloat(16, vector_bits // 16), prefix + "set1_epi16", [bf16_a]),
      ),
  ]


def make_x86_slice_patterns(vector_bits, prefix):
  """Adds x86 slice patterns.

  Args:
    vector_bits: The number of vector bits.
    prefix: The prefix for the intrinsic names (e.g., "_mm256_").

  Returns:
    A list of rules for slice patterns.
  """
  if vector_bits == 256:
    rules = []
    for v in [i8_a, u8_a, i16_a, u16_a, i32_a, u32_a, bf16_a]:
      lanes_256 = 256 // v.ty.size
      lanes_128 = 128 // v.ty.size

      val_256 = v.with_lanes(lanes_256)
      val_128 = v.with_lanes(lanes_128)

      rules.append(
          Rule(
              Op(
                  val_128.ty,
                  "slice",
                  [val_256, WildConstant(0), WildConstant(2)],
              ),
              Op(
                  val_128.ty, "wrapper" + prefix + "slice_cast_si256", [val_256]
              ),
          )
      )
      rules.append(
          Rule(
              Op(
                  val_128.ty,
                  "slice",
                  [val_256, WildConstant(1), WildConstant(2)],
              ),
              Op(
                  val_128.ty,
                  "wrapper" + prefix + "slice_extract_si256_1",
                  [val_256, i32(1)],
              ),
          )
      )

    lanes_256 = 256 // 32
    lanes_128 = 128 // 32
    val_256 = f32_a.with_lanes(lanes_256)
    val_128 = f32_a.with_lanes(lanes_128)

    rules.append(
        Rule(
            Op(
                val_128.ty, "slice", [val_256, WildConstant(0), WildConstant(2)]
            ),
            Op(val_128.ty, "wrapper" + prefix + "slice_cast_ps256", [val_256]),
        )
    )
    rules.append(
        Rule(
            Op(
                val_128.ty, "slice", [val_256, WildConstant(1), WildConstant(2)]
            ),
            Op(
                val_128.ty,
                "wrapper" + prefix + "slice_extract_ps256_1",
                [val_256, i32(1)],
            ),
        )
    )
    return rules

  if vector_bits == 512:
    rules = []
    for v in [i8_a, u8_a, i16_a, u16_a, i32_a, u32_a, bf16_a]:
      lanes_512 = 512 // v.ty.size
      lanes_256 = 256 // v.ty.size

      val_512 = v.with_lanes(lanes_512)
      val_256 = v.with_lanes(lanes_256)

      rules.append(
          Rule(
              Op(
                  val_256.ty,
                  "slice",
                  [val_512, WildConstant(0), WildConstant(2)],
              ),
              Op(
                  val_256.ty, "wrapper" + prefix + "slice_cast_si512", [val_512]
              ),
          )
      )
      rules.append(
          Rule(
              Op(
                  val_256.ty,
                  "slice",
                  [val_512, WildConstant(1), WildConstant(2)],
              ),
              Op(
                  val_256.ty,
                  "wrapper" + prefix + "slice_extract_si512_1",
                  [val_512, i32(1)],
              ),
          )
      )

    lanes_512 = 512 // 32
    lanes_256 = 256 // 32
    val_512 = f32_a.with_lanes(lanes_512)
    val_256 = f32_a.with_lanes(lanes_256)

    rules.append(
        Rule(
            Op(
                val_256.ty, "slice", [val_512, WildConstant(0), WildConstant(2)]
            ),
            Op(val_256.ty, "wrapper" + prefix + "slice_cast_ps512", [val_512]),
        )
    )
    rules.append(
        Rule(
            Op(
                val_256.ty, "slice", [val_512, WildConstant(1), WildConstant(2)]
            ),
            Op(
                val_256.ty,
                "wrapper" + prefix + "slice_extract_ps512_1",
                [val_512, i32(1)],
            ),
        )
    )
    return rules

  return []


def make_x86_float32_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(f32_a + f32_b, Op(Float(32), prefix + "add_ps", [f32_a, f32_b])),
          Rule(f32_a - f32_b, Op(Float(32), prefix + "sub_ps", [f32_a, f32_b])),
          Rule(f32_a * f32_b, Op(Float(32), prefix + "mul_ps", [f32_a, f32_b])),
          Rule(f32_a / f32_b, Op(Float(32), prefix + "div_ps", [f32_a, f32_b])),
          Rule(
              max(f32_a, f32_b),
              Op(Float(32), prefix + "max_ps", [f32_a, f32_b]),
          ),
          Rule(
              min(f32_a, f32_b),
              Op(Float(32), prefix + "min_ps", [f32_a, f32_b]),
          ),
          Rule(
              ceil(f32_a),
              Op(Float(32), prefix + "ceil_ps", [f32_a]),
              features=["SSE41"],
          ),
          Rule(
              floor(f32_a),
              Op(Float(32), prefix + "floor_ps", [f32_a]),
              features=["SSE41"],
          ),
          Rule(
              sqrt(f32_a),
              Op(Float(32), prefix + "sqrt_ps", [f32_a]),
          ),
          Rule(
              f32_a & f32_b,
              Op(Float(32), prefix + "and_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a | f32_b,
              Op(Float(32), prefix + "or_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a ^ f32_b,
              Op(Float(32), prefix + "xor_ps", [f32_a, f32_b]),
          ),
      ]
  ]


def make_x86_f16c_patterns(vector_bits, prefix):
  # TODO(vksnk): this is just a workaround, because the fp16 vector is shorter
  # than target bit width. This needs a clean-up.
  f32_to_f16_rule = Rule(
      cast(Float(16), f32_a).with_lanes(vector_bits // 32),
      Op(Float(16), "wrapper" + prefix + "cvtps_ph", [f32_a]).with_lanes(
          vector_bits // 32
      ),
      ["F16C"],
  )
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              cast(Float(32), f16_a),
              Op(Float(32), prefix + "cvtph_ps", [f16_a]),
              ["F16C"],
          ),
          Rule(
              cast(Float(16), f32_a),
              Op(Float(16), prefix + "cvtps_ph", [f32_a]),
              ["F16C"],
          ),
      ]
  ] + [f32_to_f16_rule]


def make_x86_fma_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              multiply_add(f32_a, f32_b, f32_c),
              Op(Float(32), prefix + "fmadd_ps", [f32_a, f32_b, f32_c]),
              ["FMA3", "AVX512F"],
          ),
          Rule(
              multiply_sub(f32_a, f32_b, f32_c),
              Op(Float(32), prefix + "fmsub_ps", [f32_a, f32_b, f32_c]),
              ["FMA3", "AVX512F"],
          ),
      ]
  ]


class X86(Target):
  """X86 target for elementwise kernels compiler."""

  def add_load_intrinsics(self, vector_bits, prefix):
    self.load_intrinsics[Int(8, vector_bits // 8)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[UInt(8, vector_bits // 8)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[Int(16, vector_bits // 16)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[UInt(16, vector_bits // 16)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[Int(32, vector_bits // 32)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[UInt(32, vector_bits // 32)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[Float(16, vector_bits // 16)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[BFloat(16, vector_bits // 16)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[Float(32, vector_bits // 32)] = prefix + "loadu_ps"

  def add_store_intrinsics(self, vector_bits, prefix):
    self.store_intrinsics[Int(8, vector_bits // 8)] = (
        "wrapper" + prefix + "storeu_si" + str(vector_bits)
    )
    self.store_intrinsics[UInt(8, vector_bits // 8)] = (
        "wrapper" + prefix + "storeu_si" + str(vector_bits)
    )
    self.store_intrinsics[Int(32, vector_bits // 32)] = (
        "wrapper" + prefix + "storeu_si" + str(vector_bits)
    )
    self.store_intrinsics[Float(16, vector_bits // 16)] = (
        "wrapper" + prefix + "storeu_si" + str(vector_bits)
    )
    self.store_intrinsics[BFloat(16, vector_bits // 16)] = (
        "wrapper" + prefix + "storeu_si" + str(vector_bits)
    )
    self.store_intrinsics[Float(32, vector_bits // 32)] = prefix + "storeu_ps"

  def update_for_sse2(self):
    """Updates the target for SSE2 support."""
    self.types.update({
        Int(8, 16): "__m128i",
        Int(16, 8): "__m128i",
        Int(32, 4): "__m128i",
        UInt(8, 16): "__m128i",
        UInt(16, 8): "__m128i",
        UInt(32, 4): "__m128i",
        Float(32, 4): "__m128",
    })

    self.add_load_intrinsics(128, "_mm_")
    self.add_store_intrinsics(128, "_mm_")
    self.patterns += make_x86_float32_patterns(128, "_mm_")
    self.patterns += make_x86_integer_patterns(128, "_mm_")
    self.patterns += make_x86_cast_patterns(128, "_mm_")
    self.patterns += make_x86_reinterpret_cast_patterns(128, "_mm_")
    self.patterns += make_x86_broadcast_patterns(128, "_mm_")
    self.patterns += make_x86_float_comparison_patterns(128, "_mm_")
    self.patterns += make_x86_integer_comparison_patterns(128, "_mm_")

    self.header += """
namespace {

template <typename T>
YNN_INTRINSIC __m128i wrapper_mm_loadu_si128(const T* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

template <typename T>
YNN_INTRINSIC void wrapper_mm_storeu_si128(T* ptr, __m128i v) {
  _mm_storeu_si128((__m128i*)ptr, v);
}

YNN_INTRINSIC __m128i saturating_cast_f32_to_int8(__m128 f0, __m128 f1, __m128 f2, __m128 f3) {
  const __m128 max_int16 = _mm_set1_ps((1 << 15) - 1);
  f0 = _mm_min_ps(f0, max_int16);
  f1 = _mm_min_ps(f1, max_int16);
  f2 = _mm_min_ps(f2, max_int16);
  f3 = _mm_min_ps(f3, max_int16);
  const __m128i i0 = _mm_cvtps_epi32(f0);
  const __m128i i1 = _mm_cvtps_epi32(f1);
  const __m128i i2 = _mm_cvtps_epi32(f2);
  const __m128i i3 = _mm_cvtps_epi32(f3);
  const __m128i i01_16 = _mm_packs_epi32(i0, i1);
  const __m128i i23_16 = _mm_packs_epi32(i2, i3);
  return _mm_packs_epi16(i01_16, i23_16);
}

YNN_INTRINSIC __m128i saturating_cast_f32_to_int16(__m128 f0, __m128 f1) {
  const __m128 max_int16 = _mm_set1_ps((1 << 15) - 1);
  f0 = _mm_min_ps(f0, max_int16);
  f1 = _mm_min_ps(f1, max_int16);
  const __m128i i0 = _mm_cvtps_epi32(f0);
  const __m128i i1 = _mm_cvtps_epi32(f1);
  return _mm_packs_epi32(i0, i1);
}

YNN_INTRINSIC __m128i saturating_cast_int32_to_int16(__m128i a, __m128i b) {
  return _mm_packs_epi32(a, b);
}

YNN_INTRINSIC __m128i saturating_cast_int16_to_int8(__m128i a, __m128i b) {
  return _mm_packs_epi16(a, b);
}

YNN_INTRINSIC __m128i saturating_cast_int16_to_uint8(__m128i a, __m128i b) {
  return _mm_packus_epi16(a, b);
}

YNN_INTRINSIC __m128i saturating_cast_f32_to_uint8(__m128 f0, __m128 f1, __m128 f2, __m128 f3) {
  const __m128 max_uint16 = _mm_set1_ps((1 << 16) - 1);
  f0 = _mm_min_ps(f0, max_uint16);
  f1 = _mm_min_ps(f1, max_uint16);
  const __m128i i0 = _mm_cvtps_epi32(f0);
  const __m128i i1 = _mm_cvtps_epi32(f1);
  const __m128i i2 = _mm_cvtps_epi32(f2);
  const __m128i i3 = _mm_cvtps_epi32(f3);
  const __m128i i01_16 = _mm_packs_epi32(i0, i1);
  const __m128i i23_16 = _mm_packs_epi32(i2, i3);
  return _mm_packus_epi16(i01_16, i23_16);
}

YNN_INTRINSIC __m128 bitwise_not(__m128 val) {
  __m128 all_ones = _mm_castsi128_ps(_mm_set1_epi32(-1));
  return _mm_xor_ps(val, all_ones);
}

} // namespace
"""

  def update_for_sse41(self):
    """Updates the target for SSE41 support."""
    self.header += """
namespace {

YNN_INTRINSIC __m128 round(__m128 x) {
  return _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

} // namespace

"""
    self.patterns += make_x86_float32_patterns(128, "_mm_")
    self.patterns += make_x86_integer_min_max_patterns(128, "_mm_")

  def update_for_avx(self):
    """Updates the target for AVX support."""
    self.header += """
namespace {

// Mask table used for partial load/store operations.
const int32_t mask_table_avx_f32[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                        0,  0,  0,  0,  0,  0,  0, 0};

YNN_INTRINSIC __m256 partial_load_8x(const float* ptr, size_t num_elements) {
  __m256i mask = _mm256_loadu_si256(
            (const __m256i*)(&mask_table_avx_f32[8] - num_elements));
  return _mm256_maskload_ps(ptr, mask);
}

YNN_INTRINSIC void partial_store_8x(float* output, size_t num_elements, __m256 v) {
  __m256i mask = _mm256_loadu_si256(
            (const __m256i*)(&mask_table_avx_f32[8] - num_elements));
  _mm256_maskstore_ps(output, mask, v);
}

YNN_INTRINSIC __m256i partial_load_8x(const int32_t* ptr, size_t num_elements) {
  return _mm256_castps_si256(partial_load_8x(reinterpret_cast<const float*>(ptr), num_elements));
}

YNN_INTRINSIC void partial_store_8x(int32_t* output, size_t num_elements, __m256i v) {
  partial_store_8x(reinterpret_cast<float*>(output), num_elements, _mm256_castsi256_ps(v));
}

template <typename T>
YNN_INTRINSIC __m256i wrapper_mm256_loadu_si256(const T* ptr) {
  return _mm256_loadu_si256((const __m256i*)ptr);
}

template <typename T>
YNN_INTRINSIC void wrapper_mm256_storeu_si256(T* ptr, __m256i v) {
  _mm256_storeu_si256((__m256i*)ptr, v);
}

YNN_INTRINSIC __m256 round(__m256 x) {
  return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

YNN_INTRINSIC __m256 bitwise_not(__m256 val) {
  __m256 all_ones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
  return _mm256_xor_ps(val, all_ones);
}

YNN_INTRINSIC __m256 greater_than(__m256 a, __m256 b) {
  return _mm256_cmp_ps(a, b, _CMP_GT_OS);
}

template <typename T>
YNN_INTRINSIC __m128i wrapper_mm256_slice_cast_si256(T val, int idx, int total) {
  return _mm256_castsi256_si128((__m256i)val);
}

YNN_INTRINSIC __m128 wrapper_mm256_slice_cast_ps256(__m256 val, int idx, int total) {
  return _mm256_castps256_ps128(val);
}

template <typename T>
YNN_INTRINSIC __m128i wrapper_mm256_slice_extract_si256_1(
    T val, int idx, int total) {
  return _mm256_extracti128_si256((__m256i)val, 1);
}

YNN_INTRINSIC __m128 wrapper_mm256_slice_extract_ps256_1(
    __m256 val, int idx, int total) {
  return _mm256_extractf128_ps(val, 1);
}

} // namespace

"""
    self.types.update({
        Float(32, 8): "__m256",
        Float(16, 8): "__m128i",
        Int(8, 32): "__m256i",
        Int(16, 16): "__m256i",
        Int(32, 8): "__m256i",
        UInt(8, 32): "__m256i",
        UInt(16, 16): "__m256i",
        UInt(32, 8): "__m256i",
        BFloat(16, 16): "__m256i",
        BFloat(16, 8): "__m128i",
    })
    self.add_load_intrinsics(256, "_mm256_")
    self.add_store_intrinsics(256, "_mm256_")
    self.patterns += make_x86_float32_patterns(256, "_mm256_")
    self.patterns += make_x86_reinterpret_cast_patterns(256, "_mm256_")
    self.patterns += make_x86_broadcast_patterns(256, "_mm256_")
    self.patterns += make_x86_slice_patterns(256, "_mm256_")

  def update_for_avx2(self):
    """Updates the target for AVX2 support."""
    self.patterns += make_x86_integer_patterns(256, "_mm256_")
    self.patterns += make_x86_integer_min_max_patterns(256, "_mm256_")
    self.patterns += make_x86_cast_patterns(256, "_mm256_")
    self.patterns += make_x86_bf16_patterns(256)

    self.header += """
namespace {

YNN_INTRINSIC __m256i convert_fp32_to_bf16_avx2(__m256 a, __m256 b) {
  const __m256 rounding_multiplier = _mm256_set1_ps(1.0f + 0.5f / 128.0f);
  a = _mm256_mul_ps(a, rounding_multiplier);
  b = _mm256_mul_ps(b, rounding_multiplier);
  const __m256i ai = _mm256_castps_si256(a);
  const __m256i bi = _mm256_castps_si256(b);
  const __m256i as = _mm256_srli_epi32(ai, 16);
  const __m256i bs = _mm256_srli_epi32(bi, 16);
  const __m256i r = _mm256_packus_epi32(as, bs);
  return _mm256_permute4x64_epi64(r, _MM_SHUFFLE(3, 1, 2, 0));
}

YNN_INTRINSIC __m256 convert_bf16_to_fp32_avx2(__m128i a) {
  const __m256i fp32_integers = _mm256_cvtepu16_epi32(a);
  const __m256i shifted = _mm256_slli_epi32(fp32_integers, 16);
  return _mm256_castsi256_ps(shifted);
}

YNN_INTRINSIC __m256i saturating_cast_f32_to_int8(__m256 f0, __m256 f1, __m256 f2, __m256 f3) {
  const __m256 max_int16 = _mm256_set1_ps((1 << 15) - 1);
  f0 = _mm256_min_ps(f0, max_int16);
  f1 = _mm256_min_ps(f1, max_int16);
  f2 = _mm256_min_ps(f2, max_int16);
  f3 = _mm256_min_ps(f3, max_int16);
  const __m256i i0 = _mm256_cvtps_epi32(f0);
  const __m256i i1 = _mm256_cvtps_epi32(f1);
  const __m256i i2 = _mm256_cvtps_epi32(f2);
  const __m256i i3 = _mm256_cvtps_epi32(f3);
  const __m256i i01_16 = _mm256_packs_epi32(i0, i1);
  const __m256i i23_16 = _mm256_packs_epi32(i2, i3);
  const __m256i r = _mm256_packs_epi16(i01_16, i23_16);
  return _mm256_permutevar8x32_epi32(r, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

YNN_INTRINSIC __m256i saturating_cast_f32_to_int16(__m256 f0, __m256 f1) {
  const __m256 max_int16 = _mm256_set1_ps((1 << 15) - 1);
  f0 = _mm256_min_ps(f0, max_int16);
  f1 = _mm256_min_ps(f1, max_int16);
  const __m256i i0 = _mm256_cvtps_epi32(f0);
  const __m256i i1 = _mm256_cvtps_epi32(f1);
  const __m256i i01_16 = _mm256_packs_epi32(i0, i1);
  return _mm256_permute4x64_epi64(i01_16, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

YNN_INTRINSIC __m256i saturating_cast_int32_to_int16(__m256i a, __m256i b) {
  const __m256i r = _mm256_packs_epi32(a, b);
  return _mm256_permute4x64_epi64(r, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

YNN_INTRINSIC __m256i saturating_cast_int16_to_int8(__m256i a, __m256i b) {
  const __m256i r = _mm256_packs_epi16(a, b);
  return _mm256_permute4x64_epi64(r, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

YNN_INTRINSIC __m256i saturating_cast_int16_to_uint8(__m256i a, __m256i b) {
  const __m256i r = _mm256_packus_epi16(a, b);
  return _mm256_permute4x64_epi64(r, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

YNN_INTRINSIC __m256i saturating_cast_f32_to_uint8(__m256 f0, __m256 f1, __m256 f2, __m256 f3) {
  const __m256 max_uint16 = _mm256_set1_ps((1 << 16) - 1);
  f0 = _mm256_min_ps(f0, max_uint16);
  f1 = _mm256_min_ps(f1, max_uint16);
  f2 = _mm256_min_ps(f2, max_uint16);
  f3 = _mm256_min_ps(f3, max_uint16);
  const __m256i i0 = _mm256_cvtps_epi32(f0);
  const __m256i i1 = _mm256_cvtps_epi32(f1);
  const __m256i i2 = _mm256_cvtps_epi32(f2);
  const __m256i i3 = _mm256_cvtps_epi32(f3);
  const __m256i i01_16 = _mm256_packs_epi32(i0, i1);
  const __m256i i23_16 = _mm256_packs_epi32(i2, i3);
  const __m256i r = _mm256_packus_epi16(i01_16, i23_16);
  return _mm256_permutevar8x32_epi32(r, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

}  // namespace
"""

  def update_for_fma3(self):
    """Updates the target for FMA3 support."""
    self.patterns += make_x86_fma_patterns(256, "_mm256_")

  def update_for_f16c(self):
    """Updates the target for F16C support."""
    self.header += """
namespace {

YNN_INTRINSIC __m128i wrapper_mm256_cvtps_ph(__m256 x) {
  return _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT);
}

} // namespace

"""
    self.patterns += make_x86_f16c_patterns(256, "_mm256_")

  def update_for_avx512f(self):
    """Updates the target for AVX512F support."""
    self.header += """
namespace {

YNN_INTRINSIC __m512 partial_load_16x(const float* ptr, size_t num_elements) {
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1U << num_elements) - 1U));
  return _mm512_maskz_loadu_ps(mask, ptr);
}

YNN_INTRINSIC void partial_store_16x(float* output, size_t num_elements, __m512 v) {
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1U << num_elements) - 1U));
  _mm512_mask_storeu_ps(output, mask, v);
}

YNN_INTRINSIC __m512i partial_load_16x(const int32_t* ptr, size_t num_elements) {
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1U << num_elements) - 1U));
  return _mm512_maskz_loadu_epi32(mask, ptr);
}

YNN_INTRINSIC void partial_store_16x(int32_t* output, size_t num_elements, __m512i v) {
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1U << num_elements) - 1U));
  _mm512_mask_storeu_epi32(output, mask, v);
}

YNN_INTRINSIC __m256i partial_load_16x(const uint16_t* ptr, size_t num_elements) {
  __mmask32 mask = _cvtu32_mask32((uint32_t)((1U << num_elements) - 1U));
  return _mm512_castsi512_si256(_mm512_maskz_loadu_epi16(mask, ptr));
}

template <typename T>
YNN_INTRINSIC __m512i wrapper_mm512_loadu_si512(const T* ptr) {
  return _mm512_loadu_si512((const __m512i*)ptr);
}

template <typename T>
YNN_INTRINSIC void wrapper_mm512_storeu_si512(T* ptr, __m512i v) {
  _mm512_storeu_si512((__m512i*)ptr, v);
}

YNN_INTRINSIC __m512 round(__m512 x) {
  return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

YNN_INTRINSIC __m512i bitwise_not(__m512i val) {
  __m512i all_ones = _mm512_set1_epi32(-1);
  return _mm512_xor_si512(val, all_ones);
}

YNN_INTRINSIC __m512 convert_bf16_to_fp32_avx512(__m256i a) {
  const __m512i fp32_integers = _mm512_cvtepu16_epi32(a);
  const __m512i shifted = _mm512_slli_epi32(fp32_integers, 16);
  return _mm512_castsi512_ps(shifted);
}

template <typename T>
YNN_INTRINSIC __m256i wrapper_mm512_slice_cast_si512(T val, int idx, int total) {
  return _mm512_castsi512_si256((__m512i)val);
}

YNN_INTRINSIC __m256 wrapper_mm512_slice_cast_ps512(__m512 val, int idx, int total) {
  return _mm512_castps512_ps256(val);
}

template <typename T>
YNN_INTRINSIC __m256i wrapper_mm512_slice_extract_si512_1(
    T val, int idx, int total) {
  return _mm512_extracti64x4_epi64((__m512i)val, 1);
}

YNN_INTRINSIC __m256 wrapper_mm512_slice_extract_ps512_1(
    __m512 val, int idx, int total) {
  return _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(val), 1));
}

} // namespace

"""
    self.types.update({
        Int(8, 64): "__m512i",
        Int(16, 32): "__m512i",
        Int(32, 16): "__m512i",
        UInt(8, 64): "__m512i",
        UInt(16, 32): "__m512i",
        UInt(32, 16): "__m512i",
        Float(32, 16): "__m512",
        Float(16, 16): "__m256i",
    })
    self.add_load_intrinsics(512, "_mm512_")
    self.add_store_intrinsics(512, "_mm512_")
    self.patterns += make_x86_fma_patterns(512, "_mm512_")
    self.patterns += make_x86_float32_patterns(512, "_mm512_")
    self.patterns += make_x86_reinterpret_cast_patterns(512, "_mm512_")
    self.patterns += make_x86_broadcast_patterns(512, "_mm512_")
    self.patterns += make_x86_integer_patterns(512, "_mm512_")
    self.patterns += make_x86_integer_min_max_patterns(512, "_mm512_")
    self.patterns += make_x86_cast_patterns(512, "_mm512_")
    self.patterns += make_x86_slice_patterns(512, "_mm512_")

  def update_for_avx512bf16(self):
    """Updates the target for AVX512BF16 support."""
    self.header += """
namespace {

YNN_INTRINSIC __m512i convert_fp32_to_bf16_avx512(__m512 a, __m512 b) {
  return (__m512i)_mm512_cvtne2ps_pbh(b, a);
}

YNN_INTRINSIC void partial_store_32x(bfloat16* output, size_t num_elements, __m512i v) {
  partial_store_32x((int16_t*)output, num_elements, v);
}

YNN_INTRINSIC __m512i partial_load_32x(const uint16_t* ptr, size_t num_elements) {
  return partial_load_32x((const int16_t*)ptr, num_elements);
}

} // namespace

"""
    self.types.update({
        BFloat(16, 32): "__m512i",
    })
    self.patterns += make_x86_bf16_patterns(512)

  def update_for_avx512bw(self):
    """Updates the target for AVX512BW support."""
    self.header += """
namespace {

YNN_INTRINSIC __m512i partial_load_32x(const int16_t* ptr, size_t num_elements) {
  __mmask32 mask = _cvtu32_mask32((uint32_t)((1ULL << num_elements) - 1U));
  return _mm512_maskz_loadu_epi16(mask, ptr);
}

YNN_INTRINSIC void partial_store_32x(int16_t* output, size_t num_elements, __m512i v) {
  __mmask32 mask = _cvtu32_mask32((uint32_t)((1ULL << num_elements) - 1U));
  _mm512_mask_storeu_epi16(output, mask, v);
}

YNN_INTRINSIC void partial_store_64x(int8_t* output, size_t num_elements, __m512i v) {
  __mmask64 mask = _cvtu64_mask64((1ULL << num_elements) - 1ULL);
  _mm512_mask_storeu_epi8(output, mask, v);
}

YNN_INTRINSIC void partial_store_64x(uint8_t* output, size_t num_elements, __m512i v) {
  __mmask64 mask = _cvtu64_mask64((1ULL << num_elements) - 1ULL);
  _mm512_mask_storeu_epi8(output, mask, v);
}

YNN_INTRINSIC __m512i saturating_cast_f32_to_int8(__m512 f0, __m512 f1, __m512 f2, __m512 f3) {
  const __m512 max_int16 = _mm512_set1_ps((1 << 15) - 1);
  f0 = _mm512_min_ps(f0, max_int16);
  f1 = _mm512_min_ps(f1, max_int16);
  f2 = _mm512_min_ps(f2, max_int16);
  f3 = _mm512_min_ps(f3, max_int16);
  const __m512i i0 = _mm512_cvtps_epi32(f0);
  const __m512i i1 = _mm512_cvtps_epi32(f1);
  const __m512i i2 = _mm512_cvtps_epi32(f2);
  const __m512i i3 = _mm512_cvtps_epi32(f3);
  const __m512i i01_16 = _mm512_packs_epi32(i0, i1);
  const __m512i i23_16 = _mm512_packs_epi32(i2, i3);
  const __m512i r = _mm512_packs_epi16(i01_16, i23_16);
  return _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), r);
}

YNN_INTRINSIC __m512i saturating_cast_f32_to_uint8(__m512 f0, __m512 f1, __m512 f2, __m512 f3) {
  const __m512 max_uint16 = _mm512_set1_ps((1 << 16) - 1);
  f0 = _mm512_min_ps(f0, max_uint16);
  f1 = _mm512_min_ps(f1, max_uint16);
  const __m512i i0 = _mm512_cvtps_epi32(f0);
  const __m512i i1 = _mm512_cvtps_epi32(f1);
  const __m512i i2 = _mm512_cvtps_epi32(f2);
  const __m512i i3 = _mm512_cvtps_epi32(f3);
  const __m512i i01_16 = _mm512_packus_epi32(i0, i1);
  const __m512i i23_16 = _mm512_packus_epi32(i2, i3);
  const __m512i r = _mm512_packus_epi16(i01_16, i23_16);
  return _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), r);
}

YNN_INTRINSIC __m512i saturating_cast_f32_to_int16(__m512 f0, __m512 f1) {
  const __m512 max_int16 = _mm512_set1_ps((1 << 15) - 1);
  f0 = _mm512_min_ps(f0, max_int16);
  f1 = _mm512_min_ps(f1, max_int16);
  const __m512i i0 = _mm512_cvtps_epi32(f0);
  const __m512i i1 = _mm512_cvtps_epi32(f1);
  const __m512i r = _mm512_packs_epi32(i0, i1);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

YNN_INTRINSIC __m512i saturating_cast_int32_to_int16(__m512i a, __m512i b) {
  const __m512i r = _mm512_packs_epi32(a, b);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

YNN_INTRINSIC __m512i saturating_cast_int16_to_int8(__m512i a, __m512i b) {
  const __m512i r = _mm512_packs_epi16(a, b);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

YNN_INTRINSIC __m512i saturating_cast_int16_to_uint8(__m512i a, __m512i b) {
  const __m512i r = _mm512_packus_epi16(a, b);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

} // namespace

"""

  def __init__(self, features):
    Target.__init__(self)
    self.features = features
    self.load_intrinsics = {}
    self.store_intrinsics = {}
    # These are transitive.
    implied_features = {
        "SSE41": ["SSE2"],
        "AVX": ["SSE41"],
        "AVX2": ["AVX"],
        "F16C": ["AVX"],
        "FMA3": ["AVX"],
        "AVX512F": ["AVX2", "FMA3"],
        "AVX512BW": ["AVX512F"],
        "AVX512BF16": ["AVX512BW"],
    }
    all_features = []
    self.compute_all_features(features, implied_features, all_features)

    self.header += "#include <immintrin.h>\n"

    known_features = [
        "SSE2",
        "SSE41",
        "AVX",
        "AVX2",
        "FMA3",
        "F16C",
        "AVX512F",
        "AVX512BW",
        "AVX512BF16",
    ]
    for feature in all_features:
      if feature not in known_features:
        raise ValueError(f"Unknown feature: {feature}")

    if "AVX512F" in all_features:
      self.tail_strategy = TailStrategy.MASK
      self.vector_bits = 512
    elif "AVX" in all_features:
      self.tail_strategy = TailStrategy.MEMCPY
      self.vector_bits = 256
    elif "SSE2" in all_features:
      self.tail_strategy = TailStrategy.MEMCPY
      self.vector_bits = 128

    if "AVX512BW" in all_features:
      self.update_for_avx512bw()
    if "AVX512BF16" in all_features:
      self.update_for_avx512bf16()
    if "AVX512F" in all_features:
      self.update_for_avx512f()
    if "FMA3" in all_features:
      self.update_for_fma3()
    if "F16C" in all_features:
      self.update_for_f16c()
    if "AVX2" in all_features:
      self.update_for_avx2()
    if "AVX" in all_features:
      self.update_for_avx()
    if "SSE41" in all_features:
      self.update_for_sse41()
    if "SSE2" in all_features:
      self.update_for_sse2()
