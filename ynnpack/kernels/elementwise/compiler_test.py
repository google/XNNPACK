# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


# Placeholders used for pattern matching.
i8_a = Var("a", Int(8))
i8_b = Var("b", Int(8))
u8_a = Var("a", UInt(8))
u8_b = Var("b", UInt(8))
i16_a = Var("a", Int(16))
i16_b = Var("b", Int(16))
u16_a = Var("a", UInt(16))
u16_b = Var("b", UInt(16))
i32_a = Var("a", Int(32))
i32_b = Var("b", Int(32))
i32_c = Var("c", Int(32))
u32_a = Var("a", UInt(32))
u32_b = Var("b", UInt(32))
f32_a = Var("a", Float(32))
f32_b = Var("b", Float(32))
f32_c = Var("c", Float(32))


def compare_ops(a, b):
  if a.name != b.name and a.ty == b.ty and len(a.args) == len(b.args):
    return False
  for a_arg, b_arg in zip(a.args, b.args):
    if a_arg != b_arg:
      return False
  return True

x = Var("x", Float(32))
y = Var("y", Float(32))
z = Var("z", Float(32))

r = rewrite(f32_a + f32_b, Op(Float(32), "test_add", [f32_a, f32_b]), x + y)
assert(compare_ops(r, Op(Float(32), "test_add", [x, y])))

r = rewrite(
    f32_a * f32_b + f32_c, multiply_add(f32_a, f32_b, f32_c), x * y + z
)
assert(compare_ops(r, multiply_add(x, y, z)))

r = rewrite(
    f32_a * f32_b - f32_c, multiply_sub(f32_a, f32_b, f32_c), x * y - z
)
assert(compare_ops(r, multiply_sub(x, y, z)))

r = rewrite(
    f32_a * f32_b - f32_c, multiply_sub(f32_a, f32_b, f32_c), x * y + z
)
assert(r is None)

r = find_intrinsics(f32_a * f32_b + f32_c)
assert(compare_ops(r, multiply_add(f32_a, f32_b, f32_c)))

r = find_intrinsics(i32_a * i32_b + i32_c)
assert(compare_ops(r, multiply_add(i32_a, i32_b, i32_c)))

r = find_intrinsics((i32_a + i32_b) / i32(2))
assert(compare_ops(r, halving_add(i32_a, i32_b)))

r = find_intrinsics((i32_a + i32_b) / i32(3))
assert(r is None)

