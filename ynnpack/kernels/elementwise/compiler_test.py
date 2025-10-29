import unittest

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


class RewriteTest(unittest.TestCase):
  """Check that that rewrite produces expressions we're expecting."""

  def test_simple(self):
    r = rewrite(f32_a + f32_b, Op(Float(32), "test_add", [f32_a, f32_b]), x + y)
    self.assertTrue(compare_ops(r, Op(Float(32), "test_add", [x, y])))

    r = rewrite(
        f32_a * f32_b + f32_c, multiply_add(f32_a, f32_b, f32_c), x * y + z
    )
    self.assertTrue(compare_ops(r, multiply_add(x, y, z)))

    r = rewrite(
        f32_a * f32_b - f32_c, multiply_sub(f32_a, f32_b, f32_c), x * y - z
    )
    self.assertTrue(compare_ops(r, multiply_sub(x, y, z)))

    r = rewrite(
        f32_a * f32_b - f32_c, multiply_sub(f32_a, f32_b, f32_c), x * y + z
    )
    self.assertIsNone(r)


class FindIntrinsicsTest(unittest.TestCase):
  """Check that that find_intrinsics produces expressions we're expecting."""

  def test_simple(self):
    r = find_intrinsics(f32_a * f32_b + f32_c)
    self.assertTrue(compare_ops(r, multiply_add(f32_a, f32_b, f32_c)))

    r = find_intrinsics(i32_a * i32_b + i32_c)
    self.assertTrue(compare_ops(r, multiply_add(i32_a, i32_b, i32_c)))

    r = find_intrinsics((i32_a + i32_b) / i32(2))
    self.assertTrue(compare_ops(r, halving_add(i32_a, i32_b)))

    r = find_intrinsics((i32_a + i32_b) / i32(3))
    self.assertIsNone(r)


class TestTarget(Target):
  """Test target for elementwise kernels compiler."""

  def __init__(self):
    Target.__init__(self)
    self.features = []
    self.load_intrinsics = {}
    self.store_intrinsics = {}

    self.tail_strategy = None
    self.vector_bits = 128

  def get_natural_lanes_num(self, ty):
    return self.vector_bits // ty.size


def sample_func():
  """Just a test function (it's an incomplete implementation of tanh)."""
  vmax_x = 7.9807181358e00
  vmin_x = -7.9807181358e00

  valpha_3 = 1.3412411511e-01
  valpha_5 = 3.5330520477e-03
  valpha_7 = 2.1235626264e-05
  valpha_9 = 1.4248920266e-08

  va = load(f32_a)

  vx = min(vmax_x, va)
  vx = max(vmin_x, vx)

  vx2 = vx * vx

  vp = vx2 * valpha_9 + valpha_7
  vp = vx2 * vp + valpha_5
  vp = vx2 * vp + valpha_3
  vp = vx2 * vp + 1.0
  return vx * vp


class ExpressionCachingTest(unittest.TestCase):
  """Check that passes don't introduce multiple objects for the same expression."""

  def setUp(self):
    super().setUp()
    self.target = TestTarget()

  def count_objects(self, expr):
    objects = set()

    def count_objects_impl(e):
      if isinstance(e, Op):
        for arg in e.args:
          count_objects_impl(arg)

      objects.add(e)

    count_objects_impl(expr)
    return len(objects)

  def test_vectorize(self):
    natural_lanes = 16

    c = sample_func()
    c_object_count = self.count_objects(c)

    mc = self.target.vectorize(c, natural_lanes, {})
    mc_object_count = self.count_objects(mc)

    # We need to add a number of constants in the program, because they'll be
    # broadcasted in the vectorize.
    constant_count = 7
    self.assertEqual(mc_object_count, c_object_count + constant_count)

  def test_slice_wide_types(self):
    natural_lanes = 4

    c = sample_func()
    c_object_count = self.count_objects(c)

    # At the moment slice_wide_type expects types to have number of lanes
    # divisible by the natural lanes count.
    mc = self.target.vectorize(c, natural_lanes, {})
    mc = self.target.slice_wide_types(mc, {})
    mc_object_count = self.count_objects(mc)

    # We need to add a number of constants in the program, because they'll be
    # broadcasted in the vectorize.
    constant_count = 7
    self.assertEqual(mc_object_count, c_object_count + constant_count)

  def test_optimize_slices(self):
    c = sample_func()
    c_object_count = self.count_objects(c)

    mc = self.target.optimize_slices(c, {})
    mc_object_count = self.count_objects(mc)

    self.assertEqual(mc_object_count, c_object_count)

  def test_pattern_match(self):
    c = sample_func()
    c_object_count = self.count_objects(c)

    mc = self.target.pattern_match(c, {})
    mc_object_count = self.count_objects(mc)

    # There are no patterns defined in the test target, so we should get the
    # same number of objects.
    self.assertEqual(mc_object_count, c_object_count)

  # lift_constants doesn't need and doesn't have caching, because it's only
  # supposed to modify leaf nodes, but it's still good to have the test to make
  # sure that nothing is broken.
  def test_lift_constants(self):
    c = sample_func()
    c_object_count = self.count_objects(c)

    mc = self.target.lift_constants(c, {})
    mc_object_count = self.count_objects(mc)

    self.assertEqual(mc_object_count, c_object_count)


class LiftConstantsTest(unittest.TestCase):
  """Check that lift constant pass does the right thing."""

  def setUp(self):
    super().setUp()
    self.target = TestTarget()

  def test_no_constants(self):
    c = (f32_a + f32_b) * f32_c
    constants = {}
    mc = self.target.vectorize(c, lanes=4, cache={})
    mc = self.target.lift_constants(mc, constants)
    self.assertEqual(len(constants), 0)

  def test_few_constants(self):
    c = (f32_a + f32_b * 2.0) * f32_c + 1.0
    constants = {}
    mc = self.target.vectorize(c, lanes=4, cache={})
    mc = self.target.lift_constants(mc, constants)
    self.assertEqual(len(constants), 2)

  def test_many_constants(self):
    c = sample_func()
    constants = {}
    mc = self.target.vectorize(c, lanes=4, cache={})
    mc = self.target.lift_constants(mc, constants)
    self.assertEqual(len(constants), 7)

  def test_repeating_constants(self):
    c = (f32_a * 1.0 + (f32_b - 1.0) * 2.0) * f32_c + 1.0
    constants = {}
    mc = self.target.vectorize(c, lanes=4, cache={})
    mc = self.target.lift_constants(mc, constants)
    self.assertEqual(len(constants), 2)


if __name__ == "__main__":
  unittest.main()
