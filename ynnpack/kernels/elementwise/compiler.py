"""Generator for elementwise kernels.

Enables specification of kernels as pure python functions, and generating C++
implementations using SIMD intrinsics.
"""

import builtins
import copy
import enum
import functools
import math
import types


class Type:

  def __init__(self, type_class, size, lanes):
    self.type_class = type_class
    self.size = size
    self.lanes = lanes

  def __hash__(self):
    return hash((self.type_class, self.size, self.lanes))

  def __eq__(self, other):
    return (self.type_class, self.size, self.lanes) == (
        other.type_class,
        other.size,
        other.lanes,
    )

  def is_int(self):
    return self.type_class == "int"

  def is_uint(self):
    return self.type_class == "uint"

  def is_float(self):
    return self.type_class in ("float", "bfloat")

  def __str__(self):
    return f"{self.type_class}{self.size}x{self.lanes}_t"

  def __repr__(self):
    return str(self)

  def widen(self):
    return Type(self.type_class, self.size * 2, self.lanes)

  def narrow(self):
    return Type(self.type_class, self.size // 2, self.lanes)

  def signed_widen(self):
    type_class = "int" if self.is_uint() else self.type_class
    return Type(type_class, self.size * 2, self.lanes)

  def with_lanes(self, lanes):
    return Type(self.type_class, self.size, lanes)

  def with_size(self, size):
    return Type(self.type_class, size, self.lanes)

  def to_c_decl(self, const, indirection=0):
    result = ""
    if const:
      result += "const "
    if self.size == 0:
      result += "ssize_t" if self.is_int() else "size_t"
    else:
      result += self.type_class + str(self.size)
      if self.lanes > 1:
        result += "x" + str(self.lanes)
      result += "_t"
    while indirection > 0:
      result += "*"
      indirection -= 1
    return result

  def min(self):
    if self.is_int():
      return -(2 ** (self.size - 1))
    elif self.is_uint():
      return 0
    elif self.is_float():
      return -float("inf")
    else:
      assert False

  def max(self):
    if self.is_int():
      return 2 ** (self.size - 1) - 1
    elif self.is_uint():
      return 2**self.size - 1
    elif self.is_float():
      return float("inf")
    else:
      assert False


def Index():
  return Type("size_t", 0, 0)


def Int(size=0, lanes=1):
  return Type("int", size, lanes)


def UInt(size=0, lanes=1):
  return Type("uint", size, lanes)


def Float(size, lanes=1):
  return Type("float", size, lanes)


def BFloat(size, lanes=1):  # pylint: disable=invalid-name
  return Type("bfloat", size, lanes)


fn_ops = []


def wrap(y):
  result = y
  if isinstance(y, int):
    result = Constant(Int(32), y)
  elif isinstance(y, float):
    result = Constant(Float(32), y)
  return result


def promote_types(a, b):
  """Makes x and y to have compatible types."""
  if a.ty is None and b.ty is None:
    return [a, b]

  assert a.ty.lanes == b.ty.lanes or a.ty.lanes == 1 or b.ty.lanes == 1

  if a.ty == b.ty:
    return [a, b]

  if isinstance(a, Constant) or isinstance(b, Constant):
    if not isinstance(a, Constant):
      return [a, cast(a.ty, b)]
    elif not isinstance(b, Constant):
      return [cast(b.ty, a), b]
    else:
      assert False

  if a.ty.is_float() or b.ty.is_float():
    max_size = builtins.max(a.ty.size, b.ty.size)
    max_lanes = builtins.max(a.ty.lanes, b.ty.lanes)

    return [
        cast(Float(max_size, max_lanes), a),
        cast(Float(max_size, max_lanes), b),
    ]
  else:
    assert a.ty.type_class == b.ty.type_class
    promoted_ty = Type(
        a.ty.type_class,
        builtins.max(a.ty.size, b.ty.size),
        builtins.max(a.ty.lanes, b.ty.lanes),
    )
    return [
        cast(promoted_ty, a),
        cast(promoted_ty, b),
    ]


def get_cmp_type(x):
  return x.ty


class Value:

  def __init__(self, ty):
    self.ty = ty

  def __add__(self, y):
    return Op(self.ty, "add", promote_types(self, wrap(y)))

  def __radd__(self, y):
    return Op(self.ty, "add", promote_types(self, wrap(y)))

  def __sub__(self, y):
    return Op(self.ty, "sub", promote_types(self, wrap(y)))

  def __rsub__(self, y):
    return Op(self.ty, "sub", promote_types(wrap(y), self))

  def __mul__(self, y):
    return Op(self.ty, "mul", promote_types(self, wrap(y)))

  def __rmul__(self, y):
    return Op(self.ty, "mul", promote_types(self, wrap(y)))

  def __truediv__(self, y):
    return Op(self.ty, "truediv", promote_types(self, wrap(y)))

  def __rtruediv__(self, y):
    return Op(self.ty, "truediv", promote_types(wrap(y), self))

  def __floordiv__(self, y):
    return Op(self.ty, "floordiv", promote_types(self, wrap(y)))

  def __rfloordiv__(self, y):
    return Op(self.ty, "floordiv", promote_types(wrap(y), self))

  def __mod__(self, y):
    return Op(self.ty, "mod", promote_types(self, wrap(y)))

  def __rmod__(self, y):
    return Op(self.ty, "mod", promote_types(wrap(y), self))

  def __pow__(self, y):
    return Op(self.ty, "pow", promote_types(self, wrap(y)))

  def __rpow__(self, y):
    return Op(self.ty, "pow", promote_types(wrap(y), self))

  def __rshift__(self, y):
    return Op(self.ty, "rshift", promote_types(self, wrap(y)))

  def __rrshift__(self, y):
    return Op(self.ty, "rshift", promote_types(wrap(y), self))

  def __lshift__(self, y):
    return Op(self.ty, "lshift", promote_types(self, wrap(y)))

  def __rlshift__(self, y):
    return Op(self.ty, "lshift", promote_types(wrap(y), self))

  def __and__(self, y):
    return Op(self.ty, "bitwise_and", promote_types(self, wrap(y)))

  def __rand__(self, y):
    return Op(self.ty, "bitwise_and", promote_types(self, wrap(y)))

  def __or__(self, y):
    return Op(self.ty, "bitwise_or", promote_types(self, wrap(y)))

  def __ror__(self, y):
    return Op(self.ty, "bitwise_or", promote_types(self, wrap(y)))

  def __xor__(self, y):
    return Op(self.ty, "bitwise_xor", promote_types(self, wrap(y)))

  def __rxor__(self, y):
    return Op(self.ty, "bitwise_xor", promote_types(self, wrap(y)))

  def __invert__(self):
    return Op(self.ty, "bitwise_not", [self])

  def __neg__(self):
    return Constant(self.ty, 0) - self

  def __lt__(self, y):
    x_pr, y_pr = promote_types(self, wrap(y))
    return Op(get_cmp_type(x_pr), "less_than", [x_pr, y_pr])

  def __le__(self, y):
    x_pr, y_pr = promote_types(self, wrap(y))
    return Op(get_cmp_type(x_pr), "less_equal", [x_pr, y_pr])

  def __gt__(self, y):
    x_pr, y_pr = promote_types(self, wrap(y))
    opp = Op(get_cmp_type(x_pr), "greater_than", [x_pr, y_pr])
    return opp

  def __ge__(self, y):
    x_pr, y_pr = promote_types(self, wrap(y))
    return Op(get_cmp_type(x_pr), "greater_equal", [x_pr, y_pr])


class WildCard(Value):

  def __init__(self):
    Value.__init__(self, None)


class WildConstant(Value):

  def __init__(self, value=None):
    Value.__init__(self, None)
    self.value = value


class Op(Value):

  def __init__(self, ty, name, args):
    assert isinstance(args, list)
    Value.__init__(self, ty)
    self.name = name
    self.args = args
    fn_ops.append(self)

  def __repr__(self):
    return f"{self.name}<{self.ty}>({', '.join(map(repr, self.args))})"

  def compare(self, other):
    return self.name == other.name and self.ty == other.ty

  def with_lanes(self, lanes):
    assert self.name != "combine" and self.name != "slice", self
    return Op(
        self.ty.with_lanes(lanes),
        self.name,
        [i.with_lanes(lanes) for i in self.args],
    )


class Var(Value):

  def __init__(self, name, ty):
    Value.__init__(self, ty)
    self.name = name

  def __str__(self):
    return self.name

  def __repr__(self):
    return f"{self.name}:{self.ty}"

  def __hash__(self):
    return hash((self.name, self.ty))

  def __eq__(self, other):
    if isinstance(other, Var):
      return self.name == other.name and self.ty == other.ty
    return False

  def compare(self, other):
    if type(self) != type(other):
      return False

    return self.name == other.name and self.ty == other.ty

  def with_lanes(self, lanes):
    return Var(self.name, self.ty.with_lanes(lanes))


class Load(Value):
  """A load node."""

  def __init__(self, ty, index, offset_elements):
    Value.__init__(self, ty)
    self.index = index
    self.offset_elements = offset_elements

  def __repr__(self):
    return f"load<{self.ty}>({self.index}, {self.offset_elements})"

  def with_lanes(self, lanes):
    return Load(self.ty.with_lanes(lanes), self.index, self.offset_elements)

  def compare(self, other):
    if type(self) is not type(other):
      return False

    return (
        self.index == other.index
        and self.offset_elements == other.offset_elements
        and self.ty == other.ty
    )


class Store:

  def __init__(self, value, to):
    self.value = value
    self.to = to


def intrinsic(func):
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    args = [wrap(arg) for arg in args]
    kwargs = {k: wrap(v) for k, v in kwargs.items()}
    return func(*args, **kwargs)

  return wrapper


@intrinsic
def equal(self, y):
  x_pr, y_pr = promote_types(self, wrap(y))
  return Op(get_cmp_type(x_pr), "equal", [x_pr, y_pr])


@intrinsic
def not_equal(self, y):
  x_pr, y_pr = promote_types(self, wrap(y))
  return Op(get_cmp_type(x_pr), "not_equal", [x_pr, y_pr])


@intrinsic
def abs(value):
  return Op(value.ty, "abs", [value])


def lower_abs(x):
  assert x.ty.is_float()
  return x & reinterpret_cast(Float(32), i32(0x7FFFFFFF))


@intrinsic
def logical_shift_left(x, shift):
  return Op(x.ty, "logical_shift_left", [x, shift])


@intrinsic
def round(value):
  return Op(value.ty, "round", [value])


@intrinsic
def ceil(value):
  return Op(value.ty, "ceil", [value])


@intrinsic
def floor(value):
  return Op(value.ty, "floor", [value])


@intrinsic
def sqrt(value):
  return Op(value.ty, "sqrt", [value])


@intrinsic
def cast(ty, value):
  """Casts value to a given type."""
  # This is no-op.
  if value.ty == ty:
    return value
  # We can just change the type of the constant.
  if isinstance(value, Constant):
    const = Constant(ty.with_lanes(1), value.value)
    # This is just so broadcast is inserted.
    if ty.lanes > 1:
      const = const.with_lanes(ty.lanes)
    return const
  if (
      value.ty.type_class == ty.type_class
      and ty.size == value.ty.size
      and ty.lanes > 1
      and value.ty.lanes == 1
  ):
    # If the types only differ by number of lanes we can insert a broadcast
    # instead of the full cast.
    return broadcast(value, ty.lanes)
  return Op(ty, "cast", [value])


@intrinsic
def saturating_cast(ty, value):
  return Op(ty, "saturating_cast", [value])


@intrinsic
def reinterpret_cast(ty, value):
  assert ty.size == value.ty.size and ty.lanes == value.ty.lanes
  return Op(ty, "reinterpret_cast", [value])


def widen(x):
  return cast(x.ty.widen(), x)


def signed_widen(x):
  return cast(x.ty.signed_widen(), x)


# Computes signed_widen(x) - signed_widen(y)
@intrinsic
def widening_sub(x, y):
  assert x.ty == y.ty
  return Op(x.ty.signed_widen(), "widening_sub", [x, y])


# Computes widen(x) * widen(y)
@intrinsic
def widening_mul(x, y):
  assert x.ty == y.ty
  return Op(x.ty.widen(), "widening_mul", [x, y])


# Computes saturating_narrow(widen(x) + widen(y))
@intrinsic
def saturating_add(x, y):
  assert x.ty == y.ty
  return Op(x.ty, "saturating_add", [x, y])


@intrinsic
def saturating_sub(x, y):
  assert x.ty == y.ty
  return Op(x.ty, "saturating_sub", [x, y])


# Computes x * y + z
@intrinsic
def multiply_add(x, y, z):
  assert x.ty == y.ty
  assert x.ty == z.ty
  return Op(x.ty, "multiply_add", [x, y, z])


# Computes x * y - z
@intrinsic
def multiply_sub(x, y, z):
  assert x.ty == y.ty
  assert x.ty == z.ty
  return Op(x.ty, "multiply_sub", [x, y, z])


def halving_add(x, y):
  assert x.ty == y.ty
  return Op(x.ty, "halving_add", [x, y])


def rounding_halving_add(x, y):
  assert x.ty == y.ty
  return Op(x.ty, "rounding_halving_add", [x, y])


# Computes saturating_narrow((widen(x) + (1 << (shift - 1))) >> shift)
@intrinsic
def rounding_narrowing_shift_right(x, shift):
  return Op(x.ty.narrow(), "rounding_narrowing_shift_right", [x, shift])


@intrinsic
def saturating_narrow(x):
  return Op(x.ty.narrow(), "saturating_narrow", [x])


@intrinsic
def min(x, y):
  assert x.ty == y.ty
  return Op(wrap(x).ty, "min", [x, y])


@intrinsic
def max(x, y):
  assert x.ty == y.ty
  return Op(x.ty, "max", [x, y])


@intrinsic
def select_bits(mask, x, y):
  return Op(x.ty, "select_bits", [mask, x, y])


def lower_select_bits(mask, x, y):
  return y ^ ((x ^ y) & mask)


@intrinsic
def select(cond, x, y):
  assert x.ty == y.ty
  return Op(x.ty, "select", [cond, x, y])


# We assume that cond produces a mask.
def lower_select(cond, x, y):
  return (x & cond) | (y & ~cond)


def load(x, offset_index=0):
  return Load(x.ty, x, offset_index)


def store(value, to):
  return Store(value, to)


def lower_widening_sub(x, y):
  return signed_widen(x) - signed_widen(y)


def lower_widening_mul(x, y):
  return widen(x) * widen(y)


def lower_saturating_add(x, y):
  return saturating_narrow(widen(x) + widen(y))


def lower_multiply_add(x, y, z):
  return x * y + z


def lower_multiply_sub(x, y, z):
  return x * y - z


def broadcast(x, lanes):
  """Broadcasts a scalar to a vector."""
  assert x.ty.lanes == 1
  x = wrap(x)
  # TODO(vksnk): here we push broadcast into the op. Usually, you would want to
  # the opposite, but this simplifies a lot of things. Probably, would make
  # sense to reconsider in the future.
  if isinstance(x, Op):
    return Op(
        x.ty.with_lanes(lanes),
        x.name,
        [broadcast(arg, lanes) for arg in x.args],
    )
  return Op(x.ty.with_lanes(lanes), "broadcast", [x])


def combine_vectors(args):
  total_lanes = 0
  for arg in args:
    assert arg.ty == args[0].ty
    total_lanes += arg.ty.lanes

  return Op(args[0].ty.with_lanes(total_lanes), "combine", args)


def slice_vector(arg, index, total):
  sliced_ty = arg.ty.with_lanes(arg.ty.lanes // total)
  return Op(sliced_ty, "slice", [arg, i32(index), i32(total)])


# Would it be bad if instead of using a map we looked up in a global namespace
# if there is a function called "lower" + op.name?
lowering_funcs = {
    "abs": lower_abs,
    "widening_sub": lower_widening_sub,
    "widening_mul": lower_widening_mul,
    "saturating_add": lower_saturating_add,
    "select_bits": lower_select_bits,
    "select": lower_select,
    "multiply_add": lower_multiply_add,
    "multiply_sub": lower_multiply_sub,
}


class Constant(Value):
  """Represents a constant value."""

  def __init__(self, ty, value):
    Value.__init__(self, ty)
    self.value = value

  def __str__(self):
    return f"{self.value}"

  def __repr__(self):
    return f"Constant({str(self)})"

  def __getattr__(self, name):
    # This is just for convenience, so every potential arg has a 'name' field.
    if name == "name":
      return str(self.value)
    return object.__getattribute__(self, name)

  def with_lanes(self, lanes):
    return broadcast(self, lanes)


def i8(value):
  return Constant(Int(8), value)


def i16(value):
  return Constant(Int(16), value)


def i32(value):
  return Constant(Int(32), value)


def f32(value):
  return Constant(Float(32), value)


class BroadcastMode(enum.Enum):
  """Defines how a broadcast of the buffer should be handled."""

  # The buffer is never broadcasted.
  NONE = 1
  # The buffer is always broadcasted.
  ALWAYS = 2
  # The two copies of body are created - one with broadcast and one without.
  # This option is most efficent, but doubles the code size.
  SPECIALIZE = 3
  # The broadcast is handled through making a local copy of the broadcasted
  # value and reading it as if it was a normal buffer. This has almost no
  # code size penalty, but might be slower than `specialize` option.
  LOCAL_VAR = 4
  # Let compiler decide which broadcasting mode to use.
  AUTO = 5


class Buffer:

  def __init__(
      self, name, ty, is_const, broadcast_mode=BroadcastMode.LOCAL_VAR
  ):
    self.name = name
    self.ty = ty
    self.is_const = is_const
    self.broadcast_mode = broadcast_mode


buffer_args = []
op_name = "unknown"
code = ""

header = """
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>

using bfloat16 = uint16_t;

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

template <typename T>
YNN_INTRINSIC T load(T* ptr) {
    return ptr[0];
}

template <typename T>
YNN_INTRINSIC void store(T* ptr, T val) {
    ptr[0] = val;
}

template <typename T>
YNN_INTRINSIC T add(T a, T b) {
    return a + b;
}

template <typename T>
YNN_INTRINSIC T sub(T a, T b) {
    return a - b;
}

template <typename T>
YNN_INTRINSIC T mul(T a, T b) {
    return a * b;
}

template <typename T>
YNN_INTRINSIC T truediv(T a, T b) {
    return a / b;
}

} // namespace
} // namespace ynn

"""


def scalar(name, ty):
  def actual_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # fn_args.append((name, ty, 0))
      args += (Var(name, ty),)
      return func(*args, **kwargs)

    return wrapper

  return actual_decorator


def buffer(name, ty, is_const=False, broadcast_mode=BroadcastMode.NONE):
  def actual_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      buffer_args.append(Buffer(name, ty, is_const, broadcast_mode))
      # fn_args.append((name, ty, 1))
      args += (Var(name, ty),)
      return func(*args, **kwargs)

    return wrapper

  return actual_decorator


def const_buffer(name, ty, broadcast_mode=BroadcastMode.AUTO):
  return buffer(name, ty, is_const=True, broadcast_mode=broadcast_mode)


def struct(type_name, name, fields):
  def actual_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # fn_args.append((name, type_name, fields))
      obj = types.SimpleNamespace()
      for k, v in fields.items():
        setattr(obj, k, Var(k, v))
      args += (obj,)
      return func(*args, **kwargs)

    return wrapper

  return actual_decorator


def operator_name(name):
  def actual_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      global op_name
      op_name = name
      return func(*args, **kwargs)

    return wrapper

  return actual_decorator


class TailStrategy(enum.Enum):
  SCALAR = 1
  MEMCPY = 2
  MASK = 3


def match(pattern, expr, matches):
  """Matches an expression against a pattern."""
  if isinstance(pattern, Op):
    if (
        not isinstance(expr, Op)
        or (pattern.ty is not None and pattern.ty != expr.ty)
        or pattern.name != expr.name
        or len(pattern.args) != len(expr.args)
    ):
      return False
  elif isinstance(pattern, Var):
    if pattern.ty == expr.ty:
      matches.append(expr)
      return True
    else:
      return False
  elif isinstance(pattern, WildCard):
    matches.append(expr)
    return True
  elif isinstance(pattern, WildConstant):
    if isinstance(expr, Constant) and (
        pattern.value is None or expr.value == pattern.value
    ):
      matches.append(expr)
      return True
    return False
  else:
    assert False
  for p_arg, e_arg in zip(pattern.args, expr.args):
    matched = match(p_arg, e_arg, matches)
    if not matched:
      return False

  return True


def rewrite(pattern, replacement, expr):
  """Rewrites an expression based on a pattern and result."""
  matches = []
  matched = match(pattern, expr, matches)
  if matched:
    return Op(replacement.ty, replacement.name, matches)
  else:
    return None


def find_intrinsics(e):
  """Looks for a known patterns in an expression."""
  x = WildCard()
  y = WildCard()
  z = WildCard()

  if (r := rewrite(x * y + z, multiply_add(x, y, z), e)) is not None:
    return r
  if (
      r := rewrite((x + y) / WildConstant(2), halving_add(x, y), e)
  ) is not None:
    return r

  return r


class Target:

  def __init__(self):
    self.indent_level = 0
    self.patterns = []
    self.types = {
        Int(8, 1): "int8_t",
        Int(16, 1): "int16_t",
        Int(32, 1): "int32_t",
        UInt(8, 1): "uint8_t",
        UInt(16, 1): "uint16_t",
        UInt(32, 1): "uint32_t",
        Float(16, 1): "half",
        Float(32, 1): "float",
        BFloat(16, 1): "uint16_t",
    }
    self.features = []
    self.load_intrinsics = {}
    self.store_intrinsics = {}
    self.vector_bits = 0
    self.tail_strategy = TailStrategy.SCALAR
    self.result = ""
    self.header = header

  def indent(self):
    return "  " * self.indent_level

  def get_natural_lanes_num(self, ty):
    """Returns a number of lanes of the widest type registered in target's types list which also matches class and size of the type."""
    lanes = 1
    for t in self.types:
      if t.type_class == ty.type_class and t.size == ty.size:
        lanes = builtins.max(lanes, t.lanes)

    return lanes

  def as_buffer(self, arg, buffers):
    b = None
    if isinstance(arg, Var):
      b = next((buf for buf in buffers if buf.name == arg.name), None)
    return b

  def compute_all_features(self, features, implied_features, all_features):
    for feature in features:
      if feature in all_features:
        continue
      all_features.append(feature)

      if feature in implied_features:
        self.compute_all_features(
            implied_features[feature], implied_features, all_features
        )

  def legalize_type(self, ty, is_const=True):
    return self.types.get(ty, ty.to_c_decl(is_const))

  def vectorize(self, expr, lanes, cache):
    if expr in cache:
      return cache[expr]

    v = None
    if (
        isinstance(expr, Var)
        or isinstance(expr, Constant)
        or isinstance(expr, Load)
    ):
      v = expr.with_lanes(lanes)
    else:
      # TODO(vksnk): this needs a proper fix, for example, ops checking the types
      # and doing broadcasting when neccessary.
      if expr.name == "broadcast":
        return expr
      v = Op(
          expr.ty.with_lanes(lanes),
          expr.name,
          [self.vectorize(i, lanes, cache) for i in expr.args],
      )

    cache[expr] = v
    return v

  def slice_wide_types(self, expr, cache):
    """Recursively iterate over the expression and slice it into chunks matching the target platform vector size."""
    if expr in cache:
      return cache[expr]

    if isinstance(expr, Var) or isinstance(expr, Constant):
      v = expr
    else:
      natural_lanes = self.get_natural_lanes_num(expr.ty)
      # If the number of lanes is less than what hardware vector has there is
      # nothing to slice.
      slices_num = builtins.max(expr.ty.lanes // natural_lanes, 1)

      args = []
      if not isinstance(expr, Load):
        for arg in expr.args:
          if isinstance(arg, Op) and arg.name == "broadcast":
            args.append(arg)
          else:
            args.append(self.slice_wide_types(arg, cache))

      if slices_num != 1:
        op_slices = []
        if isinstance(expr, Load):
          for slice_index in range(slices_num):
            op_slices.append(
                Load(
                    expr.ty.with_lanes(natural_lanes),
                    expr.index,
                    expr.offset_elements + natural_lanes * slice_index,
                )
            )
        else:
          for slice_index in range(slices_num):
            op_slices.append(
                Op(
                    expr.ty.with_lanes(natural_lanes),
                    expr.name,
                    [
                        slice_vector(arg, slice_index, slices_num)
                        for arg in args
                    ],
                )
            )

        v = combine_vectors(op_slices)
      else:
        if isinstance(expr, Load):
          v = expr
        else:
          v = Op(
              expr.ty,
              expr.name,
              args,
          )

    cache[expr] = v
    return v

  def optimize_slices(self, expr, cache):
    """Recursively iterate over the expression and optimize slices/combines."""
    if expr in cache:
      return cache[expr]

    if isinstance(expr, Op):
      args = [self.optimize_slices(arg, cache) for arg in expr.args]
      if expr.name == "slice":
        if isinstance(args[0], Op) and args[0].name == "combine":
          comb = args[0]
          slice_index = args[1].value
          total_slices = args[2].value
          if len(comb.args) == total_slices:
            mutated = comb.args[slice_index]
            cache[expr] = mutated
            return mutated
          elif len(comb.args) % total_slices == 0:
            num = len(comb.args) // total_slices
            mutated = combine_vectors(
                comb.args[slice_index * num : (slice_index + 1) * num]
            )
            cache[expr] = mutated
            return mutated

        elif isinstance(args[0], Op) and args[0].name == "broadcast":
          bc = args[0]
          mutated = broadcast(bc.args[0], expr.ty.lanes)
          cache[expr] = mutated
          return mutated
      mutated = Op(expr.ty, expr.name, args)
      cache[expr] = mutated
      return mutated
    else:
      return expr

  def pattern_match(self, expr, cache):
    """Recursively iterate over the expression and try to find patterns to replace."""
    if expr in cache:
      return cache[expr]

    if isinstance(expr, Op):
      for rule in self.patterns:
        replacement = rewrite(rule.pattern, rule.result, expr)
        if replacement is not None:
          mutated_expr = self.pattern_match(replacement, cache)
          cache[expr] = mutated_expr
          return mutated_expr

      if expr.name in lowering_funcs:
        lowered = lowering_funcs[expr.name](*expr.args)
        mutated = self.pattern_match(lowered, cache)
        cache[expr] = mutated
        return mutated

      args = []
      for arg in expr.args:
        args.append(self.pattern_match(arg, cache))

      mutated_expr = Op(expr.ty, expr.name, args)
      cache[expr] = mutated_expr

      return mutated_expr
    else:
      return expr

  def lift_constants(self, expr, constants):
    if isinstance(expr, Op):
      if all(isinstance(arg, Constant) for arg in expr.args):
        result = None
        # Check if we already lifted this constant.
        for var, lifted in constants.items():
          if all(
              lifted_arg.value == arg.value
              for lifted_arg, arg in zip(lifted.args, expr.args)
          ):
            result = var
            break

        if result is None:
          name = "c" + str(len(constants))
          result = Var(name, expr.ty)

          # Lifting it together with the op.
          constants[result] = expr
        return result
      else:
        for i, arg in enumerate(expr.args):
          expr.args[i] = self.lift_constants(arg, constants)

    return expr

  def linearize(self, op, ops, values):
    if isinstance(op, Var):
      values[op] = op.name
      return op
    if isinstance(op, Constant):
      values[op] = op
      return op

    if op in values:
      return values[op]

    flat_op = op
    if not isinstance(op, Constant) and not isinstance(op, Load):
      linearized_args = []
      for i in op.args:
        linearized_args += [self.linearize(i, ops, values)]
      flat_op = Op(op.ty, op.name, linearized_args)

    name = "v" + str(len(ops))
    result = Var(name, flat_op.ty)
    values[op] = result
    ops.append((result, flat_op))

    return result

  def begin_function(self, name, args):
    self.indent_level += 1
    self.result += "void " + name.lower() + "(\n"
    args_str = []
    args_str.append(f"{self.indent()}size_t m, size_t n")
    if len(args) >= 3:
      for b in args[:-1]:
        args_str.append(
            f"{self.indent()}size_t stride_{b.name}_m, size_t"
            f" stride_{b.name}_n, const void* __restrict base_{b.name}"
        )
    else:
      args_str.append(
          f"{self.indent()}size_t stride_{args[0].name}_m, const void*"
          f" __restrict base_{args[0].name}"
      )

    args_str.append(
        f"{self.indent()}size_t stride_{args[-1].name}_m, void* __restrict"
        f" base_{args[-1].name}"
    )

    if len(args) == 4:
      args_str.append(
          f"{self.indent()}const ynn::ternary_params* __restrict params"
      )
    if len(args) == 3:
      args_str.append(
          f"{self.indent()}const ynn::binary_params* __restrict params"
      )
    elif len(args) == 2:
      args_str.append(
          f"{self.indent()}const ynn::unary_params* __restrict params"
      )
    self.result += ",\n".join(args_str)
    self.result += ") {\n"

  def end_function(self):
    self.indent_level -= 1
    self.result += "}\n"

  def begin_loop(
      self,
      var,
      start,
      index,
      max_value,
      increment,
      emit_loop,
      buffers,
      prefix_before,
      prefix_after,
  ):
    # Probably bad to rely on specific var name.
    outer_loop = index == "i"
    if emit_loop:
      if outer_loop:
        self.result += (
            f"{self.indent()}for (size_t {index} = {start}; {index} <"
            f" {max_value}; {index} += {increment}) {{\n"
        )
        self.indent_level += 1
      else:
        self.result += f"{self.indent()}size_t {index} = {max_value};\n"
        self.result += f"{self.indent()}while ({index} >= {increment}) {{\n"
        self.indent_level += 1
        self.result += f"{self.indent()}{index} -= {increment};\n"
    else:
      if not outer_loop:
        self.result += f"{self.indent()}if ({index} != 0) {{\n"
        self.indent_level += 1

    if outer_loop:
      for b in buffers:
        modifier = ""
        if b.is_const:
          modifier = "const "
        self.result += (
            f"{self.indent()}{modifier}void* {prefix_after}{b.name} ="
            f" offset_bytes({prefix_before}{b.name}, stride_{b.name}_{var} *"
            f" {index});\n"
        )

        # This should've been handled before here.
        assert b.broadcast_mode != BroadcastMode.SPECIALIZE

        if not b.is_const or b.broadcast_mode == BroadcastMode.NONE:
          continue

        if b.broadcast_mode == BroadcastMode.LOCAL_VAR and increment > 1:
          self.result += (
              f"{self.indent()}size_t stride_{b.name}_m_broadcasted ="
              f" stride_{b.name}_m;\n"
          )

        broadcast_lanes = self.vector_bits // b.ty.size
        broadcast_op = self.pattern_match(
            broadcast(Var(b.name, b.ty.with_lanes(1)), broadcast_lanes),
            {},
        )
        self.result += (
            f"{self.indent()}{self.legalize_type(broadcast_op.ty)} {b.name}_broadcasted[{increment}];\n"
        )

        if b.broadcast_mode == BroadcastMode.LOCAL_VAR:
          self.result += f"{self.indent()}if (stride_{b.name}_n == 0) {{\n"

          self.indent_level += 1
          if increment > 1:
            self.result += (
                f"{self.indent()}stride_{b.name}_m_broadcasted ="
                f" {self.vector_bits // 8};\n"
            )

        for i in range(increment):
          self.result += (
              f"{self.indent()}{b.name}_broadcasted[{i}] ="
              f" {broadcast_op.name}(((const"
              f" {self.legalize_type(b.ty, True)}*)offset_bytes({prefix_after}{b.name},"
              f" min({i}, m - i - 1) * stride_{b.name}_m))[0]);\n"
          )

        if b.broadcast_mode == BroadcastMode.LOCAL_VAR:
          self.result += (
              f"{self.indent()}{prefix_after}{b.name} ="
              f" &{b.name}_broadcasted[0];\n"
          )

          self.indent_level -= 1
          self.result += f"{self.indent()}}}\n"

        self.result += "\n"

  def advance_pointers(self, buffers, var, step):
    for b in buffers:
      stride = ""
      if b.broadcast_mode == BroadcastMode.NONE:
        stride = str(b.ty.size // 8)
      elif b.broadcast_mode == BroadcastMode.ALWAYS:
        stride = "0"
      else:
        stride = f"stride_{b.name}_{var}"

      self.result += (
          f"{self.indent()} {b.name} = offset_bytes({b.name},"
          f" {stride} * {step});\n"
      )

  def end_scope(self):
    self.indent_level -= 1
    self.result += f"{self.indent()}}}\n"

  def emit_constants(self, constants):
    for k, v in constants.items():
      self.result += (
          f"{self.indent()}const {self.legalize_type(v.ty)} {k} ="
          f" {v.name}({v.args[0]});\n"
      )

  def emit_op(
      self,
      i,
      j,
      k,
      is_rem_width,
      buffers,
      constants,
      op_natural_vector_size,
      output_lanes,
  ):
    """Emits a single operation."""
    op = i[1]
    self.result += self.indent()
    if i[0] is not None:
      self.result += (
          f"{self.legalize_type(op.ty.with_lanes(op_natural_vector_size))} {i[0]}_{j}_{k}"
      )

    is_load = isinstance(op, Load)
    is_store = isinstance(op, Op) and op.name == "store"
    is_load_store = is_load or is_store

    str_args = []

    args = []

    if is_load:
      # Just for simplicity handle the broadcast here:
      # if the type of the broadcasting is always replace the load with
      # the existing broadcast value.
      b = self.as_buffer(op.index, buffers)
      if b is not None and b.broadcast_mode == BroadcastMode.ALWAYS:
        self.result += f" = {b.name}_broadcasted[{j}];\n"
        return
      # hack
      args = [op.index]
    else:
      args = op.args
    for arg in args:
      b = self.as_buffer(arg, buffers)
      if b is not None:
        t = self.legalize_type(b.ty, False)
        if is_load:
          t = "const " + t
        row_offset = ""
        stride_n = ""
        if b.broadcast_mode == BroadcastMode.NONE:
          # If there is no broadcasting for this b we can just use
          # sizeof(type) instead of the stride.
          stride_n = str(b.ty.size // 8)
        else:
          stride_n = f"stride_{arg.name}_n"
        if j > 0:
          # Only add stride if this is not the first row of the tile.
          if b.is_const and b.broadcast_mode == BroadcastMode.LOCAL_VAR:
            row_offset = (
                f" + stride_{arg.name}_m_broadcasted * min({j}, m - i - 1)"
            )
          else:
            row_offset = f" + stride_{arg.name}_m * min({j}, m - i - 1)"
        offset = op.offset_elements if is_load else 0
        str_args.append(
            f"({t}*)offset_bytes({arg.name}, {stride_n} *"
            f" ({k * output_lanes}{row_offset} + {offset}))"
        )
      else:
        if isinstance(arg, Constant) or (
            isinstance(arg, Var) and arg in constants
        ):
          str_args.append(f"{arg}")
        else:
          str_args.append(f"{arg}_{j}_{k}")

    offset = op.offset_elements if is_load else 0
    lanes = (
        f"min({op_natural_vector_size}, (size_t)std::max<int>(j -"
        f" {k * output_lanes} - {offset}, 0))"
    )

    if (
        is_load_store
        and is_rem_width
        and self.tail_strategy == TailStrategy.MEMCPY
    ):
      dst = str_args[0] if is_store else f"(void*)&{i[0]}_{j}_{k}"
      src = f"(void*)&{str_args[1]}" if is_store else str_args[0]

      if is_load:
        self.result += f";\n{self.indent()}"
      # TODO(vksnk): this can be unified with the code below by
      # adding a separate function (something like memcpy_load).
      # However, I am not sure if it'd have worse performance.
      self.result += (
          f"memcpy({dst}, {src},"
          # TODO(vksnk): this expression can be simplified once
          # we switch to while loop for tracking n.
          f" {lanes} *"
          f" {i[1].ty.size // 8});\n"
      )
    elif (
        is_load_store
        and is_rem_width
        and self.tail_strategy == TailStrategy.MASK
    ):
      mask_op = ""
      if is_load:
        self.result += " = "
        mask_op = "load"
      else:
        mask_op = "store"
      self.result += (
          f"partial_{mask_op}_{op_natural_vector_size}x({str_args[0]}, {lanes}"
      )

      if is_store:
        self.result += f", {str_args[1]}"
      self.result += ");\n"
    else:
      if not is_store:
        self.result += " = "
      mem_op = ""

      if is_load:
        mem_op = self.load_intrinsics.get(op.ty, "load")
      elif is_store:
        mem_op = self.store_intrinsics.get(op.ty, "store")
      else:
        mem_op = op.name
      self.result += f"{mem_op}({', '.join(str_args)});\n"

  def emit_inner_loop_body(
      self,
      ops,
      rows_num,
      is_rem_width,
      output_vector_num,
      output_lanes,
      buffers,
      constants,
      step,
  ):
    """Emits the body of the innermost loop."""
    for op in ops:
      for j in range(rows_num):
        op_natural_vector_size = (
            1
            if is_rem_width and self.tail_strategy == TailStrategy.SCALAR
            else op[1].ty.lanes
        )
        for k in range(output_vector_num):
          self.emit_op(
              op,
              j,
              k,
              is_rem_width,
              buffers,
              constants,
              op_natural_vector_size,
              output_lanes,
          )

    if not is_rem_width:
      self.advance_pointers(buffers, "n", step)

  def emit_body(
      self,
      ops,
      constants,
      buffers,
      natural_lanes,
      tile_height,
      tile_width,
  ):
    """Emits the main body of the generated kernel function."""
    for is_rem_height in [False]:  # , True] if tile_height > 1 else [False]:
      self.begin_loop(
          "m",
          "0" if not is_rem_height else f"m - m % {tile_height}",
          "i",
          "m"  # f"m - m % {tile_height}"
          if not is_rem_height and tile_height > 1
          else "m",
          tile_height if not is_rem_height else 1,
          True,
          buffers,
          "base_",
          "",
      )

      rows_num = tile_height if not is_rem_height else 1
      for is_rem_width in [False, True] if tile_width > 1 else [False]:
        emit_loop = (
            not is_rem_width or self.tail_strategy == TailStrategy.SCALAR
        )
        # if not is_rem_width:
        step = tile_width if not is_rem_width else 1
        self.begin_loop(
            "n",
            "0",
            "j",
            "n",
            step,
            emit_loop,
            buffers,
            "",
            "",
        )
        output_vector_num = (
            1
            if is_rem_width and self.tail_strategy == TailStrategy.SCALAR
            else tile_width // natural_lanes
        )

        self.emit_inner_loop_body(
            ops,
            rows_num,
            is_rem_width,
            output_vector_num,
            natural_lanes,
            buffers,
            constants,
            step,
        )

        self.end_scope()

      self.end_scope()

  def emit_asserts(self, buffers):
    """Emit asserts to check that strides match broadcasting modes."""
    for b in buffers:
      if not b.is_const:
        continue

      assert_body = ""
      if b.broadcast_mode == BroadcastMode.ALWAYS:
        assert_body = f"stride_{b.name}_n == 0"
      elif b.broadcast_mode == BroadcastMode.NONE:
        assert_body = f"stride_{b.name}_n == {b.ty.size // 8}"
      else:
        assert_body = (
            f"stride_{b.name}_n == 0 || stride_{b.name}_n == {b.ty.size // 8}"
        )
      self.result += f"{self.indent()}assert({assert_body} || n == 1);\n"

  def handle_specialize(
      self,
      ops,
      constants,
      buffers,
      natural_lanes,
      tile_height,
      tile_width,
  ):
    """Handles broadcast specializations of the kernel."""
    for i, b in enumerate(buffers):
      if b.is_const and b.broadcast_mode == BroadcastMode.SPECIALIZE:
        # If the buffer has SPECIALIZE type of the broadcast we produce two
        # branches of the code to handle cases with and without broadcasting.
        new_buffers = copy.deepcopy(buffers)
        self.result += f"{self.indent()}if (stride_{b.name}_n == 0) {{\n"
        self.indent_level += 1
        new_buffers[i].broadcast_mode = BroadcastMode.ALWAYS

        self.handle_specialize(
            ops,
            constants,
            new_buffers,
            natural_lanes,
            tile_height,
            tile_width,
        )

        self.indent_level -= 1
        self.result += f"{self.indent()}}} else {{\n"
        self.indent_level += 1
        new_buffers[i].broadcast_mode = BroadcastMode.NONE

        self.handle_specialize(
            ops,
            constants,
            new_buffers,
            natural_lanes,
            tile_height,
            tile_width,
        )

        self.indent_level -= 1

        self.result += f"{self.indent()}}}\n"
        return

    # Just produce body for every other type of broadcasting.
    self.emit_body(
        ops,
        constants,
        buffers,
        natural_lanes,
        tile_height,
        tile_width,
    )

  def compile(self, name, buffers, func, tile_shapes):
    """Generates a function for a range of tile sizes."""
    assert (
        func.value.ty == func.to.ty
    ), "Mismatching types of the output value and buffer."

    ast = copy.deepcopy(func.value)

    natural_lanes = self.get_natural_lanes_num(func.value.ty)
    ast = self.vectorize(ast, natural_lanes, {})
    ast = self.slice_wide_types(ast, {})
    ast = self.optimize_slices(ast, {})
    ast = self.pattern_match(ast, {})

    constants = {}
    ast = self.lift_constants(ast, constants)

    values = {}
    ops = []
    self.linearize(ast, ops, values)

    ops.append((
        None,
        Op(
            func.value.ty.with_lanes(natural_lanes),
            "store",
            [func.to, values[ast]],
        ),
    ))

    for tile in tile_shapes:
      tile_width = tile[1]
      tile_height = tile[0]

      self.begin_function(
          name,
          buffers,
      )

      if len(buffers) > 2:
        self.emit_asserts(buffers)

      self.emit_constants(constants)

      self.handle_specialize(
          ops,
          constants,
          buffer_args,
          natural_lanes,
          tile_height,
          tile_width,
      )

      self.end_function()
      self.result += "\n"

      return self.result

  def arch_flags(self):
    return "|".join(["arch_flag::" + i.lower() for i in self.features])

  def arch_string(self):
    return "x86_" + "_".join([i.lower() for i in self.features])

  def compile_function(self, name, fn, tile_shapes):
    self.result = ""
    buffer_args.clear()
    global op_name
    op_name = "unknown"
    result = fn()

    for b in buffer_args:
      if b.broadcast_mode == BroadcastMode.AUTO:
        if len(buffer_args) == 2:
          b.broadcast_mode = BroadcastMode.NONE
        else:
          b.broadcast_mode = BroadcastMode.SPECIALIZE

    func_name = (
        name
        + "_"
        + "x".join([str(i) for i in tile_shapes[0]])
        + "_"
        + self.arch_string()
    )

    src = '#include "ynnpack/kernels/'
    if len(buffer_args) == 4:
      src += "ternary/ternary.h"
    elif len(buffer_args) == 3:
      src += "binary/binary.h"
    elif len(buffer_args) == 2:
      src += "unary/unary.h"
    src += '"\n'
    src += "namespace ynn {\n"
    src += self.compile(func_name, buffer_args, result, tile_shapes)
    src += "} // namespace ynn\n"

    tps = []
    for b in buffer_args:
      tps.append(str(b.ty))
    types = ", ".join(tps)

    init_params_fn = "nullptr"

    inc = (
        f"YNN_ELEMENTWISE_KERNEL({self.arch_flags()}, {func_name}, {op_name},"
        f" {init_params_fn}, {types})\n"
    )

    return src, inc


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
u32_a = Var("a", UInt(32))
u32_b = Var("b", UInt(32))
f16_a = Var("a", Float(16))
f16_b = Var("b", Float(16))
f32_a = Var("a", Float(32))
f32_b = Var("b", Float(32))
f32_c = Var("c", Float(32))
f32_d = Var("d", Float(32))
bf16_a = Var("a", BFloat(16))
bf16_b = Var("b", BFloat(16))


class Rule:
  # The first argument to the result should be split into the low and high parts.
  split_operand_0_lo_hi = 1

  def __init__(self, pattern, result, features=[], flags=0):
    self.pattern = pattern
    self.result = result
    self.features = features
    self.flags = flags

  def __str__(self):
    return f"{self.pattern} -> {self.result} ({', '.join(self.features)})"

  def __repr__(self):
    return str(self)

  def vectorize(self, vector_bits):

    return Rule(
        self.pattern.with_lanes(vector_bits // (self.pattern.ty.size)),
        self.result.with_lanes(vector_bits // (self.result.ty.size)),
        self.features,
        self.flags,
    )
