// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/binary/binary.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

namespace {

// These "reference" microkernels are not designed to be fast, only to support
// all possible operators and types with reasonable performance. We just
// intend to give the compiler a reasonable chance at optimizing them.
template <typename T, typename Operator>
void binary_impl(size_t m, size_t n, size_t stride_a_m, size_t stride_a_n,
                 const void* va, size_t stride_b_m, size_t stride_b_n,
                 const void* vb, size_t stride_x_m, void* vx,
                 const binary_params*) {
  auto a = reinterpret_cast<const T*>(va);
  auto b = reinterpret_cast<const T*>(vb);
  auto x = reinterpret_cast<T*>(vx);
  Operator op;
  for (size_t i = 0; i < m; ++i) {
    if (stride_a_n == 0) {
      const T a_0 = a[0];
      for (size_t j = 0; j < n; ++j) {
        x[j] = static_cast<T>(op(a_0, b[j]));
      }
    } else if (stride_b_n == 0) {
      // TODO: If Operator::commutative is true, we could swap the arguments and
      // avoid the code size cost of the third variant.
      const T b_0 = b[0];
      for (size_t j = 0; j < n; ++j) {
        x[j] = static_cast<T>(op(a[j], b_0));
      }
    } else {
      assert(stride_a_n == sizeof(T));
      assert(stride_b_n == sizeof(T));
      for (size_t j = 0; j < n; ++j) {
        x[j] = static_cast<T>(op(a[j], b[j]));
      }
    }
    a = offset_bytes(a, stride_a_m);
    b = offset_bytes(b, stride_b_m);
    x = offset_bytes(x, stride_x_m);
  }
}

// We can't use STL's functional ops for integers because they will crash if
// they overflow.
struct AddOp {
  float operator()(float a, float b) const { return a + b; }
  double operator()(double a, double b) const { return a + b; }
  int32_t operator()(int32_t a, int32_t b) const {
    return static_cast<int32_t>(static_cast<int64_t>(a) +
                                static_cast<int64_t>(b));
  }
  using commutative = std::true_type;
};

struct SubtractOp {
  float operator()(float a, float b) const { return a - b; }
  double operator()(double a, double b) const { return a - b; }
  int32_t operator()(int32_t a, int32_t b) const {
    return static_cast<int64_t>(a) - static_cast<int64_t>(b);
  }
  using commutative = std::false_type;
};

struct MultiplyOp {
  float operator()(float a, float b) const { return a * b; }
  double operator()(double a, double b) const { return a * b; }
  int32_t operator()(int32_t a, int32_t b) const {
    return static_cast<int64_t>(a) * static_cast<int64_t>(b);
  }
  using commutative = std::true_type;
};

struct DivideOp {
  float operator()(float a, float b) const { return a / b; }
  double operator()(double a, double b) const { return a / b; }
  int32_t operator()(int32_t a, int32_t b) const { return euclidean_div(a, b); }
  using commutative = std::false_type;
};

struct MaxOp {
  template <typename T>
  T operator()(T a, T b) const {
    return a > b ? a : b;
  }
  using commutative = std::true_type;
};

struct MinOp {
  template <typename T>
  T operator()(T a, T b) const {
    return a < b ? a : b;
  }
  using commutative = std::true_type;
};

struct SquaredDifferenceOp {
  float operator()(float a, float b) const { return (a - b) * (a - b); }
  double operator()(double a, double b) const { return (a - b) * (a - b); }
  using commutative = std::true_type;
};

struct PowOp {
  float operator()(float a, float b) const { return std::pow(a, b); }
  double operator()(double a, double b) const { return std::pow(a, b); }
  int32_t operator()(int32_t a, int32_t b) const { return integer_pow(a, b); }
  using commutative = std::false_type;
};

struct LeakyReluOp {
  float operator()(float a, float b) const { return (a < 0) ? a * b : a; }
  double operator()(double a, double b) const { return (a < 0) ? a * b : a; }
  using commutative = std::false_type;
};

struct CopysignOp {
  template <typename T>
  T operator()(T a, T b) const {
    return std::copysign(a, b);
  }
  using commutative = std::false_type;
};

template <typename T>
binary_kernel_fn get_float_binary_reference_kernel(ynn_binary_operator op) {
  switch (op) {
    case ynn_binary_add:
      return binary_impl<T, AddOp>;
    case ynn_binary_subtract:
      return binary_impl<T, SubtractOp>;
    case ynn_binary_multiply:
      return binary_impl<T, MultiplyOp>;
    case ynn_binary_divide:
      return binary_impl<T, DivideOp>;
    case ynn_binary_max:
      return binary_impl<T, MaxOp>;
    case ynn_binary_min:
      return binary_impl<T, MinOp>;
    case ynn_binary_copysign:
      return binary_impl<T, CopysignOp>;
    case ynn_binary_squared_difference:
      return binary_impl<T, SquaredDifferenceOp>;
    case ynn_binary_pow:
      return binary_impl<T, PowOp>;
    case ynn_binary_leaky_relu:
      return binary_impl<T, LeakyReluOp>;
    default:
      break;
  }
  return nullptr;
}

}  // namespace

binary_kernel_fn get_binary_reference_kernel(ynn_binary_operator op,
                                             ynn_type type) {
  if (type == ynn_type_fp32) {
    return get_float_binary_reference_kernel<float>(op);
  } else if (type == ynn_type_fp64) {
    return get_float_binary_reference_kernel<double>(op);
  } else if (type == ynn_type_int32) {
    switch (op) {
      case ynn_binary_add:
        return binary_impl<int32_t, AddOp>;
      case ynn_binary_subtract:
        return binary_impl<int32_t, SubtractOp>;
      case ynn_binary_multiply:
        return binary_impl<int32_t, MultiplyOp>;
      case ynn_binary_divide:
        return binary_impl<int32_t, DivideOp>;
      case ynn_binary_max:
        return binary_impl<int32_t, MaxOp>;
      case ynn_binary_min:
        return binary_impl<int32_t, MinOp>;
      case ynn_binary_copysign:
        return binary_impl<int32_t, CopysignOp>;
      case ynn_binary_pow:
        return binary_impl<int32_t, PowOp>;
      default:
        return nullptr;
    }
  }
  return nullptr;
}

binary_kernel_fn get_binary_kernel(ynn_binary_operator op, ynn_type type_a,
                                   ynn_type type_b, ynn_type type_x,
                                   uint64_t supported_arch_flags) {
#define YNN_ELEMENTWISE_KERNEL(arch, name, op_type, A, B, X) \
  if (is_arch_supported(arch, supported_arch_flags) &&       \
      op == ynn_binary_##op_type) {                          \
    if (type_of<A>() == type_a && type_of<B>() == type_b &&  \
        type_of<X>() == type_x) {                            \
      YNN_LOG_DEBUG() << "Using binary kernel " << #name;    \
      return name;                                           \
    }                                                        \
  }
#include "ynnpack/kernels/binary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL
  if (type_a == ynn_type_fp64 && type_b == ynn_type_fp64 &&
      type_x == ynn_type_fp64) {
    return get_binary_reference_kernel(op, type_x);
  } else if (type_a == ynn_type_fp32 && type_b == ynn_type_fp32 &&
             type_x == ynn_type_fp32) {
    return get_binary_reference_kernel(op, type_x);
  } else if (type_a == ynn_type_int32 && type_b == ynn_type_int32 &&
             type_x == ynn_type_int32) {
    return get_binary_reference_kernel(op, type_x);
  }
  return nullptr;
}

}  // namespace ynn
