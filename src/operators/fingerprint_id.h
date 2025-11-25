// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_OPERATORS_FINGERPRINT_ID_H_
#define XNNPACK_SRC_OPERATORS_FINGERPRINT_ID_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define XNN_EXPAND(x) x
#define XNN_GLUE(x, y) x y

#define XNN_COUNT_ARGS_HELPER(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, ...) _9
#define XNN_COUNT_ARGS_EXPAND(args) XNN_COUNT_ARGS_HELPER args
#define XNN_COUNT_ARGS(...) \
  XNN_COUNT_ARGS_EXPAND((__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

#define XNN_CONCAT_FLAGS_4(prefix, a, b, c, d) \
  prefix##a | prefix##b | prefix##c | prefix##d

#define XNN_CONCAT_FLAGS_5(prefix, a, b, c, d, e) \
  prefix##a | prefix##b | prefix##c | prefix##d | prefix##e

#define XNN_CONCAT_FLAGS_6(prefix, a, b, c, d, e, f) \
  prefix##a | prefix##b | prefix##c | prefix##d | prefix##e | prefix##f

#define XNN_CONCAT_FLAGS_7(prefix, a, b, c, d, e, f, g)                   \
  prefix##a | prefix##b | prefix##c | prefix##d | prefix##e | prefix##f | \
      prefix##g

#define XNN_CONCAT_FLAGS_8(prefix, a, b, c, d, e, f, g, h)                \
  prefix##a | prefix##b | prefix##c | prefix##d | prefix##e | prefix##f | \
      prefix##g | prefix##h

#define XNN_CONCAT_FLAGS_9(prefix, a, b, c, d, e, f, g, h, i)             \
  prefix##a | prefix##b | prefix##c | prefix##d | prefix##e | prefix##f | \
      prefix##g | prefix##h | prefix##i

#define XNN_CONCAT_1(a) a
#define XNN_CONCAT_2(a, b) a##_##b
#define XNN_CONCAT_3(a, b, c) a##_##b##_##c
#define XNN_CONCAT_4(a, b, c, d) a##_##b##_##c##_##d
#define XNN_CONCAT_5(a, b, c, d, e) a##_##b##_##c##_##d##_##e
#define XNN_CONCAT_6(a, b, c, d, e, f) a##_##b##_##c##_##d##_##e##_##f
#define XNN_CONCAT_7(a, b, c, d, e, f, g) a##_##b##_##c##_##d##_##e##_##f##_##g
#define XNN_CONCAT_8(a, b, c, d, e, f, g, h) \
  a##_##b##_##c##_##d##_##e##_##f##_##g##_##h
#define XNN_CONCAT_9(a, b, c, d, e, f, g, h, i) \
  a##_##b##_##c##_##d##_##e##_##f##_##g##_##h##_##i

#define XNN_OVERLOAD_2(a, b) a##b
#define XNN_OVERLOAD_1(a, b) XNN_OVERLOAD_2(a, b)
#define XNN_OVERLOAD(X, C) XNN_OVERLOAD_1(X, C)

#define XNN_FINGERPRINT_ID_OP_OFFSET 26
#define XNN_FINGERPRINT_ID_IN_OFFSET 20
#define XNN_FINGERPRINT_ID_OUT_OFFSET 14
#define XNN_FINGERPRINT_ID_WEIGHTS_OFFSET 8

// Creates a fingerprint id value.
//
// ```cpp
// XNN_FINGERPRINT_ID_VALUE(operator, in_type, out_type, weights_type
//                          [, flags...])
// ```
//
// A fingerprint id value is built using 32 bits.
//
//     op    in   out weights flags
//  ├─────┼─────┼─────┼─────┼───────┤
// 32    26    20    14     8       0
//
// - operator: 6 bits -> 64 operators possible
// - input, output, weights: 6 bits -> 64 types possible
// - flags:  8 bits ->  8 flags possible
//
// These values should be used with the `XNN_DEFINE_FINGERPRINT_VALUES()` macro
// without the `xnn_fingerprint_id_helper_` prefix.
//
// Example:
// ```cpp
// xnn_fingerprint_id_fully_connected_nc_f16 =
//     XNN_FINGERPRINT_ID_VALUE(fully_connected_nc, f16, f16, f16),
// ```
#define XNN_FINGERPRINT_ID_VALUE(...) \
  XNN_FINGERPRINT_ID_VALUE_EXPAND((__VA_ARGS__))

#define XNN_FINGERPRINT_ID_VALUE_EXPAND(args) XNN_FINGERPRINT_ID_VALUE_IMPL args

#define XNN_FINGERPRINT_ID_VALUE_IMPL(op, in, out, weights, ...)               \
  XNN_EXPAND(XNN_GLUE(                                                         \
      XNN_OVERLOAD(XNN_CONCAT_FLAGS_,                                          \
                   XNN_COUNT_ARGS(op, in, out, weights, __VA_ARGS__)),         \
      ((uint32_t)xnn_fingerprint_id_helper_,                                   \
       op << XNN_FINGERPRINT_ID_OP_OFFSET, in << XNN_FINGERPRINT_ID_IN_OFFSET, \
       out << XNN_FINGERPRINT_ID_OUT_OFFSET,                                   \
       weights << XNN_FINGERPRINT_ID_WEIGHTS_OFFSET, __VA_ARGS__)))

#define XNN_EXPAND_TYPES(prefix, ...)                                    \
  XNN_EXPAND(XNN_GLUE(                                                   \
      XNN_OVERLOAD(XNN_EXPAND_TYPES_IMPL_, XNN_COUNT_ARGS(__VA_ARGS__)), \
      (prefix, __VA_ARGS__)))

#define XNN_EXPAND_TYPES_IMPL_1(prefix, a) XNN_CONCAT_4(prefix, a, a, a)
#define XNN_EXPAND_TYPES_IMPL_2(prefix, a, b) XNN_CONCAT_4(prefix, a, a, b)
#define XNN_EXPAND_TYPES_IMPL_3(prefix, a, b, c) XNN_CONCAT_4(prefix, a, b, c)
#define XNN_EXPAND_TYPES_IMPL_4(prefix, a, b, c, d) \
  XNN_CONCAT_5(prefix, a, b, c, d)

#define XNN_CONCAT_TYPES(prefix, ...)                                          \
  XNN_EXPAND(                                                                  \
      XNN_GLUE(XNN_OVERLOAD(XNN_CONCAT_, XNN_COUNT_ARGS(prefix, __VA_ARGS__)), \
               (prefix, __VA_ARGS__)))

#define XNN_FINGERPRINT_ID_NAME(...) \
  XNN_CONCAT_TYPES(xnn_fingerprint_id, __VA_ARGS__)

// Used to build fingerprint ids.
//
// Warning: Because fingerprints may be stored to be compared to past/future
// versions of XNNPack, **THESE VALUES MUST NOT CHANGE**. You can add new
// values.
enum xnn_fingerprint_id_helper {
  // Operator values
  xnn_fingerprint_id_helper_unknown = 0,
  xnn_fingerprint_id_helper_test = 1,
  xnn_fingerprint_id_helper_no_fingerprint = 2,
  xnn_fingerprint_id_helper_fully_connected_nc = 3,
  xnn_fingerprint_id_helper_convolution2d_nchw = 4,
  // Type values
  xnn_fingerprint_id_helper_bf16 = 0,
  xnn_fingerprint_id_helper_f16 = 1,
  xnn_fingerprint_id_helper_f32 = 2,
  xnn_fingerprint_id_helper_pf16 = 3,
  xnn_fingerprint_id_helper_pf32 = 4,
  xnn_fingerprint_id_helper_qb4w = 5,
  xnn_fingerprint_id_helper_qc2w = 6,
  xnn_fingerprint_id_helper_qc4w = 7,
  xnn_fingerprint_id_helper_qc8w = 8,
  xnn_fingerprint_id_helper_qd8 = 9,
  xnn_fingerprint_id_helper_qdu8 = 10,
  xnn_fingerprint_id_helper_qp8 = 11,
  xnn_fingerprint_id_helper_qs8 = 12,
  xnn_fingerprint_id_helper_pqs8 = 13,
  xnn_fingerprint_id_helper_qu8 = 14,
  // Flag values
  //
  // Flag values are OR-ed so they need to avoid colliding. Not all operators
  // use the same flags values. Flags values that don't overlap between
  // operators may reuse the same value.
  xnn_fingerprint_id_helper_nr2 = (1 << 0),
  xnn_fingerprint_id_helper_fp32_static_weights = (1 << 1),
  xnn_fingerprint_id_helper_conv2d_hwc2chw = (1 << 2),
  xnn_fingerprint_id_helper_dwconv = (1 << 3),
  // The C preprocessor is obnoxious. For variadic arguments, there's no way to
  // differentiate between an empty argument list and one argument. This value
  // allows us to avoid bending around this issue when generating the
  // fingerprint ID values.
  //
  // This value is OR-ed to the value by the COMBINE macro when no flag is
  // passed.
  xnn_fingerprint_id_helper_ = 0,
};

// Identifies a fingerprint.
//
// This should be retrievable from a fingerprint and allows identifying what
// that fingerprint identifies.
//
// Because fingerprints may be stored to be compared to past/future versions of
// XNNPack, **these values MUST NOT CHANGE**.
enum xnn_fingerprint_id {
#define XNN_FINGERPRINT_ID(operator, in, out, ...)          \
  XNN_FINGERPRINT_ID_NAME(operator, in, out, __VA_ARGS__) = \
      XNN_FINGERPRINT_ID_VALUE(operator, in, out, __VA_ARGS__),
#include "fingerprint_id.h.inc"
  // Manually defined values.
  XNN_FINGERPRINT_ID(test, f16, f32, qc8w, nr2)
  xnn_fingerprint_id_unknown =
      XNN_FINGERPRINT_ID_VALUE(unknown, unknown, unknown, unknown),
  xnn_fingerprint_id_no_fingerprint =
      XNN_FINGERPRINT_ID_VALUE(no_fingerprint, unknown, unknown, unknown),
#undef XNN_FINGERPRINT_ID
};

// Computes the fingerprint id value from the given helper values.
//
// Warning: All calls should have the last variadic argument be 0.
//
// Note: `weights` is defined as an `uint32_t` because variadic lists in C must
// start with a type that doesn't go through integral promotion.
enum xnn_fingerprint_id xnn_compute_fingerprint_id_value(
    enum xnn_fingerprint_id_helper op, enum xnn_fingerprint_id_helper in,
    enum xnn_fingerprint_id_helper out, uint32_t weights, ...);

const char* xnn_fingerprint_id_to_string(
    enum xnn_fingerprint_id fingerprint_id);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // XNNPACK_SRC_OPERATORS_FINGERPRINT_ID_H_
