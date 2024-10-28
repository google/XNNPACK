// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/datatype.h"

#include "xnnpack.h"

bool xnn_datatype_is_real(enum xnn_datatype t) {
  switch (t) {
    case xnn_datatype_invalid:
    case xnn_datatype_int32:
      return false;
    case xnn_datatype_fp32:
    case xnn_datatype_fp16:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qint32:
    case xnn_datatype_qcint8:
    case xnn_datatype_qcint32:
    case xnn_datatype_qcint4:
    case xnn_datatype_qdint8:
    case xnn_datatype_qpint8:
    case xnn_datatype_qbint4:
    case xnn_datatype_pfp32:
      return true;
  }
  XNN_UNREACHABLE;
  return false;
}

bool xnn_datatype_is_integral(enum xnn_datatype t) {
  switch (t) {
    case xnn_datatype_invalid:
    case xnn_datatype_fp32:
    case xnn_datatype_fp16:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qint32:
    case xnn_datatype_qcint8:
    case xnn_datatype_qcint32:
    case xnn_datatype_qcint4:
    case xnn_datatype_qdint8:
    case xnn_datatype_qpint8:
    case xnn_datatype_qbint4:
    case xnn_datatype_pfp32:
      return false;
    case xnn_datatype_int32:
      return true;
  }
  XNN_UNREACHABLE;
  return false;
}

bool xnn_datatype_is_quantized(enum xnn_datatype t) {
  switch (t) {
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qint32:
    case xnn_datatype_qcint8:
    case xnn_datatype_qcint32:
    case xnn_datatype_qcint4:
    case xnn_datatype_qdint8:
    case xnn_datatype_qpint8:
    case xnn_datatype_qbint4:
      return true;
    case xnn_datatype_invalid:
    case xnn_datatype_fp32:
    case xnn_datatype_fp16:
    case xnn_datatype_int32:
    case xnn_datatype_pfp32:
      return false;
  }
  XNN_UNREACHABLE;
  return false;
}


size_t xnn_datatype_log2_size_bits(enum xnn_datatype t) {
  switch (t) {
    case xnn_datatype_invalid:
      return -1;
    case xnn_datatype_qcint4:
    case xnn_datatype_qbint4:
      return 2;
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qcint8:
    case xnn_datatype_qdint8:
    case xnn_datatype_qpint8:
      return 3;
    case xnn_datatype_fp16:
      return 4;
    case xnn_datatype_qint32:
    case xnn_datatype_qcint32:
    case xnn_datatype_int32:
    case xnn_datatype_fp32:
    case xnn_datatype_pfp32:
      return 5;
  }
  XNN_UNREACHABLE;
  return -1;
}

size_t xnn_datatype_log2_size_bytes(enum xnn_datatype t) {
  size_t size_bits = xnn_datatype_log2_size_bits(t);
  assert(size_bits >= 3);
  return size_bits - 3;
}

size_t xnn_datatype_size_bits(enum xnn_datatype t) {
  return 1 << xnn_datatype_log2_size_bits(t);
}

size_t xnn_datatype_size_bytes(enum xnn_datatype t) {
  return 1 << xnn_datatype_log2_size_bytes(t);
}

bool xnn_datatype_is_byte_addressable(enum xnn_datatype t) {
  switch (t) {
    case xnn_datatype_invalid:
    case xnn_datatype_qcint4:
    case xnn_datatype_qbint4:
    case xnn_datatype_pfp32:
    case xnn_datatype_qpint8:
      return false;
    case xnn_datatype_fp16:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qint32:
    case xnn_datatype_qcint8:
    case xnn_datatype_qcint32:
    case xnn_datatype_qdint8:
    case xnn_datatype_int32:
    case xnn_datatype_fp32:
      return true;
  }
  XNN_UNREACHABLE;
  return false;
}
