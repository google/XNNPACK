// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/log.h>


// This function is defined inline when logging is disabled
#if XNN_LOG_LEVEL > 0
const char* xnn_datatype_to_string(enum xnn_datatype type) {
  switch (type) {
    case xnn_datatype_invalid:
      return "Invalid";
    case xnn_datatype_fp32:
      return "FP32";
    case xnn_datatype_fp16:
      return "FP16";
    case xnn_datatype_qint8:
      return "QINT8";
    case xnn_datatype_quint8:
      return "QUINT8";
    case xnn_datatype_qint32:
      return "QINT32";
    case xnn_datatype_qcint4:
      return "QCINT4";
    case xnn_datatype_qcint8:
      return "QCINT8";
    case xnn_datatype_qcint32:
      return "QCINT32";
    case xnn_datatype_qdint8:
      return "QDINT8";
  }
  XNN_UNREACHABLE;
  return NULL;
}
#endif  // XNN_LOG_LEVEL > 0
