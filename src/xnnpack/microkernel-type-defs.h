// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNN_ENUM_ITEM_0
#define XNN_ENUM_ITEM_0(enum_name, enum_string) XNN_ENUM_ITEM(enum_name, enum_string)
#define XNN_DEFINED_ENUM_ITEM_0
#endif

XNN_ENUM_ITEM_0(xnn_microkernel_type_default, "Default")
XNN_ENUM_ITEM(xnn_microkernel_type_average_pooling, "Average Pooling")
XNN_ENUM_ITEM(xnn_microkernel_type_conv2d_hwc2chw, "Conv2D HWC2CHW")
XNN_ENUM_ITEM(xnn_microkernel_type_dwconv, "DWConv")
XNN_ENUM_ITEM(xnn_microkernel_type_gemm, "GEMM")
XNN_ENUM_ITEM(xnn_microkernel_type_global_average_pooling, "Global Average Pooling")
XNN_ENUM_ITEM(xnn_microkernel_type_igemm, "IGEMM")
XNN_ENUM_ITEM(xnn_microkernel_type_mean, "Mean")
XNN_ENUM_ITEM(xnn_microkernel_type_pixelwise_average_pooling, "Pixelwise Average Pooling")
XNN_ENUM_ITEM(xnn_microkernel_type_spmm, "SPMM")
XNN_ENUM_ITEM(xnn_microkernel_type_subconv2d, "Subconv2D")
XNN_ENUM_ITEM(xnn_microkernel_type_transpose, "Transpose")
XNN_ENUM_ITEM(xnn_microkernel_type_vmulcaddc, "VMulCAddC")


#ifdef XNN_DEFINED_ENUM_ITEM_0
#undef XNN_DEFINED_ENUM_ITEM_0
#undef XNN_ENUM_ITEM_0
#endif
