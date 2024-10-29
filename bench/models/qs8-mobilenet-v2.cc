// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!

#include "xnnpack.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "xnnpack.h"

// align a size up to XNN_EXTRA_BYTES
#define XNN_PAD_EXTRA_BYTES(s, t) (((s) + XNN_EXTRA_BYTES / sizeof(t) - 1) & ~(XNN_EXTRA_BYTES / sizeof(t) - 1))

namespace models {

xnn_subgraph_t QS8MobileNetV2() {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgrpah" << std::endl;
    return nullptr;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  uint32_t v0 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v0_dims = {{1, 224, 224, 3}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v0_dims.size(), v0_dims.data(),
    /*data=*/nullptr,
    0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &v0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v0" << std::endl;
    return nullptr;
  }

  uint32_t v1 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v1_dims = {{1, 224, 224, 3}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-14,
    /*scale=*/0.01865844801068306f,
    v1_dims.size(), v1_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v1" << std::endl;
    return nullptr;
  }

  uint32_t v2 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v2_dims = {{1, 112, 112, 32}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v2_dims.size(), v2_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v2" << std::endl;
    return nullptr;
  }

  uint32_t v3 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v3_dims = {{1, 112, 112, 32}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v3_dims.size(), v3_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v3" << std::endl;
    return nullptr;
  }

  uint32_t v4 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v4_dims = {{1, 112, 112, 16}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-6,
    /*scale=*/0.3295263648033142f,
    v4_dims.size(), v4_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v4" << std::endl;
    return nullptr;
  }

  uint32_t v5 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v5_dims = {{1, 112, 112, 96}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v5_dims.size(), v5_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v5" << std::endl;
    return nullptr;
  }

  uint32_t v6 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v6_dims = {{1, 56, 56, 96}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v6_dims.size(), v6_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v6" << std::endl;
    return nullptr;
  }

  uint32_t v7 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v7_dims = {{1, 56, 56, 24}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/1,
    /*scale=*/0.26354169845581055f,
    v7_dims.size(), v7_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v7" << std::endl;
    return nullptr;
  }

  uint32_t v8 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v8_dims = {{1, 56, 56, 144}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v8_dims.size(), v8_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v8" << std::endl;
    return nullptr;
  }

  uint32_t v9 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v9_dims = {{1, 56, 56, 144}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v9_dims.size(), v9_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v9" << std::endl;
    return nullptr;
  }

  uint32_t v10 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v10_dims = {{1, 56, 56, 24}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-1,
    /*scale=*/0.361392080783844f,
    v10_dims.size(), v10_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v10" << std::endl;
    return nullptr;
  }

  uint32_t v11 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v11_dims = {{1, 56, 56, 24}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/5,
    /*scale=*/0.39553302526474f,
    v11_dims.size(), v11_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v11" << std::endl;
    return nullptr;
  }

  uint32_t v12 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v12_dims = {{1, 56, 56, 144}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v12_dims.size(), v12_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v12" << std::endl;
    return nullptr;
  }

  uint32_t v13 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v13_dims = {{1, 28, 28, 144}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v13_dims.size(), v13_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v13" << std::endl;
    return nullptr;
  }

  uint32_t v14 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v14_dims = {{1, 28, 28, 32}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/1,
    /*scale=*/0.24222400784492493f,
    v14_dims.size(), v14_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v14" << std::endl;
    return nullptr;
  }

  uint32_t v15 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v15_dims = {{1, 28, 28, 192}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v15_dims.size(), v15_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v15" << std::endl;
    return nullptr;
  }

  uint32_t v16 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v16_dims = {{1, 28, 28, 192}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v16_dims.size(), v16_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v16" << std::endl;
    return nullptr;
  }

  uint32_t v17 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v17_dims = {{1, 28, 28, 32}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-22,
    /*scale=*/0.2548377215862274f,
    v17_dims.size(), v17_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v17" << std::endl;
    return nullptr;
  }

  uint32_t v18 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v18_dims = {{1, 28, 28, 32}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-2,
    /*scale=*/0.32090887427330017f,
    v18_dims.size(), v18_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v18" << std::endl;
    return nullptr;
  }

  uint32_t v19 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v19_dims = {{1, 28, 28, 192}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v19_dims.size(), v19_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v19" << std::endl;
    return nullptr;
  }

  uint32_t v20 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v20_dims = {{1, 28, 28, 192}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v20_dims.size(), v20_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v20" << std::endl;
    return nullptr;
  }

  uint32_t v21 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v21_dims = {{1, 28, 28, 32}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/0,
    /*scale=*/0.22142209112644196f,
    v21_dims.size(), v21_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v21" << std::endl;
    return nullptr;
  }

  uint32_t v22 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v22_dims = {{1, 28, 28, 32}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-1,
    /*scale=*/0.3504160940647125f,
    v22_dims.size(), v22_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v22" << std::endl;
    return nullptr;
  }

  uint32_t v23 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v23_dims = {{1, 28, 28, 192}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v23_dims.size(), v23_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v23" << std::endl;
    return nullptr;
  }

  uint32_t v24 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v24_dims = {{1, 14, 14, 192}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v24_dims.size(), v24_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v24" << std::endl;
    return nullptr;
  }

  uint32_t v25 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v25_dims = {{1, 14, 14, 64}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-5,
    /*scale=*/0.1933557689189911f,
    v25_dims.size(), v25_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v25" << std::endl;
    return nullptr;
  }

  uint32_t v26 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v26_dims = {{1, 14, 14, 384}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v26_dims.size(), v26_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v26" << std::endl;
    return nullptr;
  }

  uint32_t v27 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v27_dims = {{1, 14, 14, 384}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v27_dims.size(), v27_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v27" << std::endl;
    return nullptr;
  }

  uint32_t v28 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v28_dims = {{1, 14, 14, 64}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-14,
    /*scale=*/0.17820045351982117f,
    v28_dims.size(), v28_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v28" << std::endl;
    return nullptr;
  }

  uint32_t v29 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v29_dims = {{1, 14, 14, 64}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-3,
    /*scale=*/0.2269556224346161f,
    v29_dims.size(), v29_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v29" << std::endl;
    return nullptr;
  }

  uint32_t v30 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v30_dims = {{1, 14, 14, 384}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v30_dims.size(), v30_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v30" << std::endl;
    return nullptr;
  }

  uint32_t v31 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v31_dims = {{1, 14, 14, 384}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v31_dims.size(), v31_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v31" << std::endl;
    return nullptr;
  }

  uint32_t v32 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v32_dims = {{1, 14, 14, 64}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/4,
    /*scale=*/0.13266333937644958f,
    v32_dims.size(), v32_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v32" << std::endl;
    return nullptr;
  }

  uint32_t v33 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v33_dims = {{1, 14, 14, 64}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-4,
    /*scale=*/0.24267108738422394f,
    v33_dims.size(), v33_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v33" << std::endl;
    return nullptr;
  }

  uint32_t v34 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v34_dims = {{1, 14, 14, 384}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v34_dims.size(), v34_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v34" << std::endl;
    return nullptr;
  }

  uint32_t v35 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v35_dims = {{1, 14, 14, 384}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v35_dims.size(), v35_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v35" << std::endl;
    return nullptr;
  }

  uint32_t v36 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v36_dims = {{1, 14, 14, 64}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-5,
    /*scale=*/0.143711119890213f,
    v36_dims.size(), v36_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v36" << std::endl;
    return nullptr;
  }

  uint32_t v37 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v37_dims = {{1, 14, 14, 64}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-3,
    /*scale=*/0.24989819526672363f,
    v37_dims.size(), v37_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v37" << std::endl;
    return nullptr;
  }

  uint32_t v38 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v38_dims = {{1, 14, 14, 384}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v38_dims.size(), v38_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v38" << std::endl;
    return nullptr;
  }

  uint32_t v39 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v39_dims = {{1, 14, 14, 384}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v39_dims.size(), v39_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v39" << std::endl;
    return nullptr;
  }

  uint32_t v40 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v40_dims = {{1, 14, 14, 96}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-3,
    /*scale=*/0.199631929397583f,
    v40_dims.size(), v40_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v40" << std::endl;
    return nullptr;
  }

  uint32_t v41 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v41_dims = {{1, 14, 14, 576}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v41_dims.size(), v41_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v41" << std::endl;
    return nullptr;
  }

  uint32_t v42 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v42_dims = {{1, 14, 14, 576}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v42_dims.size(), v42_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v42" << std::endl;
    return nullptr;
  }

  uint32_t v43 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v43_dims = {{1, 14, 14, 96}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-5,
    /*scale=*/0.1339312046766281f,
    v43_dims.size(), v43_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v43" << std::endl;
    return nullptr;
  }

  uint32_t v44 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v44_dims = {{1, 14, 14, 96}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-3,
    /*scale=*/0.2115112543106079f,
    v44_dims.size(), v44_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v44" << std::endl;
    return nullptr;
  }

  uint32_t v45 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v45_dims = {{1, 14, 14, 576}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v45_dims.size(), v45_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v45" << std::endl;
    return nullptr;
  }

  uint32_t v46 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v46_dims = {{1, 14, 14, 576}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v46_dims.size(), v46_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v46" << std::endl;
    return nullptr;
  }

  uint32_t v47 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v47_dims = {{1, 14, 14, 96}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/13,
    /*scale=*/0.20608632266521454f,
    v47_dims.size(), v47_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v47" << std::endl;
    return nullptr;
  }

  uint32_t v48 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v48_dims = {{1, 14, 14, 96}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/4,
    /*scale=*/0.2694852650165558f,
    v48_dims.size(), v48_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v48" << std::endl;
    return nullptr;
  }

  uint32_t v49 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v49_dims = {{1, 14, 14, 576}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v49_dims.size(), v49_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v49" << std::endl;
    return nullptr;
  }

  uint32_t v50 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v50_dims = {{1, 7, 7, 576}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v50_dims.size(), v50_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v50" << std::endl;
    return nullptr;
  }

  uint32_t v51 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v51_dims = {{1, 7, 7, 160}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-3,
    /*scale=*/0.15931324660778046f,
    v51_dims.size(), v51_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v51" << std::endl;
    return nullptr;
  }

  uint32_t v52 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v52_dims = {{1, 7, 7, 960}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v52_dims.size(), v52_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v52" << std::endl;
    return nullptr;
  }

  uint32_t v53 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v53_dims = {{1, 7, 7, 960}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v53_dims.size(), v53_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v53" << std::endl;
    return nullptr;
  }

  uint32_t v54 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v54_dims = {{1, 7, 7, 160}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/1,
    /*scale=*/0.10275092720985413f,
    v54_dims.size(), v54_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v54" << std::endl;
    return nullptr;
  }

  uint32_t v55 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v55_dims = {{1, 7, 7, 160}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-6,
    /*scale=*/0.1888202577829361f,
    v55_dims.size(), v55_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v55" << std::endl;
    return nullptr;
  }

  uint32_t v56 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v56_dims = {{1, 7, 7, 960}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v56_dims.size(), v56_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v56" << std::endl;
    return nullptr;
  }

  uint32_t v57 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v57_dims = {{1, 7, 7, 960}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v57_dims.size(), v57_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v57" << std::endl;
    return nullptr;
  }

  uint32_t v58 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v58_dims = {{1, 7, 7, 160}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/7,
    /*scale=*/0.22013002634048462f,
    v58_dims.size(), v58_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v58" << std::endl;
    return nullptr;
  }

  uint32_t v59 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v59_dims = {{1, 7, 7, 160}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-5,
    /*scale=*/0.3162474036216736f,
    v59_dims.size(), v59_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v59" << std::endl;
    return nullptr;
  }

  uint32_t v60 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v60_dims = {{1, 7, 7, 960}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v60_dims.size(), v60_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v60" << std::endl;
    return nullptr;
  }

  uint32_t v61 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v61_dims = {{1, 7, 7, 960}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v61_dims.size(), v61_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v61" << std::endl;
    return nullptr;
  }

  uint32_t v62 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v62_dims = {{1, 7, 7, 320}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-1,
    /*scale=*/0.1037522703409195f,
    v62_dims.size(), v62_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v62);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v62" << std::endl;
    return nullptr;
  }

  uint32_t v63 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> v63_dims = {{1, 7, 7, 1280}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.0235294122248888f,
    v63_dims.size(), v63_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v63" << std::endl;
    return nullptr;
  }

  uint32_t v64 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> v64_dims = {{1, 1280}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-128,
    /*scale=*/0.018486851826310158f,
    v64_dims.size(), v64_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v64);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v64" << std::endl;
    return nullptr;
  }

  uint32_t v65 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> v65_dims = {{1, 1001}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/-53,
    /*scale=*/0.07755904644727707f,
    v65_dims.size(), v65_dims.data(),
    /*data=*/nullptr,
    XNN_INVALID_VALUE_ID, /*flags=*/0, &v65);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v65" << std::endl;
    return nullptr;
  }

  uint32_t v66 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> v66_dims = {{1, 1001}};
  status = xnn_define_tensor_value(
    subgraph, xnn_datatype_fp32,
    v66_dims.size(), v66_dims.data(),
    /*data=*/nullptr,
    1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &v66);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor v66" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(864, int8_t)> w67_data;
  uint32_t w67 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w67_dims = {{32, 3, 3, 3}};
  static std::array<float, 32> w67_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w67_scale.begin(), w67_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w67_scale.data(),
    w67_dims.size(), 0, w67_dims.data(),
    /*data=*/w67_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w67);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w67" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(32, int32_t)> w68_data;
  uint32_t w68 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w68_dims = {{32}};
  static std::array<float, 32> w68_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w68_scale.begin(), w68_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w68_scale.data(),
    w68_dims.size(), 0, w68_dims.data(),
    /*data=*/w68_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w68);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w68" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(288, int8_t)> w69_data;
  uint32_t w69 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w69_dims = {{1, 3, 3, 32}};
  static std::array<float, 32> w69_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w69_scale.begin(), w69_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w69_scale.data(),
    w69_dims.size(), 3, w69_dims.data(),
    /*data=*/w69_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w69);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w69" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(32, int32_t)> w70_data;
  uint32_t w70 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w70_dims = {{32}};
  static std::array<float, 32> w70_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w70_scale.begin(), w70_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w70_scale.data(),
    w70_dims.size(), 0, w70_dims.data(),
    /*data=*/w70_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w70);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w70" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(512, int8_t)> w71_data;
  uint32_t w71 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w71_dims = {{16, 1, 1, 32}};
  static std::array<float, 16> w71_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w71_scale.begin(), w71_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w71_scale.data(),
    w71_dims.size(), 0, w71_dims.data(),
    /*data=*/w71_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w71);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w71" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(16, int32_t)> w72_data;
  uint32_t w72 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w72_dims = {{16}};
  static std::array<float, 16> w72_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w72_scale.begin(), w72_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w72_scale.data(),
    w72_dims.size(), 0, w72_dims.data(),
    /*data=*/w72_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w72);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w72" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(1536, int8_t)> w73_data;
  uint32_t w73 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w73_dims = {{96, 1, 1, 16}};
  static std::array<float, 96> w73_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w73_scale.begin(), w73_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w73_scale.data(),
    w73_dims.size(), 0, w73_dims.data(),
    /*data=*/w73_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w73);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w73" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(96, int32_t)> w74_data;
  uint32_t w74 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w74_dims = {{96}};
  static std::array<float, 96> w74_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w74_scale.begin(), w74_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w74_scale.data(),
    w74_dims.size(), 0, w74_dims.data(),
    /*data=*/w74_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w74);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w74" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(864, int8_t)> w75_data;
  uint32_t w75 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w75_dims = {{1, 3, 3, 96}};
  static std::array<float, 96> w75_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w75_scale.begin(), w75_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w75_scale.data(),
    w75_dims.size(), 3, w75_dims.data(),
    /*data=*/w75_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w75);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w75" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(96, int32_t)> w76_data;
  uint32_t w76 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w76_dims = {{96}};
  static std::array<float, 96> w76_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w76_scale.begin(), w76_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w76_scale.data(),
    w76_dims.size(), 0, w76_dims.data(),
    /*data=*/w76_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w76);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w76" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(2304, int8_t)> w77_data;
  uint32_t w77 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w77_dims = {{24, 1, 1, 96}};
  static std::array<float, 24> w77_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w77_scale.begin(), w77_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w77_scale.data(),
    w77_dims.size(), 0, w77_dims.data(),
    /*data=*/w77_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w77);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w77" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(24, int32_t)> w78_data;
  uint32_t w78 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w78_dims = {{24}};
  static std::array<float, 24> w78_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w78_scale.begin(), w78_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w78_scale.data(),
    w78_dims.size(), 0, w78_dims.data(),
    /*data=*/w78_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w78);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w78" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(3456, int8_t)> w79_data;
  uint32_t w79 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w79_dims = {{144, 1, 1, 24}};
  static std::array<float, 144> w79_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w79_scale.begin(), w79_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w79_scale.data(),
    w79_dims.size(), 0, w79_dims.data(),
    /*data=*/w79_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w79);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w79" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(144, int32_t)> w80_data;
  uint32_t w80 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w80_dims = {{144}};
  static std::array<float, 144> w80_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w80_scale.begin(), w80_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w80_scale.data(),
    w80_dims.size(), 0, w80_dims.data(),
    /*data=*/w80_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w80);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w80" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(1296, int8_t)> w81_data;
  uint32_t w81 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w81_dims = {{1, 3, 3, 144}};
  static std::array<float, 144> w81_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w81_scale.begin(), w81_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w81_scale.data(),
    w81_dims.size(), 3, w81_dims.data(),
    /*data=*/w81_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w81);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w81" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(144, int32_t)> w82_data;
  uint32_t w82 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w82_dims = {{144}};
  static std::array<float, 144> w82_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w82_scale.begin(), w82_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w82_scale.data(),
    w82_dims.size(), 0, w82_dims.data(),
    /*data=*/w82_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w82);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w82" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(3456, int8_t)> w83_data;
  uint32_t w83 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w83_dims = {{24, 1, 1, 144}};
  static std::array<float, 24> w83_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w83_scale.begin(), w83_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w83_scale.data(),
    w83_dims.size(), 0, w83_dims.data(),
    /*data=*/w83_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w83);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w83" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(24, int32_t)> w84_data;
  uint32_t w84 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w84_dims = {{24}};
  static std::array<float, 24> w84_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w84_scale.begin(), w84_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w84_scale.data(),
    w84_dims.size(), 0, w84_dims.data(),
    /*data=*/w84_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w84);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w84" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(3456, int8_t)> w85_data;
  uint32_t w85 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w85_dims = {{144, 1, 1, 24}};
  static std::array<float, 144> w85_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w85_scale.begin(), w85_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w85_scale.data(),
    w85_dims.size(), 0, w85_dims.data(),
    /*data=*/w85_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w85);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w85" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(144, int32_t)> w86_data;
  uint32_t w86 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w86_dims = {{144}};
  static std::array<float, 144> w86_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w86_scale.begin(), w86_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w86_scale.data(),
    w86_dims.size(), 0, w86_dims.data(),
    /*data=*/w86_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w86);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w86" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(1296, int8_t)> w87_data;
  uint32_t w87 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w87_dims = {{1, 3, 3, 144}};
  static std::array<float, 144> w87_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w87_scale.begin(), w87_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w87_scale.data(),
    w87_dims.size(), 3, w87_dims.data(),
    /*data=*/w87_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w87);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w87" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(144, int32_t)> w88_data;
  uint32_t w88 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w88_dims = {{144}};
  static std::array<float, 144> w88_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w88_scale.begin(), w88_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w88_scale.data(),
    w88_dims.size(), 0, w88_dims.data(),
    /*data=*/w88_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w88);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w88" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(4608, int8_t)> w89_data;
  uint32_t w89 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w89_dims = {{32, 1, 1, 144}};
  static std::array<float, 32> w89_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w89_scale.begin(), w89_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w89_scale.data(),
    w89_dims.size(), 0, w89_dims.data(),
    /*data=*/w89_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w89);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w89" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(32, int32_t)> w90_data;
  uint32_t w90 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w90_dims = {{32}};
  static std::array<float, 32> w90_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w90_scale.begin(), w90_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w90_scale.data(),
    w90_dims.size(), 0, w90_dims.data(),
    /*data=*/w90_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w90);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w90" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(6144, int8_t)> w91_data;
  uint32_t w91 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w91_dims = {{192, 1, 1, 32}};
  static std::array<float, 192> w91_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w91_scale.begin(), w91_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w91_scale.data(),
    w91_dims.size(), 0, w91_dims.data(),
    /*data=*/w91_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w91);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w91" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w92_data;
  uint32_t w92 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w92_dims = {{192}};
  static std::array<float, 192> w92_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w92_scale.begin(), w92_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w92_scale.data(),
    w92_dims.size(), 0, w92_dims.data(),
    /*data=*/w92_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w92);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w92" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(1728, int8_t)> w93_data;
  uint32_t w93 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w93_dims = {{1, 3, 3, 192}};
  static std::array<float, 192> w93_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w93_scale.begin(), w93_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w93_scale.data(),
    w93_dims.size(), 3, w93_dims.data(),
    /*data=*/w93_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w93);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w93" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w94_data;
  uint32_t w94 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w94_dims = {{192}};
  static std::array<float, 192> w94_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w94_scale.begin(), w94_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w94_scale.data(),
    w94_dims.size(), 0, w94_dims.data(),
    /*data=*/w94_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w94);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w94" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(6144, int8_t)> w95_data;
  uint32_t w95 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w95_dims = {{32, 1, 1, 192}};
  static std::array<float, 32> w95_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w95_scale.begin(), w95_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w95_scale.data(),
    w95_dims.size(), 0, w95_dims.data(),
    /*data=*/w95_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w95);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w95" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(32, int32_t)> w96_data;
  uint32_t w96 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w96_dims = {{32}};
  static std::array<float, 32> w96_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w96_scale.begin(), w96_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w96_scale.data(),
    w96_dims.size(), 0, w96_dims.data(),
    /*data=*/w96_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w96);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w96" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(6144, int8_t)> w97_data;
  uint32_t w97 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w97_dims = {{192, 1, 1, 32}};
  static std::array<float, 192> w97_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w97_scale.begin(), w97_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w97_scale.data(),
    w97_dims.size(), 0, w97_dims.data(),
    /*data=*/w97_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w97);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w97" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w98_data;
  uint32_t w98 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w98_dims = {{192}};
  static std::array<float, 192> w98_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w98_scale.begin(), w98_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w98_scale.data(),
    w98_dims.size(), 0, w98_dims.data(),
    /*data=*/w98_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w98);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w98" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(1728, int8_t)> w99_data;
  uint32_t w99 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w99_dims = {{1, 3, 3, 192}};
  static std::array<float, 192> w99_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w99_scale.begin(), w99_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w99_scale.data(),
    w99_dims.size(), 3, w99_dims.data(),
    /*data=*/w99_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w99);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w99" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w100_data;
  uint32_t w100 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w100_dims = {{192}};
  static std::array<float, 192> w100_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w100_scale.begin(), w100_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w100_scale.data(),
    w100_dims.size(), 0, w100_dims.data(),
    /*data=*/w100_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w100);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w100" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(6144, int8_t)> w101_data;
  uint32_t w101 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w101_dims = {{32, 1, 1, 192}};
  static std::array<float, 32> w101_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w101_scale.begin(), w101_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w101_scale.data(),
    w101_dims.size(), 0, w101_dims.data(),
    /*data=*/w101_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w101);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w101" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(32, int32_t)> w102_data;
  uint32_t w102 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w102_dims = {{32}};
  static std::array<float, 32> w102_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w102_scale.begin(), w102_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w102_scale.data(),
    w102_dims.size(), 0, w102_dims.data(),
    /*data=*/w102_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w102);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w102" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(6144, int8_t)> w103_data;
  uint32_t w103 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w103_dims = {{192, 1, 1, 32}};
  static std::array<float, 192> w103_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w103_scale.begin(), w103_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w103_scale.data(),
    w103_dims.size(), 0, w103_dims.data(),
    /*data=*/w103_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w103);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w103" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w104_data;
  uint32_t w104 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w104_dims = {{192}};
  static std::array<float, 192> w104_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w104_scale.begin(), w104_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w104_scale.data(),
    w104_dims.size(), 0, w104_dims.data(),
    /*data=*/w104_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w104);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w104" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(1728, int8_t)> w105_data;
  uint32_t w105 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w105_dims = {{1, 3, 3, 192}};
  static std::array<float, 192> w105_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w105_scale.begin(), w105_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w105_scale.data(),
    w105_dims.size(), 3, w105_dims.data(),
    /*data=*/w105_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w105);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w105" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w106_data;
  uint32_t w106 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w106_dims = {{192}};
  static std::array<float, 192> w106_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w106_scale.begin(), w106_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w106_scale.data(),
    w106_dims.size(), 0, w106_dims.data(),
    /*data=*/w106_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w106);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w106" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(12288, int8_t)> w107_data;
  uint32_t w107 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w107_dims = {{64, 1, 1, 192}};
  static std::array<float, 64> w107_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w107_scale.begin(), w107_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w107_scale.data(),
    w107_dims.size(), 0, w107_dims.data(),
    /*data=*/w107_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w107);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w107" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(64, int32_t)> w108_data;
  uint32_t w108 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w108_dims = {{64}};
  static std::array<float, 64> w108_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w108_scale.begin(), w108_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w108_scale.data(),
    w108_dims.size(), 0, w108_dims.data(),
    /*data=*/w108_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w108);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w108" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(24576, int8_t)> w109_data;
  uint32_t w109 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w109_dims = {{384, 1, 1, 64}};
  static std::array<float, 384> w109_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w109_scale.begin(), w109_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w109_scale.data(),
    w109_dims.size(), 0, w109_dims.data(),
    /*data=*/w109_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w109);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w109" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w110_data;
  uint32_t w110 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w110_dims = {{384}};
  static std::array<float, 384> w110_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w110_scale.begin(), w110_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w110_scale.data(),
    w110_dims.size(), 0, w110_dims.data(),
    /*data=*/w110_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w110);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w110" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(3456, int8_t)> w111_data;
  uint32_t w111 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w111_dims = {{1, 3, 3, 384}};
  static std::array<float, 384> w111_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w111_scale.begin(), w111_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w111_scale.data(),
    w111_dims.size(), 3, w111_dims.data(),
    /*data=*/w111_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w111);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w111" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w112_data;
  uint32_t w112 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w112_dims = {{384}};
  static std::array<float, 384> w112_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w112_scale.begin(), w112_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w112_scale.data(),
    w112_dims.size(), 0, w112_dims.data(),
    /*data=*/w112_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w112);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w112" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(24576, int8_t)> w113_data;
  uint32_t w113 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w113_dims = {{64, 1, 1, 384}};
  static std::array<float, 64> w113_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w113_scale.begin(), w113_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w113_scale.data(),
    w113_dims.size(), 0, w113_dims.data(),
    /*data=*/w113_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w113);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w113" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(64, int32_t)> w114_data;
  uint32_t w114 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w114_dims = {{64}};
  static std::array<float, 64> w114_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w114_scale.begin(), w114_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w114_scale.data(),
    w114_dims.size(), 0, w114_dims.data(),
    /*data=*/w114_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w114);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w114" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(24576, int8_t)> w115_data;
  uint32_t w115 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w115_dims = {{384, 1, 1, 64}};
  static std::array<float, 384> w115_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w115_scale.begin(), w115_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w115_scale.data(),
    w115_dims.size(), 0, w115_dims.data(),
    /*data=*/w115_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w115);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w115" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w116_data;
  uint32_t w116 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w116_dims = {{384}};
  static std::array<float, 384> w116_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w116_scale.begin(), w116_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w116_scale.data(),
    w116_dims.size(), 0, w116_dims.data(),
    /*data=*/w116_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w116);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w116" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(3456, int8_t)> w117_data;
  uint32_t w117 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w117_dims = {{1, 3, 3, 384}};
  static std::array<float, 384> w117_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w117_scale.begin(), w117_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w117_scale.data(),
    w117_dims.size(), 3, w117_dims.data(),
    /*data=*/w117_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w117);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w117" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w118_data;
  uint32_t w118 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w118_dims = {{384}};
  static std::array<float, 384> w118_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w118_scale.begin(), w118_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w118_scale.data(),
    w118_dims.size(), 0, w118_dims.data(),
    /*data=*/w118_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w118);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w118" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(24576, int8_t)> w119_data;
  uint32_t w119 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w119_dims = {{64, 1, 1, 384}};
  static std::array<float, 64> w119_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w119_scale.begin(), w119_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w119_scale.data(),
    w119_dims.size(), 0, w119_dims.data(),
    /*data=*/w119_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w119);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w119" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(64, int32_t)> w120_data;
  uint32_t w120 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w120_dims = {{64}};
  static std::array<float, 64> w120_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w120_scale.begin(), w120_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w120_scale.data(),
    w120_dims.size(), 0, w120_dims.data(),
    /*data=*/w120_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w120);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w120" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(24576, int8_t)> w121_data;
  uint32_t w121 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w121_dims = {{384, 1, 1, 64}};
  static std::array<float, 384> w121_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w121_scale.begin(), w121_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w121_scale.data(),
    w121_dims.size(), 0, w121_dims.data(),
    /*data=*/w121_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w121);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w121" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w122_data;
  uint32_t w122 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w122_dims = {{384}};
  static std::array<float, 384> w122_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w122_scale.begin(), w122_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w122_scale.data(),
    w122_dims.size(), 0, w122_dims.data(),
    /*data=*/w122_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w122);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w122" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(3456, int8_t)> w123_data;
  uint32_t w123 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w123_dims = {{1, 3, 3, 384}};
  static std::array<float, 384> w123_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w123_scale.begin(), w123_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w123_scale.data(),
    w123_dims.size(), 3, w123_dims.data(),
    /*data=*/w123_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w123);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w123" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w124_data;
  uint32_t w124 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w124_dims = {{384}};
  static std::array<float, 384> w124_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w124_scale.begin(), w124_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w124_scale.data(),
    w124_dims.size(), 0, w124_dims.data(),
    /*data=*/w124_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w124);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w124" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(24576, int8_t)> w125_data;
  uint32_t w125 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w125_dims = {{64, 1, 1, 384}};
  static std::array<float, 64> w125_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w125_scale.begin(), w125_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w125_scale.data(),
    w125_dims.size(), 0, w125_dims.data(),
    /*data=*/w125_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w125);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w125" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(64, int32_t)> w126_data;
  uint32_t w126 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w126_dims = {{64}};
  static std::array<float, 64> w126_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w126_scale.begin(), w126_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w126_scale.data(),
    w126_dims.size(), 0, w126_dims.data(),
    /*data=*/w126_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w126);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w126" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(24576, int8_t)> w127_data;
  uint32_t w127 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w127_dims = {{384, 1, 1, 64}};
  static std::array<float, 384> w127_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w127_scale.begin(), w127_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w127_scale.data(),
    w127_dims.size(), 0, w127_dims.data(),
    /*data=*/w127_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w127);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w127" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w128_data;
  uint32_t w128 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w128_dims = {{384}};
  static std::array<float, 384> w128_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w128_scale.begin(), w128_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w128_scale.data(),
    w128_dims.size(), 0, w128_dims.data(),
    /*data=*/w128_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w128);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w128" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(3456, int8_t)> w129_data;
  uint32_t w129 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w129_dims = {{1, 3, 3, 384}};
  static std::array<float, 384> w129_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w129_scale.begin(), w129_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w129_scale.data(),
    w129_dims.size(), 3, w129_dims.data(),
    /*data=*/w129_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w129);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w129" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w130_data;
  uint32_t w130 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w130_dims = {{384}};
  static std::array<float, 384> w130_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w130_scale.begin(), w130_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w130_scale.data(),
    w130_dims.size(), 0, w130_dims.data(),
    /*data=*/w130_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w130);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w130" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(36864, int8_t)> w131_data;
  uint32_t w131 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w131_dims = {{96, 1, 1, 384}};
  static std::array<float, 96> w131_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w131_scale.begin(), w131_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w131_scale.data(),
    w131_dims.size(), 0, w131_dims.data(),
    /*data=*/w131_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w131);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w131" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(96, int32_t)> w132_data;
  uint32_t w132 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w132_dims = {{96}};
  static std::array<float, 96> w132_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w132_scale.begin(), w132_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w132_scale.data(),
    w132_dims.size(), 0, w132_dims.data(),
    /*data=*/w132_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w132);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w132" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(55296, int8_t)> w133_data;
  uint32_t w133 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w133_dims = {{576, 1, 1, 96}};
  static std::array<float, 576> w133_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w133_scale.begin(), w133_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w133_scale.data(),
    w133_dims.size(), 0, w133_dims.data(),
    /*data=*/w133_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w133);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w133" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w134_data;
  uint32_t w134 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w134_dims = {{576}};
  static std::array<float, 576> w134_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w134_scale.begin(), w134_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w134_scale.data(),
    w134_dims.size(), 0, w134_dims.data(),
    /*data=*/w134_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w134);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w134" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(5184, int8_t)> w135_data;
  uint32_t w135 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w135_dims = {{1, 3, 3, 576}};
  static std::array<float, 576> w135_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w135_scale.begin(), w135_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w135_scale.data(),
    w135_dims.size(), 3, w135_dims.data(),
    /*data=*/w135_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w135);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w135" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w136_data;
  uint32_t w136 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w136_dims = {{576}};
  static std::array<float, 576> w136_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w136_scale.begin(), w136_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w136_scale.data(),
    w136_dims.size(), 0, w136_dims.data(),
    /*data=*/w136_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w136);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w136" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(55296, int8_t)> w137_data;
  uint32_t w137 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w137_dims = {{96, 1, 1, 576}};
  static std::array<float, 96> w137_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w137_scale.begin(), w137_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w137_scale.data(),
    w137_dims.size(), 0, w137_dims.data(),
    /*data=*/w137_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w137);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w137" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(96, int32_t)> w138_data;
  uint32_t w138 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w138_dims = {{96}};
  static std::array<float, 96> w138_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w138_scale.begin(), w138_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w138_scale.data(),
    w138_dims.size(), 0, w138_dims.data(),
    /*data=*/w138_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w138);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w138" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(55296, int8_t)> w139_data;
  uint32_t w139 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w139_dims = {{576, 1, 1, 96}};
  static std::array<float, 576> w139_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w139_scale.begin(), w139_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w139_scale.data(),
    w139_dims.size(), 0, w139_dims.data(),
    /*data=*/w139_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w139);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w139" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w140_data;
  uint32_t w140 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w140_dims = {{576}};
  static std::array<float, 576> w140_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w140_scale.begin(), w140_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w140_scale.data(),
    w140_dims.size(), 0, w140_dims.data(),
    /*data=*/w140_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w140);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w140" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(5184, int8_t)> w141_data;
  uint32_t w141 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w141_dims = {{1, 3, 3, 576}};
  static std::array<float, 576> w141_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w141_scale.begin(), w141_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w141_scale.data(),
    w141_dims.size(), 3, w141_dims.data(),
    /*data=*/w141_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w141);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w141" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w142_data;
  uint32_t w142 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w142_dims = {{576}};
  static std::array<float, 576> w142_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w142_scale.begin(), w142_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w142_scale.data(),
    w142_dims.size(), 0, w142_dims.data(),
    /*data=*/w142_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w142);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w142" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(55296, int8_t)> w143_data;
  uint32_t w143 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w143_dims = {{96, 1, 1, 576}};
  static std::array<float, 96> w143_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w143_scale.begin(), w143_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w143_scale.data(),
    w143_dims.size(), 0, w143_dims.data(),
    /*data=*/w143_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w143);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w143" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(96, int32_t)> w144_data;
  uint32_t w144 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w144_dims = {{96}};
  static std::array<float, 96> w144_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w144_scale.begin(), w144_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w144_scale.data(),
    w144_dims.size(), 0, w144_dims.data(),
    /*data=*/w144_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w144);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w144" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(55296, int8_t)> w145_data;
  uint32_t w145 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w145_dims = {{576, 1, 1, 96}};
  static std::array<float, 576> w145_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w145_scale.begin(), w145_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w145_scale.data(),
    w145_dims.size(), 0, w145_dims.data(),
    /*data=*/w145_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w145);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w145" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w146_data;
  uint32_t w146 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w146_dims = {{576}};
  static std::array<float, 576> w146_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w146_scale.begin(), w146_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w146_scale.data(),
    w146_dims.size(), 0, w146_dims.data(),
    /*data=*/w146_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w146);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w146" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(5184, int8_t)> w147_data;
  uint32_t w147 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w147_dims = {{1, 3, 3, 576}};
  static std::array<float, 576> w147_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w147_scale.begin(), w147_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w147_scale.data(),
    w147_dims.size(), 3, w147_dims.data(),
    /*data=*/w147_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w147);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w147" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w148_data;
  uint32_t w148 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w148_dims = {{576}};
  static std::array<float, 576> w148_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w148_scale.begin(), w148_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w148_scale.data(),
    w148_dims.size(), 0, w148_dims.data(),
    /*data=*/w148_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w148);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w148" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(92160, int8_t)> w149_data;
  uint32_t w149 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w149_dims = {{160, 1, 1, 576}};
  static std::array<float, 160> w149_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w149_scale.begin(), w149_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w149_scale.data(),
    w149_dims.size(), 0, w149_dims.data(),
    /*data=*/w149_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w149);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w149" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(160, int32_t)> w150_data;
  uint32_t w150 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w150_dims = {{160}};
  static std::array<float, 160> w150_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w150_scale.begin(), w150_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w150_scale.data(),
    w150_dims.size(), 0, w150_dims.data(),
    /*data=*/w150_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w150);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w150" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(153600, int8_t)> w151_data;
  uint32_t w151 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w151_dims = {{960, 1, 1, 160}};
  static std::array<float, 960> w151_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w151_scale.begin(), w151_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w151_scale.data(),
    w151_dims.size(), 0, w151_dims.data(),
    /*data=*/w151_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w151);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w151" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w152_data;
  uint32_t w152 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w152_dims = {{960}};
  static std::array<float, 960> w152_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w152_scale.begin(), w152_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w152_scale.data(),
    w152_dims.size(), 0, w152_dims.data(),
    /*data=*/w152_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w152);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w152" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(8640, int8_t)> w153_data;
  uint32_t w153 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w153_dims = {{1, 3, 3, 960}};
  static std::array<float, 960> w153_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w153_scale.begin(), w153_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w153_scale.data(),
    w153_dims.size(), 3, w153_dims.data(),
    /*data=*/w153_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w153);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w153" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w154_data;
  uint32_t w154 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w154_dims = {{960}};
  static std::array<float, 960> w154_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w154_scale.begin(), w154_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w154_scale.data(),
    w154_dims.size(), 0, w154_dims.data(),
    /*data=*/w154_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w154);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w154" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(153600, int8_t)> w155_data;
  uint32_t w155 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w155_dims = {{160, 1, 1, 960}};
  static std::array<float, 160> w155_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w155_scale.begin(), w155_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w155_scale.data(),
    w155_dims.size(), 0, w155_dims.data(),
    /*data=*/w155_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w155);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w155" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(160, int32_t)> w156_data;
  uint32_t w156 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w156_dims = {{160}};
  static std::array<float, 160> w156_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w156_scale.begin(), w156_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w156_scale.data(),
    w156_dims.size(), 0, w156_dims.data(),
    /*data=*/w156_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w156);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w156" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(153600, int8_t)> w157_data;
  uint32_t w157 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w157_dims = {{960, 1, 1, 160}};
  static std::array<float, 960> w157_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w157_scale.begin(), w157_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w157_scale.data(),
    w157_dims.size(), 0, w157_dims.data(),
    /*data=*/w157_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w157);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w157" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w158_data;
  uint32_t w158 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w158_dims = {{960}};
  static std::array<float, 960> w158_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w158_scale.begin(), w158_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w158_scale.data(),
    w158_dims.size(), 0, w158_dims.data(),
    /*data=*/w158_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w158);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w158" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(8640, int8_t)> w159_data;
  uint32_t w159 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w159_dims = {{1, 3, 3, 960}};
  static std::array<float, 960> w159_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w159_scale.begin(), w159_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w159_scale.data(),
    w159_dims.size(), 3, w159_dims.data(),
    /*data=*/w159_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w159);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w159" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w160_data;
  uint32_t w160 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w160_dims = {{960}};
  static std::array<float, 960> w160_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w160_scale.begin(), w160_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w160_scale.data(),
    w160_dims.size(), 0, w160_dims.data(),
    /*data=*/w160_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w160);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w160" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(153600, int8_t)> w161_data;
  uint32_t w161 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w161_dims = {{160, 1, 1, 960}};
  static std::array<float, 160> w161_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w161_scale.begin(), w161_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w161_scale.data(),
    w161_dims.size(), 0, w161_dims.data(),
    /*data=*/w161_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w161);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w161" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(160, int32_t)> w162_data;
  uint32_t w162 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w162_dims = {{160}};
  static std::array<float, 160> w162_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w162_scale.begin(), w162_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w162_scale.data(),
    w162_dims.size(), 0, w162_dims.data(),
    /*data=*/w162_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w162);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w162" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(153600, int8_t)> w163_data;
  uint32_t w163 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w163_dims = {{960, 1, 1, 160}};
  static std::array<float, 960> w163_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w163_scale.begin(), w163_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w163_scale.data(),
    w163_dims.size(), 0, w163_dims.data(),
    /*data=*/w163_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w163);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w163" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w164_data;
  uint32_t w164 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w164_dims = {{960}};
  static std::array<float, 960> w164_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w164_scale.begin(), w164_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w164_scale.data(),
    w164_dims.size(), 0, w164_dims.data(),
    /*data=*/w164_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w164);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w164" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(8640, int8_t)> w165_data;
  uint32_t w165 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w165_dims = {{1, 3, 3, 960}};
  static std::array<float, 960> w165_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w165_scale.begin(), w165_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w165_scale.data(),
    w165_dims.size(), 3, w165_dims.data(),
    /*data=*/w165_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w165);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w165" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w166_data;
  uint32_t w166 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w166_dims = {{960}};
  static std::array<float, 960> w166_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w166_scale.begin(), w166_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w166_scale.data(),
    w166_dims.size(), 0, w166_dims.data(),
    /*data=*/w166_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w166);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w166" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(307200, int8_t)> w167_data;
  uint32_t w167 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w167_dims = {{320, 1, 1, 960}};
  static std::array<float, 320> w167_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w167_scale.begin(), w167_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w167_scale.data(),
    w167_dims.size(), 0, w167_dims.data(),
    /*data=*/w167_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w167);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w167" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(320, int32_t)> w168_data;
  uint32_t w168 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w168_dims = {{320}};
  static std::array<float, 320> w168_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w168_scale.begin(), w168_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w168_scale.data(),
    w168_dims.size(), 0, w168_dims.data(),
    /*data=*/w168_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w168);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w168" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(409600, int8_t)> w169_data;
  uint32_t w169 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 4> w169_dims = {{1280, 1, 1, 320}};
  static std::array<float, 1280> w169_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w169_scale.begin(), w169_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint8,
    /*scale=*/w169_scale.data(),
    w169_dims.size(), 0, w169_dims.data(),
    /*data=*/w169_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w169);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w169" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(1280, int32_t)> w170_data;
  uint32_t w170 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w170_dims = {{1280}};
  static std::array<float, 1280> w170_scale;
  {
    auto scalerng = std::bind(std::uniform_real_distribution<float>(0.01f, 1.0f), std::ref(rng));
    std::generate(w170_scale.begin(), w170_scale.end(), std::ref(scalerng));
  }
  status = xnn_define_channelwise_quantized_tensor_value(
    subgraph, xnn_datatype_qcint32,
    /*scale=*/w170_scale.data(),
    w170_dims.size(), 0, w170_dims.data(),
    /*data=*/w170_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w170);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w170" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int8_t, XNN_PAD_EXTRA_BYTES(1281280, int8_t)> w171_data;
  uint32_t w171 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 2> w171_dims = {{1001, 1280}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint8,
    /*zero_point=*/0,
    /*scale=*/0.004167426843196154f,
    w171_dims.size(), w171_dims.data(),
    /*data=*/w171_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w171);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w171" << std::endl;
    return nullptr;
  }

  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(1001, int32_t)> w172_data;
  uint32_t w172 = XNN_INVALID_VALUE_ID;
  std::array<size_t, 1> w172_dims = {{1001}};
  status = xnn_define_quantized_tensor_value(
    subgraph, xnn_datatype_qint32,
    /*zero_point=*/0,
    /*scale=*/7.704259769525379e-05f,
    w172_dims.size(), w172_dims.data(),
    /*data=*/w172_data.data(),
    XNN_INVALID_VALUE_ID, /*flags=*/0, &w172);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor w172" << std::endl;
    return nullptr;
  }

  auto qs8rng = std::bind(std::uniform_int_distribution<int>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()), std::ref(rng));
  auto qc8rng = std::bind(std::uniform_int_distribution<int>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()), std::ref(rng));
  auto qs32rng = std::bind(std::uniform_int_distribution<int>(-10000, 10000), std::ref(rng));
  auto qc32rng = std::bind(std::uniform_int_distribution<int>(-10000, 10000), std::ref(rng));
  std::generate(w67_data.begin(), w67_data.end(), std::ref(qc8rng));
  std::generate(w68_data.begin(), w68_data.end(), std::ref(qc32rng));
  std::generate(w69_data.begin(), w69_data.end(), std::ref(qc8rng));
  std::generate(w70_data.begin(), w70_data.end(), std::ref(qc32rng));
  std::generate(w71_data.begin(), w71_data.end(), std::ref(qc8rng));
  std::generate(w72_data.begin(), w72_data.end(), std::ref(qc32rng));
  std::generate(w73_data.begin(), w73_data.end(), std::ref(qc8rng));
  std::generate(w74_data.begin(), w74_data.end(), std::ref(qc32rng));
  std::generate(w75_data.begin(), w75_data.end(), std::ref(qc8rng));
  std::generate(w76_data.begin(), w76_data.end(), std::ref(qc32rng));
  std::generate(w77_data.begin(), w77_data.end(), std::ref(qc8rng));
  std::generate(w78_data.begin(), w78_data.end(), std::ref(qc32rng));
  std::generate(w79_data.begin(), w79_data.end(), std::ref(qc8rng));
  std::generate(w80_data.begin(), w80_data.end(), std::ref(qc32rng));
  std::generate(w81_data.begin(), w81_data.end(), std::ref(qc8rng));
  std::generate(w82_data.begin(), w82_data.end(), std::ref(qc32rng));
  std::generate(w83_data.begin(), w83_data.end(), std::ref(qc8rng));
  std::generate(w84_data.begin(), w84_data.end(), std::ref(qc32rng));
  std::generate(w85_data.begin(), w85_data.end(), std::ref(qc8rng));
  std::generate(w86_data.begin(), w86_data.end(), std::ref(qc32rng));
  std::generate(w87_data.begin(), w87_data.end(), std::ref(qc8rng));
  std::generate(w88_data.begin(), w88_data.end(), std::ref(qc32rng));
  std::generate(w89_data.begin(), w89_data.end(), std::ref(qc8rng));
  std::generate(w90_data.begin(), w90_data.end(), std::ref(qc32rng));
  std::generate(w91_data.begin(), w91_data.end(), std::ref(qc8rng));
  std::generate(w92_data.begin(), w92_data.end(), std::ref(qc32rng));
  std::generate(w93_data.begin(), w93_data.end(), std::ref(qc8rng));
  std::generate(w94_data.begin(), w94_data.end(), std::ref(qc32rng));
  std::generate(w95_data.begin(), w95_data.end(), std::ref(qc8rng));
  std::generate(w96_data.begin(), w96_data.end(), std::ref(qc32rng));
  std::generate(w97_data.begin(), w97_data.end(), std::ref(qc8rng));
  std::generate(w98_data.begin(), w98_data.end(), std::ref(qc32rng));
  std::generate(w99_data.begin(), w99_data.end(), std::ref(qc8rng));
  std::generate(w100_data.begin(), w100_data.end(), std::ref(qc32rng));
  std::generate(w101_data.begin(), w101_data.end(), std::ref(qc8rng));
  std::generate(w102_data.begin(), w102_data.end(), std::ref(qc32rng));
  std::generate(w103_data.begin(), w103_data.end(), std::ref(qc8rng));
  std::generate(w104_data.begin(), w104_data.end(), std::ref(qc32rng));
  std::generate(w105_data.begin(), w105_data.end(), std::ref(qc8rng));
  std::generate(w106_data.begin(), w106_data.end(), std::ref(qc32rng));
  std::generate(w107_data.begin(), w107_data.end(), std::ref(qc8rng));
  std::generate(w108_data.begin(), w108_data.end(), std::ref(qc32rng));
  std::generate(w109_data.begin(), w109_data.end(), std::ref(qc8rng));
  std::generate(w110_data.begin(), w110_data.end(), std::ref(qc32rng));
  std::generate(w111_data.begin(), w111_data.end(), std::ref(qc8rng));
  std::generate(w112_data.begin(), w112_data.end(), std::ref(qc32rng));
  std::generate(w113_data.begin(), w113_data.end(), std::ref(qc8rng));
  std::generate(w114_data.begin(), w114_data.end(), std::ref(qc32rng));
  std::generate(w115_data.begin(), w115_data.end(), std::ref(qc8rng));
  std::generate(w116_data.begin(), w116_data.end(), std::ref(qc32rng));
  std::generate(w117_data.begin(), w117_data.end(), std::ref(qc8rng));
  std::generate(w118_data.begin(), w118_data.end(), std::ref(qc32rng));
  std::generate(w119_data.begin(), w119_data.end(), std::ref(qc8rng));
  std::generate(w120_data.begin(), w120_data.end(), std::ref(qc32rng));
  std::generate(w121_data.begin(), w121_data.end(), std::ref(qc8rng));
  std::generate(w122_data.begin(), w122_data.end(), std::ref(qc32rng));
  std::generate(w123_data.begin(), w123_data.end(), std::ref(qc8rng));
  std::generate(w124_data.begin(), w124_data.end(), std::ref(qc32rng));
  std::generate(w125_data.begin(), w125_data.end(), std::ref(qc8rng));
  std::generate(w126_data.begin(), w126_data.end(), std::ref(qc32rng));
  std::generate(w127_data.begin(), w127_data.end(), std::ref(qc8rng));
  std::generate(w128_data.begin(), w128_data.end(), std::ref(qc32rng));
  std::generate(w129_data.begin(), w129_data.end(), std::ref(qc8rng));
  std::generate(w130_data.begin(), w130_data.end(), std::ref(qc32rng));
  std::generate(w131_data.begin(), w131_data.end(), std::ref(qc8rng));
  std::generate(w132_data.begin(), w132_data.end(), std::ref(qc32rng));
  std::generate(w133_data.begin(), w133_data.end(), std::ref(qc8rng));
  std::generate(w134_data.begin(), w134_data.end(), std::ref(qc32rng));
  std::generate(w135_data.begin(), w135_data.end(), std::ref(qc8rng));
  std::generate(w136_data.begin(), w136_data.end(), std::ref(qc32rng));
  std::generate(w137_data.begin(), w137_data.end(), std::ref(qc8rng));
  std::generate(w138_data.begin(), w138_data.end(), std::ref(qc32rng));
  std::generate(w139_data.begin(), w139_data.end(), std::ref(qc8rng));
  std::generate(w140_data.begin(), w140_data.end(), std::ref(qc32rng));
  std::generate(w141_data.begin(), w141_data.end(), std::ref(qc8rng));
  std::generate(w142_data.begin(), w142_data.end(), std::ref(qc32rng));
  std::generate(w143_data.begin(), w143_data.end(), std::ref(qc8rng));
  std::generate(w144_data.begin(), w144_data.end(), std::ref(qc32rng));
  std::generate(w145_data.begin(), w145_data.end(), std::ref(qc8rng));
  std::generate(w146_data.begin(), w146_data.end(), std::ref(qc32rng));
  std::generate(w147_data.begin(), w147_data.end(), std::ref(qc8rng));
  std::generate(w148_data.begin(), w148_data.end(), std::ref(qc32rng));
  std::generate(w149_data.begin(), w149_data.end(), std::ref(qc8rng));
  std::generate(w150_data.begin(), w150_data.end(), std::ref(qc32rng));
  std::generate(w151_data.begin(), w151_data.end(), std::ref(qc8rng));
  std::generate(w152_data.begin(), w152_data.end(), std::ref(qc32rng));
  std::generate(w153_data.begin(), w153_data.end(), std::ref(qc8rng));
  std::generate(w154_data.begin(), w154_data.end(), std::ref(qc32rng));
  std::generate(w155_data.begin(), w155_data.end(), std::ref(qc8rng));
  std::generate(w156_data.begin(), w156_data.end(), std::ref(qc32rng));
  std::generate(w157_data.begin(), w157_data.end(), std::ref(qc8rng));
  std::generate(w158_data.begin(), w158_data.end(), std::ref(qc32rng));
  std::generate(w159_data.begin(), w159_data.end(), std::ref(qc8rng));
  std::generate(w160_data.begin(), w160_data.end(), std::ref(qc32rng));
  std::generate(w161_data.begin(), w161_data.end(), std::ref(qc8rng));
  std::generate(w162_data.begin(), w162_data.end(), std::ref(qc32rng));
  std::generate(w163_data.begin(), w163_data.end(), std::ref(qc8rng));
  std::generate(w164_data.begin(), w164_data.end(), std::ref(qc32rng));
  std::generate(w165_data.begin(), w165_data.end(), std::ref(qc8rng));
  std::generate(w166_data.begin(), w166_data.end(), std::ref(qc32rng));
  std::generate(w167_data.begin(), w167_data.end(), std::ref(qc8rng));
  std::generate(w168_data.begin(), w168_data.end(), std::ref(qc32rng));
  std::generate(w169_data.begin(), w169_data.end(), std::ref(qc8rng));
  std::generate(w170_data.begin(), w170_data.end(), std::ref(qc32rng));
  std::generate(w171_data.begin(), w171_data.end(), std::ref(qs8rng));
  std::generate(w172_data.begin(), w172_data.end(), std::ref(qs32rng));

  status = xnn_define_unary(
    subgraph,
    xnn_unary_convert,
    /*params=*/nullptr,
    v0,
    v1,
    0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #0" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/3,
    /*group_output_channels=*/32,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v1,
    w67,
    w68,
    v2,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #1" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/32,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v2,
    w69,
    w70,
    v3,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #2" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/16,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v3,
    w71,
    w72,
    v4,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #3" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/16,
    /*group_output_channels=*/96,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v4,
    w73,
    w74,
    v5,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #4" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/96,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v5,
    w75,
    w76,
    v6,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #5" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/24,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v6,
    w77,
    w78,
    v7,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #6" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/24,
    /*group_output_channels=*/144,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v7,
    w79,
    w80,
    v8,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #7" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/144,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v8,
    w81,
    w82,
    v9,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #8" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/144,
    /*group_output_channels=*/24,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v9,
    w83,
    w84,
    v10,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #9" << std::endl;
    return nullptr;
  }

  xnn_binary_params v11_params = { std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_add,
    &v11_params,
    v10,
    v7,
    v11,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #10" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/24,
    /*group_output_channels=*/144,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v11,
    w85,
    w86,
    v12,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #11" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/144,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v12,
    w87,
    w88,
    v13,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #12" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/144,
    /*group_output_channels=*/32,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v13,
    w89,
    w90,
    v14,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #13" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/192,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v14,
    w91,
    w92,
    v15,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #14" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/192,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v15,
    w93,
    w94,
    v16,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #15" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/192,
    /*group_output_channels=*/32,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v16,
    w95,
    w96,
    v17,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #16" << std::endl;
    return nullptr;
  }

  xnn_binary_params v18_params = { std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_add,
    &v18_params,
    v17,
    v14,
    v18,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #17" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/192,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v18,
    w97,
    w98,
    v19,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #18" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/192,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v19,
    w99,
    w100,
    v20,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #19" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/192,
    /*group_output_channels=*/32,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v20,
    w101,
    w102,
    v21,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #20" << std::endl;
    return nullptr;
  }

  xnn_binary_params v22_params = { std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_add,
    &v22_params,
    v21,
    v18,
    v22,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #21" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/192,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v22,
    w103,
    w104,
    v23,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #22" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/192,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v23,
    w105,
    w106,
    v24,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #23" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/192,
    /*group_output_channels=*/64,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v24,
    w107,
    w108,
    v25,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #24" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/384,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v25,
    w109,
    w110,
    v26,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #25" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/384,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v26,
    w111,
    w112,
    v27,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #26" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/384,
    /*group_output_channels=*/64,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v27,
    w113,
    w114,
    v28,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #27" << std::endl;
    return nullptr;
  }

  xnn_binary_params v29_params = { std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_add,
    &v29_params,
    v28,
    v25,
    v29,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #28" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/384,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v29,
    w115,
    w116,
    v30,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #29" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/384,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v30,
    w117,
    w118,
    v31,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #30" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/384,
    /*group_output_channels=*/64,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v31,
    w119,
    w120,
    v32,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #31" << std::endl;
    return nullptr;
  }

  xnn_binary_params v33_params = { std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_add,
    &v33_params,
    v32,
    v29,
    v33,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #32" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/384,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v33,
    w121,
    w122,
    v34,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #33" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/384,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v34,
    w123,
    w124,
    v35,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #34" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/384,
    /*group_output_channels=*/64,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v35,
    w125,
    w126,
    v36,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #35" << std::endl;
    return nullptr;
  }

  xnn_binary_params v37_params = { std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_add,
    &v37_params,
    v36,
    v33,
    v37,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #36" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/384,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v37,
    w127,
    w128,
    v38,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #37" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/384,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v38,
    w129,
    w130,
    v39,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #38" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/384,
    /*group_output_channels=*/96,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v39,
    w131,
    w132,
    v40,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #39" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/576,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v40,
    w133,
    w134,
    v41,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #40" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/576,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v41,
    w135,
    w136,
    v42,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #41" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/96,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v42,
    w137,
    w138,
    v43,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #42" << std::endl;
    return nullptr;
  }

  xnn_binary_params v44_params = { std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_add,
    &v44_params,
    v43,
    v40,
    v44,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #43" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/576,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v44,
    w139,
    w140,
    v45,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #44" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/576,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v45,
    w141,
    w142,
    v46,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #45" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/96,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v46,
    w143,
    w144,
    v47,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #46" << std::endl;
    return nullptr;
  }

  xnn_binary_params v48_params = { std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_add,
    &v48_params,
    v47,
    v44,
    v48,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #47" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/576,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v48,
    w145,
    w146,
    v49,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #48" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/576,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v49,
    w147,
    w148,
    v50,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #49" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/160,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v50,
    w149,
    w150,
    v51,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #50" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/160,
    /*group_output_channels=*/960,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v51,
    w151,
    w152,
    v52,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #51" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/960,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v52,
    w153,
    w154,
    v53,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #52" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/960,
    /*group_output_channels=*/160,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v53,
    w155,
    w156,
    v54,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #53" << std::endl;
    return nullptr;
  }

  xnn_binary_params v55_params = { std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_add,
    &v55_params,
    v54,
    v51,
    v55,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #54" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/160,
    /*group_output_channels=*/960,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v55,
    w157,
    w158,
    v56,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #55" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/960,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v56,
    w159,
    w160,
    v57,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #56" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/960,
    /*group_output_channels=*/160,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v57,
    w161,
    w162,
    v58,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #57" << std::endl;
    return nullptr;
  }

  xnn_binary_params v59_params = { std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max() };
  status = xnn_define_binary(
    subgraph,
    xnn_binary_add,
    &v59_params,
    v58,
    v55,
    v59,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #58" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/160,
    /*group_output_channels=*/960,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v59,
    w163,
    w164,
    v60,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #59" << std::endl;
    return nullptr;
  }

  status = xnn_define_depthwise_convolution_2d(
    subgraph,
    /*padding_top=*/1, /*padding_right=*/1, /*padding_bottom=*/1, /*padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*depth_multiplier=*/1,
    /*input_channels=*/960,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v60,
    w165,
    w166,
    v61,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #60" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/960,
    /*group_output_channels=*/320,
    /*output_min=*/-std::numeric_limits<float>::infinity(), /*output_max=*/std::numeric_limits<float>::infinity(),
    v61,
    w167,
    w168,
    v62,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #61" << std::endl;
    return nullptr;
  }

  status = xnn_define_convolution_2d(
    subgraph,
    /*padding_top=*/0, /*padding_right=*/0, /*padding_bottom=*/0, /*padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/320,
    /*group_output_channels=*/1280,
    /*output_min=*/0.0f, /*output_max=*/6.0f,
    v62,
    w169,
    w170,
    v63,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #62" << std::endl;
    return nullptr;
  }

  size_t reduction_axes_v64[2] = {1, 2};
  status = xnn_define_static_reduce(
    subgraph,
    xnn_reduce_mean,
    2,
    &reduction_axes_v64[0],
    v63,
    v64,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #63" << std::endl;
    return nullptr;
  }

  status = xnn_define_fully_connected(
    subgraph,
    /*output_min=*/std::numeric_limits<int8_t>::min(), /*output_max=*/std::numeric_limits<int8_t>::max(),
    v64,
    w171,
    w172,
    v65,
    /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #64" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(
    subgraph,
    xnn_unary_convert,
    /*params=*/nullptr,
    v65,
    v66,
    0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create node #65" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models
