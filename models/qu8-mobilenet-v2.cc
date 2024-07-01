// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!

#include "xnnpack.h"

#include <array>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "xnnpack/cache.h"
#include "xnnpack/common.h"
#include "xnnpack/models.h"

namespace models {

ExecutionPlan QU8MobileNetV2(pthreadpool_t threadpool) {
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(150528, uint8_t)> v0;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(401408, uint8_t)> v1;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(401408, uint8_t)> v2;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(200704, uint8_t)> v3;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1204224, uint8_t)> v4;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(301056, uint8_t)> v5;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v6;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(451584, uint8_t)> v7;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(451584, uint8_t)> v8;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v9;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v10;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(451584, uint8_t)> v11;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(112896, uint8_t)> v12;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(25088, uint8_t)> v13;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(150528, uint8_t)> v14;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(150528, uint8_t)> v15;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(25088, uint8_t)> v16;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(25088, uint8_t)> v17;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(150528, uint8_t)> v18;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(150528, uint8_t)> v19;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(25088, uint8_t)> v20;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(25088, uint8_t)> v21;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(150528, uint8_t)> v22;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(37632, uint8_t)> v23;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(12544, uint8_t)> v24;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v25;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v26;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(12544, uint8_t)> v27;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(12544, uint8_t)> v28;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v29;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v30;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(12544, uint8_t)> v31;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(12544, uint8_t)> v32;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v33;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v34;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(12544, uint8_t)> v35;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(12544, uint8_t)> v36;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v37;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(75264, uint8_t)> v38;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(18816, uint8_t)> v39;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(112896, uint8_t)> v40;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(112896, uint8_t)> v41;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(18816, uint8_t)> v42;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(18816, uint8_t)> v43;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(112896, uint8_t)> v44;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(112896, uint8_t)> v45;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(18816, uint8_t)> v46;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(18816, uint8_t)> v47;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(112896, uint8_t)> v48;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(28224, uint8_t)> v49;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(7840, uint8_t)> v50;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(47040, uint8_t)> v51;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(47040, uint8_t)> v52;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(7840, uint8_t)> v53;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(7840, uint8_t)> v54;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(47040, uint8_t)> v55;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(47040, uint8_t)> v56;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(7840, uint8_t)> v57;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(7840, uint8_t)> v58;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(47040, uint8_t)> v59;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(47040, uint8_t)> v60;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(15680, uint8_t)> v61;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(62720, uint8_t)> v62;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1280, uint8_t)> v63;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1001, uint8_t)> v64;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1001, uint8_t)> v65;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(864, uint8_t)> w66;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(32, int32_t)> w67;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(288, uint8_t)> w68;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(32, int32_t)> w69;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(512, uint8_t)> w70;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(16, int32_t)> w71;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1536, uint8_t)> w72;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(96, int32_t)> w73;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(864, uint8_t)> w74;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(96, int32_t)> w75;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(2304, uint8_t)> w76;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(24, int32_t)> w77;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(3456, uint8_t)> w78;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(144, int32_t)> w79;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1296, uint8_t)> w80;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(144, int32_t)> w81;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(3456, uint8_t)> w82;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(24, int32_t)> w83;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(3456, uint8_t)> w84;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(144, int32_t)> w85;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1296, uint8_t)> w86;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(144, int32_t)> w87;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(4608, uint8_t)> w88;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(32, int32_t)> w89;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(6144, uint8_t)> w90;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w91;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1728, uint8_t)> w92;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w93;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(6144, uint8_t)> w94;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(32, int32_t)> w95;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(6144, uint8_t)> w96;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w97;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1728, uint8_t)> w98;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w99;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(6144, uint8_t)> w100;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(32, int32_t)> w101;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(6144, uint8_t)> w102;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w103;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1728, uint8_t)> w104;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(192, int32_t)> w105;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(12288, uint8_t)> w106;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(64, int32_t)> w107;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(24576, uint8_t)> w108;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w109;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(3456, uint8_t)> w110;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w111;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(24576, uint8_t)> w112;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(64, int32_t)> w113;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(24576, uint8_t)> w114;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w115;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(3456, uint8_t)> w116;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w117;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(24576, uint8_t)> w118;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(64, int32_t)> w119;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(24576, uint8_t)> w120;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w121;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(3456, uint8_t)> w122;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w123;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(24576, uint8_t)> w124;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(64, int32_t)> w125;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(24576, uint8_t)> w126;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w127;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(3456, uint8_t)> w128;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(384, int32_t)> w129;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(36864, uint8_t)> w130;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(96, int32_t)> w131;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(55296, uint8_t)> w132;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w133;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(5184, uint8_t)> w134;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w135;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(55296, uint8_t)> w136;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(96, int32_t)> w137;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(55296, uint8_t)> w138;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w139;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(5184, uint8_t)> w140;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w141;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(55296, uint8_t)> w142;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(96, int32_t)> w143;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(55296, uint8_t)> w144;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w145;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(5184, uint8_t)> w146;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(576, int32_t)> w147;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(92160, uint8_t)> w148;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(160, int32_t)> w149;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(153600, uint8_t)> w150;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w151;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(8640, uint8_t)> w152;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w153;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(153600, uint8_t)> w154;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(160, int32_t)> w155;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(153600, uint8_t)> w156;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w157;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(8640, uint8_t)> w158;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w159;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(153600, uint8_t)> w160;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(160, int32_t)> w161;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(153600, uint8_t)> w162;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w163;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(8640, uint8_t)> w164;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(960, int32_t)> w165;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(307200, uint8_t)> w166;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(320, int32_t)> w167;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(409600, uint8_t)> w168;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(1280, int32_t)> w169;
  alignas(16) static std::array<uint8_t, XNN_PAD_EXTRA_BYTES(1281280, uint8_t)> w170;
  alignas(16) static std::array<int32_t, XNN_PAD_EXTRA_BYTES(1001, int32_t)> w171;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto qu8rng = std::bind(std::uniform_int_distribution<int>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()), std::ref(rng));
  auto qs32rng = std::bind(std::uniform_int_distribution<int>(-10000, 10000), std::ref(rng));
  std::generate(v0.begin(), v0.end(), std::ref(qu8rng));
  std::generate(v1.begin(), v1.end(), std::ref(qu8rng));
  std::generate(v2.begin(), v2.end(), std::ref(qu8rng));
  std::generate(v3.begin(), v3.end(), std::ref(qu8rng));
  std::generate(v4.begin(), v4.end(), std::ref(qu8rng));
  std::generate(v5.begin(), v5.end(), std::ref(qu8rng));
  std::generate(v6.begin(), v6.end(), std::ref(qu8rng));
  std::generate(v7.begin(), v7.end(), std::ref(qu8rng));
  std::generate(v8.begin(), v8.end(), std::ref(qu8rng));
  std::generate(v9.begin(), v9.end(), std::ref(qu8rng));
  std::generate(v10.begin(), v10.end(), std::ref(qu8rng));
  std::generate(v11.begin(), v11.end(), std::ref(qu8rng));
  std::generate(v12.begin(), v12.end(), std::ref(qu8rng));
  std::generate(v13.begin(), v13.end(), std::ref(qu8rng));
  std::generate(v14.begin(), v14.end(), std::ref(qu8rng));
  std::generate(v15.begin(), v15.end(), std::ref(qu8rng));
  std::generate(v16.begin(), v16.end(), std::ref(qu8rng));
  std::generate(v17.begin(), v17.end(), std::ref(qu8rng));
  std::generate(v18.begin(), v18.end(), std::ref(qu8rng));
  std::generate(v19.begin(), v19.end(), std::ref(qu8rng));
  std::generate(v20.begin(), v20.end(), std::ref(qu8rng));
  std::generate(v21.begin(), v21.end(), std::ref(qu8rng));
  std::generate(v22.begin(), v22.end(), std::ref(qu8rng));
  std::generate(v23.begin(), v23.end(), std::ref(qu8rng));
  std::generate(v24.begin(), v24.end(), std::ref(qu8rng));
  std::generate(v25.begin(), v25.end(), std::ref(qu8rng));
  std::generate(v26.begin(), v26.end(), std::ref(qu8rng));
  std::generate(v27.begin(), v27.end(), std::ref(qu8rng));
  std::generate(v28.begin(), v28.end(), std::ref(qu8rng));
  std::generate(v29.begin(), v29.end(), std::ref(qu8rng));
  std::generate(v30.begin(), v30.end(), std::ref(qu8rng));
  std::generate(v31.begin(), v31.end(), std::ref(qu8rng));
  std::generate(v32.begin(), v32.end(), std::ref(qu8rng));
  std::generate(v33.begin(), v33.end(), std::ref(qu8rng));
  std::generate(v34.begin(), v34.end(), std::ref(qu8rng));
  std::generate(v35.begin(), v35.end(), std::ref(qu8rng));
  std::generate(v36.begin(), v36.end(), std::ref(qu8rng));
  std::generate(v37.begin(), v37.end(), std::ref(qu8rng));
  std::generate(v38.begin(), v38.end(), std::ref(qu8rng));
  std::generate(v39.begin(), v39.end(), std::ref(qu8rng));
  std::generate(v40.begin(), v40.end(), std::ref(qu8rng));
  std::generate(v41.begin(), v41.end(), std::ref(qu8rng));
  std::generate(v42.begin(), v42.end(), std::ref(qu8rng));
  std::generate(v43.begin(), v43.end(), std::ref(qu8rng));
  std::generate(v44.begin(), v44.end(), std::ref(qu8rng));
  std::generate(v45.begin(), v45.end(), std::ref(qu8rng));
  std::generate(v46.begin(), v46.end(), std::ref(qu8rng));
  std::generate(v47.begin(), v47.end(), std::ref(qu8rng));
  std::generate(v48.begin(), v48.end(), std::ref(qu8rng));
  std::generate(v49.begin(), v49.end(), std::ref(qu8rng));
  std::generate(v50.begin(), v50.end(), std::ref(qu8rng));
  std::generate(v51.begin(), v51.end(), std::ref(qu8rng));
  std::generate(v52.begin(), v52.end(), std::ref(qu8rng));
  std::generate(v53.begin(), v53.end(), std::ref(qu8rng));
  std::generate(v54.begin(), v54.end(), std::ref(qu8rng));
  std::generate(v55.begin(), v55.end(), std::ref(qu8rng));
  std::generate(v56.begin(), v56.end(), std::ref(qu8rng));
  std::generate(v57.begin(), v57.end(), std::ref(qu8rng));
  std::generate(v58.begin(), v58.end(), std::ref(qu8rng));
  std::generate(v59.begin(), v59.end(), std::ref(qu8rng));
  std::generate(v60.begin(), v60.end(), std::ref(qu8rng));
  std::generate(v61.begin(), v61.end(), std::ref(qu8rng));
  std::generate(v62.begin(), v62.end(), std::ref(qu8rng));
  std::generate(v63.begin(), v63.end(), std::ref(qu8rng));
  std::generate(v64.begin(), v64.end(), std::ref(qu8rng));
  std::generate(v65.begin(), v65.end(), std::ref(qu8rng));
  std::generate(w66.begin(), w66.end(), std::ref(qu8rng));
  std::generate(w67.begin(), w67.end(), std::ref(qs32rng));
  std::generate(w68.begin(), w68.end(), std::ref(qu8rng));
  std::generate(w69.begin(), w69.end(), std::ref(qs32rng));
  std::generate(w70.begin(), w70.end(), std::ref(qu8rng));
  std::generate(w71.begin(), w71.end(), std::ref(qs32rng));
  std::generate(w72.begin(), w72.end(), std::ref(qu8rng));
  std::generate(w73.begin(), w73.end(), std::ref(qs32rng));
  std::generate(w74.begin(), w74.end(), std::ref(qu8rng));
  std::generate(w75.begin(), w75.end(), std::ref(qs32rng));
  std::generate(w76.begin(), w76.end(), std::ref(qu8rng));
  std::generate(w77.begin(), w77.end(), std::ref(qs32rng));
  std::generate(w78.begin(), w78.end(), std::ref(qu8rng));
  std::generate(w79.begin(), w79.end(), std::ref(qs32rng));
  std::generate(w80.begin(), w80.end(), std::ref(qu8rng));
  std::generate(w81.begin(), w81.end(), std::ref(qs32rng));
  std::generate(w82.begin(), w82.end(), std::ref(qu8rng));
  std::generate(w83.begin(), w83.end(), std::ref(qs32rng));
  std::generate(w84.begin(), w84.end(), std::ref(qu8rng));
  std::generate(w85.begin(), w85.end(), std::ref(qs32rng));
  std::generate(w86.begin(), w86.end(), std::ref(qu8rng));
  std::generate(w87.begin(), w87.end(), std::ref(qs32rng));
  std::generate(w88.begin(), w88.end(), std::ref(qu8rng));
  std::generate(w89.begin(), w89.end(), std::ref(qs32rng));
  std::generate(w90.begin(), w90.end(), std::ref(qu8rng));
  std::generate(w91.begin(), w91.end(), std::ref(qs32rng));
  std::generate(w92.begin(), w92.end(), std::ref(qu8rng));
  std::generate(w93.begin(), w93.end(), std::ref(qs32rng));
  std::generate(w94.begin(), w94.end(), std::ref(qu8rng));
  std::generate(w95.begin(), w95.end(), std::ref(qs32rng));
  std::generate(w96.begin(), w96.end(), std::ref(qu8rng));
  std::generate(w97.begin(), w97.end(), std::ref(qs32rng));
  std::generate(w98.begin(), w98.end(), std::ref(qu8rng));
  std::generate(w99.begin(), w99.end(), std::ref(qs32rng));
  std::generate(w100.begin(), w100.end(), std::ref(qu8rng));
  std::generate(w101.begin(), w101.end(), std::ref(qs32rng));
  std::generate(w102.begin(), w102.end(), std::ref(qu8rng));
  std::generate(w103.begin(), w103.end(), std::ref(qs32rng));
  std::generate(w104.begin(), w104.end(), std::ref(qu8rng));
  std::generate(w105.begin(), w105.end(), std::ref(qs32rng));
  std::generate(w106.begin(), w106.end(), std::ref(qu8rng));
  std::generate(w107.begin(), w107.end(), std::ref(qs32rng));
  std::generate(w108.begin(), w108.end(), std::ref(qu8rng));
  std::generate(w109.begin(), w109.end(), std::ref(qs32rng));
  std::generate(w110.begin(), w110.end(), std::ref(qu8rng));
  std::generate(w111.begin(), w111.end(), std::ref(qs32rng));
  std::generate(w112.begin(), w112.end(), std::ref(qu8rng));
  std::generate(w113.begin(), w113.end(), std::ref(qs32rng));
  std::generate(w114.begin(), w114.end(), std::ref(qu8rng));
  std::generate(w115.begin(), w115.end(), std::ref(qs32rng));
  std::generate(w116.begin(), w116.end(), std::ref(qu8rng));
  std::generate(w117.begin(), w117.end(), std::ref(qs32rng));
  std::generate(w118.begin(), w118.end(), std::ref(qu8rng));
  std::generate(w119.begin(), w119.end(), std::ref(qs32rng));
  std::generate(w120.begin(), w120.end(), std::ref(qu8rng));
  std::generate(w121.begin(), w121.end(), std::ref(qs32rng));
  std::generate(w122.begin(), w122.end(), std::ref(qu8rng));
  std::generate(w123.begin(), w123.end(), std::ref(qs32rng));
  std::generate(w124.begin(), w124.end(), std::ref(qu8rng));
  std::generate(w125.begin(), w125.end(), std::ref(qs32rng));
  std::generate(w126.begin(), w126.end(), std::ref(qu8rng));
  std::generate(w127.begin(), w127.end(), std::ref(qs32rng));
  std::generate(w128.begin(), w128.end(), std::ref(qu8rng));
  std::generate(w129.begin(), w129.end(), std::ref(qs32rng));
  std::generate(w130.begin(), w130.end(), std::ref(qu8rng));
  std::generate(w131.begin(), w131.end(), std::ref(qs32rng));
  std::generate(w132.begin(), w132.end(), std::ref(qu8rng));
  std::generate(w133.begin(), w133.end(), std::ref(qs32rng));
  std::generate(w134.begin(), w134.end(), std::ref(qu8rng));
  std::generate(w135.begin(), w135.end(), std::ref(qs32rng));
  std::generate(w136.begin(), w136.end(), std::ref(qu8rng));
  std::generate(w137.begin(), w137.end(), std::ref(qs32rng));
  std::generate(w138.begin(), w138.end(), std::ref(qu8rng));
  std::generate(w139.begin(), w139.end(), std::ref(qs32rng));
  std::generate(w140.begin(), w140.end(), std::ref(qu8rng));
  std::generate(w141.begin(), w141.end(), std::ref(qs32rng));
  std::generate(w142.begin(), w142.end(), std::ref(qu8rng));
  std::generate(w143.begin(), w143.end(), std::ref(qs32rng));
  std::generate(w144.begin(), w144.end(), std::ref(qu8rng));
  std::generate(w145.begin(), w145.end(), std::ref(qs32rng));
  std::generate(w146.begin(), w146.end(), std::ref(qu8rng));
  std::generate(w147.begin(), w147.end(), std::ref(qs32rng));
  std::generate(w148.begin(), w148.end(), std::ref(qu8rng));
  std::generate(w149.begin(), w149.end(), std::ref(qs32rng));
  std::generate(w150.begin(), w150.end(), std::ref(qu8rng));
  std::generate(w151.begin(), w151.end(), std::ref(qs32rng));
  std::generate(w152.begin(), w152.end(), std::ref(qu8rng));
  std::generate(w153.begin(), w153.end(), std::ref(qs32rng));
  std::generate(w154.begin(), w154.end(), std::ref(qu8rng));
  std::generate(w155.begin(), w155.end(), std::ref(qs32rng));
  std::generate(w156.begin(), w156.end(), std::ref(qu8rng));
  std::generate(w157.begin(), w157.end(), std::ref(qs32rng));
  std::generate(w158.begin(), w158.end(), std::ref(qu8rng));
  std::generate(w159.begin(), w159.end(), std::ref(qs32rng));
  std::generate(w160.begin(), w160.end(), std::ref(qu8rng));
  std::generate(w161.begin(), w161.end(), std::ref(qs32rng));
  std::generate(w162.begin(), w162.end(), std::ref(qu8rng));
  std::generate(w163.begin(), w163.end(), std::ref(qs32rng));
  std::generate(w164.begin(), w164.end(), std::ref(qu8rng));
  std::generate(w165.begin(), w165.end(), std::ref(qs32rng));
  std::generate(w166.begin(), w166.end(), std::ref(qu8rng));
  std::generate(w167.begin(), w167.end(), std::ref(qs32rng));
  std::generate(w168.begin(), w168.end(), std::ref(qu8rng));
  std::generate(w169.begin(), w169.end(), std::ref(qs32rng));
  std::generate(w170.begin(), w170.end(), std::ref(qu8rng));
  std::generate(w171.begin(), w171.end(), std::ref(qs32rng));

  Operators operators;
  xnn_status status;
  xnn_code_cache* code_cache_ptr = nullptr;
  size_t max_workspace_size = 0;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/3,
    /*group_output_channels=*/32,
    /*input_channel_stride=*/3,
    /*output_channel_stride=*/32,
    /*input_zero_point=*/(uint8_t) 128,
    /*input_scale=*/0.0078125,
    /*kernel_zero_point=*/(uint8_t) 122,
    /*kernel_scale=*/0.03396892547607422,
    /*kernel=*/w66.data(), /*bias=*/w67.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/32,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/32,
    /*output_channel_stride=*/32,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 165,
    /*kernel_scale=*/0.3436955213546753,
    /*kernel=*/w68.data(), /*bias=*/w69.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/16,
    /*input_channel_stride=*/32,
    /*output_channel_stride=*/16,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 140,
    /*kernel_scale=*/0.03737175464630127,
    /*kernel=*/w70.data(), /*bias=*/w71.data(),
    /*output_zero_point=*/(uint8_t) 129,
    /*output_scale=*/0.35441333055496216,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/16,
    /*group_output_channels=*/96,
    /*input_channel_stride=*/16,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 129,
    /*input_scale=*/0.35441333055496216,
    /*kernel_zero_point=*/(uint8_t) 127,
    /*kernel_scale=*/0.009758507832884789,
    /*kernel=*/w72.data(), /*bias=*/w73.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/96,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 109,
    /*kernel_scale=*/0.020969120785593987,
    /*kernel=*/w74.data(), /*bias=*/w75.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/24,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/24,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 156,
    /*kernel_scale=*/0.0225360207259655,
    /*kernel=*/w76.data(), /*bias=*/w77.data(),
    /*output_zero_point=*/(uint8_t) 119,
    /*output_scale=*/0.2758343517780304,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/24,
    /*group_output_channels=*/144,
    /*input_channel_stride=*/24,
    /*output_channel_stride=*/144,
    /*input_zero_point=*/(uint8_t) 119,
    /*input_scale=*/0.2758343517780304,
    /*kernel_zero_point=*/(uint8_t) 144,
    /*kernel_scale=*/0.0036556976847350597,
    /*kernel=*/w78.data(), /*bias=*/w79.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/144,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/144,
    /*output_channel_stride=*/144,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 52,
    /*kernel_scale=*/0.16981913149356842,
    /*kernel=*/w80.data(), /*bias=*/w81.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/144,
    /*group_output_channels=*/24,
    /*input_channel_stride=*/144,
    /*output_channel_stride=*/24,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 122,
    /*kernel_scale=*/0.02740888111293316,
    /*kernel=*/w82.data(), /*bias=*/w83.data(),
    /*output_zero_point=*/(uint8_t) 136,
    /*output_scale=*/0.4014929533004761,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_add_nd_qu8(
    136 /* input1 zero point */, 0.4014929533004761 /* input1 scale */,
    119 /* input2 zero point */, 0.2758343517780304 /* input2 scale */,
    133 /* output zero point */, 0.43216896057128906 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/24,
    /*group_output_channels=*/144,
    /*input_channel_stride=*/24,
    /*output_channel_stride=*/144,
    /*input_zero_point=*/(uint8_t) 133,
    /*input_scale=*/0.43216896057128906,
    /*kernel_zero_point=*/(uint8_t) 104,
    /*kernel_scale=*/0.0029988749884068966,
    /*kernel=*/w84.data(), /*bias=*/w85.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/144,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/144,
    /*output_channel_stride=*/144,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 143,
    /*kernel_scale=*/0.017202870920300484,
    /*kernel=*/w86.data(), /*bias=*/w87.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/144,
    /*group_output_channels=*/32,
    /*input_channel_stride=*/144,
    /*output_channel_stride=*/32,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 111,
    /*kernel_scale=*/0.016844693571329117,
    /*kernel=*/w88.data(), /*bias=*/w89.data(),
    /*output_zero_point=*/(uint8_t) 127,
    /*output_scale=*/0.21836242079734802,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/192,
    /*input_channel_stride=*/32,
    /*output_channel_stride=*/192,
    /*input_zero_point=*/(uint8_t) 127,
    /*input_scale=*/0.21836242079734802,
    /*kernel_zero_point=*/(uint8_t) 128,
    /*kernel_scale=*/0.0019244228024035692,
    /*kernel=*/w90.data(), /*bias=*/w91.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/192,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/192,
    /*output_channel_stride=*/192,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 118,
    /*kernel_scale=*/0.06525065749883652,
    /*kernel=*/w92.data(), /*bias=*/w93.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/192,
    /*group_output_channels=*/32,
    /*input_channel_stride=*/192,
    /*output_channel_stride=*/32,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 146,
    /*kernel_scale=*/0.019062912091612816,
    /*kernel=*/w94.data(), /*bias=*/w95.data(),
    /*output_zero_point=*/(uint8_t) 121,
    /*output_scale=*/0.2279418408870697,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_add_nd_qu8(
    121 /* input1 zero point */, 0.2279418408870697 /* input1 scale */,
    127 /* input2 zero point */, 0.21836242079734802 /* input2 scale */,
    130 /* output zero point */, 0.25968998670578003 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/192,
    /*input_channel_stride=*/32,
    /*output_channel_stride=*/192,
    /*input_zero_point=*/(uint8_t) 130,
    /*input_scale=*/0.25968998670578003,
    /*kernel_zero_point=*/(uint8_t) 135,
    /*kernel_scale=*/0.0013649158645421267,
    /*kernel=*/w96.data(), /*bias=*/w97.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/192,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/192,
    /*output_channel_stride=*/192,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 95,
    /*kernel_scale=*/0.07909784466028214,
    /*kernel=*/w98.data(), /*bias=*/w99.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/192,
    /*group_output_channels=*/32,
    /*input_channel_stride=*/192,
    /*output_channel_stride=*/32,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 128,
    /*kernel_scale=*/0.018293123692274094,
    /*kernel=*/w100.data(), /*bias=*/w101.data(),
    /*output_zero_point=*/(uint8_t) 124,
    /*output_scale=*/0.25774890184402466,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_add_nd_qu8(
    124 /* input1 zero point */, 0.25774890184402466 /* input1 scale */,
    130 /* input2 zero point */, 0.25968998670578003 /* input2 scale */,
    124 /* output zero point */, 0.331714928150177 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/32,
    /*group_output_channels=*/192,
    /*input_channel_stride=*/32,
    /*output_channel_stride=*/192,
    /*input_zero_point=*/(uint8_t) 124,
    /*input_scale=*/0.331714928150177,
    /*kernel_zero_point=*/(uint8_t) 127,
    /*kernel_scale=*/0.0019170437008142471,
    /*kernel=*/w102.data(), /*bias=*/w103.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/192,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/192,
    /*output_channel_stride=*/192,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 127,
    /*kernel_scale=*/0.010087885893881321,
    /*kernel=*/w104.data(), /*bias=*/w105.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/192,
    /*group_output_channels=*/64,
    /*input_channel_stride=*/192,
    /*output_channel_stride=*/64,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 147,
    /*kernel_scale=*/0.014601286500692368,
    /*kernel=*/w106.data(), /*bias=*/w107.data(),
    /*output_zero_point=*/(uint8_t) 126,
    /*output_scale=*/0.18540528416633606,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/384,
    /*input_channel_stride=*/64,
    /*output_channel_stride=*/384,
    /*input_zero_point=*/(uint8_t) 126,
    /*input_scale=*/0.18540528416633606,
    /*kernel_zero_point=*/(uint8_t) 125,
    /*kernel_scale=*/0.0015538912266492844,
    /*kernel=*/w108.data(), /*bias=*/w109.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/384,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/384,
    /*output_channel_stride=*/384,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 110,
    /*kernel_scale=*/0.06092711538076401,
    /*kernel=*/w110.data(), /*bias=*/w111.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/384,
    /*group_output_channels=*/64,
    /*input_channel_stride=*/384,
    /*output_channel_stride=*/64,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 124,
    /*kernel_scale=*/0.016782939434051514,
    /*kernel=*/w112.data(), /*bias=*/w113.data(),
    /*output_zero_point=*/(uint8_t) 109,
    /*output_scale=*/0.17263489961624146,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_add_nd_qu8(
    109 /* input1 zero point */, 0.17263489961624146 /* input1 scale */,
    126 /* input2 zero point */, 0.18540528416633606 /* input2 scale */,
    122 /* output zero point */, 0.18911026418209076 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/384,
    /*input_channel_stride=*/64,
    /*output_channel_stride=*/384,
    /*input_zero_point=*/(uint8_t) 122,
    /*input_scale=*/0.18911026418209076,
    /*kernel_zero_point=*/(uint8_t) 134,
    /*kernel_scale=*/0.0014702979242429137,
    /*kernel=*/w114.data(), /*bias=*/w115.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  xnn_operator_t op29 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/384,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/384,
    /*output_channel_stride=*/384,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 133,
    /*kernel_scale=*/0.052407849580049515,
    /*kernel=*/w116.data(), /*bias=*/w117.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #29" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op29, xnn_delete_operator);

  xnn_operator_t op30 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/384,
    /*group_output_channels=*/64,
    /*input_channel_stride=*/384,
    /*output_channel_stride=*/64,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 125,
    /*kernel_scale=*/0.012898261658847332,
    /*kernel=*/w118.data(), /*bias=*/w119.data(),
    /*output_zero_point=*/(uint8_t) 123,
    /*output_scale=*/0.14715521037578583,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #30" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op30, xnn_delete_operator);

  xnn_operator_t op31 = nullptr;
  status = xnn_create_add_nd_qu8(
    123 /* input1 zero point */, 0.14715521037578583 /* input1 scale */,
    122 /* input2 zero point */, 0.18911026418209076 /* input2 scale */,
    124 /* output zero point */, 0.1996811032295227 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #31" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op31, xnn_delete_operator);

  xnn_operator_t op32 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/384,
    /*input_channel_stride=*/64,
    /*output_channel_stride=*/384,
    /*input_zero_point=*/(uint8_t) 124,
    /*input_scale=*/0.1996811032295227,
    /*kernel_zero_point=*/(uint8_t) 127,
    /*kernel_scale=*/0.0013733493397012353,
    /*kernel=*/w120.data(), /*bias=*/w121.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #32" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op32, xnn_delete_operator);

  xnn_operator_t op33 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/384,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/384,
    /*output_channel_stride=*/384,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 155,
    /*kernel_scale=*/0.04077887907624245,
    /*kernel=*/w122.data(), /*bias=*/w123.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #33" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op33, xnn_delete_operator);

  xnn_operator_t op34 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/384,
    /*group_output_channels=*/64,
    /*input_channel_stride=*/384,
    /*output_channel_stride=*/64,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 144,
    /*kernel_scale=*/0.019561484456062317,
    /*kernel=*/w124.data(), /*bias=*/w125.data(),
    /*output_zero_point=*/(uint8_t) 122,
    /*output_scale=*/0.15627601742744446,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #34" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op34, xnn_delete_operator);

  xnn_operator_t op35 = nullptr;
  status = xnn_create_add_nd_qu8(
    122 /* input1 zero point */, 0.15627601742744446 /* input1 scale */,
    124 /* input2 zero point */, 0.1996811032295227 /* input2 scale */,
    120 /* output zero point */, 0.22027325630187988 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #35" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op35, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/64,
    /*group_output_channels=*/384,
    /*input_channel_stride=*/64,
    /*output_channel_stride=*/384,
    /*input_zero_point=*/(uint8_t) 120,
    /*input_scale=*/0.22027325630187988,
    /*kernel_zero_point=*/(uint8_t) 131,
    /*kernel_scale=*/0.0016282502328976989,
    /*kernel=*/w126.data(), /*bias=*/w127.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #36" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op36, xnn_delete_operator);

  xnn_operator_t op37 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/384,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/384,
    /*output_channel_stride=*/384,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 143,
    /*kernel_scale=*/0.031107846647500992,
    /*kernel=*/w128.data(), /*bias=*/w129.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #37" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op37, xnn_delete_operator);

  xnn_operator_t op38 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/384,
    /*group_output_channels=*/96,
    /*input_channel_stride=*/384,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 129,
    /*kernel_scale=*/0.007436311338096857,
    /*kernel=*/w130.data(), /*bias=*/w131.data(),
    /*output_zero_point=*/(uint8_t) 129,
    /*output_scale=*/0.17061053216457367,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #38" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op38, xnn_delete_operator);

  xnn_operator_t op39 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/576,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 129,
    /*input_scale=*/0.17061053216457367,
    /*kernel_zero_point=*/(uint8_t) 134,
    /*kernel_scale=*/0.0016309921629726887,
    /*kernel=*/w132.data(), /*bias=*/w133.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #39" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op39, xnn_delete_operator);

  xnn_operator_t op40 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/576,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 66,
    /*kernel_scale=*/0.07080810517072678,
    /*kernel=*/w134.data(), /*bias=*/w135.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #40" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op40, xnn_delete_operator);

  xnn_operator_t op41 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/96,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 136,
    /*kernel_scale=*/0.00838223285973072,
    /*kernel=*/w136.data(), /*bias=*/w137.data(),
    /*output_zero_point=*/(uint8_t) 127,
    /*output_scale=*/0.12332822382450104,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #41" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op41, xnn_delete_operator);

  xnn_operator_t op42 = nullptr;
  status = xnn_create_add_nd_qu8(
    127 /* input1 zero point */, 0.12332822382450104 /* input1 scale */,
    129 /* input2 zero point */, 0.17061053216457367 /* input2 scale */,
    127 /* output zero point */, 0.17615799605846405 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #42" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op42, xnn_delete_operator);

  xnn_operator_t op43 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/576,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 127,
    /*input_scale=*/0.17615799605846405,
    /*kernel_zero_point=*/(uint8_t) 138,
    /*kernel_scale=*/0.0018258779309689999,
    /*kernel=*/w138.data(), /*bias=*/w139.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #43" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op43, xnn_delete_operator);

  xnn_operator_t op44 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/576,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 159,
    /*kernel_scale=*/0.07448793947696686,
    /*kernel=*/w140.data(), /*bias=*/w141.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op45 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/96,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/96,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 154,
    /*kernel_scale=*/0.023982593789696693,
    /*kernel=*/w142.data(), /*bias=*/w143.data(),
    /*output_zero_point=*/(uint8_t) 127,
    /*output_scale=*/0.18619607388973236,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #45" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op45, xnn_delete_operator);

  xnn_operator_t op46 = nullptr;
  status = xnn_create_add_nd_qu8(
    127 /* input1 zero point */, 0.18619607388973236 /* input1 scale */,
    127 /* input2 zero point */, 0.17615799605846405 /* input2 scale */,
    126 /* output zero point */, 0.23340091109275818 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #46" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op46, xnn_delete_operator);

  xnn_operator_t op47 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/96,
    /*group_output_channels=*/576,
    /*input_channel_stride=*/96,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 126,
    /*input_scale=*/0.23340091109275818,
    /*kernel_zero_point=*/(uint8_t) 123,
    /*kernel_scale=*/0.0013828007504343987,
    /*kernel=*/w144.data(), /*bias=*/w145.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #47" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op47, xnn_delete_operator);

  xnn_operator_t op48 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/0,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/2, /*subsampling_width=*/2,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/576,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/576,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 92,
    /*kernel_scale=*/0.01525793131440878,
    /*kernel=*/w146.data(), /*bias=*/w147.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #48" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op48, xnn_delete_operator);

  xnn_operator_t op49 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/576,
    /*group_output_channels=*/160,
    /*input_channel_stride=*/576,
    /*output_channel_stride=*/160,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 140,
    /*kernel_scale=*/0.009447949007153511,
    /*kernel=*/w148.data(), /*bias=*/w149.data(),
    /*output_zero_point=*/(uint8_t) 132,
    /*output_scale=*/0.13237787783145905,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #49" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op49, xnn_delete_operator);

  xnn_operator_t op50 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/160,
    /*group_output_channels=*/960,
    /*input_channel_stride=*/160,
    /*output_channel_stride=*/960,
    /*input_zero_point=*/(uint8_t) 132,
    /*input_scale=*/0.13237787783145905,
    /*kernel_zero_point=*/(uint8_t) 135,
    /*kernel_scale=*/0.0020222084131091833,
    /*kernel=*/w150.data(), /*bias=*/w151.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #50" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op50, xnn_delete_operator);

  xnn_operator_t op51 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/960,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/960,
    /*output_channel_stride=*/960,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 147,
    /*kernel_scale=*/0.04166752099990845,
    /*kernel=*/w152.data(), /*bias=*/w153.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #51" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op51, xnn_delete_operator);

  xnn_operator_t op52 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/960,
    /*group_output_channels=*/160,
    /*input_channel_stride=*/960,
    /*output_channel_stride=*/160,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 139,
    /*kernel_scale=*/0.00789870135486126,
    /*kernel=*/w154.data(), /*bias=*/w155.data(),
    /*output_zero_point=*/(uint8_t) 129,
    /*output_scale=*/0.10045691579580307,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #52" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op52, xnn_delete_operator);

  xnn_operator_t op53 = nullptr;
  status = xnn_create_add_nd_qu8(
    129 /* input1 zero point */, 0.10045691579580307 /* input1 scale */,
    132 /* input2 zero point */, 0.13237787783145905 /* input2 scale */,
    134 /* output zero point */, 0.15070965886116028 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op54 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/160,
    /*group_output_channels=*/960,
    /*input_channel_stride=*/160,
    /*output_channel_stride=*/960,
    /*input_zero_point=*/(uint8_t) 134,
    /*input_scale=*/0.15070965886116028,
    /*kernel_zero_point=*/(uint8_t) 127,
    /*kernel_scale=*/0.0015944414772093296,
    /*kernel=*/w156.data(), /*bias=*/w157.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #54" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op54, xnn_delete_operator);

  xnn_operator_t op55 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/960,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/960,
    /*output_channel_stride=*/960,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 102,
    /*kernel_scale=*/0.04281935095787048,
    /*kernel=*/w158.data(), /*bias=*/w159.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #55" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op55, xnn_delete_operator);

  xnn_operator_t op56 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/960,
    /*group_output_channels=*/160,
    /*input_channel_stride=*/960,
    /*output_channel_stride=*/160,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 131,
    /*kernel_scale=*/0.03697410225868225,
    /*kernel=*/w160.data(), /*bias=*/w161.data(),
    /*output_zero_point=*/(uint8_t) 133,
    /*output_scale=*/0.1696060746908188,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #56" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op56, xnn_delete_operator);

  xnn_operator_t op57 = nullptr;
  status = xnn_create_add_nd_qu8(
    133 /* input1 zero point */, 0.1696060746908188 /* input1 scale */,
    134 /* input2 zero point */, 0.15070965886116028 /* input2 scale */,
    131 /* output zero point */, 0.21005140244960785 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #57" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op57, xnn_delete_operator);

  xnn_operator_t op58 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/160,
    /*group_output_channels=*/960,
    /*input_channel_stride=*/160,
    /*output_channel_stride=*/960,
    /*input_zero_point=*/(uint8_t) 131,
    /*input_scale=*/0.21005140244960785,
    /*kernel_zero_point=*/(uint8_t) 135,
    /*kernel_scale=*/0.002046825597062707,
    /*kernel=*/w162.data(), /*bias=*/w163.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #58" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op58, xnn_delete_operator);

  xnn_operator_t op59 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/1, /*input_padding_right=*/1,
    /*input_padding_bottom=*/1, /*input_padding_left=*/1,
    /*kernel_height=*/3, /*kernel_width=*/3,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/960,
    /*group_input_channels=*/1,
    /*group_output_channels=*/1,
    /*input_channel_stride=*/960,
    /*output_channel_stride=*/960,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 201,
    /*kernel_scale=*/0.16456253826618195,
    /*kernel=*/w164.data(), /*bias=*/w165.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #59" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op59, xnn_delete_operator);

  xnn_operator_t op60 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/960,
    /*group_output_channels=*/320,
    /*input_channel_stride=*/960,
    /*output_channel_stride=*/320,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 111,
    /*kernel_scale=*/0.008009289391338825,
    /*kernel=*/w166.data(), /*bias=*/w167.data(),
    /*output_zero_point=*/(uint8_t) 130,
    /*output_scale=*/0.11694499105215073,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #60" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op60, xnn_delete_operator);

  xnn_operator_t op61 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/320,
    /*group_output_channels=*/1280,
    /*input_channel_stride=*/320,
    /*output_channel_stride=*/1280,
    /*input_zero_point=*/(uint8_t) 130,
    /*input_scale=*/0.11694499105215073,
    /*kernel_zero_point=*/(uint8_t) 125,
    /*kernel_scale=*/0.005167067516595125,
    /*kernel=*/w168.data(), /*bias=*/w169.data(),
    /*output_zero_point=*/(uint8_t) 0,
    /*output_scale=*/0.023528477177023888,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #61" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op61, xnn_delete_operator);

  xnn_operator_t op62 = nullptr;
  status = xnn_create_global_average_pooling_nwc_qu8(
    0 /* input zero point */, 0.023528477177023888 /* input scale */,
    0 /* output zero point */, 0.023528477177023888 /* output scale */,
    0 /* output min */, 255 /* output max */,
    0 /* flags */,
    &op62);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #62" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op62, xnn_delete_operator);

  xnn_operator_t op63 = nullptr;
  status = xnn_create_convolution2d_nhwc_qu8(
    /*input_padding_top=*/0, /*input_padding_right=*/0,
    /*input_padding_bottom=*/0, /*input_padding_left=*/0,
    /*kernel_height=*/1, /*kernel_width=*/1,
    /*subsampling_height=*/1, /*subsampling_width=*/1,
    /*dilation_height=*/1, /*dilation_width=*/1,
    /*groups=*/1,
    /*group_input_channels=*/1280,
    /*group_output_channels=*/1001,
    /*input_channel_stride=*/1280,
    /*output_channel_stride=*/1001,
    /*input_zero_point=*/(uint8_t) 0,
    /*input_scale=*/0.023528477177023888,
    /*kernel_zero_point=*/(uint8_t) 113,
    /*kernel_scale=*/0.0016910821432247758,
    /*kernel=*/w170.data(), /*bias=*/w171.data(),
    /*output_zero_point=*/(uint8_t) 58,
    /*output_scale=*/0.09889253973960876,
    /*output_min=*/(uint8_t) 0, /*output_max=*/(uint8_t) 255,
    /*flags=*/0,
    /*code_cache=*/code_cache_ptr,
    /*weights_cache=*/nullptr,
    &op63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #63" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op63, xnn_delete_operator);

  xnn_operator_t op64 = nullptr;
  status = xnn_create_copy_nc_x8(
    0 /* flags */,
    &op64);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #64" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op64, xnn_delete_operator);

  size_t op0_workspace_size = 0;
  size_t op0_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op0,
    /*batch_size=*/1, /*input_height=*/224, /*input_width=*/224,
    &op0_workspace_size, &op0_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op0_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #0" << std::endl;
    return ExecutionPlan();
  }

  size_t op1_workspace_size = 0;
  size_t op1_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op1,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    &op1_workspace_size, &op1_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op1_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #1" << std::endl;
    return ExecutionPlan();
  }

  size_t op2_workspace_size = 0;
  size_t op2_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op2,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    &op2_workspace_size, &op2_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op2_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #2" << std::endl;
    return ExecutionPlan();
  }

  size_t op3_workspace_size = 0;
  size_t op3_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op3,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    &op3_workspace_size, &op3_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op3_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #3" << std::endl;
    return ExecutionPlan();
  }

  size_t op4_workspace_size = 0;
  size_t op4_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op4,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    &op4_workspace_size, &op4_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op4_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #4" << std::endl;
    return ExecutionPlan();
  }

  size_t op5_workspace_size = 0;
  size_t op5_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op5,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op5_workspace_size, &op5_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op5_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #5" << std::endl;
    return ExecutionPlan();
  }

  size_t op6_workspace_size = 0;
  size_t op6_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op6,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op6_workspace_size, &op6_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op6_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #6" << std::endl;
    return ExecutionPlan();
  }

  size_t op7_workspace_size = 0;
  size_t op7_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op7,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op7_workspace_size, &op7_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op7_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #7" << std::endl;
    return ExecutionPlan();
  }

  size_t op8_workspace_size = 0;
  size_t op8_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op8,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op8_workspace_size, &op8_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op8_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #8" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 56, 56, 24 };
    const size_t b_shape[] = { 1, 56, 56, 24 };
    status = xnn_reshape_add_nd_qu8(
      op9,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #9" << std::endl;
    return ExecutionPlan();
  }

  size_t op10_workspace_size = 0;
  size_t op10_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op10,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op10_workspace_size, &op10_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op10_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #10" << std::endl;
    return ExecutionPlan();
  }

  size_t op11_workspace_size = 0;
  size_t op11_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op11,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    &op11_workspace_size, &op11_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op11_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #11" << std::endl;
    return ExecutionPlan();
  }

  size_t op12_workspace_size = 0;
  size_t op12_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op12,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op12_workspace_size, &op12_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op12_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #12" << std::endl;
    return ExecutionPlan();
  }

  size_t op13_workspace_size = 0;
  size_t op13_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op13,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op13_workspace_size, &op13_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op13_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #13" << std::endl;
    return ExecutionPlan();
  }

  size_t op14_workspace_size = 0;
  size_t op14_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op14,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op14_workspace_size, &op14_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op14_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #14" << std::endl;
    return ExecutionPlan();
  }

  size_t op15_workspace_size = 0;
  size_t op15_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op15,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op15_workspace_size, &op15_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op15_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #15" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 32 };
    const size_t b_shape[] = { 1, 28, 28, 32 };
    status = xnn_reshape_add_nd_qu8(
      op16,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #16" << std::endl;
    return ExecutionPlan();
  }

  size_t op17_workspace_size = 0;
  size_t op17_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op17,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op17_workspace_size, &op17_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op17_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #17" << std::endl;
    return ExecutionPlan();
  }

  size_t op18_workspace_size = 0;
  size_t op18_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op18,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op18_workspace_size, &op18_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op18_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #18" << std::endl;
    return ExecutionPlan();
  }

  size_t op19_workspace_size = 0;
  size_t op19_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op19,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op19_workspace_size, &op19_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op19_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #19" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 32 };
    const size_t b_shape[] = { 1, 28, 28, 32 };
    status = xnn_reshape_add_nd_qu8(
      op20,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #20" << std::endl;
    return ExecutionPlan();
  }

  size_t op21_workspace_size = 0;
  size_t op21_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op21,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op21_workspace_size, &op21_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op21_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #21" << std::endl;
    return ExecutionPlan();
  }

  size_t op22_workspace_size = 0;
  size_t op22_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op22,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    &op22_workspace_size, &op22_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op22_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #22" << std::endl;
    return ExecutionPlan();
  }

  size_t op23_workspace_size = 0;
  size_t op23_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op23,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op23_workspace_size, &op23_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op23_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #23" << std::endl;
    return ExecutionPlan();
  }

  size_t op24_workspace_size = 0;
  size_t op24_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op24,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op24_workspace_size, &op24_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op24_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #24" << std::endl;
    return ExecutionPlan();
  }

  size_t op25_workspace_size = 0;
  size_t op25_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op25,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op25_workspace_size, &op25_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op25_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #25" << std::endl;
    return ExecutionPlan();
  }

  size_t op26_workspace_size = 0;
  size_t op26_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op26,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op26_workspace_size, &op26_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op26_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #26" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 64 };
    const size_t b_shape[] = { 1, 14, 14, 64 };
    status = xnn_reshape_add_nd_qu8(
      op27,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #27" << std::endl;
    return ExecutionPlan();
  }

  size_t op28_workspace_size = 0;
  size_t op28_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op28,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op28_workspace_size, &op28_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op28_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #28" << std::endl;
    return ExecutionPlan();
  }

  size_t op29_workspace_size = 0;
  size_t op29_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op29,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op29_workspace_size, &op29_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op29_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #29" << std::endl;
    return ExecutionPlan();
  }

  size_t op30_workspace_size = 0;
  size_t op30_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op30,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op30_workspace_size, &op30_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op30_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #30" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 64 };
    const size_t b_shape[] = { 1, 14, 14, 64 };
    status = xnn_reshape_add_nd_qu8(
      op31,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #31" << std::endl;
    return ExecutionPlan();
  }

  size_t op32_workspace_size = 0;
  size_t op32_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op32,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op32_workspace_size, &op32_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op32_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #32" << std::endl;
    return ExecutionPlan();
  }

  size_t op33_workspace_size = 0;
  size_t op33_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op33,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op33_workspace_size, &op33_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op33_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #33" << std::endl;
    return ExecutionPlan();
  }

  size_t op34_workspace_size = 0;
  size_t op34_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op34,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op34_workspace_size, &op34_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op34_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #34" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 64 };
    const size_t b_shape[] = { 1, 14, 14, 64 };
    status = xnn_reshape_add_nd_qu8(
      op35,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #35" << std::endl;
    return ExecutionPlan();
  }

  size_t op36_workspace_size = 0;
  size_t op36_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op36,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op36_workspace_size, &op36_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op36_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #36" << std::endl;
    return ExecutionPlan();
  }

  size_t op37_workspace_size = 0;
  size_t op37_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op37,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op37_workspace_size, &op37_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op37_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #37" << std::endl;
    return ExecutionPlan();
  }

  size_t op38_workspace_size = 0;
  size_t op38_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op38,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op38_workspace_size, &op38_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op38_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #38" << std::endl;
    return ExecutionPlan();
  }

  size_t op39_workspace_size = 0;
  size_t op39_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op39,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op39_workspace_size, &op39_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op39_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #39" << std::endl;
    return ExecutionPlan();
  }

  size_t op40_workspace_size = 0;
  size_t op40_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op40,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op40_workspace_size, &op40_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op40_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #40" << std::endl;
    return ExecutionPlan();
  }

  size_t op41_workspace_size = 0;
  size_t op41_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op41,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op41_workspace_size, &op41_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op41_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #41" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 96 };
    const size_t b_shape[] = { 1, 14, 14, 96 };
    status = xnn_reshape_add_nd_qu8(
      op42,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #42" << std::endl;
    return ExecutionPlan();
  }

  size_t op43_workspace_size = 0;
  size_t op43_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op43,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op43_workspace_size, &op43_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op43_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #43" << std::endl;
    return ExecutionPlan();
  }

  size_t op44_workspace_size = 0;
  size_t op44_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op44,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op44_workspace_size, &op44_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op44_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #44" << std::endl;
    return ExecutionPlan();
  }

  size_t op45_workspace_size = 0;
  size_t op45_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op45,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op45_workspace_size, &op45_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op45_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #45" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 96 };
    const size_t b_shape[] = { 1, 14, 14, 96 };
    status = xnn_reshape_add_nd_qu8(
      op46,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #46" << std::endl;
    return ExecutionPlan();
  }

  size_t op47_workspace_size = 0;
  size_t op47_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op47,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op47_workspace_size, &op47_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op47_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #47" << std::endl;
    return ExecutionPlan();
  }

  size_t op48_workspace_size = 0;
  size_t op48_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op48,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    &op48_workspace_size, &op48_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op48_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #48" << std::endl;
    return ExecutionPlan();
  }

  size_t op49_workspace_size = 0;
  size_t op49_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op49,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op49_workspace_size, &op49_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op49_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #49" << std::endl;
    return ExecutionPlan();
  }

  size_t op50_workspace_size = 0;
  size_t op50_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op50,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op50_workspace_size, &op50_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op50_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #50" << std::endl;
    return ExecutionPlan();
  }

  size_t op51_workspace_size = 0;
  size_t op51_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op51,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op51_workspace_size, &op51_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op51_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #51" << std::endl;
    return ExecutionPlan();
  }

  size_t op52_workspace_size = 0;
  size_t op52_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op52,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op52_workspace_size, &op52_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op52_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #52" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 160 };
    const size_t b_shape[] = { 1, 7, 7, 160 };
    status = xnn_reshape_add_nd_qu8(
      op53,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #53" << std::endl;
    return ExecutionPlan();
  }

  size_t op54_workspace_size = 0;
  size_t op54_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op54,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op54_workspace_size, &op54_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op54_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #54" << std::endl;
    return ExecutionPlan();
  }

  size_t op55_workspace_size = 0;
  size_t op55_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op55,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op55_workspace_size, &op55_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op55_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #55" << std::endl;
    return ExecutionPlan();
  }

  size_t op56_workspace_size = 0;
  size_t op56_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op56,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op56_workspace_size, &op56_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op56_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #56" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 160 };
    const size_t b_shape[] = { 1, 7, 7, 160 };
    status = xnn_reshape_add_nd_qu8(
      op57,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #57" << std::endl;
    return ExecutionPlan();
  }

  size_t op58_workspace_size = 0;
  size_t op58_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op58,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op58_workspace_size, &op58_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op58_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #58" << std::endl;
    return ExecutionPlan();
  }

  size_t op59_workspace_size = 0;
  size_t op59_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op59,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op59_workspace_size, &op59_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op59_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #59" << std::endl;
    return ExecutionPlan();
  }

  size_t op60_workspace_size = 0;
  size_t op60_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op60,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op60_workspace_size, &op60_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op60_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #60" << std::endl;
    return ExecutionPlan();
  }

  size_t op61_workspace_size = 0;
  size_t op61_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op61,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    &op61_workspace_size, &op61_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op61_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #61" << std::endl;
    return ExecutionPlan();
  }

  size_t op62_workspace_size = 0;
  size_t op62_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_qu8(
    op62,
    /*batch_size=*/1, 49 /* width */,
    1280 /* channels */, 1280 /* input stride */, 1280 /* output stride */,
    &op62_workspace_size, &op62_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op62_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #62" << std::endl;
    return ExecutionPlan();
  }

  size_t op63_workspace_size = 0;
  size_t op63_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_qu8(
    op63,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    &op63_workspace_size, &op63_workspace_alignment,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op63_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_copy_nc_x8(
    op64,
    /*batch_size=*/1001,
    1 /* channels */,
    1 /* input stride */,
    1 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #64" << std::endl;
    return ExecutionPlan();
  }

  Workspace workspace(max_workspace_size);

  status = xnn_setup_convolution2d_nhwc_qu8(
    op0,
    workspace.data(), /*input=*/v0.data(), /*output=*/v1.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op1,
    workspace.data(), /*input=*/v1.data(), /*output=*/v2.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op2,
    workspace.data(), /*input=*/v2.data(), /*output=*/v3.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op3,
    workspace.data(), /*input=*/v3.data(), /*output=*/v4.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op4,
    workspace.data(), /*input=*/v4.data(), /*output=*/v5.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op5,
    workspace.data(), /*input=*/v5.data(), /*output=*/v6.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op6,
    workspace.data(), /*input=*/v6.data(), /*output=*/v7.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op7,
    workspace.data(), /*input=*/v7.data(), /*output=*/v8.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op8,
    workspace.data(), /*input=*/v8.data(), /*output=*/v9.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op9,
    v9.data() /* a */, v6.data() /* b */, /*output=*/v10.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op10,
    workspace.data(), /*input=*/v10.data(), /*output=*/v11.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op11,
    workspace.data(), /*input=*/v11.data(), /*output=*/v12.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op12,
    workspace.data(), /*input=*/v12.data(), /*output=*/v13.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op13,
    workspace.data(), /*input=*/v13.data(), /*output=*/v14.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op14,
    workspace.data(), /*input=*/v14.data(), /*output=*/v15.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op15,
    workspace.data(), /*input=*/v15.data(), /*output=*/v16.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op16,
    v16.data() /* a */, v13.data() /* b */, /*output=*/v17.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op17,
    workspace.data(), /*input=*/v17.data(), /*output=*/v18.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op18,
    workspace.data(), /*input=*/v18.data(), /*output=*/v19.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op19,
    workspace.data(), /*input=*/v19.data(), /*output=*/v20.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op20,
    v20.data() /* a */, v17.data() /* b */, /*output=*/v21.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op21,
    workspace.data(), /*input=*/v21.data(), /*output=*/v22.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op22,
    workspace.data(), /*input=*/v22.data(), /*output=*/v23.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op23,
    workspace.data(), /*input=*/v23.data(), /*output=*/v24.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op24,
    workspace.data(), /*input=*/v24.data(), /*output=*/v25.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op25,
    workspace.data(), /*input=*/v25.data(), /*output=*/v26.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op26,
    workspace.data(), /*input=*/v26.data(), /*output=*/v27.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op27,
    v27.data() /* a */, v24.data() /* b */, /*output=*/v28.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op28,
    workspace.data(), /*input=*/v28.data(), /*output=*/v29.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op29,
    workspace.data(), /*input=*/v29.data(), /*output=*/v30.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op30,
    workspace.data(), /*input=*/v30.data(), /*output=*/v31.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op31,
    v31.data() /* a */, v28.data() /* b */, /*output=*/v32.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op32,
    workspace.data(), /*input=*/v32.data(), /*output=*/v33.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op33,
    workspace.data(), /*input=*/v33.data(), /*output=*/v34.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op34,
    workspace.data(), /*input=*/v34.data(), /*output=*/v35.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op35,
    v35.data() /* a */, v32.data() /* b */, /*output=*/v36.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op36,
    workspace.data(), /*input=*/v36.data(), /*output=*/v37.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op37,
    workspace.data(), /*input=*/v37.data(), /*output=*/v38.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op38,
    workspace.data(), /*input=*/v38.data(), /*output=*/v39.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op39,
    workspace.data(), /*input=*/v39.data(), /*output=*/v40.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op40,
    workspace.data(), /*input=*/v40.data(), /*output=*/v41.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op41,
    workspace.data(), /*input=*/v41.data(), /*output=*/v42.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op42,
    v42.data() /* a */, v39.data() /* b */, /*output=*/v43.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op43,
    workspace.data(), /*input=*/v43.data(), /*output=*/v44.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op44,
    workspace.data(), /*input=*/v44.data(), /*output=*/v45.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op45,
    workspace.data(), /*input=*/v45.data(), /*output=*/v46.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op46,
    v46.data() /* a */, v43.data() /* b */, /*output=*/v47.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op47,
    workspace.data(), /*input=*/v47.data(), /*output=*/v48.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op48,
    workspace.data(), /*input=*/v48.data(), /*output=*/v49.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op49,
    workspace.data(), /*input=*/v49.data(), /*output=*/v50.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op50,
    workspace.data(), /*input=*/v50.data(), /*output=*/v51.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op51,
    workspace.data(), /*input=*/v51.data(), /*output=*/v52.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op52,
    workspace.data(), /*input=*/v52.data(), /*output=*/v53.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op53,
    v53.data() /* a */, v50.data() /* b */, /*output=*/v54.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op54,
    workspace.data(), /*input=*/v54.data(), /*output=*/v55.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op55,
    workspace.data(), /*input=*/v55.data(), /*output=*/v56.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op56,
    workspace.data(), /*input=*/v56.data(), /*output=*/v57.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_qu8(
    op57,
    v57.data() /* a */, v54.data() /* b */, /*output=*/v58.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op58,
    workspace.data(), /*input=*/v58.data(), /*output=*/v59.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op59,
    workspace.data(), /*input=*/v59.data(), /*output=*/v60.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op60,
    workspace.data(), /*input=*/v60.data(), /*output=*/v61.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op61,
    workspace.data(), /*input=*/v61.data(), /*output=*/v62.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_qu8(
    op62,
    workspace.data(),
    /*input=*/v62.data(), /*output=*/v63.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_qu8(
    op63,
    workspace.data(), /*input=*/v63.data(), /*output=*/v64.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_copy_nc_x8(
    op64,
    /*input=*/v64.data(), /*output=*/v65.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #64" << std::endl;
    return ExecutionPlan();
  }

  XNN_PRAGMA_CLANG("clang diagnostic push")
  XNN_PRAGMA_CLANG("clang diagnostic ignored \"-Wpessimizing-move\"")
  return ExecutionPlan{operators, workspace};
  XNN_PRAGMA_CLANG("clang diagnostic pop")
}

}  // namespace models
