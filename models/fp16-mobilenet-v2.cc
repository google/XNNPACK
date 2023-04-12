// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>

#include <array>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include <xnnpack/cache.h>
#include <xnnpack/common.h>

#include <fp16/fp16.h>

#include "models/models.h"

namespace models {

ExecutionPlan FP16MobileNetV2(bool use_jit, pthreadpool_t threadpool) {
  alignas(16) static std::array<uint16_t, 150528 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v0;
  alignas(16) static std::array<uint16_t, 401408 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v1;
  alignas(16) static std::array<uint16_t, 401408 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v2;
  alignas(16) static std::array<uint16_t, 200704 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v3;
  alignas(16) static std::array<uint16_t, 1204224 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v4;
  alignas(16) static std::array<uint16_t, 301056 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v5;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v6;
  alignas(16) static std::array<uint16_t, 451584 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v7;
  alignas(16) static std::array<uint16_t, 451584 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v8;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v9;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v10;
  alignas(16) static std::array<uint16_t, 451584 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v11;
  alignas(16) static std::array<uint16_t, 112896 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v12;
  alignas(16) static std::array<uint16_t, 25088 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v13;
  alignas(16) static std::array<uint16_t, 150528 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v14;
  alignas(16) static std::array<uint16_t, 150528 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v15;
  alignas(16) static std::array<uint16_t, 25088 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v16;
  alignas(16) static std::array<uint16_t, 25088 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v17;
  alignas(16) static std::array<uint16_t, 150528 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v18;
  alignas(16) static std::array<uint16_t, 150528 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v19;
  alignas(16) static std::array<uint16_t, 25088 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v20;
  alignas(16) static std::array<uint16_t, 25088 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v21;
  alignas(16) static std::array<uint16_t, 150528 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v22;
  alignas(16) static std::array<uint16_t, 37632 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v23;
  alignas(16) static std::array<uint16_t, 12544 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v24;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v25;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v26;
  alignas(16) static std::array<uint16_t, 12544 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v27;
  alignas(16) static std::array<uint16_t, 12544 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v28;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v29;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v30;
  alignas(16) static std::array<uint16_t, 12544 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v31;
  alignas(16) static std::array<uint16_t, 12544 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v32;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v33;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v34;
  alignas(16) static std::array<uint16_t, 12544 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v35;
  alignas(16) static std::array<uint16_t, 12544 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v36;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v37;
  alignas(16) static std::array<uint16_t, 75264 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v38;
  alignas(16) static std::array<uint16_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v39;
  alignas(16) static std::array<uint16_t, 112896 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v40;
  alignas(16) static std::array<uint16_t, 112896 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v41;
  alignas(16) static std::array<uint16_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v42;
  alignas(16) static std::array<uint16_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v43;
  alignas(16) static std::array<uint16_t, 112896 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v44;
  alignas(16) static std::array<uint16_t, 112896 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v45;
  alignas(16) static std::array<uint16_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v46;
  alignas(16) static std::array<uint16_t, 18816 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v47;
  alignas(16) static std::array<uint16_t, 112896 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v48;
  alignas(16) static std::array<uint16_t, 28224 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v49;
  alignas(16) static std::array<uint16_t, 7840 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v50;
  alignas(16) static std::array<uint16_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v51;
  alignas(16) static std::array<uint16_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v52;
  alignas(16) static std::array<uint16_t, 7840 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v53;
  alignas(16) static std::array<uint16_t, 7840 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v54;
  alignas(16) static std::array<uint16_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v55;
  alignas(16) static std::array<uint16_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v56;
  alignas(16) static std::array<uint16_t, 7840 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v57;
  alignas(16) static std::array<uint16_t, 7840 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v58;
  alignas(16) static std::array<uint16_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v59;
  alignas(16) static std::array<uint16_t, 47040 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v60;
  alignas(16) static std::array<uint16_t, 15680 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v61;
  alignas(16) static std::array<uint16_t, 62720 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v62;
  alignas(16) static std::array<uint16_t, 1280 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v63;
  alignas(16) static std::array<uint16_t, 1001 + XNN_EXTRA_BYTES / sizeof(uint16_t)> v64;
  alignas(16) static std::array<uint16_t, 864 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w65;
  alignas(16) static std::array<uint16_t, 32 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w66;
  alignas(16) static std::array<uint16_t, 288 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w67;
  alignas(16) static std::array<uint16_t, 32 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w68;
  alignas(16) static std::array<uint16_t, 512 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w69;
  alignas(16) static std::array<uint16_t, 16 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w70;
  alignas(16) static std::array<uint16_t, 1536 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w71;
  alignas(16) static std::array<uint16_t, 96 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w72;
  alignas(16) static std::array<uint16_t, 864 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w73;
  alignas(16) static std::array<uint16_t, 96 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w74;
  alignas(16) static std::array<uint16_t, 2304 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w75;
  alignas(16) static std::array<uint16_t, 24 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w76;
  alignas(16) static std::array<uint16_t, 3456 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w77;
  alignas(16) static std::array<uint16_t, 144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w78;
  alignas(16) static std::array<uint16_t, 1296 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w79;
  alignas(16) static std::array<uint16_t, 144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w80;
  alignas(16) static std::array<uint16_t, 3456 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w81;
  alignas(16) static std::array<uint16_t, 24 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w82;
  alignas(16) static std::array<uint16_t, 3456 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w83;
  alignas(16) static std::array<uint16_t, 144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w84;
  alignas(16) static std::array<uint16_t, 1296 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w85;
  alignas(16) static std::array<uint16_t, 144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w86;
  alignas(16) static std::array<uint16_t, 4608 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w87;
  alignas(16) static std::array<uint16_t, 32 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w88;
  alignas(16) static std::array<uint16_t, 6144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w89;
  alignas(16) static std::array<uint16_t, 192 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w90;
  alignas(16) static std::array<uint16_t, 1728 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w91;
  alignas(16) static std::array<uint16_t, 192 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w92;
  alignas(16) static std::array<uint16_t, 6144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w93;
  alignas(16) static std::array<uint16_t, 32 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w94;
  alignas(16) static std::array<uint16_t, 6144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w95;
  alignas(16) static std::array<uint16_t, 192 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w96;
  alignas(16) static std::array<uint16_t, 1728 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w97;
  alignas(16) static std::array<uint16_t, 192 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w98;
  alignas(16) static std::array<uint16_t, 6144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w99;
  alignas(16) static std::array<uint16_t, 32 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w100;
  alignas(16) static std::array<uint16_t, 6144 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w101;
  alignas(16) static std::array<uint16_t, 192 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w102;
  alignas(16) static std::array<uint16_t, 1728 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w103;
  alignas(16) static std::array<uint16_t, 192 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w104;
  alignas(16) static std::array<uint16_t, 12288 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w105;
  alignas(16) static std::array<uint16_t, 64 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w106;
  alignas(16) static std::array<uint16_t, 24576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w107;
  alignas(16) static std::array<uint16_t, 384 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w108;
  alignas(16) static std::array<uint16_t, 3456 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w109;
  alignas(16) static std::array<uint16_t, 384 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w110;
  alignas(16) static std::array<uint16_t, 24576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w111;
  alignas(16) static std::array<uint16_t, 64 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w112;
  alignas(16) static std::array<uint16_t, 24576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w113;
  alignas(16) static std::array<uint16_t, 384 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w114;
  alignas(16) static std::array<uint16_t, 3456 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w115;
  alignas(16) static std::array<uint16_t, 384 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w116;
  alignas(16) static std::array<uint16_t, 24576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w117;
  alignas(16) static std::array<uint16_t, 64 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w118;
  alignas(16) static std::array<uint16_t, 24576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w119;
  alignas(16) static std::array<uint16_t, 384 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w120;
  alignas(16) static std::array<uint16_t, 3456 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w121;
  alignas(16) static std::array<uint16_t, 384 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w122;
  alignas(16) static std::array<uint16_t, 24576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w123;
  alignas(16) static std::array<uint16_t, 64 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w124;
  alignas(16) static std::array<uint16_t, 24576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w125;
  alignas(16) static std::array<uint16_t, 384 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w126;
  alignas(16) static std::array<uint16_t, 3456 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w127;
  alignas(16) static std::array<uint16_t, 384 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w128;
  alignas(16) static std::array<uint16_t, 36864 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w129;
  alignas(16) static std::array<uint16_t, 96 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w130;
  alignas(16) static std::array<uint16_t, 55296 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w131;
  alignas(16) static std::array<uint16_t, 576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w132;
  alignas(16) static std::array<uint16_t, 5184 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w133;
  alignas(16) static std::array<uint16_t, 576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w134;
  alignas(16) static std::array<uint16_t, 55296 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w135;
  alignas(16) static std::array<uint16_t, 96 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w136;
  alignas(16) static std::array<uint16_t, 55296 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w137;
  alignas(16) static std::array<uint16_t, 576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w138;
  alignas(16) static std::array<uint16_t, 5184 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w139;
  alignas(16) static std::array<uint16_t, 576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w140;
  alignas(16) static std::array<uint16_t, 55296 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w141;
  alignas(16) static std::array<uint16_t, 96 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w142;
  alignas(16) static std::array<uint16_t, 55296 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w143;
  alignas(16) static std::array<uint16_t, 576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w144;
  alignas(16) static std::array<uint16_t, 5184 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w145;
  alignas(16) static std::array<uint16_t, 576 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w146;
  alignas(16) static std::array<uint16_t, 92160 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w147;
  alignas(16) static std::array<uint16_t, 160 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w148;
  alignas(16) static std::array<uint16_t, 153600 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w149;
  alignas(16) static std::array<uint16_t, 960 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w150;
  alignas(16) static std::array<uint16_t, 8640 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w151;
  alignas(16) static std::array<uint16_t, 960 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w152;
  alignas(16) static std::array<uint16_t, 153600 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w153;
  alignas(16) static std::array<uint16_t, 160 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w154;
  alignas(16) static std::array<uint16_t, 153600 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w155;
  alignas(16) static std::array<uint16_t, 960 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w156;
  alignas(16) static std::array<uint16_t, 8640 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w157;
  alignas(16) static std::array<uint16_t, 960 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w158;
  alignas(16) static std::array<uint16_t, 153600 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w159;
  alignas(16) static std::array<uint16_t, 160 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w160;
  alignas(16) static std::array<uint16_t, 153600 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w161;
  alignas(16) static std::array<uint16_t, 960 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w162;
  alignas(16) static std::array<uint16_t, 8640 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w163;
  alignas(16) static std::array<uint16_t, 960 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w164;
  alignas(16) static std::array<uint16_t, 307200 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w165;
  alignas(16) static std::array<uint16_t, 320 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w166;
  alignas(16) static std::array<uint16_t, 409600 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w167;
  alignas(16) static std::array<uint16_t, 1280 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w168;
  alignas(16) static std::array<uint16_t, 1281280 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w169;
  alignas(16) static std::array<uint16_t, 1001 + XNN_EXTRA_BYTES / sizeof(uint16_t)> w170;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);
  std::generate(v0.begin(), v0.end(), std::ref(f16rng));
  std::generate(v1.begin(), v1.end(), std::ref(f16rng));
  std::generate(v2.begin(), v2.end(), std::ref(f16rng));
  std::generate(v3.begin(), v3.end(), std::ref(f16rng));
  std::generate(v4.begin(), v4.end(), std::ref(f16rng));
  std::generate(v5.begin(), v5.end(), std::ref(f16rng));
  std::generate(v6.begin(), v6.end(), std::ref(f16rng));
  std::generate(v7.begin(), v7.end(), std::ref(f16rng));
  std::generate(v8.begin(), v8.end(), std::ref(f16rng));
  std::generate(v9.begin(), v9.end(), std::ref(f16rng));
  std::generate(v10.begin(), v10.end(), std::ref(f16rng));
  std::generate(v11.begin(), v11.end(), std::ref(f16rng));
  std::generate(v12.begin(), v12.end(), std::ref(f16rng));
  std::generate(v13.begin(), v13.end(), std::ref(f16rng));
  std::generate(v14.begin(), v14.end(), std::ref(f16rng));
  std::generate(v15.begin(), v15.end(), std::ref(f16rng));
  std::generate(v16.begin(), v16.end(), std::ref(f16rng));
  std::generate(v17.begin(), v17.end(), std::ref(f16rng));
  std::generate(v18.begin(), v18.end(), std::ref(f16rng));
  std::generate(v19.begin(), v19.end(), std::ref(f16rng));
  std::generate(v20.begin(), v20.end(), std::ref(f16rng));
  std::generate(v21.begin(), v21.end(), std::ref(f16rng));
  std::generate(v22.begin(), v22.end(), std::ref(f16rng));
  std::generate(v23.begin(), v23.end(), std::ref(f16rng));
  std::generate(v24.begin(), v24.end(), std::ref(f16rng));
  std::generate(v25.begin(), v25.end(), std::ref(f16rng));
  std::generate(v26.begin(), v26.end(), std::ref(f16rng));
  std::generate(v27.begin(), v27.end(), std::ref(f16rng));
  std::generate(v28.begin(), v28.end(), std::ref(f16rng));
  std::generate(v29.begin(), v29.end(), std::ref(f16rng));
  std::generate(v30.begin(), v30.end(), std::ref(f16rng));
  std::generate(v31.begin(), v31.end(), std::ref(f16rng));
  std::generate(v32.begin(), v32.end(), std::ref(f16rng));
  std::generate(v33.begin(), v33.end(), std::ref(f16rng));
  std::generate(v34.begin(), v34.end(), std::ref(f16rng));
  std::generate(v35.begin(), v35.end(), std::ref(f16rng));
  std::generate(v36.begin(), v36.end(), std::ref(f16rng));
  std::generate(v37.begin(), v37.end(), std::ref(f16rng));
  std::generate(v38.begin(), v38.end(), std::ref(f16rng));
  std::generate(v39.begin(), v39.end(), std::ref(f16rng));
  std::generate(v40.begin(), v40.end(), std::ref(f16rng));
  std::generate(v41.begin(), v41.end(), std::ref(f16rng));
  std::generate(v42.begin(), v42.end(), std::ref(f16rng));
  std::generate(v43.begin(), v43.end(), std::ref(f16rng));
  std::generate(v44.begin(), v44.end(), std::ref(f16rng));
  std::generate(v45.begin(), v45.end(), std::ref(f16rng));
  std::generate(v46.begin(), v46.end(), std::ref(f16rng));
  std::generate(v47.begin(), v47.end(), std::ref(f16rng));
  std::generate(v48.begin(), v48.end(), std::ref(f16rng));
  std::generate(v49.begin(), v49.end(), std::ref(f16rng));
  std::generate(v50.begin(), v50.end(), std::ref(f16rng));
  std::generate(v51.begin(), v51.end(), std::ref(f16rng));
  std::generate(v52.begin(), v52.end(), std::ref(f16rng));
  std::generate(v53.begin(), v53.end(), std::ref(f16rng));
  std::generate(v54.begin(), v54.end(), std::ref(f16rng));
  std::generate(v55.begin(), v55.end(), std::ref(f16rng));
  std::generate(v56.begin(), v56.end(), std::ref(f16rng));
  std::generate(v57.begin(), v57.end(), std::ref(f16rng));
  std::generate(v58.begin(), v58.end(), std::ref(f16rng));
  std::generate(v59.begin(), v59.end(), std::ref(f16rng));
  std::generate(v60.begin(), v60.end(), std::ref(f16rng));
  std::generate(v61.begin(), v61.end(), std::ref(f16rng));
  std::generate(v62.begin(), v62.end(), std::ref(f16rng));
  std::generate(v63.begin(), v63.end(), std::ref(f16rng));
  std::generate(v64.begin(), v64.end(), std::ref(f16rng));
  std::generate(w65.begin(), w65.end(), std::ref(f16rng));
  std::generate(w66.begin(), w66.end(), std::ref(f16rng));
  std::generate(w67.begin(), w67.end(), std::ref(f16rng));
  std::generate(w68.begin(), w68.end(), std::ref(f16rng));
  std::generate(w69.begin(), w69.end(), std::ref(f16rng));
  std::generate(w70.begin(), w70.end(), std::ref(f16rng));
  std::generate(w71.begin(), w71.end(), std::ref(f16rng));
  std::generate(w72.begin(), w72.end(), std::ref(f16rng));
  std::generate(w73.begin(), w73.end(), std::ref(f16rng));
  std::generate(w74.begin(), w74.end(), std::ref(f16rng));
  std::generate(w75.begin(), w75.end(), std::ref(f16rng));
  std::generate(w76.begin(), w76.end(), std::ref(f16rng));
  std::generate(w77.begin(), w77.end(), std::ref(f16rng));
  std::generate(w78.begin(), w78.end(), std::ref(f16rng));
  std::generate(w79.begin(), w79.end(), std::ref(f16rng));
  std::generate(w80.begin(), w80.end(), std::ref(f16rng));
  std::generate(w81.begin(), w81.end(), std::ref(f16rng));
  std::generate(w82.begin(), w82.end(), std::ref(f16rng));
  std::generate(w83.begin(), w83.end(), std::ref(f16rng));
  std::generate(w84.begin(), w84.end(), std::ref(f16rng));
  std::generate(w85.begin(), w85.end(), std::ref(f16rng));
  std::generate(w86.begin(), w86.end(), std::ref(f16rng));
  std::generate(w87.begin(), w87.end(), std::ref(f16rng));
  std::generate(w88.begin(), w88.end(), std::ref(f16rng));
  std::generate(w89.begin(), w89.end(), std::ref(f16rng));
  std::generate(w90.begin(), w90.end(), std::ref(f16rng));
  std::generate(w91.begin(), w91.end(), std::ref(f16rng));
  std::generate(w92.begin(), w92.end(), std::ref(f16rng));
  std::generate(w93.begin(), w93.end(), std::ref(f16rng));
  std::generate(w94.begin(), w94.end(), std::ref(f16rng));
  std::generate(w95.begin(), w95.end(), std::ref(f16rng));
  std::generate(w96.begin(), w96.end(), std::ref(f16rng));
  std::generate(w97.begin(), w97.end(), std::ref(f16rng));
  std::generate(w98.begin(), w98.end(), std::ref(f16rng));
  std::generate(w99.begin(), w99.end(), std::ref(f16rng));
  std::generate(w100.begin(), w100.end(), std::ref(f16rng));
  std::generate(w101.begin(), w101.end(), std::ref(f16rng));
  std::generate(w102.begin(), w102.end(), std::ref(f16rng));
  std::generate(w103.begin(), w103.end(), std::ref(f16rng));
  std::generate(w104.begin(), w104.end(), std::ref(f16rng));
  std::generate(w105.begin(), w105.end(), std::ref(f16rng));
  std::generate(w106.begin(), w106.end(), std::ref(f16rng));
  std::generate(w107.begin(), w107.end(), std::ref(f16rng));
  std::generate(w108.begin(), w108.end(), std::ref(f16rng));
  std::generate(w109.begin(), w109.end(), std::ref(f16rng));
  std::generate(w110.begin(), w110.end(), std::ref(f16rng));
  std::generate(w111.begin(), w111.end(), std::ref(f16rng));
  std::generate(w112.begin(), w112.end(), std::ref(f16rng));
  std::generate(w113.begin(), w113.end(), std::ref(f16rng));
  std::generate(w114.begin(), w114.end(), std::ref(f16rng));
  std::generate(w115.begin(), w115.end(), std::ref(f16rng));
  std::generate(w116.begin(), w116.end(), std::ref(f16rng));
  std::generate(w117.begin(), w117.end(), std::ref(f16rng));
  std::generate(w118.begin(), w118.end(), std::ref(f16rng));
  std::generate(w119.begin(), w119.end(), std::ref(f16rng));
  std::generate(w120.begin(), w120.end(), std::ref(f16rng));
  std::generate(w121.begin(), w121.end(), std::ref(f16rng));
  std::generate(w122.begin(), w122.end(), std::ref(f16rng));
  std::generate(w123.begin(), w123.end(), std::ref(f16rng));
  std::generate(w124.begin(), w124.end(), std::ref(f16rng));
  std::generate(w125.begin(), w125.end(), std::ref(f16rng));
  std::generate(w126.begin(), w126.end(), std::ref(f16rng));
  std::generate(w127.begin(), w127.end(), std::ref(f16rng));
  std::generate(w128.begin(), w128.end(), std::ref(f16rng));
  std::generate(w129.begin(), w129.end(), std::ref(f16rng));
  std::generate(w130.begin(), w130.end(), std::ref(f16rng));
  std::generate(w131.begin(), w131.end(), std::ref(f16rng));
  std::generate(w132.begin(), w132.end(), std::ref(f16rng));
  std::generate(w133.begin(), w133.end(), std::ref(f16rng));
  std::generate(w134.begin(), w134.end(), std::ref(f16rng));
  std::generate(w135.begin(), w135.end(), std::ref(f16rng));
  std::generate(w136.begin(), w136.end(), std::ref(f16rng));
  std::generate(w137.begin(), w137.end(), std::ref(f16rng));
  std::generate(w138.begin(), w138.end(), std::ref(f16rng));
  std::generate(w139.begin(), w139.end(), std::ref(f16rng));
  std::generate(w140.begin(), w140.end(), std::ref(f16rng));
  std::generate(w141.begin(), w141.end(), std::ref(f16rng));
  std::generate(w142.begin(), w142.end(), std::ref(f16rng));
  std::generate(w143.begin(), w143.end(), std::ref(f16rng));
  std::generate(w144.begin(), w144.end(), std::ref(f16rng));
  std::generate(w145.begin(), w145.end(), std::ref(f16rng));
  std::generate(w146.begin(), w146.end(), std::ref(f16rng));
  std::generate(w147.begin(), w147.end(), std::ref(f16rng));
  std::generate(w148.begin(), w148.end(), std::ref(f16rng));
  std::generate(w149.begin(), w149.end(), std::ref(f16rng));
  std::generate(w150.begin(), w150.end(), std::ref(f16rng));
  std::generate(w151.begin(), w151.end(), std::ref(f16rng));
  std::generate(w152.begin(), w152.end(), std::ref(f16rng));
  std::generate(w153.begin(), w153.end(), std::ref(f16rng));
  std::generate(w154.begin(), w154.end(), std::ref(f16rng));
  std::generate(w155.begin(), w155.end(), std::ref(f16rng));
  std::generate(w156.begin(), w156.end(), std::ref(f16rng));
  std::generate(w157.begin(), w157.end(), std::ref(f16rng));
  std::generate(w158.begin(), w158.end(), std::ref(f16rng));
  std::generate(w159.begin(), w159.end(), std::ref(f16rng));
  std::generate(w160.begin(), w160.end(), std::ref(f16rng));
  std::generate(w161.begin(), w161.end(), std::ref(f16rng));
  std::generate(w162.begin(), w162.end(), std::ref(f16rng));
  std::generate(w163.begin(), w163.end(), std::ref(f16rng));
  std::generate(w164.begin(), w164.end(), std::ref(f16rng));
  std::generate(w165.begin(), w165.end(), std::ref(f16rng));
  std::generate(w166.begin(), w166.end(), std::ref(f16rng));
  std::generate(w167.begin(), w167.end(), std::ref(f16rng));
  std::generate(w168.begin(), w168.end(), std::ref(f16rng));
  std::generate(w169.begin(), w169.end(), std::ref(f16rng));
  std::generate(w170.begin(), w170.end(), std::ref(f16rng));

  ExecutionPlan operators;
  xnn_status status;
  xnn_caches caches = {};
#if XNN_PLATFORM_JIT
  xnn_code_cache code_cache;
  if (use_jit) {
    status = xnn_init_code_cache(&code_cache);
    if (status != xnn_status_success) {
      std::cerr << "failed to initailize code cache" << std::endl;
      return ExecutionPlan();
    }
    caches.code_cache = &code_cache;
  }
#endif  // XNN_PLATFORM_JIT

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    3 /* input channels per group */,
    32 /* output_channels_per_group */,
    3 /* input pixel stride */,
    32 /* output pixel stride */,
    w65.data(), w66.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    32 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    32 /* input pixel stride */,
    32 /* output pixel stride */,
    w67.data(), w68.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    16 /* output_channels_per_group */,
    32 /* input pixel stride */,
    16 /* output pixel stride */,
    w69.data(), w70.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    16 /* input channels per group */,
    96 /* output_channels_per_group */,
    16 /* input pixel stride */,
    96 /* output pixel stride */,
    w71.data(), w72.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    96 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    96 /* input pixel stride */,
    96 /* output pixel stride */,
    w73.data(), w74.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    96 /* input channels per group */,
    24 /* output_channels_per_group */,
    96 /* input pixel stride */,
    24 /* output pixel stride */,
    w75.data(), w76.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    24 /* input channels per group */,
    144 /* output_channels_per_group */,
    24 /* input pixel stride */,
    144 /* output pixel stride */,
    w77.data(), w78.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    144 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    144 /* input pixel stride */,
    144 /* output pixel stride */,
    w79.data(), w80.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    144 /* input channels per group */,
    24 /* output_channels_per_group */,
    144 /* input pixel stride */,
    24 /* output pixel stride */,
    w81.data(), w82.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    24 /* input channels per group */,
    144 /* output_channels_per_group */,
    24 /* input pixel stride */,
    144 /* output pixel stride */,
    w83.data(), w84.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    144 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    144 /* input pixel stride */,
    144 /* output pixel stride */,
    w85.data(), w86.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    144 /* input channels per group */,
    32 /* output_channels_per_group */,
    144 /* input pixel stride */,
    32 /* output pixel stride */,
    w87.data(), w88.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    192 /* output_channels_per_group */,
    32 /* input pixel stride */,
    192 /* output pixel stride */,
    w89.data(), w90.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    192 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    192 /* input pixel stride */,
    192 /* output pixel stride */,
    w91.data(), w92.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    192 /* input channels per group */,
    32 /* output_channels_per_group */,
    192 /* input pixel stride */,
    32 /* output pixel stride */,
    w93.data(), w94.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    192 /* output_channels_per_group */,
    32 /* input pixel stride */,
    192 /* output pixel stride */,
    w95.data(), w96.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    192 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    192 /* input pixel stride */,
    192 /* output pixel stride */,
    w97.data(), w98.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    192 /* input channels per group */,
    32 /* output_channels_per_group */,
    192 /* input pixel stride */,
    32 /* output pixel stride */,
    w99.data(), w100.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    192 /* output_channels_per_group */,
    32 /* input pixel stride */,
    192 /* output pixel stride */,
    w101.data(), w102.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    192 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    192 /* input pixel stride */,
    192 /* output pixel stride */,
    w103.data(), w104.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    192 /* input channels per group */,
    64 /* output_channels_per_group */,
    192 /* input pixel stride */,
    64 /* output pixel stride */,
    w105.data(), w106.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    384 /* output_channels_per_group */,
    64 /* input pixel stride */,
    384 /* output pixel stride */,
    w107.data(), w108.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    384 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    384 /* input pixel stride */,
    384 /* output pixel stride */,
    w109.data(), w110.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    384 /* input channels per group */,
    64 /* output_channels_per_group */,
    384 /* input pixel stride */,
    64 /* output pixel stride */,
    w111.data(), w112.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    384 /* output_channels_per_group */,
    64 /* input pixel stride */,
    384 /* output pixel stride */,
    w113.data(), w114.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  xnn_operator_t op29 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    384 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    384 /* input pixel stride */,
    384 /* output pixel stride */,
    w115.data(), w116.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #29" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op29, xnn_delete_operator);

  xnn_operator_t op30 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    384 /* input channels per group */,
    64 /* output_channels_per_group */,
    384 /* input pixel stride */,
    64 /* output pixel stride */,
    w117.data(), w118.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #30" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op30, xnn_delete_operator);

  xnn_operator_t op31 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #31" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op31, xnn_delete_operator);

  xnn_operator_t op32 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    384 /* output_channels_per_group */,
    64 /* input pixel stride */,
    384 /* output pixel stride */,
    w119.data(), w120.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #32" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op32, xnn_delete_operator);

  xnn_operator_t op33 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    384 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    384 /* input pixel stride */,
    384 /* output pixel stride */,
    w121.data(), w122.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #33" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op33, xnn_delete_operator);

  xnn_operator_t op34 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    384 /* input channels per group */,
    64 /* output_channels_per_group */,
    384 /* input pixel stride */,
    64 /* output pixel stride */,
    w123.data(), w124.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #34" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op34, xnn_delete_operator);

  xnn_operator_t op35 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #35" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op35, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    384 /* output_channels_per_group */,
    64 /* input pixel stride */,
    384 /* output pixel stride */,
    w125.data(), w126.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #36" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op36, xnn_delete_operator);

  xnn_operator_t op37 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    384 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    384 /* input pixel stride */,
    384 /* output pixel stride */,
    w127.data(), w128.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #37" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op37, xnn_delete_operator);

  xnn_operator_t op38 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    384 /* input channels per group */,
    96 /* output_channels_per_group */,
    384 /* input pixel stride */,
    96 /* output pixel stride */,
    w129.data(), w130.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #38" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op38, xnn_delete_operator);

  xnn_operator_t op39 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    96 /* input channels per group */,
    576 /* output_channels_per_group */,
    96 /* input pixel stride */,
    576 /* output pixel stride */,
    w131.data(), w132.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #39" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op39, xnn_delete_operator);

  xnn_operator_t op40 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    576 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    576 /* input pixel stride */,
    576 /* output pixel stride */,
    w133.data(), w134.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #40" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op40, xnn_delete_operator);

  xnn_operator_t op41 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    96 /* output_channels_per_group */,
    576 /* input pixel stride */,
    96 /* output pixel stride */,
    w135.data(), w136.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #41" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op41, xnn_delete_operator);

  xnn_operator_t op42 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #42" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op42, xnn_delete_operator);

  xnn_operator_t op43 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    96 /* input channels per group */,
    576 /* output_channels_per_group */,
    96 /* input pixel stride */,
    576 /* output pixel stride */,
    w137.data(), w138.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #43" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op43, xnn_delete_operator);

  xnn_operator_t op44 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    576 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    576 /* input pixel stride */,
    576 /* output pixel stride */,
    w139.data(), w140.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op45 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    96 /* output_channels_per_group */,
    576 /* input pixel stride */,
    96 /* output pixel stride */,
    w141.data(), w142.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #45" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op45, xnn_delete_operator);

  xnn_operator_t op46 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #46" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op46, xnn_delete_operator);

  xnn_operator_t op47 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    96 /* input channels per group */,
    576 /* output_channels_per_group */,
    96 /* input pixel stride */,
    576 /* output pixel stride */,
    w143.data(), w144.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #47" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op47, xnn_delete_operator);

  xnn_operator_t op48 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    576 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    576 /* input pixel stride */,
    576 /* output pixel stride */,
    w145.data(), w146.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #48" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op48, xnn_delete_operator);

  xnn_operator_t op49 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    160 /* output_channels_per_group */,
    576 /* input pixel stride */,
    160 /* output pixel stride */,
    w147.data(), w148.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #49" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op49, xnn_delete_operator);

  xnn_operator_t op50 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    160 /* input channels per group */,
    960 /* output_channels_per_group */,
    160 /* input pixel stride */,
    960 /* output pixel stride */,
    w149.data(), w150.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #50" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op50, xnn_delete_operator);

  xnn_operator_t op51 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    960 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    960 /* input pixel stride */,
    960 /* output pixel stride */,
    w151.data(), w152.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #51" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op51, xnn_delete_operator);

  xnn_operator_t op52 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    960 /* input channels per group */,
    160 /* output_channels_per_group */,
    960 /* input pixel stride */,
    160 /* output pixel stride */,
    w153.data(), w154.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #52" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op52, xnn_delete_operator);

  xnn_operator_t op53 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op54 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    160 /* input channels per group */,
    960 /* output_channels_per_group */,
    160 /* input pixel stride */,
    960 /* output pixel stride */,
    w155.data(), w156.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #54" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op54, xnn_delete_operator);

  xnn_operator_t op55 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    960 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    960 /* input pixel stride */,
    960 /* output pixel stride */,
    w157.data(), w158.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #55" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op55, xnn_delete_operator);

  xnn_operator_t op56 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    960 /* input channels per group */,
    160 /* output_channels_per_group */,
    960 /* input pixel stride */,
    160 /* output pixel stride */,
    w159.data(), w160.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #56" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op56, xnn_delete_operator);

  xnn_operator_t op57 = nullptr;
  status = xnn_create_add_nd_f16(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #57" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op57, xnn_delete_operator);

  xnn_operator_t op58 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    160 /* input channels per group */,
    960 /* output_channels_per_group */,
    160 /* input pixel stride */,
    960 /* output pixel stride */,
    w161.data(), w162.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #58" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op58, xnn_delete_operator);

  xnn_operator_t op59 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    960 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    960 /* input pixel stride */,
    960 /* output pixel stride */,
    w163.data(), w164.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #59" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op59, xnn_delete_operator);

  xnn_operator_t op60 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    960 /* input channels per group */,
    320 /* output_channels_per_group */,
    960 /* input pixel stride */,
    320 /* output pixel stride */,
    w165.data(), w166.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #60" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op60, xnn_delete_operator);

  xnn_operator_t op61 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    320 /* input channels per group */,
    1280 /* output_channels_per_group */,
    320 /* input pixel stride */,
    1280 /* output pixel stride */,
    w167.data(), w168.data(),
    0.0f /* output min */, 6.0f /* output max */,
    0 /* flags */,
    &caches,
    &op61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #61" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op61, xnn_delete_operator);

  xnn_operator_t op62 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f16(
    1280 /* channels */, 1280 /* input stride */, 1280 /* output stride */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op62);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #62" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op62, xnn_delete_operator);

  xnn_operator_t op63 = nullptr;
  status = xnn_create_convolution2d_nhwc_f16(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    1280 /* input channels per group */,
    1001 /* output_channels_per_group */,
    1280 /* input pixel stride */,
    1001 /* output pixel stride */,
    w169.data(), w170.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &caches,
    &op63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #63" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op63, xnn_delete_operator);

#if XNN_PLATFORM_JIT
  if (use_jit) {
    xnn_finalize_code_memory(&code_cache.cache.code);
  }
#endif  // XNN_PLATFORM_JIT

  status = xnn_setup_convolution2d_nhwc_f16(
    op0,
    1 /* batch size */, 224 /* input height */, 224 /* input width */,
    v0.data() /* input */, v1.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op1,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v1.data() /* input */, v2.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op2,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v2.data() /* input */, v3.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op3,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v3.data() /* input */, v4.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op4,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v4.data() /* input */, v5.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op5,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v5.data() /* input */, v6.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op6,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v6.data() /* input */, v7.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op7,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v7.data() /* input */, v8.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op8,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v8.data() /* input */, v9.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 56, 56, 24 };
    const size_t b_shape[] = { 1, 56, 56, 24 };
    status = xnn_setup_add_nd_f16(
      op9,
      4, a_shape, 4, b_shape,
      v9.data() /* a */, v6.data() /* b */, v10.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op10,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v10.data() /* input */, v11.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op11,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v11.data() /* input */, v12.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op12,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v12.data() /* input */, v13.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op13,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v13.data() /* input */, v14.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op14,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v14.data() /* input */, v15.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op15,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v15.data() /* input */, v16.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 32 };
    const size_t b_shape[] = { 1, 28, 28, 32 };
    status = xnn_setup_add_nd_f16(
      op16,
      4, a_shape, 4, b_shape,
      v16.data() /* a */, v13.data() /* b */, v17.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op17,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v17.data() /* input */, v18.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op18,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v18.data() /* input */, v19.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op19,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v19.data() /* input */, v20.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 32 };
    const size_t b_shape[] = { 1, 28, 28, 32 };
    status = xnn_setup_add_nd_f16(
      op20,
      4, a_shape, 4, b_shape,
      v20.data() /* a */, v17.data() /* b */, v21.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op21,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v21.data() /* input */, v22.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op22,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v22.data() /* input */, v23.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op23,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v23.data() /* input */, v24.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op24,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v24.data() /* input */, v25.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op25,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v25.data() /* input */, v26.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op26,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v26.data() /* input */, v27.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 64 };
    const size_t b_shape[] = { 1, 14, 14, 64 };
    status = xnn_setup_add_nd_f16(
      op27,
      4, a_shape, 4, b_shape,
      v27.data() /* a */, v24.data() /* b */, v28.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op28,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v28.data() /* input */, v29.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op29,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v29.data() /* input */, v30.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op30,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v30.data() /* input */, v31.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 64 };
    const size_t b_shape[] = { 1, 14, 14, 64 };
    status = xnn_setup_add_nd_f16(
      op31,
      4, a_shape, 4, b_shape,
      v31.data() /* a */, v28.data() /* b */, v32.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op32,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v32.data() /* input */, v33.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op33,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v33.data() /* input */, v34.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op34,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v34.data() /* input */, v35.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 64 };
    const size_t b_shape[] = { 1, 14, 14, 64 };
    status = xnn_setup_add_nd_f16(
      op35,
      4, a_shape, 4, b_shape,
      v35.data() /* a */, v32.data() /* b */, v36.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op36,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v36.data() /* input */, v37.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op37,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v37.data() /* input */, v38.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op38,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v38.data() /* input */, v39.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op39,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v39.data() /* input */, v40.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op40,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v40.data() /* input */, v41.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op41,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v41.data() /* input */, v42.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 96 };
    const size_t b_shape[] = { 1, 14, 14, 96 };
    status = xnn_setup_add_nd_f16(
      op42,
      4, a_shape, 4, b_shape,
      v42.data() /* a */, v39.data() /* b */, v43.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op43,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v43.data() /* input */, v44.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op44,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v44.data() /* input */, v45.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op45,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v45.data() /* input */, v46.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 96 };
    const size_t b_shape[] = { 1, 14, 14, 96 };
    status = xnn_setup_add_nd_f16(
      op46,
      4, a_shape, 4, b_shape,
      v46.data() /* a */, v43.data() /* b */, v47.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op47,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v47.data() /* input */, v48.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op48,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v48.data() /* input */, v49.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op49,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v49.data() /* input */, v50.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op50,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v50.data() /* input */, v51.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op51,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v51.data() /* input */, v52.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op52,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v52.data() /* input */, v53.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 160 };
    const size_t b_shape[] = { 1, 7, 7, 160 };
    status = xnn_setup_add_nd_f16(
      op53,
      4, a_shape, 4, b_shape,
      v53.data() /* a */, v50.data() /* b */, v54.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op54,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v54.data() /* input */, v55.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op55,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v55.data() /* input */, v56.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op56,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v56.data() /* input */, v57.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 160 };
    const size_t b_shape[] = { 1, 7, 7, 160 };
    status = xnn_setup_add_nd_f16(
      op57,
      4, a_shape, 4, b_shape,
      v57.data() /* a */, v54.data() /* b */, v58.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op58,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v58.data() /* input */, v59.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op59,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v59.data() /* input */, v60.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op60,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v60.data() /* input */, v61.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op61,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v61.data() /* input */, v62.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f16(
    op62,
    1 /* batch size */, 49 /* width */,
    v62.data() /* input */, v63.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f16(
    op63,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v63.data() /* input */, v64.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #63" << std::endl;
    return ExecutionPlan();
  }

  XNN_PRAGMA_CLANG("clang diagnostic push")
  XNN_PRAGMA_CLANG("clang diagnostic ignored \"-Wpessimizing-move\"")
  return operators;
  XNN_PRAGMA_CLANG("clang diagnostic pop")
}

ExecutionPlan FP16MobileNetV2(pthreadpool_t threadpool) {
  return FP16MobileNetV2(/*use_jit=*/false, threadpool);
}

ExecutionPlan FP16MobileNetV2Jit(pthreadpool_t threadpool) {
  return FP16MobileNetV2(/*use_jit=*/true, threadpool);
}

}  // namespace models
