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
#include <xnnpack/models.h>

namespace models {

ExecutionPlan FP32SparseMobileNetV2(float sparsity, pthreadpool_t threadpool) {
  alignas(16) static std::array<float, 150528 + XNN_EXTRA_BYTES / sizeof(float)> v0;
  alignas(16) static std::array<float, 401408 + XNN_EXTRA_BYTES / sizeof(float)> v1;
  alignas(16) static std::array<float, 401408 + XNN_EXTRA_BYTES / sizeof(float)> v2;
  alignas(16) static std::array<float, 200704 + XNN_EXTRA_BYTES / sizeof(float)> v3;
  alignas(16) static std::array<float, 1204224 + XNN_EXTRA_BYTES / sizeof(float)> v4;
  alignas(16) static std::array<float, 301056 + XNN_EXTRA_BYTES / sizeof(float)> v5;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v6;
  alignas(16) static std::array<float, 451584 + XNN_EXTRA_BYTES / sizeof(float)> v7;
  alignas(16) static std::array<float, 451584 + XNN_EXTRA_BYTES / sizeof(float)> v8;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v9;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v10;
  alignas(16) static std::array<float, 451584 + XNN_EXTRA_BYTES / sizeof(float)> v11;
  alignas(16) static std::array<float, 112896 + XNN_EXTRA_BYTES / sizeof(float)> v12;
  alignas(16) static std::array<float, 25088 + XNN_EXTRA_BYTES / sizeof(float)> v13;
  alignas(16) static std::array<float, 150528 + XNN_EXTRA_BYTES / sizeof(float)> v14;
  alignas(16) static std::array<float, 150528 + XNN_EXTRA_BYTES / sizeof(float)> v15;
  alignas(16) static std::array<float, 25088 + XNN_EXTRA_BYTES / sizeof(float)> v16;
  alignas(16) static std::array<float, 25088 + XNN_EXTRA_BYTES / sizeof(float)> v17;
  alignas(16) static std::array<float, 150528 + XNN_EXTRA_BYTES / sizeof(float)> v18;
  alignas(16) static std::array<float, 150528 + XNN_EXTRA_BYTES / sizeof(float)> v19;
  alignas(16) static std::array<float, 25088 + XNN_EXTRA_BYTES / sizeof(float)> v20;
  alignas(16) static std::array<float, 25088 + XNN_EXTRA_BYTES / sizeof(float)> v21;
  alignas(16) static std::array<float, 150528 + XNN_EXTRA_BYTES / sizeof(float)> v22;
  alignas(16) static std::array<float, 37632 + XNN_EXTRA_BYTES / sizeof(float)> v23;
  alignas(16) static std::array<float, 12544 + XNN_EXTRA_BYTES / sizeof(float)> v24;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v25;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v26;
  alignas(16) static std::array<float, 12544 + XNN_EXTRA_BYTES / sizeof(float)> v27;
  alignas(16) static std::array<float, 12544 + XNN_EXTRA_BYTES / sizeof(float)> v28;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v29;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v30;
  alignas(16) static std::array<float, 12544 + XNN_EXTRA_BYTES / sizeof(float)> v31;
  alignas(16) static std::array<float, 12544 + XNN_EXTRA_BYTES / sizeof(float)> v32;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v33;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v34;
  alignas(16) static std::array<float, 12544 + XNN_EXTRA_BYTES / sizeof(float)> v35;
  alignas(16) static std::array<float, 12544 + XNN_EXTRA_BYTES / sizeof(float)> v36;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v37;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v38;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v39;
  alignas(16) static std::array<float, 112896 + XNN_EXTRA_BYTES / sizeof(float)> v40;
  alignas(16) static std::array<float, 112896 + XNN_EXTRA_BYTES / sizeof(float)> v41;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v42;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v43;
  alignas(16) static std::array<float, 112896 + XNN_EXTRA_BYTES / sizeof(float)> v44;
  alignas(16) static std::array<float, 112896 + XNN_EXTRA_BYTES / sizeof(float)> v45;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v46;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v47;
  alignas(16) static std::array<float, 112896 + XNN_EXTRA_BYTES / sizeof(float)> v48;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v49;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v50;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v51;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v52;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v53;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v54;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v55;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v56;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v57;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v58;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v59;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v60;
  alignas(16) static std::array<float, 15680 + XNN_EXTRA_BYTES / sizeof(float)> v61;
  alignas(16) static std::array<float, 62720 + XNN_EXTRA_BYTES / sizeof(float)> v62;
  alignas(16) static std::array<float, 1280 + XNN_EXTRA_BYTES / sizeof(float)> v63;
  alignas(16) static std::array<float, 1001 + XNN_EXTRA_BYTES / sizeof(float)> v64;
  alignas(16) static std::array<float, 864 + XNN_EXTRA_BYTES / sizeof(float)> w65;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> w66;
  alignas(16) static std::array<float, 288 + XNN_EXTRA_BYTES / sizeof(float)> w67;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> w68;
  alignas(16) static std::array<float, 512 + XNN_EXTRA_BYTES / sizeof(float)> w69;
  alignas(16) static std::array<float, 16 + XNN_EXTRA_BYTES / sizeof(float)> w70;
  alignas(16) static std::array<float, 1536 + XNN_EXTRA_BYTES / sizeof(float)> w71;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w72;
  alignas(16) static std::array<float, 864 + XNN_EXTRA_BYTES / sizeof(float)> w73;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w74;
  alignas(16) static std::array<float, 2304 + XNN_EXTRA_BYTES / sizeof(float)> w75;
  alignas(16) static std::array<float, 24 + XNN_EXTRA_BYTES / sizeof(float)> w76;
  alignas(16) static std::array<float, 3456 + XNN_EXTRA_BYTES / sizeof(float)> w77;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w78;
  alignas(16) static std::array<float, 1296 + XNN_EXTRA_BYTES / sizeof(float)> w79;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w80;
  alignas(16) static std::array<float, 3456 + XNN_EXTRA_BYTES / sizeof(float)> w81;
  alignas(16) static std::array<float, 24 + XNN_EXTRA_BYTES / sizeof(float)> w82;
  alignas(16) static std::array<float, 3456 + XNN_EXTRA_BYTES / sizeof(float)> w83;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w84;
  alignas(16) static std::array<float, 1296 + XNN_EXTRA_BYTES / sizeof(float)> w85;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w86;
  alignas(16) static std::array<float, 4608 + XNN_EXTRA_BYTES / sizeof(float)> w87;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> w88;
  alignas(16) static std::array<float, 6144 + XNN_EXTRA_BYTES / sizeof(float)> w89;
  alignas(16) static std::array<float, 192 + XNN_EXTRA_BYTES / sizeof(float)> w90;
  alignas(16) static std::array<float, 1728 + XNN_EXTRA_BYTES / sizeof(float)> w91;
  alignas(16) static std::array<float, 192 + XNN_EXTRA_BYTES / sizeof(float)> w92;
  alignas(16) static std::array<float, 6144 + XNN_EXTRA_BYTES / sizeof(float)> w93;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> w94;
  alignas(16) static std::array<float, 6144 + XNN_EXTRA_BYTES / sizeof(float)> w95;
  alignas(16) static std::array<float, 192 + XNN_EXTRA_BYTES / sizeof(float)> w96;
  alignas(16) static std::array<float, 1728 + XNN_EXTRA_BYTES / sizeof(float)> w97;
  alignas(16) static std::array<float, 192 + XNN_EXTRA_BYTES / sizeof(float)> w98;
  alignas(16) static std::array<float, 6144 + XNN_EXTRA_BYTES / sizeof(float)> w99;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> w100;
  alignas(16) static std::array<float, 6144 + XNN_EXTRA_BYTES / sizeof(float)> w101;
  alignas(16) static std::array<float, 192 + XNN_EXTRA_BYTES / sizeof(float)> w102;
  alignas(16) static std::array<float, 1728 + XNN_EXTRA_BYTES / sizeof(float)> w103;
  alignas(16) static std::array<float, 192 + XNN_EXTRA_BYTES / sizeof(float)> w104;
  alignas(16) static std::array<float, 12288 + XNN_EXTRA_BYTES / sizeof(float)> w105;
  alignas(16) static std::array<float, 64 + XNN_EXTRA_BYTES / sizeof(float)> w106;
  alignas(16) static std::array<float, 24576 + XNN_EXTRA_BYTES / sizeof(float)> w107;
  alignas(16) static std::array<float, 384 + XNN_EXTRA_BYTES / sizeof(float)> w108;
  alignas(16) static std::array<float, 3456 + XNN_EXTRA_BYTES / sizeof(float)> w109;
  alignas(16) static std::array<float, 384 + XNN_EXTRA_BYTES / sizeof(float)> w110;
  alignas(16) static std::array<float, 24576 + XNN_EXTRA_BYTES / sizeof(float)> w111;
  alignas(16) static std::array<float, 64 + XNN_EXTRA_BYTES / sizeof(float)> w112;
  alignas(16) static std::array<float, 24576 + XNN_EXTRA_BYTES / sizeof(float)> w113;
  alignas(16) static std::array<float, 384 + XNN_EXTRA_BYTES / sizeof(float)> w114;
  alignas(16) static std::array<float, 3456 + XNN_EXTRA_BYTES / sizeof(float)> w115;
  alignas(16) static std::array<float, 384 + XNN_EXTRA_BYTES / sizeof(float)> w116;
  alignas(16) static std::array<float, 24576 + XNN_EXTRA_BYTES / sizeof(float)> w117;
  alignas(16) static std::array<float, 64 + XNN_EXTRA_BYTES / sizeof(float)> w118;
  alignas(16) static std::array<float, 24576 + XNN_EXTRA_BYTES / sizeof(float)> w119;
  alignas(16) static std::array<float, 384 + XNN_EXTRA_BYTES / sizeof(float)> w120;
  alignas(16) static std::array<float, 3456 + XNN_EXTRA_BYTES / sizeof(float)> w121;
  alignas(16) static std::array<float, 384 + XNN_EXTRA_BYTES / sizeof(float)> w122;
  alignas(16) static std::array<float, 24576 + XNN_EXTRA_BYTES / sizeof(float)> w123;
  alignas(16) static std::array<float, 64 + XNN_EXTRA_BYTES / sizeof(float)> w124;
  alignas(16) static std::array<float, 24576 + XNN_EXTRA_BYTES / sizeof(float)> w125;
  alignas(16) static std::array<float, 384 + XNN_EXTRA_BYTES / sizeof(float)> w126;
  alignas(16) static std::array<float, 3456 + XNN_EXTRA_BYTES / sizeof(float)> w127;
  alignas(16) static std::array<float, 384 + XNN_EXTRA_BYTES / sizeof(float)> w128;
  alignas(16) static std::array<float, 36864 + XNN_EXTRA_BYTES / sizeof(float)> w129;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w130;
  alignas(16) static std::array<float, 55296 + XNN_EXTRA_BYTES / sizeof(float)> w131;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w132;
  alignas(16) static std::array<float, 5184 + XNN_EXTRA_BYTES / sizeof(float)> w133;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w134;
  alignas(16) static std::array<float, 55296 + XNN_EXTRA_BYTES / sizeof(float)> w135;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w136;
  alignas(16) static std::array<float, 55296 + XNN_EXTRA_BYTES / sizeof(float)> w137;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w138;
  alignas(16) static std::array<float, 5184 + XNN_EXTRA_BYTES / sizeof(float)> w139;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w140;
  alignas(16) static std::array<float, 55296 + XNN_EXTRA_BYTES / sizeof(float)> w141;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w142;
  alignas(16) static std::array<float, 55296 + XNN_EXTRA_BYTES / sizeof(float)> w143;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w144;
  alignas(16) static std::array<float, 5184 + XNN_EXTRA_BYTES / sizeof(float)> w145;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w146;
  alignas(16) static std::array<float, 92160 + XNN_EXTRA_BYTES / sizeof(float)> w147;
  alignas(16) static std::array<float, 160 + XNN_EXTRA_BYTES / sizeof(float)> w148;
  alignas(16) static std::array<float, 153600 + XNN_EXTRA_BYTES / sizeof(float)> w149;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w150;
  alignas(16) static std::array<float, 8640 + XNN_EXTRA_BYTES / sizeof(float)> w151;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w152;
  alignas(16) static std::array<float, 153600 + XNN_EXTRA_BYTES / sizeof(float)> w153;
  alignas(16) static std::array<float, 160 + XNN_EXTRA_BYTES / sizeof(float)> w154;
  alignas(16) static std::array<float, 153600 + XNN_EXTRA_BYTES / sizeof(float)> w155;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w156;
  alignas(16) static std::array<float, 8640 + XNN_EXTRA_BYTES / sizeof(float)> w157;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w158;
  alignas(16) static std::array<float, 153600 + XNN_EXTRA_BYTES / sizeof(float)> w159;
  alignas(16) static std::array<float, 160 + XNN_EXTRA_BYTES / sizeof(float)> w160;
  alignas(16) static std::array<float, 153600 + XNN_EXTRA_BYTES / sizeof(float)> w161;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w162;
  alignas(16) static std::array<float, 8640 + XNN_EXTRA_BYTES / sizeof(float)> w163;
  alignas(16) static std::array<float, 960 + XNN_EXTRA_BYTES / sizeof(float)> w164;
  alignas(16) static std::array<float, 307200 + XNN_EXTRA_BYTES / sizeof(float)> w165;
  alignas(16) static std::array<float, 320 + XNN_EXTRA_BYTES / sizeof(float)> w166;
  alignas(16) static std::array<float, 409600 + XNN_EXTRA_BYTES / sizeof(float)> w167;
  alignas(16) static std::array<float, 1280 + XNN_EXTRA_BYTES / sizeof(float)> w168;
  alignas(16) static std::array<float, 1281280 + XNN_EXTRA_BYTES / sizeof(float)> w169;
  alignas(16) static std::array<float, 1001 + XNN_EXTRA_BYTES / sizeof(float)> w170;

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, +1.0f), std::ref(rng));
  std::generate(v0.begin(), v0.end(), std::ref(f32rng));
  std::generate(v1.begin(), v1.end(), std::ref(f32rng));
  std::generate(v2.begin(), v2.end(), std::ref(f32rng));
  std::generate(v3.begin(), v3.end(), std::ref(f32rng));
  std::generate(v4.begin(), v4.end(), std::ref(f32rng));
  std::generate(v5.begin(), v5.end(), std::ref(f32rng));
  std::generate(v6.begin(), v6.end(), std::ref(f32rng));
  std::generate(v7.begin(), v7.end(), std::ref(f32rng));
  std::generate(v8.begin(), v8.end(), std::ref(f32rng));
  std::generate(v9.begin(), v9.end(), std::ref(f32rng));
  std::generate(v10.begin(), v10.end(), std::ref(f32rng));
  std::generate(v11.begin(), v11.end(), std::ref(f32rng));
  std::generate(v12.begin(), v12.end(), std::ref(f32rng));
  std::generate(v13.begin(), v13.end(), std::ref(f32rng));
  std::generate(v14.begin(), v14.end(), std::ref(f32rng));
  std::generate(v15.begin(), v15.end(), std::ref(f32rng));
  std::generate(v16.begin(), v16.end(), std::ref(f32rng));
  std::generate(v17.begin(), v17.end(), std::ref(f32rng));
  std::generate(v18.begin(), v18.end(), std::ref(f32rng));
  std::generate(v19.begin(), v19.end(), std::ref(f32rng));
  std::generate(v20.begin(), v20.end(), std::ref(f32rng));
  std::generate(v21.begin(), v21.end(), std::ref(f32rng));
  std::generate(v22.begin(), v22.end(), std::ref(f32rng));
  std::generate(v23.begin(), v23.end(), std::ref(f32rng));
  std::generate(v24.begin(), v24.end(), std::ref(f32rng));
  std::generate(v25.begin(), v25.end(), std::ref(f32rng));
  std::generate(v26.begin(), v26.end(), std::ref(f32rng));
  std::generate(v27.begin(), v27.end(), std::ref(f32rng));
  std::generate(v28.begin(), v28.end(), std::ref(f32rng));
  std::generate(v29.begin(), v29.end(), std::ref(f32rng));
  std::generate(v30.begin(), v30.end(), std::ref(f32rng));
  std::generate(v31.begin(), v31.end(), std::ref(f32rng));
  std::generate(v32.begin(), v32.end(), std::ref(f32rng));
  std::generate(v33.begin(), v33.end(), std::ref(f32rng));
  std::generate(v34.begin(), v34.end(), std::ref(f32rng));
  std::generate(v35.begin(), v35.end(), std::ref(f32rng));
  std::generate(v36.begin(), v36.end(), std::ref(f32rng));
  std::generate(v37.begin(), v37.end(), std::ref(f32rng));
  std::generate(v38.begin(), v38.end(), std::ref(f32rng));
  std::generate(v39.begin(), v39.end(), std::ref(f32rng));
  std::generate(v40.begin(), v40.end(), std::ref(f32rng));
  std::generate(v41.begin(), v41.end(), std::ref(f32rng));
  std::generate(v42.begin(), v42.end(), std::ref(f32rng));
  std::generate(v43.begin(), v43.end(), std::ref(f32rng));
  std::generate(v44.begin(), v44.end(), std::ref(f32rng));
  std::generate(v45.begin(), v45.end(), std::ref(f32rng));
  std::generate(v46.begin(), v46.end(), std::ref(f32rng));
  std::generate(v47.begin(), v47.end(), std::ref(f32rng));
  std::generate(v48.begin(), v48.end(), std::ref(f32rng));
  std::generate(v49.begin(), v49.end(), std::ref(f32rng));
  std::generate(v50.begin(), v50.end(), std::ref(f32rng));
  std::generate(v51.begin(), v51.end(), std::ref(f32rng));
  std::generate(v52.begin(), v52.end(), std::ref(f32rng));
  std::generate(v53.begin(), v53.end(), std::ref(f32rng));
  std::generate(v54.begin(), v54.end(), std::ref(f32rng));
  std::generate(v55.begin(), v55.end(), std::ref(f32rng));
  std::generate(v56.begin(), v56.end(), std::ref(f32rng));
  std::generate(v57.begin(), v57.end(), std::ref(f32rng));
  std::generate(v58.begin(), v58.end(), std::ref(f32rng));
  std::generate(v59.begin(), v59.end(), std::ref(f32rng));
  std::generate(v60.begin(), v60.end(), std::ref(f32rng));
  std::generate(v61.begin(), v61.end(), std::ref(f32rng));
  std::generate(v62.begin(), v62.end(), std::ref(f32rng));
  std::generate(v63.begin(), v63.end(), std::ref(f32rng));
  std::generate(v64.begin(), v64.end(), std::ref(f32rng));
  std::generate(w65.begin(), w65.end(), std::ref(f32rng));
  std::generate(w66.begin(), w66.end(), std::ref(f32rng));
  std::generate(w67.begin(), w67.end(), std::ref(f32rng));
  std::generate(w68.begin(), w68.end(), std::ref(f32rng));
  std::fill(w69.begin(), w69.end(), 0.0f);
  std::generate(w69.begin(), w69.end() - size_t(sparsity * w69.size()), std::ref(f32rng));
  std::shuffle(w69.begin(), w69.end(), rng);
  std::generate(w70.begin(), w70.end(), std::ref(f32rng));
  std::fill(w71.begin(), w71.end(), 0.0f);
  std::generate(w71.begin(), w71.end() - size_t(sparsity * w71.size()), std::ref(f32rng));
  std::shuffle(w71.begin(), w71.end(), rng);
  std::generate(w72.begin(), w72.end(), std::ref(f32rng));
  std::generate(w73.begin(), w73.end(), std::ref(f32rng));
  std::generate(w74.begin(), w74.end(), std::ref(f32rng));
  std::fill(w75.begin(), w75.end(), 0.0f);
  std::generate(w75.begin(), w75.end() - size_t(sparsity * w75.size()), std::ref(f32rng));
  std::shuffle(w75.begin(), w75.end(), rng);
  std::generate(w76.begin(), w76.end(), std::ref(f32rng));
  std::fill(w77.begin(), w77.end(), 0.0f);
  std::generate(w77.begin(), w77.end() - size_t(sparsity * w77.size()), std::ref(f32rng));
  std::shuffle(w77.begin(), w77.end(), rng);
  std::generate(w78.begin(), w78.end(), std::ref(f32rng));
  std::generate(w79.begin(), w79.end(), std::ref(f32rng));
  std::generate(w80.begin(), w80.end(), std::ref(f32rng));
  std::fill(w81.begin(), w81.end(), 0.0f);
  std::generate(w81.begin(), w81.end() - size_t(sparsity * w81.size()), std::ref(f32rng));
  std::shuffle(w81.begin(), w81.end(), rng);
  std::generate(w82.begin(), w82.end(), std::ref(f32rng));
  std::fill(w83.begin(), w83.end(), 0.0f);
  std::generate(w83.begin(), w83.end() - size_t(sparsity * w83.size()), std::ref(f32rng));
  std::shuffle(w83.begin(), w83.end(), rng);
  std::generate(w84.begin(), w84.end(), std::ref(f32rng));
  std::generate(w85.begin(), w85.end(), std::ref(f32rng));
  std::generate(w86.begin(), w86.end(), std::ref(f32rng));
  std::fill(w87.begin(), w87.end(), 0.0f);
  std::generate(w87.begin(), w87.end() - size_t(sparsity * w87.size()), std::ref(f32rng));
  std::shuffle(w87.begin(), w87.end(), rng);
  std::generate(w88.begin(), w88.end(), std::ref(f32rng));
  std::fill(w89.begin(), w89.end(), 0.0f);
  std::generate(w89.begin(), w89.end() - size_t(sparsity * w89.size()), std::ref(f32rng));
  std::shuffle(w89.begin(), w89.end(), rng);
  std::generate(w90.begin(), w90.end(), std::ref(f32rng));
  std::generate(w91.begin(), w91.end(), std::ref(f32rng));
  std::generate(w92.begin(), w92.end(), std::ref(f32rng));
  std::fill(w93.begin(), w93.end(), 0.0f);
  std::generate(w93.begin(), w93.end() - size_t(sparsity * w93.size()), std::ref(f32rng));
  std::shuffle(w93.begin(), w93.end(), rng);
  std::generate(w94.begin(), w94.end(), std::ref(f32rng));
  std::fill(w95.begin(), w95.end(), 0.0f);
  std::generate(w95.begin(), w95.end() - size_t(sparsity * w95.size()), std::ref(f32rng));
  std::shuffle(w95.begin(), w95.end(), rng);
  std::generate(w96.begin(), w96.end(), std::ref(f32rng));
  std::generate(w97.begin(), w97.end(), std::ref(f32rng));
  std::generate(w98.begin(), w98.end(), std::ref(f32rng));
  std::fill(w99.begin(), w99.end(), 0.0f);
  std::generate(w99.begin(), w99.end() - size_t(sparsity * w99.size()), std::ref(f32rng));
  std::shuffle(w99.begin(), w99.end(), rng);
  std::generate(w100.begin(), w100.end(), std::ref(f32rng));
  std::fill(w101.begin(), w101.end(), 0.0f);
  std::generate(w101.begin(), w101.end() - size_t(sparsity * w101.size()), std::ref(f32rng));
  std::shuffle(w101.begin(), w101.end(), rng);
  std::generate(w102.begin(), w102.end(), std::ref(f32rng));
  std::generate(w103.begin(), w103.end(), std::ref(f32rng));
  std::generate(w104.begin(), w104.end(), std::ref(f32rng));
  std::fill(w105.begin(), w105.end(), 0.0f);
  std::generate(w105.begin(), w105.end() - size_t(sparsity * w105.size()), std::ref(f32rng));
  std::shuffle(w105.begin(), w105.end(), rng);
  std::generate(w106.begin(), w106.end(), std::ref(f32rng));
  std::fill(w107.begin(), w107.end(), 0.0f);
  std::generate(w107.begin(), w107.end() - size_t(sparsity * w107.size()), std::ref(f32rng));
  std::shuffle(w107.begin(), w107.end(), rng);
  std::generate(w108.begin(), w108.end(), std::ref(f32rng));
  std::generate(w109.begin(), w109.end(), std::ref(f32rng));
  std::generate(w110.begin(), w110.end(), std::ref(f32rng));
  std::fill(w111.begin(), w111.end(), 0.0f);
  std::generate(w111.begin(), w111.end() - size_t(sparsity * w111.size()), std::ref(f32rng));
  std::shuffle(w111.begin(), w111.end(), rng);
  std::generate(w112.begin(), w112.end(), std::ref(f32rng));
  std::fill(w113.begin(), w113.end(), 0.0f);
  std::generate(w113.begin(), w113.end() - size_t(sparsity * w113.size()), std::ref(f32rng));
  std::shuffle(w113.begin(), w113.end(), rng);
  std::generate(w114.begin(), w114.end(), std::ref(f32rng));
  std::generate(w115.begin(), w115.end(), std::ref(f32rng));
  std::generate(w116.begin(), w116.end(), std::ref(f32rng));
  std::fill(w117.begin(), w117.end(), 0.0f);
  std::generate(w117.begin(), w117.end() - size_t(sparsity * w117.size()), std::ref(f32rng));
  std::shuffle(w117.begin(), w117.end(), rng);
  std::generate(w118.begin(), w118.end(), std::ref(f32rng));
  std::fill(w119.begin(), w119.end(), 0.0f);
  std::generate(w119.begin(), w119.end() - size_t(sparsity * w119.size()), std::ref(f32rng));
  std::shuffle(w119.begin(), w119.end(), rng);
  std::generate(w120.begin(), w120.end(), std::ref(f32rng));
  std::generate(w121.begin(), w121.end(), std::ref(f32rng));
  std::generate(w122.begin(), w122.end(), std::ref(f32rng));
  std::fill(w123.begin(), w123.end(), 0.0f);
  std::generate(w123.begin(), w123.end() - size_t(sparsity * w123.size()), std::ref(f32rng));
  std::shuffle(w123.begin(), w123.end(), rng);
  std::generate(w124.begin(), w124.end(), std::ref(f32rng));
  std::fill(w125.begin(), w125.end(), 0.0f);
  std::generate(w125.begin(), w125.end() - size_t(sparsity * w125.size()), std::ref(f32rng));
  std::shuffle(w125.begin(), w125.end(), rng);
  std::generate(w126.begin(), w126.end(), std::ref(f32rng));
  std::generate(w127.begin(), w127.end(), std::ref(f32rng));
  std::generate(w128.begin(), w128.end(), std::ref(f32rng));
  std::fill(w129.begin(), w129.end(), 0.0f);
  std::generate(w129.begin(), w129.end() - size_t(sparsity * w129.size()), std::ref(f32rng));
  std::shuffle(w129.begin(), w129.end(), rng);
  std::generate(w130.begin(), w130.end(), std::ref(f32rng));
  std::fill(w131.begin(), w131.end(), 0.0f);
  std::generate(w131.begin(), w131.end() - size_t(sparsity * w131.size()), std::ref(f32rng));
  std::shuffle(w131.begin(), w131.end(), rng);
  std::generate(w132.begin(), w132.end(), std::ref(f32rng));
  std::generate(w133.begin(), w133.end(), std::ref(f32rng));
  std::generate(w134.begin(), w134.end(), std::ref(f32rng));
  std::fill(w135.begin(), w135.end(), 0.0f);
  std::generate(w135.begin(), w135.end() - size_t(sparsity * w135.size()), std::ref(f32rng));
  std::shuffle(w135.begin(), w135.end(), rng);
  std::generate(w136.begin(), w136.end(), std::ref(f32rng));
  std::fill(w137.begin(), w137.end(), 0.0f);
  std::generate(w137.begin(), w137.end() - size_t(sparsity * w137.size()), std::ref(f32rng));
  std::shuffle(w137.begin(), w137.end(), rng);
  std::generate(w138.begin(), w138.end(), std::ref(f32rng));
  std::generate(w139.begin(), w139.end(), std::ref(f32rng));
  std::generate(w140.begin(), w140.end(), std::ref(f32rng));
  std::fill(w141.begin(), w141.end(), 0.0f);
  std::generate(w141.begin(), w141.end() - size_t(sparsity * w141.size()), std::ref(f32rng));
  std::shuffle(w141.begin(), w141.end(), rng);
  std::generate(w142.begin(), w142.end(), std::ref(f32rng));
  std::fill(w143.begin(), w143.end(), 0.0f);
  std::generate(w143.begin(), w143.end() - size_t(sparsity * w143.size()), std::ref(f32rng));
  std::shuffle(w143.begin(), w143.end(), rng);
  std::generate(w144.begin(), w144.end(), std::ref(f32rng));
  std::generate(w145.begin(), w145.end(), std::ref(f32rng));
  std::generate(w146.begin(), w146.end(), std::ref(f32rng));
  std::fill(w147.begin(), w147.end(), 0.0f);
  std::generate(w147.begin(), w147.end() - size_t(sparsity * w147.size()), std::ref(f32rng));
  std::shuffle(w147.begin(), w147.end(), rng);
  std::generate(w148.begin(), w148.end(), std::ref(f32rng));
  std::fill(w149.begin(), w149.end(), 0.0f);
  std::generate(w149.begin(), w149.end() - size_t(sparsity * w149.size()), std::ref(f32rng));
  std::shuffle(w149.begin(), w149.end(), rng);
  std::generate(w150.begin(), w150.end(), std::ref(f32rng));
  std::generate(w151.begin(), w151.end(), std::ref(f32rng));
  std::generate(w152.begin(), w152.end(), std::ref(f32rng));
  std::fill(w153.begin(), w153.end(), 0.0f);
  std::generate(w153.begin(), w153.end() - size_t(sparsity * w153.size()), std::ref(f32rng));
  std::shuffle(w153.begin(), w153.end(), rng);
  std::generate(w154.begin(), w154.end(), std::ref(f32rng));
  std::fill(w155.begin(), w155.end(), 0.0f);
  std::generate(w155.begin(), w155.end() - size_t(sparsity * w155.size()), std::ref(f32rng));
  std::shuffle(w155.begin(), w155.end(), rng);
  std::generate(w156.begin(), w156.end(), std::ref(f32rng));
  std::generate(w157.begin(), w157.end(), std::ref(f32rng));
  std::generate(w158.begin(), w158.end(), std::ref(f32rng));
  std::fill(w159.begin(), w159.end(), 0.0f);
  std::generate(w159.begin(), w159.end() - size_t(sparsity * w159.size()), std::ref(f32rng));
  std::shuffle(w159.begin(), w159.end(), rng);
  std::generate(w160.begin(), w160.end(), std::ref(f32rng));
  std::fill(w161.begin(), w161.end(), 0.0f);
  std::generate(w161.begin(), w161.end() - size_t(sparsity * w161.size()), std::ref(f32rng));
  std::shuffle(w161.begin(), w161.end(), rng);
  std::generate(w162.begin(), w162.end(), std::ref(f32rng));
  std::generate(w163.begin(), w163.end(), std::ref(f32rng));
  std::generate(w164.begin(), w164.end(), std::ref(f32rng));
  std::fill(w165.begin(), w165.end(), 0.0f);
  std::generate(w165.begin(), w165.end() - size_t(sparsity * w165.size()), std::ref(f32rng));
  std::shuffle(w165.begin(), w165.end(), rng);
  std::generate(w166.begin(), w166.end(), std::ref(f32rng));
  std::fill(w167.begin(), w167.end(), 0.0f);
  std::generate(w167.begin(), w167.end() - size_t(sparsity * w167.size()), std::ref(f32rng));
  std::shuffle(w167.begin(), w167.end(), rng);
  std::generate(w168.begin(), w168.end(), std::ref(f32rng));
  std::fill(w169.begin(), w169.end(), 0.0f);
  std::generate(w169.begin(), w169.end() - size_t(sparsity * w169.size()), std::ref(f32rng));
  std::shuffle(w169.begin(), w169.end(), rng);
  std::generate(w170.begin(), w170.end(), std::ref(f32rng));

  size_t max_workspace_size = 0;
  Operators operators;
  xnn_status status;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
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
    XNN_FLAG_INPUT_NHWC /* flags */,
    nullptr,
    nullptr,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
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
    nullptr,
    nullptr,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
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
    nullptr,
    nullptr,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
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
    nullptr,
    nullptr,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  xnn_operator_t op29 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #29" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op29, xnn_delete_operator);

  xnn_operator_t op30 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #30" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op30, xnn_delete_operator);

  xnn_operator_t op31 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #31" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op31, xnn_delete_operator);

  xnn_operator_t op32 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #32" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op32, xnn_delete_operator);

  xnn_operator_t op33 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #33" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op33, xnn_delete_operator);

  xnn_operator_t op34 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #34" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op34, xnn_delete_operator);

  xnn_operator_t op35 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #35" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op35, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #36" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op36, xnn_delete_operator);

  xnn_operator_t op37 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #37" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op37, xnn_delete_operator);

  xnn_operator_t op38 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #38" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op38, xnn_delete_operator);

  xnn_operator_t op39 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #39" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op39, xnn_delete_operator);

  xnn_operator_t op40 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #40" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op40, xnn_delete_operator);

  xnn_operator_t op41 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #41" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op41, xnn_delete_operator);

  xnn_operator_t op42 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #42" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op42, xnn_delete_operator);

  xnn_operator_t op43 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #43" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op43, xnn_delete_operator);

  xnn_operator_t op44 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op45 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #45" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op45, xnn_delete_operator);

  xnn_operator_t op46 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #46" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op46, xnn_delete_operator);

  xnn_operator_t op47 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #47" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op47, xnn_delete_operator);

  xnn_operator_t op48 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
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
    nullptr,
    nullptr,
    &op48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #48" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op48, xnn_delete_operator);

  xnn_operator_t op49 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #49" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op49, xnn_delete_operator);

  xnn_operator_t op50 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #50" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op50, xnn_delete_operator);

  xnn_operator_t op51 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #51" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op51, xnn_delete_operator);

  xnn_operator_t op52 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #52" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op52, xnn_delete_operator);

  xnn_operator_t op53 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op54 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #54" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op54, xnn_delete_operator);

  xnn_operator_t op55 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #55" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op55, xnn_delete_operator);

  xnn_operator_t op56 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #56" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op56, xnn_delete_operator);

  xnn_operator_t op57 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #57" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op57, xnn_delete_operator);

  xnn_operator_t op58 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #58" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op58, xnn_delete_operator);

  xnn_operator_t op59 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #59" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op59, xnn_delete_operator);

  xnn_operator_t op60 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #60" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op60, xnn_delete_operator);

  xnn_operator_t op61 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    nullptr,
    nullptr,
    &op61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #61" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op61, xnn_delete_operator);

  xnn_operator_t op62 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op62);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #62" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op62, xnn_delete_operator);

  xnn_operator_t op63 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
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
    nullptr,
    nullptr,
    &op63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #63" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op63, xnn_delete_operator);

  status = xnn_reshape_convolution2d_nchw_f32(
    op0,
    /*batch_size=*/1, /*input_height=*/224, /*input_width=*/224,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op1,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op2,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op3,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op4,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op5,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op6,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op7,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op8,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #8" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 24, 56, 56 };
    const size_t b_shape[] = { 1, 24, 56, 56 };
    status = xnn_reshape_add_nd_f32(
      op9,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op10,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op11,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op12,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op13,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op14,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op15,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #15" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 32, 28, 28 };
    const size_t b_shape[] = { 1, 32, 28, 28 };
    status = xnn_reshape_add_nd_f32(
      op16,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op17,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op18,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op19,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #19" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 32, 28, 28 };
    const size_t b_shape[] = { 1, 32, 28, 28 };
    status = xnn_reshape_add_nd_f32(
      op20,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op21,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op22,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op23,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op24,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op25,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op26,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #26" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 64, 14, 14 };
    const size_t b_shape[] = { 1, 64, 14, 14 };
    status = xnn_reshape_add_nd_f32(
      op27,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op28,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op29,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op30,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #30" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 64, 14, 14 };
    const size_t b_shape[] = { 1, 64, 14, 14 };
    status = xnn_reshape_add_nd_f32(
      op31,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op32,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op33,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op34,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #34" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 64, 14, 14 };
    const size_t b_shape[] = { 1, 64, 14, 14 };
    status = xnn_reshape_add_nd_f32(
      op35,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op36,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op37,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op38,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op39,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op40,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op41,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #41" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 96, 14, 14 };
    const size_t b_shape[] = { 1, 96, 14, 14 };
    status = xnn_reshape_add_nd_f32(
      op42,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op43,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op44,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op45,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #45" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 96, 14, 14 };
    const size_t b_shape[] = { 1, 96, 14, 14 };
    status = xnn_reshape_add_nd_f32(
      op46,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op47,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op48,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op49,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op50,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #50" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op51,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op52,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #52" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 160, 7, 7 };
    const size_t b_shape[] = { 1, 160, 7, 7 };
    status = xnn_reshape_add_nd_f32(
      op53,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op54,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op55,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op56,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #56" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 160, 7, 7 };
    const size_t b_shape[] = { 1, 160, 7, 7 };
    status = xnn_reshape_add_nd_f32(
      op57,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op58,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op59,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op60,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nchw_f32(
    op61,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_global_average_pooling_ncw_f32(
    op62,
    /*batch_size=*/1, 49 /* width */,
    1280 /* channels */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #62" << std::endl;
    return ExecutionPlan();
  }

  size_t op63_workspace_size = 0;
  size_t op63_workspace_alignment = 0;
  status = xnn_reshape_convolution2d_nhwc_f32(
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

  Workspace workspace(max_workspace_size);

  status = xnn_setup_convolution2d_nchw_f32(
    op0,
    /*input=*/v0.data(), /*output=*/v1.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op1,
    /*input=*/v1.data(), /*output=*/v2.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op2,
    /*input=*/v2.data(), /*output=*/v3.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op3,
    /*input=*/v3.data(), /*output=*/v4.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op4,
    /*input=*/v4.data(), /*output=*/v5.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op5,
    /*input=*/v5.data(), /*output=*/v6.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op6,
    /*input=*/v6.data(), /*output=*/v7.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op7,
    /*input=*/v7.data(), /*output=*/v8.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op8,
    /*input=*/v8.data(), /*output=*/v9.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op9,
    v9.data() /* a */, v6.data() /* b */, /*output=*/v10.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op10,
    /*input=*/v10.data(), /*output=*/v11.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op11,
    /*input=*/v11.data(), /*output=*/v12.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op12,
    /*input=*/v12.data(), /*output=*/v13.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op13,
    /*input=*/v13.data(), /*output=*/v14.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op14,
    /*input=*/v14.data(), /*output=*/v15.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op15,
    /*input=*/v15.data(), /*output=*/v16.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op16,
    v16.data() /* a */, v13.data() /* b */, /*output=*/v17.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op17,
    /*input=*/v17.data(), /*output=*/v18.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op18,
    /*input=*/v18.data(), /*output=*/v19.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op19,
    /*input=*/v19.data(), /*output=*/v20.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op20,
    v20.data() /* a */, v17.data() /* b */, /*output=*/v21.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op21,
    /*input=*/v21.data(), /*output=*/v22.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op22,
    /*input=*/v22.data(), /*output=*/v23.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op23,
    /*input=*/v23.data(), /*output=*/v24.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op24,
    /*input=*/v24.data(), /*output=*/v25.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op25,
    /*input=*/v25.data(), /*output=*/v26.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op26,
    /*input=*/v26.data(), /*output=*/v27.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op27,
    v27.data() /* a */, v24.data() /* b */, /*output=*/v28.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op28,
    /*input=*/v28.data(), /*output=*/v29.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op29,
    /*input=*/v29.data(), /*output=*/v30.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op30,
    /*input=*/v30.data(), /*output=*/v31.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op31,
    v31.data() /* a */, v28.data() /* b */, /*output=*/v32.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op32,
    /*input=*/v32.data(), /*output=*/v33.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op33,
    /*input=*/v33.data(), /*output=*/v34.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op34,
    /*input=*/v34.data(), /*output=*/v35.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op35,
    v35.data() /* a */, v32.data() /* b */, /*output=*/v36.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op36,
    /*input=*/v36.data(), /*output=*/v37.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op37,
    /*input=*/v37.data(), /*output=*/v38.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op38,
    /*input=*/v38.data(), /*output=*/v39.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op39,
    /*input=*/v39.data(), /*output=*/v40.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op40,
    /*input=*/v40.data(), /*output=*/v41.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op41,
    /*input=*/v41.data(), /*output=*/v42.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op42,
    v42.data() /* a */, v39.data() /* b */, /*output=*/v43.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op43,
    /*input=*/v43.data(), /*output=*/v44.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op44,
    /*input=*/v44.data(), /*output=*/v45.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op45,
    /*input=*/v45.data(), /*output=*/v46.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op46,
    v46.data() /* a */, v43.data() /* b */, /*output=*/v47.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op47,
    /*input=*/v47.data(), /*output=*/v48.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op48,
    /*input=*/v48.data(), /*output=*/v49.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op49,
    /*input=*/v49.data(), /*output=*/v50.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op50,
    /*input=*/v50.data(), /*output=*/v51.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op51,
    /*input=*/v51.data(), /*output=*/v52.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op52,
    /*input=*/v52.data(), /*output=*/v53.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op53,
    v53.data() /* a */, v50.data() /* b */, /*output=*/v54.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op54,
    /*input=*/v54.data(), /*output=*/v55.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op55,
    /*input=*/v55.data(), /*output=*/v56.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op56,
    /*input=*/v56.data(), /*output=*/v57.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op57,
    v57.data() /* a */, v54.data() /* b */, /*output=*/v58.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op58,
    /*input=*/v58.data(), /*output=*/v59.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op59,
    /*input=*/v59.data(), /*output=*/v60.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op60,
    /*input=*/v60.data(), /*output=*/v61.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op61,
    /*input=*/v61.data(), /*output=*/v62.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op62,
    /*input=*/v62.data(), /*output=*/v63.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op63,
    workspace.data(),
    /*input=*/v63.data(), /*output=*/v64.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #63" << std::endl;
    return ExecutionPlan();
  }

  XNN_PRAGMA_CLANG("clang diagnostic push")
  XNN_PRAGMA_CLANG("clang diagnostic ignored \"-Wpessimizing-move\"")
  return ExecutionPlan{operators, workspace};
  XNN_PRAGMA_CLANG("clang diagnostic pop")
}

}  // namespace models
