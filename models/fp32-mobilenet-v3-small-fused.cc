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
#include <xnnpack/post-operation.h>

namespace models {

ExecutionPlan FP32MobileNetV3SmallFused(pthreadpool_t threadpool) {
  alignas(16) static std::array<float, 150528 + XNN_EXTRA_BYTES / sizeof(float)> v0;
  alignas(16) static std::array<float, 200704 + XNN_EXTRA_BYTES / sizeof(float)> v1;
  alignas(16) static std::array<float, 200704 + XNN_EXTRA_BYTES / sizeof(float)> v2;
  alignas(16) static std::array<float, 50176 + XNN_EXTRA_BYTES / sizeof(float)> v3;
  alignas(16) static std::array<float, 16 + XNN_EXTRA_BYTES / sizeof(float)> v4;
  alignas(16) static std::array<float, 8 + XNN_EXTRA_BYTES / sizeof(float)> v5;
  alignas(16) static std::array<float, 16 + XNN_EXTRA_BYTES / sizeof(float)> v6;
  alignas(16) static std::array<float, 50176 + XNN_EXTRA_BYTES / sizeof(float)> v7;
  alignas(16) static std::array<float, 50176 + XNN_EXTRA_BYTES / sizeof(float)> v8;
  alignas(16) static std::array<float, 225792 + XNN_EXTRA_BYTES / sizeof(float)> v9;
  alignas(16) static std::array<float, 56448 + XNN_EXTRA_BYTES / sizeof(float)> v10;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v11;
  alignas(16) static std::array<float, 68992 + XNN_EXTRA_BYTES / sizeof(float)> v12;
  alignas(16) static std::array<float, 68992 + XNN_EXTRA_BYTES / sizeof(float)> v13;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v14;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v15;
  alignas(16) static std::array<float, 75264 + XNN_EXTRA_BYTES / sizeof(float)> v17;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v18;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v19;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> v20;
  alignas(16) static std::array<float, 24 + XNN_EXTRA_BYTES / sizeof(float)> v21;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> v22;
  alignas(16) static std::array<float, 18816 + XNN_EXTRA_BYTES / sizeof(float)> v23;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v24;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v26;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v27;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v28;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> v29;
  alignas(16) static std::array<float, 64 + XNN_EXTRA_BYTES / sizeof(float)> v30;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> v31;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v32;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v33;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v34;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v36;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v37;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v38;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> v39;
  alignas(16) static std::array<float, 64 + XNN_EXTRA_BYTES / sizeof(float)> v40;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> v41;
  alignas(16) static std::array<float, 47040 + XNN_EXTRA_BYTES / sizeof(float)> v42;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v43;
  alignas(16) static std::array<float, 7840 + XNN_EXTRA_BYTES / sizeof(float)> v44;
  alignas(16) static std::array<float, 23520 + XNN_EXTRA_BYTES / sizeof(float)> v46;
  alignas(16) static std::array<float, 23520 + XNN_EXTRA_BYTES / sizeof(float)> v47;
  alignas(16) static std::array<float, 23520 + XNN_EXTRA_BYTES / sizeof(float)> v48;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> v49;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> v50;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> v51;
  alignas(16) static std::array<float, 23520 + XNN_EXTRA_BYTES / sizeof(float)> v52;
  alignas(16) static std::array<float, 9408 + XNN_EXTRA_BYTES / sizeof(float)> v53;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v55;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v56;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v57;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> v58;
  alignas(16) static std::array<float, 40 + XNN_EXTRA_BYTES / sizeof(float)> v59;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> v60;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v61;
  alignas(16) static std::array<float, 9408 + XNN_EXTRA_BYTES / sizeof(float)> v62;
  alignas(16) static std::array<float, 9408 + XNN_EXTRA_BYTES / sizeof(float)> v63;
  alignas(16) static std::array<float, 56448 + XNN_EXTRA_BYTES / sizeof(float)> v65;
  alignas(16) static std::array<float, 14112 + XNN_EXTRA_BYTES / sizeof(float)> v66;
  alignas(16) static std::array<float, 14112 + XNN_EXTRA_BYTES / sizeof(float)> v67;
  alignas(16) static std::array<float, 288 + XNN_EXTRA_BYTES / sizeof(float)> v68;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> v69;
  alignas(16) static std::array<float, 288 + XNN_EXTRA_BYTES / sizeof(float)> v70;
  alignas(16) static std::array<float, 14112 + XNN_EXTRA_BYTES / sizeof(float)> v71;
  alignas(16) static std::array<float, 4704 + XNN_EXTRA_BYTES / sizeof(float)> v72;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v74;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v75;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v76;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> v77;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> v78;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> v79;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v80;
  alignas(16) static std::array<float, 4704 + XNN_EXTRA_BYTES / sizeof(float)> v81;
  alignas(16) static std::array<float, 4704 + XNN_EXTRA_BYTES / sizeof(float)> v82;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v84;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v85;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v86;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> v87;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> v88;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> v89;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v90;
  alignas(16) static std::array<float, 4704 + XNN_EXTRA_BYTES / sizeof(float)> v91;
  alignas(16) static std::array<float, 4704 + XNN_EXTRA_BYTES / sizeof(float)> v92;
  alignas(16) static std::array<float, 28224 + XNN_EXTRA_BYTES / sizeof(float)> v94;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> v95;
  alignas(16) static std::array<float, 1024 + XNN_EXTRA_BYTES / sizeof(float)> v97;
  alignas(16) static std::array<float, 1024 + XNN_EXTRA_BYTES / sizeof(float)> v98;
  alignas(16) static std::array<float, 1001 + XNN_EXTRA_BYTES / sizeof(float)> v99;
  alignas(16) static std::array<float, 432 + XNN_EXTRA_BYTES / sizeof(float)> w100;
  alignas(16) static std::array<float, 16 + XNN_EXTRA_BYTES / sizeof(float)> w101;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w102;
  alignas(16) static std::array<float, 16 + XNN_EXTRA_BYTES / sizeof(float)> w103;
  alignas(16) static std::array<float, 128 + XNN_EXTRA_BYTES / sizeof(float)> w104;
  alignas(16) static std::array<float, 8 + XNN_EXTRA_BYTES / sizeof(float)> w105;
  alignas(16) static std::array<float, 128 + XNN_EXTRA_BYTES / sizeof(float)> w106;
  alignas(16) static std::array<float, 16 + XNN_EXTRA_BYTES / sizeof(float)> w107;
  alignas(16) static std::array<float, 256 + XNN_EXTRA_BYTES / sizeof(float)> w108;
  alignas(16) static std::array<float, 16 + XNN_EXTRA_BYTES / sizeof(float)> w109;
  alignas(16) static std::array<float, 1152 + XNN_EXTRA_BYTES / sizeof(float)> w110;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> w111;
  alignas(16) static std::array<float, 648 + XNN_EXTRA_BYTES / sizeof(float)> w112;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> w113;
  alignas(16) static std::array<float, 1728 + XNN_EXTRA_BYTES / sizeof(float)> w114;
  alignas(16) static std::array<float, 24 + XNN_EXTRA_BYTES / sizeof(float)> w115;
  alignas(16) static std::array<float, 2112 + XNN_EXTRA_BYTES / sizeof(float)> w116;
  alignas(16) static std::array<float, 88 + XNN_EXTRA_BYTES / sizeof(float)> w117;
  alignas(16) static std::array<float, 792 + XNN_EXTRA_BYTES / sizeof(float)> w118;
  alignas(16) static std::array<float, 88 + XNN_EXTRA_BYTES / sizeof(float)> w119;
  alignas(16) static std::array<float, 2112 + XNN_EXTRA_BYTES / sizeof(float)> w120;
  alignas(16) static std::array<float, 24 + XNN_EXTRA_BYTES / sizeof(float)> w121;
  alignas(16) static std::array<float, 2304 + XNN_EXTRA_BYTES / sizeof(float)> w122;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w123;
  alignas(16) static std::array<float, 2400 + XNN_EXTRA_BYTES / sizeof(float)> w124;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w125;
  alignas(16) static std::array<float, 2304 + XNN_EXTRA_BYTES / sizeof(float)> w126;
  alignas(16) static std::array<float, 24 + XNN_EXTRA_BYTES / sizeof(float)> w127;
  alignas(16) static std::array<float, 2304 + XNN_EXTRA_BYTES / sizeof(float)> w128;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w129;
  alignas(16) static std::array<float, 3840 + XNN_EXTRA_BYTES / sizeof(float)> w130;
  alignas(16) static std::array<float, 40 + XNN_EXTRA_BYTES / sizeof(float)> w131;
  alignas(16) static std::array<float, 9600 + XNN_EXTRA_BYTES / sizeof(float)> w132;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> w133;
  alignas(16) static std::array<float, 6000 + XNN_EXTRA_BYTES / sizeof(float)> w134;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> w135;
  alignas(16) static std::array<float, 15360 + XNN_EXTRA_BYTES / sizeof(float)> w136;
  alignas(16) static std::array<float, 64 + XNN_EXTRA_BYTES / sizeof(float)> w137;
  alignas(16) static std::array<float, 15360 + XNN_EXTRA_BYTES / sizeof(float)> w138;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> w139;
  alignas(16) static std::array<float, 9600 + XNN_EXTRA_BYTES / sizeof(float)> w140;
  alignas(16) static std::array<float, 40 + XNN_EXTRA_BYTES / sizeof(float)> w141;
  alignas(16) static std::array<float, 9600 + XNN_EXTRA_BYTES / sizeof(float)> w142;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> w143;
  alignas(16) static std::array<float, 6000 + XNN_EXTRA_BYTES / sizeof(float)> w144;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> w145;
  alignas(16) static std::array<float, 15360 + XNN_EXTRA_BYTES / sizeof(float)> w146;
  alignas(16) static std::array<float, 64 + XNN_EXTRA_BYTES / sizeof(float)> w147;
  alignas(16) static std::array<float, 15360 + XNN_EXTRA_BYTES / sizeof(float)> w148;
  alignas(16) static std::array<float, 240 + XNN_EXTRA_BYTES / sizeof(float)> w149;
  alignas(16) static std::array<float, 9600 + XNN_EXTRA_BYTES / sizeof(float)> w150;
  alignas(16) static std::array<float, 40 + XNN_EXTRA_BYTES / sizeof(float)> w151;
  alignas(16) static std::array<float, 4800 + XNN_EXTRA_BYTES / sizeof(float)> w152;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> w153;
  alignas(16) static std::array<float, 3000 + XNN_EXTRA_BYTES / sizeof(float)> w154;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> w155;
  alignas(16) static std::array<float, 3840 + XNN_EXTRA_BYTES / sizeof(float)> w156;
  alignas(16) static std::array<float, 32 + XNN_EXTRA_BYTES / sizeof(float)> w157;
  alignas(16) static std::array<float, 3840 + XNN_EXTRA_BYTES / sizeof(float)> w158;
  alignas(16) static std::array<float, 120 + XNN_EXTRA_BYTES / sizeof(float)> w159;
  alignas(16) static std::array<float, 5760 + XNN_EXTRA_BYTES / sizeof(float)> w160;
  alignas(16) static std::array<float, 48 + XNN_EXTRA_BYTES / sizeof(float)> w161;
  alignas(16) static std::array<float, 6912 + XNN_EXTRA_BYTES / sizeof(float)> w162;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w163;
  alignas(16) static std::array<float, 3600 + XNN_EXTRA_BYTES / sizeof(float)> w164;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w165;
  alignas(16) static std::array<float, 5760 + XNN_EXTRA_BYTES / sizeof(float)> w166;
  alignas(16) static std::array<float, 40 + XNN_EXTRA_BYTES / sizeof(float)> w167;
  alignas(16) static std::array<float, 5760 + XNN_EXTRA_BYTES / sizeof(float)> w168;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w169;
  alignas(16) static std::array<float, 6912 + XNN_EXTRA_BYTES / sizeof(float)> w170;
  alignas(16) static std::array<float, 48 + XNN_EXTRA_BYTES / sizeof(float)> w171;
  alignas(16) static std::array<float, 13824 + XNN_EXTRA_BYTES / sizeof(float)> w172;
  alignas(16) static std::array<float, 288 + XNN_EXTRA_BYTES / sizeof(float)> w173;
  alignas(16) static std::array<float, 7200 + XNN_EXTRA_BYTES / sizeof(float)> w174;
  alignas(16) static std::array<float, 288 + XNN_EXTRA_BYTES / sizeof(float)> w175;
  alignas(16) static std::array<float, 20736 + XNN_EXTRA_BYTES / sizeof(float)> w176;
  alignas(16) static std::array<float, 72 + XNN_EXTRA_BYTES / sizeof(float)> w177;
  alignas(16) static std::array<float, 20736 + XNN_EXTRA_BYTES / sizeof(float)> w178;
  alignas(16) static std::array<float, 288 + XNN_EXTRA_BYTES / sizeof(float)> w179;
  alignas(16) static std::array<float, 27648 + XNN_EXTRA_BYTES / sizeof(float)> w180;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w181;
  alignas(16) static std::array<float, 55296 + XNN_EXTRA_BYTES / sizeof(float)> w182;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w183;
  alignas(16) static std::array<float, 14400 + XNN_EXTRA_BYTES / sizeof(float)> w184;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w185;
  alignas(16) static std::array<float, 82944 + XNN_EXTRA_BYTES / sizeof(float)> w186;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w187;
  alignas(16) static std::array<float, 82944 + XNN_EXTRA_BYTES / sizeof(float)> w188;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w189;
  alignas(16) static std::array<float, 55296 + XNN_EXTRA_BYTES / sizeof(float)> w190;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w191;
  alignas(16) static std::array<float, 55296 + XNN_EXTRA_BYTES / sizeof(float)> w192;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w193;
  alignas(16) static std::array<float, 14400 + XNN_EXTRA_BYTES / sizeof(float)> w194;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w195;
  alignas(16) static std::array<float, 82944 + XNN_EXTRA_BYTES / sizeof(float)> w196;
  alignas(16) static std::array<float, 144 + XNN_EXTRA_BYTES / sizeof(float)> w197;
  alignas(16) static std::array<float, 82944 + XNN_EXTRA_BYTES / sizeof(float)> w198;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w199;
  alignas(16) static std::array<float, 55296 + XNN_EXTRA_BYTES / sizeof(float)> w200;
  alignas(16) static std::array<float, 96 + XNN_EXTRA_BYTES / sizeof(float)> w201;
  alignas(16) static std::array<float, 55296 + XNN_EXTRA_BYTES / sizeof(float)> w202;
  alignas(16) static std::array<float, 576 + XNN_EXTRA_BYTES / sizeof(float)> w203;
  alignas(16) static std::array<float, 589824 + XNN_EXTRA_BYTES / sizeof(float)> w204;
  alignas(16) static std::array<float, 1024 + XNN_EXTRA_BYTES / sizeof(float)> w205;
  alignas(16) static std::array<float, 1025024 + XNN_EXTRA_BYTES / sizeof(float)> w206;
  alignas(16) static std::array<float, 1001 + XNN_EXTRA_BYTES / sizeof(float)> w207;

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
  std::generate(v17.begin(), v17.end(), std::ref(f32rng));
  std::generate(v18.begin(), v18.end(), std::ref(f32rng));
  std::generate(v19.begin(), v19.end(), std::ref(f32rng));
  std::generate(v20.begin(), v20.end(), std::ref(f32rng));
  std::generate(v21.begin(), v21.end(), std::ref(f32rng));
  std::generate(v22.begin(), v22.end(), std::ref(f32rng));
  std::generate(v23.begin(), v23.end(), std::ref(f32rng));
  std::generate(v24.begin(), v24.end(), std::ref(f32rng));
  std::generate(v26.begin(), v26.end(), std::ref(f32rng));
  std::generate(v27.begin(), v27.end(), std::ref(f32rng));
  std::generate(v28.begin(), v28.end(), std::ref(f32rng));
  std::generate(v29.begin(), v29.end(), std::ref(f32rng));
  std::generate(v30.begin(), v30.end(), std::ref(f32rng));
  std::generate(v31.begin(), v31.end(), std::ref(f32rng));
  std::generate(v32.begin(), v32.end(), std::ref(f32rng));
  std::generate(v33.begin(), v33.end(), std::ref(f32rng));
  std::generate(v34.begin(), v34.end(), std::ref(f32rng));
  std::generate(v36.begin(), v36.end(), std::ref(f32rng));
  std::generate(v37.begin(), v37.end(), std::ref(f32rng));
  std::generate(v38.begin(), v38.end(), std::ref(f32rng));
  std::generate(v39.begin(), v39.end(), std::ref(f32rng));
  std::generate(v40.begin(), v40.end(), std::ref(f32rng));
  std::generate(v41.begin(), v41.end(), std::ref(f32rng));
  std::generate(v42.begin(), v42.end(), std::ref(f32rng));
  std::generate(v43.begin(), v43.end(), std::ref(f32rng));
  std::generate(v44.begin(), v44.end(), std::ref(f32rng));
  std::generate(v46.begin(), v46.end(), std::ref(f32rng));
  std::generate(v47.begin(), v47.end(), std::ref(f32rng));
  std::generate(v48.begin(), v48.end(), std::ref(f32rng));
  std::generate(v49.begin(), v49.end(), std::ref(f32rng));
  std::generate(v50.begin(), v50.end(), std::ref(f32rng));
  std::generate(v51.begin(), v51.end(), std::ref(f32rng));
  std::generate(v52.begin(), v52.end(), std::ref(f32rng));
  std::generate(v53.begin(), v53.end(), std::ref(f32rng));
  std::generate(v55.begin(), v55.end(), std::ref(f32rng));
  std::generate(v56.begin(), v56.end(), std::ref(f32rng));
  std::generate(v57.begin(), v57.end(), std::ref(f32rng));
  std::generate(v58.begin(), v58.end(), std::ref(f32rng));
  std::generate(v59.begin(), v59.end(), std::ref(f32rng));
  std::generate(v60.begin(), v60.end(), std::ref(f32rng));
  std::generate(v61.begin(), v61.end(), std::ref(f32rng));
  std::generate(v62.begin(), v62.end(), std::ref(f32rng));
  std::generate(v63.begin(), v63.end(), std::ref(f32rng));
  std::generate(v65.begin(), v65.end(), std::ref(f32rng));
  std::generate(v66.begin(), v66.end(), std::ref(f32rng));
  std::generate(v67.begin(), v67.end(), std::ref(f32rng));
  std::generate(v68.begin(), v68.end(), std::ref(f32rng));
  std::generate(v69.begin(), v69.end(), std::ref(f32rng));
  std::generate(v70.begin(), v70.end(), std::ref(f32rng));
  std::generate(v71.begin(), v71.end(), std::ref(f32rng));
  std::generate(v72.begin(), v72.end(), std::ref(f32rng));
  std::generate(v74.begin(), v74.end(), std::ref(f32rng));
  std::generate(v75.begin(), v75.end(), std::ref(f32rng));
  std::generate(v76.begin(), v76.end(), std::ref(f32rng));
  std::generate(v77.begin(), v77.end(), std::ref(f32rng));
  std::generate(v78.begin(), v78.end(), std::ref(f32rng));
  std::generate(v79.begin(), v79.end(), std::ref(f32rng));
  std::generate(v80.begin(), v80.end(), std::ref(f32rng));
  std::generate(v81.begin(), v81.end(), std::ref(f32rng));
  std::generate(v82.begin(), v82.end(), std::ref(f32rng));
  std::generate(v84.begin(), v84.end(), std::ref(f32rng));
  std::generate(v85.begin(), v85.end(), std::ref(f32rng));
  std::generate(v86.begin(), v86.end(), std::ref(f32rng));
  std::generate(v87.begin(), v87.end(), std::ref(f32rng));
  std::generate(v88.begin(), v88.end(), std::ref(f32rng));
  std::generate(v89.begin(), v89.end(), std::ref(f32rng));
  std::generate(v90.begin(), v90.end(), std::ref(f32rng));
  std::generate(v91.begin(), v91.end(), std::ref(f32rng));
  std::generate(v92.begin(), v92.end(), std::ref(f32rng));
  std::generate(v94.begin(), v94.end(), std::ref(f32rng));
  std::generate(v95.begin(), v95.end(), std::ref(f32rng));
  std::generate(v97.begin(), v97.end(), std::ref(f32rng));
  std::generate(v98.begin(), v98.end(), std::ref(f32rng));
  std::generate(v99.begin(), v99.end(), std::ref(f32rng));
  std::generate(w100.begin(), w100.end(), std::ref(f32rng));
  std::generate(w101.begin(), w101.end(), std::ref(f32rng));
  std::generate(w102.begin(), w102.end(), std::ref(f32rng));
  std::generate(w103.begin(), w103.end(), std::ref(f32rng));
  std::generate(w104.begin(), w104.end(), std::ref(f32rng));
  std::generate(w105.begin(), w105.end(), std::ref(f32rng));
  std::generate(w106.begin(), w106.end(), std::ref(f32rng));
  std::generate(w107.begin(), w107.end(), std::ref(f32rng));
  std::generate(w108.begin(), w108.end(), std::ref(f32rng));
  std::generate(w109.begin(), w109.end(), std::ref(f32rng));
  std::generate(w110.begin(), w110.end(), std::ref(f32rng));
  std::generate(w111.begin(), w111.end(), std::ref(f32rng));
  std::generate(w112.begin(), w112.end(), std::ref(f32rng));
  std::generate(w113.begin(), w113.end(), std::ref(f32rng));
  std::generate(w114.begin(), w114.end(), std::ref(f32rng));
  std::generate(w115.begin(), w115.end(), std::ref(f32rng));
  std::generate(w116.begin(), w116.end(), std::ref(f32rng));
  std::generate(w117.begin(), w117.end(), std::ref(f32rng));
  std::generate(w118.begin(), w118.end(), std::ref(f32rng));
  std::generate(w119.begin(), w119.end(), std::ref(f32rng));
  std::generate(w120.begin(), w120.end(), std::ref(f32rng));
  std::generate(w121.begin(), w121.end(), std::ref(f32rng));
  std::generate(w122.begin(), w122.end(), std::ref(f32rng));
  std::generate(w123.begin(), w123.end(), std::ref(f32rng));
  std::generate(w124.begin(), w124.end(), std::ref(f32rng));
  std::generate(w125.begin(), w125.end(), std::ref(f32rng));
  std::generate(w126.begin(), w126.end(), std::ref(f32rng));
  std::generate(w127.begin(), w127.end(), std::ref(f32rng));
  std::generate(w128.begin(), w128.end(), std::ref(f32rng));
  std::generate(w129.begin(), w129.end(), std::ref(f32rng));
  std::generate(w130.begin(), w130.end(), std::ref(f32rng));
  std::generate(w131.begin(), w131.end(), std::ref(f32rng));
  std::generate(w132.begin(), w132.end(), std::ref(f32rng));
  std::generate(w133.begin(), w133.end(), std::ref(f32rng));
  std::generate(w134.begin(), w134.end(), std::ref(f32rng));
  std::generate(w135.begin(), w135.end(), std::ref(f32rng));
  std::generate(w136.begin(), w136.end(), std::ref(f32rng));
  std::generate(w137.begin(), w137.end(), std::ref(f32rng));
  std::generate(w138.begin(), w138.end(), std::ref(f32rng));
  std::generate(w139.begin(), w139.end(), std::ref(f32rng));
  std::generate(w140.begin(), w140.end(), std::ref(f32rng));
  std::generate(w141.begin(), w141.end(), std::ref(f32rng));
  std::generate(w142.begin(), w142.end(), std::ref(f32rng));
  std::generate(w143.begin(), w143.end(), std::ref(f32rng));
  std::generate(w144.begin(), w144.end(), std::ref(f32rng));
  std::generate(w145.begin(), w145.end(), std::ref(f32rng));
  std::generate(w146.begin(), w146.end(), std::ref(f32rng));
  std::generate(w147.begin(), w147.end(), std::ref(f32rng));
  std::generate(w148.begin(), w148.end(), std::ref(f32rng));
  std::generate(w149.begin(), w149.end(), std::ref(f32rng));
  std::generate(w150.begin(), w150.end(), std::ref(f32rng));
  std::generate(w151.begin(), w151.end(), std::ref(f32rng));
  std::generate(w152.begin(), w152.end(), std::ref(f32rng));
  std::generate(w153.begin(), w153.end(), std::ref(f32rng));
  std::generate(w154.begin(), w154.end(), std::ref(f32rng));
  std::generate(w155.begin(), w155.end(), std::ref(f32rng));
  std::generate(w156.begin(), w156.end(), std::ref(f32rng));
  std::generate(w157.begin(), w157.end(), std::ref(f32rng));
  std::generate(w158.begin(), w158.end(), std::ref(f32rng));
  std::generate(w159.begin(), w159.end(), std::ref(f32rng));
  std::generate(w160.begin(), w160.end(), std::ref(f32rng));
  std::generate(w161.begin(), w161.end(), std::ref(f32rng));
  std::generate(w162.begin(), w162.end(), std::ref(f32rng));
  std::generate(w163.begin(), w163.end(), std::ref(f32rng));
  std::generate(w164.begin(), w164.end(), std::ref(f32rng));
  std::generate(w165.begin(), w165.end(), std::ref(f32rng));
  std::generate(w166.begin(), w166.end(), std::ref(f32rng));
  std::generate(w167.begin(), w167.end(), std::ref(f32rng));
  std::generate(w168.begin(), w168.end(), std::ref(f32rng));
  std::generate(w169.begin(), w169.end(), std::ref(f32rng));
  std::generate(w170.begin(), w170.end(), std::ref(f32rng));
  std::generate(w171.begin(), w171.end(), std::ref(f32rng));
  std::generate(w172.begin(), w172.end(), std::ref(f32rng));
  std::generate(w173.begin(), w173.end(), std::ref(f32rng));
  std::generate(w174.begin(), w174.end(), std::ref(f32rng));
  std::generate(w175.begin(), w175.end(), std::ref(f32rng));
  std::generate(w176.begin(), w176.end(), std::ref(f32rng));
  std::generate(w177.begin(), w177.end(), std::ref(f32rng));
  std::generate(w178.begin(), w178.end(), std::ref(f32rng));
  std::generate(w179.begin(), w179.end(), std::ref(f32rng));
  std::generate(w180.begin(), w180.end(), std::ref(f32rng));
  std::generate(w181.begin(), w181.end(), std::ref(f32rng));
  std::generate(w182.begin(), w182.end(), std::ref(f32rng));
  std::generate(w183.begin(), w183.end(), std::ref(f32rng));
  std::generate(w184.begin(), w184.end(), std::ref(f32rng));
  std::generate(w185.begin(), w185.end(), std::ref(f32rng));
  std::generate(w186.begin(), w186.end(), std::ref(f32rng));
  std::generate(w187.begin(), w187.end(), std::ref(f32rng));
  std::generate(w188.begin(), w188.end(), std::ref(f32rng));
  std::generate(w189.begin(), w189.end(), std::ref(f32rng));
  std::generate(w190.begin(), w190.end(), std::ref(f32rng));
  std::generate(w191.begin(), w191.end(), std::ref(f32rng));
  std::generate(w192.begin(), w192.end(), std::ref(f32rng));
  std::generate(w193.begin(), w193.end(), std::ref(f32rng));
  std::generate(w194.begin(), w194.end(), std::ref(f32rng));
  std::generate(w195.begin(), w195.end(), std::ref(f32rng));
  std::generate(w196.begin(), w196.end(), std::ref(f32rng));
  std::generate(w197.begin(), w197.end(), std::ref(f32rng));
  std::generate(w198.begin(), w198.end(), std::ref(f32rng));
  std::generate(w199.begin(), w199.end(), std::ref(f32rng));
  std::generate(w200.begin(), w200.end(), std::ref(f32rng));
  std::generate(w201.begin(), w201.end(), std::ref(f32rng));
  std::generate(w202.begin(), w202.end(), std::ref(f32rng));
  std::generate(w203.begin(), w203.end(), std::ref(f32rng));
  std::generate(w204.begin(), w204.end(), std::ref(f32rng));
  std::generate(w205.begin(), w205.end(), std::ref(f32rng));
  std::generate(w206.begin(), w206.end(), std::ref(f32rng));
  std::generate(w207.begin(), w207.end(), std::ref(f32rng));

  Operators operators;
  xnn_status status;
  xnn_code_cache* code_cache_ptr = nullptr;
#if XNN_PLATFORM_JIT
  xnn_code_cache code_cache;
  xnn_init_code_cache(&code_cache);
  code_cache_ptr = &code_cache;
#endif
  std::array<xnn_post_operation, 1> post_ops {xnn_post_operation{xnn_post_operation_type_hardswish}};
  size_t max_workspace_size = 0;

  xnn_operator_t op0 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    3 /* input channels per group */,
    16 /* output_channels_per_group */,
    3 /* input pixel stride */,
    16 /* output pixel stride */,
    w100.data(), w101.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    16 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    16 /* input pixel stride */,
    16 /* output pixel stride */,
    w102.data(), w103.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    16 /* input channels per group */,
    8 /* output_channels_per_group */,
    16 /* input pixel stride */,
    8 /* output pixel stride */,
    w104.data(), w105.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op4);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #4" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op4, xnn_delete_operator);

  xnn_operator_t op5 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    8 /* input channels per group */,
    16 /* output_channels_per_group */,
    8 /* input pixel stride */,
    16 /* output pixel stride */,
    w106.data(), w107.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op5);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #5" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op5, xnn_delete_operator);

  xnn_operator_t op6 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op6);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #6" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op6, xnn_delete_operator);

  xnn_operator_t op7 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    16 /* input channels per group */,
    16 /* output_channels_per_group */,
    16 /* input pixel stride */,
    16 /* output pixel stride */,
    w108.data(), w109.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op7);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #7" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op7, xnn_delete_operator);

  xnn_operator_t op8 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    16 /* input channels per group */,
    72 /* output_channels_per_group */,
    16 /* input pixel stride */,
    72 /* output pixel stride */,
    w110.data(), w111.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 0 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    72 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    72 /* input pixel stride */,
    72 /* output pixel stride */,
    w112.data(), w113.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op9);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #9" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op9, xnn_delete_operator);

  xnn_operator_t op10 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    72 /* input channels per group */,
    24 /* output_channels_per_group */,
    72 /* input pixel stride */,
    24 /* output pixel stride */,
    w114.data(), w115.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    24 /* input channels per group */,
    88 /* output_channels_per_group */,
    24 /* input pixel stride */,
    88 /* output pixel stride */,
    w116.data(), w117.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
    3 /* kernel height */, 3 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    88 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    88 /* input pixel stride */,
    88 /* output pixel stride */,
    w118.data(), w119.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op12);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #12" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op12, xnn_delete_operator);

  xnn_operator_t op13 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    88 /* input channels per group */,
    24 /* output_channels_per_group */,
    88 /* input pixel stride */,
    24 /* output pixel stride */,
    w120.data(), w121.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op13);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #13" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op13, xnn_delete_operator);

  xnn_operator_t op14 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op14);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #14" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op14, xnn_delete_operator);

  xnn_operator_t op15 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    24 /* input channels per group */,
    96 /* output_channels_per_group */,
    24 /* input pixel stride */,
    96 /* output pixel stride */,
    w122.data(), w123.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 1 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    96 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    96 /* input pixel stride */,
    96 /* output pixel stride */,
    w124.data(), w125.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    0 /* flags */,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
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
    w126.data(), w127.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op20);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #20" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op20, xnn_delete_operator);

  xnn_operator_t op21 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    24 /* input channels per group */,
    96 /* output_channels_per_group */,
    24 /* input pixel stride */,
    96 /* output pixel stride */,
    w128.data(), w129.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op21);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #21" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op21, xnn_delete_operator);

  xnn_operator_t op22 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op22);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #22" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op22, xnn_delete_operator);

  xnn_operator_t op23 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    96 /* input channels per group */,
    40 /* output_channels_per_group */,
    96 /* input pixel stride */,
    40 /* output pixel stride */,
    w130.data(), w131.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op23);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #23" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op23, xnn_delete_operator);

  xnn_operator_t op24 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    40 /* input channels per group */,
    240 /* output_channels_per_group */,
    40 /* input pixel stride */,
    240 /* output pixel stride */,
    w132.data(), w133.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    240 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    240 /* input pixel stride */,
    240 /* output pixel stride */,
    w134.data(), w135.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  xnn_operator_t op29 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    240 /* input channels per group */,
    64 /* output_channels_per_group */,
    240 /* input pixel stride */,
    64 /* output pixel stride */,
    w136.data(), w137.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op29);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #29" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op29, xnn_delete_operator);

  xnn_operator_t op30 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    240 /* output_channels_per_group */,
    64 /* input pixel stride */,
    240 /* output pixel stride */,
    w138.data(), w139.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op30);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #30" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op30, xnn_delete_operator);

  xnn_operator_t op31 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op31);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #31" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op31, xnn_delete_operator);

  xnn_operator_t op32 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    240 /* input channels per group */,
    40 /* output_channels_per_group */,
    240 /* input pixel stride */,
    40 /* output pixel stride */,
    w140.data(), w141.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op32);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #32" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op32, xnn_delete_operator);

  xnn_operator_t op33 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op33);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #33" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op33, xnn_delete_operator);

  xnn_operator_t op34 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    40 /* input channels per group */,
    240 /* output_channels_per_group */,
    40 /* input pixel stride */,
    240 /* output pixel stride */,
    w142.data(), w143.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #34" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op34, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    240 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    240 /* input pixel stride */,
    240 /* output pixel stride */,
    w144.data(), w145.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #36" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op36, xnn_delete_operator);

  xnn_operator_t op37 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    0 /* flags */,
    &op37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #37" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op37, xnn_delete_operator);

  xnn_operator_t op38 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op38);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #38" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op38, xnn_delete_operator);

  xnn_operator_t op39 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    240 /* input channels per group */,
    64 /* output_channels_per_group */,
    240 /* input pixel stride */,
    64 /* output pixel stride */,
    w146.data(), w147.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #39" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op39, xnn_delete_operator);

  xnn_operator_t op40 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    64 /* input channels per group */,
    240 /* output_channels_per_group */,
    64 /* input pixel stride */,
    240 /* output pixel stride */,
    w148.data(), w149.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op40);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #40" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op40, xnn_delete_operator);

  xnn_operator_t op41 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op41);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #41" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op41, xnn_delete_operator);

  xnn_operator_t op42 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    240 /* input channels per group */,
    40 /* output_channels_per_group */,
    240 /* input pixel stride */,
    40 /* output pixel stride */,
    w150.data(), w151.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op42);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #42" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op42, xnn_delete_operator);

  xnn_operator_t op43 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op43);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #43" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op43, xnn_delete_operator);

  xnn_operator_t op44 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    40 /* input channels per group */,
    120 /* output_channels_per_group */,
    40 /* input pixel stride */,
    120 /* output pixel stride */,
    w152.data(), w153.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op46 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    120 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    120 /* input pixel stride */,
    120 /* output pixel stride */,
    w154.data(), w155.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #46" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op46, xnn_delete_operator);

  xnn_operator_t op47 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    0 /* flags */,
    &op47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #47" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op47, xnn_delete_operator);

  xnn_operator_t op48 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op48);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #48" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op48, xnn_delete_operator);

  xnn_operator_t op49 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    120 /* input channels per group */,
    32 /* output_channels_per_group */,
    120 /* input pixel stride */,
    32 /* output pixel stride */,
    w156.data(), w157.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op49);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #49" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op49, xnn_delete_operator);

  xnn_operator_t op50 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    32 /* input channels per group */,
    120 /* output_channels_per_group */,
    32 /* input pixel stride */,
    120 /* output pixel stride */,
    w158.data(), w159.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op50);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #50" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op50, xnn_delete_operator);

  xnn_operator_t op51 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op51);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #51" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op51, xnn_delete_operator);

  xnn_operator_t op52 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    120 /* input channels per group */,
    48 /* output_channels_per_group */,
    120 /* input pixel stride */,
    48 /* output pixel stride */,
    w160.data(), w161.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #52" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op52, xnn_delete_operator);

  xnn_operator_t op53 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    48 /* input channels per group */,
    144 /* output_channels_per_group */,
    48 /* input pixel stride */,
    144 /* output pixel stride */,
    w162.data(), w163.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op55 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    144 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    144 /* input pixel stride */,
    144 /* output pixel stride */,
    w164.data(), w165.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #55" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op55, xnn_delete_operator);

  xnn_operator_t op56 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    0 /* flags */,
    &op56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #56" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op56, xnn_delete_operator);

  xnn_operator_t op57 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op57);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #57" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op57, xnn_delete_operator);

  xnn_operator_t op58 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    144 /* input channels per group */,
    40 /* output_channels_per_group */,
    144 /* input pixel stride */,
    40 /* output pixel stride */,
    w166.data(), w167.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #58" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op58, xnn_delete_operator);

  xnn_operator_t op59 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    40 /* input channels per group */,
    144 /* output_channels_per_group */,
    40 /* input pixel stride */,
    144 /* output pixel stride */,
    w168.data(), w169.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op59);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #59" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op59, xnn_delete_operator);

  xnn_operator_t op60 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op60);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #60" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op60, xnn_delete_operator);

  xnn_operator_t op61 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    144 /* input channels per group */,
    48 /* output_channels_per_group */,
    144 /* input pixel stride */,
    48 /* output pixel stride */,
    w170.data(), w171.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op61);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #61" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op61, xnn_delete_operator);

  xnn_operator_t op62 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op62);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #62" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op62, xnn_delete_operator);

  xnn_operator_t op63 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    48 /* input channels per group */,
    288 /* output_channels_per_group */,
    48 /* input pixel stride */,
    288 /* output pixel stride */,
    w172.data(), w173.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #63" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op63, xnn_delete_operator);

  xnn_operator_t op65 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    1 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 1 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    2 /* subsampling height */, 2 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    288 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    288 /* input pixel stride */,
    288 /* output pixel stride */,
    w174.data(), w175.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op65);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #65" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op65, xnn_delete_operator);

  xnn_operator_t op66 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    0 /* flags */,
    &op66);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #66" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op66, xnn_delete_operator);

  xnn_operator_t op67 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op67);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #67" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op67, xnn_delete_operator);

  xnn_operator_t op68 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    288 /* input channels per group */,
    72 /* output_channels_per_group */,
    288 /* input pixel stride */,
    72 /* output pixel stride */,
    w176.data(), w177.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op68);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #68" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op68, xnn_delete_operator);

  xnn_operator_t op69 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    72 /* input channels per group */,
    288 /* output_channels_per_group */,
    72 /* input pixel stride */,
    288 /* output pixel stride */,
    w178.data(), w179.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op69);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #69" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op69, xnn_delete_operator);

  xnn_operator_t op70 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op70);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #70" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op70, xnn_delete_operator);

  xnn_operator_t op71 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    288 /* input channels per group */,
    96 /* output_channels_per_group */,
    288 /* input pixel stride */,
    96 /* output pixel stride */,
    w180.data(), w181.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op71);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #71" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op71, xnn_delete_operator);

  xnn_operator_t op72 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
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
    w182.data(), w183.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op72);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #72" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op72, xnn_delete_operator);

  xnn_operator_t op74 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    576 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    576 /* input pixel stride */,
    576 /* output pixel stride */,
    w184.data(), w185.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op74);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #74" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op74, xnn_delete_operator);

  xnn_operator_t op75 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    0 /* flags */,
    &op75);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #75" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op75, xnn_delete_operator);

  xnn_operator_t op76 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op76);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #76" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op76, xnn_delete_operator);

  xnn_operator_t op77 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    144 /* output_channels_per_group */,
    576 /* input pixel stride */,
    144 /* output pixel stride */,
    w186.data(), w187.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op77);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #77" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op77, xnn_delete_operator);

  xnn_operator_t op78 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    144 /* input channels per group */,
    576 /* output_channels_per_group */,
    144 /* input pixel stride */,
    576 /* output pixel stride */,
    w188.data(), w189.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op78);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #78" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op78, xnn_delete_operator);

  xnn_operator_t op79 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op79);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #79" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op79, xnn_delete_operator);

  xnn_operator_t op80 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
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
    w190.data(), w191.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op80);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #80" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op80, xnn_delete_operator);

  xnn_operator_t op81 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op81);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #81" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op81, xnn_delete_operator);

  xnn_operator_t op82 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
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
    w192.data(), w193.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op82);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #82" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op82, xnn_delete_operator);

  xnn_operator_t op84 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
    5 /* kernel height */, 5 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    576 /* groups */,
    1 /* input channels per group */,
    1 /* output_channels_per_group */,
    576 /* input pixel stride */,
    576 /* output pixel stride */,
    w194.data(), w195.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op84);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #84" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op84, xnn_delete_operator);

  xnn_operator_t op85 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    0 /* flags */,
    &op85);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #85" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op85, xnn_delete_operator);

  xnn_operator_t op86 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op86);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #86" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op86, xnn_delete_operator);

  xnn_operator_t op87 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    144 /* output_channels_per_group */,
    576 /* input pixel stride */,
    144 /* output pixel stride */,
    w196.data(), w197.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op87);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #87" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op87, xnn_delete_operator);

  xnn_operator_t op88 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    144 /* input channels per group */,
    576 /* output_channels_per_group */,
    144 /* input pixel stride */,
    576 /* output pixel stride */,
    w198.data(), w199.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op88);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #88" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op88, xnn_delete_operator);

  xnn_operator_t op89 = nullptr;
  status = xnn_create_multiply_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op89);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #89" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op89, xnn_delete_operator);

  xnn_operator_t op90 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
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
    w200.data(), w201.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op90);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #90" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op90, xnn_delete_operator);

  xnn_operator_t op91 = nullptr;
  status = xnn_create_add_nd_f32(
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op91);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #91" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op91, xnn_delete_operator);

  xnn_operator_t op92 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
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
    w202.data(), w203.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op92);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #92" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op92, xnn_delete_operator);

  xnn_operator_t op94 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op94);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #94" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op94, xnn_delete_operator);

  xnn_operator_t op95 = nullptr;
  status = xnn_create_fused_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    576 /* input channels per group */,
    1024 /* output_channels_per_group */,
    576 /* input pixel stride */,
    1024 /* output pixel stride */,
    w204.data(), w205.data(),
    post_ops.size(), post_ops.data(),
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op95);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #95" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op95, xnn_delete_operator);

  xnn_operator_t op97 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op97);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #97" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op97, xnn_delete_operator);

  xnn_operator_t op98 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
    0 /* top padding */, 0 /* right padding */,
    0 /* bottom padding */, 0 /* left padding */,
    1 /* kernel height */, 1 /* kernel width */,
    1 /* subsampling height */, 1 /* subsampling width */,
    1 /* dilation_height */, 1 /* dilation_width */,
    1 /* groups */,
    1024 /* input channels per group */,
    1001 /* output_channels_per_group */,
    1024 /* input pixel stride */,
    1001 /* output pixel stride */,
    w206.data(), w207.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    code_cache_ptr,
    nullptr,
    &op98);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #98" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op98, xnn_delete_operator);

#if XNN_PLATFORM_JIT
  xnn_finalize_code_memory(&code_cache.cache.code);
#endif

  size_t workspace_size, workspace_alignment;
  status = xnn_reshape_convolution2d_nhwc_f32(
    op0,
    /*batch_size=*/1, /*input_height=*/224, /*input_width=*/224,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op2,
    /*batch_size=*/1, /*input_height=*/112, /*input_width=*/112,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #2" << std::endl;
    return ExecutionPlan();
  }

  size_t op3_workspace_size = 0;
  size_t op3_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op3,
    /*batch_size=*/1, 3136 /* width */,
    16 /* channels */, 16 /* input stride */, 16 /* output stride */,
    &op3_workspace_size, &op3_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op3_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op4,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op5,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #5" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 56, 56, 16 };
    const size_t b_shape[] = { 1, 1, 1, 16 };
    status = xnn_reshape_multiply_nd_f32(
      op6,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op7,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op8,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op9,
    /*batch_size=*/1, /*input_height=*/56, /*input_width=*/56,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op10,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op11,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op12,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op13,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #13" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 28, 28, 24 };
    const size_t b_shape[] = { 1, 28, 28, 24 };
    status = xnn_reshape_add_nd_f32(
      op14,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op15,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op17,
    /*batch_size=*/1, /*input_height=*/28, /*input_width=*/28,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f32(
    op18,
    /*batch_size=*/196,
    96 /* channels */,
    96 /* input stride */,
    96 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #18" << std::endl;
    return ExecutionPlan();
  }

  size_t op19_workspace_size = 0;
  size_t op19_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op19,
    /*batch_size=*/1, 196 /* width */,
    96 /* channels */, 96 /* input stride */, 96 /* output stride */,
    &op19_workspace_size, &op19_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op19_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op20,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op21,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #21" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 96 };
    const size_t b_shape[] = { 1, 1, 1, 96 };
    status = xnn_reshape_multiply_nd_f32(
      op22,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op23,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op24,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op26,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f32(
    op27,
    /*batch_size=*/196,
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #27" << std::endl;
    return ExecutionPlan();
  }

  size_t op28_workspace_size = 0;
  size_t op28_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op28,
    /*batch_size=*/1, 196 /* width */,
    240 /* channels */, 240 /* input stride */, 240 /* output stride */,
    &op28_workspace_size, &op28_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op28_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op29,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op30,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #30" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 240 };
    const size_t b_shape[] = { 1, 1, 1, 240 };
    status = xnn_reshape_multiply_nd_f32(
      op31,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op32,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #32" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 40 };
    const size_t b_shape[] = { 1, 14, 14, 40 };
    status = xnn_reshape_add_nd_f32(
      op33,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op34,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op36,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f32(
    op37,
    /*batch_size=*/196,
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #37" << std::endl;
    return ExecutionPlan();
  }

  size_t op38_workspace_size = 0;
  size_t op38_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op38,
    /*batch_size=*/1, 196 /* width */,
    240 /* channels */, 240 /* input stride */, 240 /* output stride */,
    &op38_workspace_size, &op38_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op38_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op39,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op40,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #40" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 240 };
    const size_t b_shape[] = { 1, 1, 1, 240 };
    status = xnn_reshape_multiply_nd_f32(
      op41,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op42,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #42" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 40 };
    const size_t b_shape[] = { 1, 14, 14, 40 };
    status = xnn_reshape_add_nd_f32(
      op43,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op44,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op46,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f32(
    op47,
    /*batch_size=*/196,
    120 /* channels */,
    120 /* input stride */,
    120 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #47" << std::endl;
    return ExecutionPlan();
  }

  size_t op48_workspace_size = 0;
  size_t op48_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op48,
    /*batch_size=*/1, 196 /* width */,
    120 /* channels */, 120 /* input stride */, 120 /* output stride */,
    &op48_workspace_size, &op48_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op48_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op49,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op50,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #50" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 120 };
    const size_t b_shape[] = { 1, 1, 1, 120 };
    status = xnn_reshape_multiply_nd_f32(
      op51,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op52,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op53,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op55,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f32(
    op56,
    /*batch_size=*/196,
    144 /* channels */,
    144 /* input stride */,
    144 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #56" << std::endl;
    return ExecutionPlan();
  }

  size_t op57_workspace_size = 0;
  size_t op57_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op57,
    /*batch_size=*/1, 196 /* width */,
    144 /* channels */, 144 /* input stride */, 144 /* output stride */,
    &op57_workspace_size, &op57_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op57_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op58,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op59,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #59" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 144 };
    const size_t b_shape[] = { 1, 1, 1, 144 };
    status = xnn_reshape_multiply_nd_f32(
      op60,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op61,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #61" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 14, 14, 48 };
    const size_t b_shape[] = { 1, 14, 14, 48 };
    status = xnn_reshape_add_nd_f32(
      op62,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op63,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op65,
    /*batch_size=*/1, /*input_height=*/14, /*input_width=*/14,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #65" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f32(
    op66,
    /*batch_size=*/49,
    288 /* channels */,
    288 /* input stride */,
    288 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #66" << std::endl;
    return ExecutionPlan();
  }

  size_t op67_workspace_size = 0;
  size_t op67_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op67,
    /*batch_size=*/1, 49 /* width */,
    288 /* channels */, 288 /* input stride */, 288 /* output stride */,
    &op67_workspace_size, &op67_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op67_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #67" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op68,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #68" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op69,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #69" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 288 };
    const size_t b_shape[] = { 1, 1, 1, 288 };
    status = xnn_reshape_multiply_nd_f32(
      op70,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #70" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op71,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #71" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op72,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #72" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op74,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #74" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f32(
    op75,
    /*batch_size=*/49,
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #75" << std::endl;
    return ExecutionPlan();
  }

  size_t op76_workspace_size = 0;
  size_t op76_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op76,
    /*batch_size=*/1, 49 /* width */,
    576 /* channels */, 576 /* input stride */, 576 /* output stride */,
    &op76_workspace_size, &op76_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op76_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #76" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op77,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #77" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op78,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #78" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 576 };
    const size_t b_shape[] = { 1, 1, 1, 576 };
    status = xnn_reshape_multiply_nd_f32(
      op79,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #79" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op80,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #80" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 96 };
    const size_t b_shape[] = { 1, 7, 7, 96 };
    status = xnn_reshape_add_nd_f32(
      op81,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #81" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op82,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #82" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op84,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #84" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_hardswish_nc_f32(
    op85,
    /*batch_size=*/49,
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #85" << std::endl;
    return ExecutionPlan();
  }

  size_t op86_workspace_size = 0;
  size_t op86_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op86,
    /*batch_size=*/1, 49 /* width */,
    576 /* channels */, 576 /* input stride */, 576 /* output stride */,
    &op86_workspace_size, &op86_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op86_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #86" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op87,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #87" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op88,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #88" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 576 };
    const size_t b_shape[] = { 1, 1, 1, 576 };
    status = xnn_reshape_multiply_nd_f32(
      op89,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #89" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op90,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #90" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 7, 7, 96 };
    const size_t b_shape[] = { 1, 7, 7, 96 };
    status = xnn_reshape_add_nd_f32(
      op91,
      4, a_shape, 4, b_shape,
      /*threadpool=*/threadpool);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #91" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op92,
    /*batch_size=*/1, /*input_height=*/7, /*input_width=*/7,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #92" << std::endl;
    return ExecutionPlan();
  }

  size_t op94_workspace_size = 0;
  size_t op94_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op94,
    /*batch_size=*/1, 49 /* width */,
    576 /* channels */, 576 /* input stride */, 576 /* output stride */,
    &op94_workspace_size, &op94_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op94_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #94" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op95,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #95" << std::endl;
    return ExecutionPlan();
  }

  size_t op97_workspace_size = 0;
  size_t op97_workspace_alignment = 0;
  status = xnn_reshape_global_average_pooling_nwc_f32(
    op97,
    /*batch_size=*/1, 1 /* width */,
    1024 /* channels */, 1024 /* input stride */, 1024 /* output stride */,
    &op97_workspace_size, &op97_workspace_alignment,
    /*threadpool=*/threadpool);
  max_workspace_size = std::max(max_workspace_size, op97_workspace_size);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #97" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_reshape_convolution2d_nhwc_f32(
    op98,
    /*batch_size=*/1, /*input_height=*/1, /*input_width=*/1,
    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
    &workspace_size, &workspace_alignment,
    /*threadpool=*/threadpool);
  if (status != xnn_status_success) {
    std::cerr << "failed to reshape operation #98" << std::endl;
    return ExecutionPlan();
  }

  Workspace workspace(max_workspace_size);

  status = xnn_setup_convolution2d_nhwc_f32(
    op0,
    /*workspace=*/nullptr, /*input=*/v0.data(), /*output=*/v2.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op2,
    /*workspace=*/nullptr, /*input=*/v2.data(), /*output=*/v3.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op3,
    workspace.data(),
    /*input=*/v3.data(), /*output=*/v4.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op4,
    /*workspace=*/nullptr, /*input=*/v4.data(), /*output=*/v5.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op5,
    /*workspace=*/nullptr, /*input=*/v5.data(), /*output=*/v6.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f32(
    op6,
    v3.data() /* a */, v6.data() /* b */, /*output=*/v7.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op7,
    /*workspace=*/nullptr, /*input=*/v7.data(), /*output=*/v8.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op8,
    /*workspace=*/nullptr, /*input=*/v8.data(), /*output=*/v9.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op9,
    /*workspace=*/nullptr, /*input=*/v9.data(), /*output=*/v10.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op10,
    /*workspace=*/nullptr, /*input=*/v10.data(), /*output=*/v11.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op11,
    /*workspace=*/nullptr, /*input=*/v11.data(), /*output=*/v12.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op12,
    /*workspace=*/nullptr, /*input=*/v12.data(), /*output=*/v13.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op13,
    /*workspace=*/nullptr, /*input=*/v13.data(), /*output=*/v14.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op14,
    v14.data() /* a */, v11.data() /* b */, /*output=*/v15.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op15,
    /*workspace=*/nullptr, /*input=*/v15.data(), /*output=*/v17.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op17,
    /*workspace=*/nullptr, /*input=*/v17.data(), /*output=*/v18.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op18,
    /*input=*/v18.data(), /*output=*/v19.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op19,
    workspace.data(),
    /*input=*/v19.data(), /*output=*/v20.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op20,
    /*workspace=*/nullptr, /*input=*/v20.data(), /*output=*/v21.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op21,
    /*workspace=*/nullptr, /*input=*/v21.data(), /*output=*/v22.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f32(
    op22,
    v19.data() /* a */, v22.data() /* b */, /*output=*/v23.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op23,
    /*workspace=*/nullptr, /*input=*/v23.data(), /*output=*/v24.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op24,
    /*workspace=*/nullptr, /*input=*/v24.data(), /*output=*/v26.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op26,
    /*workspace=*/nullptr, /*input=*/v26.data(), /*output=*/v27.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op27,
    /*input=*/v27.data(), /*output=*/v28.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op28,
    workspace.data(),
    /*input=*/v28.data(), /*output=*/v29.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op29,
    /*workspace=*/nullptr, /*input=*/v29.data(), /*output=*/v30.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op30,
    /*workspace=*/nullptr, /*input=*/v30.data(), /*output=*/v31.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f32(
    op31,
    v28.data() /* a */, v31.data() /* b */, /*output=*/v32.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op32,
    /*workspace=*/nullptr, /*input=*/v32.data(), /*output=*/v33.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op33,
    v33.data() /* a */, v24.data() /* b */, /*output=*/v34.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op34,
    /*workspace=*/nullptr, /*input=*/v34.data(), /*output=*/v36.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op36,
    /*workspace=*/nullptr, /*input=*/v36.data(), /*output=*/v37.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op37,
    /*input=*/v37.data(), /*output=*/v38.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op38,
    workspace.data(),
    /*input=*/v38.data(), /*output=*/v39.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op39,
    /*workspace=*/nullptr, /*input=*/v39.data(), /*output=*/v40.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op40,
    /*workspace=*/nullptr, /*input=*/v40.data(), /*output=*/v41.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f32(
    op41,
    v38.data() /* a */, v41.data() /* b */, /*output=*/v42.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op42,
    /*workspace=*/nullptr, /*input=*/v42.data(), /*output=*/v43.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op43,
    v43.data() /* a */, v34.data() /* b */, /*output=*/v44.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op44,
    /*workspace=*/nullptr, /*input=*/v44.data(), /*output=*/v46.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op46,
    /*workspace=*/nullptr, /*input=*/v46.data(), /*output=*/v47.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op47,
    /*input=*/v47.data(), /*output=*/v48.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op48,
    workspace.data(),
    /*input=*/v48.data(), /*output=*/v49.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op49,
    /*workspace=*/nullptr, /*input=*/v49.data(), /*output=*/v50.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op50,
    /*workspace=*/nullptr, /*input=*/v50.data(), /*output=*/v51.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f32(
    op51,
    v48.data() /* a */, v51.data() /* b */, /*output=*/v52.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op52,
    /*workspace=*/nullptr, /*input=*/v52.data(), /*output=*/v53.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op53,
    /*workspace=*/nullptr, /*input=*/v53.data(), /*output=*/v55.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op55,
    /*workspace=*/nullptr, /*input=*/v55.data(), /*output=*/v56.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op56,
    /*input=*/v56.data(), /*output=*/v57.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op57,
    workspace.data(),
    /*input=*/v57.data(), /*output=*/v58.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op58,
    /*workspace=*/nullptr, /*input=*/v58.data(), /*output=*/v59.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op59,
    /*workspace=*/nullptr, /*input=*/v59.data(), /*output=*/v60.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f32(
    op60,
    v57.data() /* a */, v60.data() /* b */, /*output=*/v61.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op61,
    /*workspace=*/nullptr, /*input=*/v61.data(), /*output=*/v62.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op62,
    v62.data() /* a */, v53.data() /* b */, /*output=*/v63.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op63,
    /*workspace=*/nullptr, /*input=*/v63.data(), /*output=*/v65.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op65,
    /*workspace=*/nullptr, /*input=*/v65.data(), /*output=*/v66.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #65" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op66,
    /*input=*/v66.data(), /*output=*/v67.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #66" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op67,
    workspace.data(),
    /*input=*/v67.data(), /*output=*/v68.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #67" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op68,
    /*workspace=*/nullptr, /*input=*/v68.data(), /*output=*/v69.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #68" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op69,
    /*workspace=*/nullptr, /*input=*/v69.data(), /*output=*/v70.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #69" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f32(
    op70,
    v67.data() /* a */, v70.data() /* b */, /*output=*/v71.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #70" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op71,
    /*workspace=*/nullptr, /*input=*/v71.data(), /*output=*/v72.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #71" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op72,
    /*workspace=*/nullptr, /*input=*/v72.data(), /*output=*/v74.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #72" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op74,
    /*workspace=*/nullptr, /*input=*/v74.data(), /*output=*/v75.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #74" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op75,
    /*input=*/v75.data(), /*output=*/v76.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #75" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op76,
    workspace.data(),
    /*input=*/v76.data(), /*output=*/v77.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #76" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op77,
    /*workspace=*/nullptr, /*input=*/v77.data(), /*output=*/v78.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #77" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op78,
    /*workspace=*/nullptr, /*input=*/v78.data(), /*output=*/v79.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #78" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f32(
    op79,
    v76.data() /* a */, v79.data() /* b */, /*output=*/v80.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #79" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op80,
    /*workspace=*/nullptr, /*input=*/v80.data(), /*output=*/v81.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #80" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op81,
    v81.data() /* a */, v72.data() /* b */, /*output=*/v82.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #81" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op82,
    /*workspace=*/nullptr, /*input=*/v82.data(), /*output=*/v84.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #82" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op84,
    /*workspace=*/nullptr, /*input=*/v84.data(), /*output=*/v85.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #84" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op85,
    /*input=*/v85.data(), /*output=*/v86.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #85" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op86,
    workspace.data(),
    /*input=*/v86.data(), /*output=*/v87.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #86" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op87,
    /*workspace=*/nullptr, /*input=*/v87.data(), /*output=*/v88.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #87" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op88,
    /*workspace=*/nullptr, /*input=*/v88.data(), /*output=*/v89.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #88" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_multiply_nd_f32(
    op89,
    v86.data() /* a */, v89.data() /* b */, /*output=*/v90.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #89" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op90,
    /*workspace=*/nullptr, /*input=*/v90.data(), /*output=*/v91.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #90" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_add_nd_f32(
    op91,
    v91.data() /* a */, v82.data() /* b */, /*output=*/v92.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #91" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op92,
    /*workspace=*/nullptr, /*input=*/v92.data(), /*output=*/v94.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #92" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op94,
    workspace.data(),
    /*input=*/v94.data(), /*output=*/v95.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #94" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op95,
    /*workspace=*/nullptr, /*input=*/v95.data(), /*output=*/v97.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #95" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op97,
    workspace.data(),
    /*input=*/v97.data(), /*output=*/v98.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #97" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op98,
    /*workspace=*/nullptr, /*input=*/v98.data(), /*output=*/v99.data());
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #98" << std::endl;
    return ExecutionPlan();
  }

  XNN_PRAGMA_CLANG("clang diagnostic push")
  XNN_PRAGMA_CLANG("clang diagnostic ignored \"-Wpessimizing-move\"")
  return ExecutionPlan{operators, workspace};
  XNN_PRAGMA_CLANG("clang diagnostic pop")
}

}  // namespace models
