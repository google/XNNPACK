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

#include "models/models.h"

namespace models {

ExecutionPlan FP32SparseMobileNetV3Small(float sparsity, pthreadpool_t threadpool) {
  alignas(16) static std::array<float, 150528> v0;
  alignas(16) static std::array<float, 200704> v1;
  alignas(16) static std::array<float, 200704> v2;
  alignas(16) static std::array<float, 50176> v3;
  alignas(16) static std::array<float, 16> v4;
  alignas(16) static std::array<float, 8> v5;
  alignas(16) static std::array<float, 16> v6;
  alignas(16) static std::array<float, 50176> v7;
  alignas(16) static std::array<float, 50176> v8;
  alignas(16) static std::array<float, 225792> v9;
  alignas(16) static std::array<float, 56448> v10;
  alignas(16) static std::array<float, 18816> v11;
  alignas(16) static std::array<float, 68992> v12;
  alignas(16) static std::array<float, 68992> v13;
  alignas(16) static std::array<float, 18816> v14;
  alignas(16) static std::array<float, 18816> v15;
  alignas(16) static std::array<float, 75264> v16;
  alignas(16) static std::array<float, 75264> v17;
  alignas(16) static std::array<float, 18816> v18;
  alignas(16) static std::array<float, 18816> v19;
  alignas(16) static std::array<float, 96> v20;
  alignas(16) static std::array<float, 24> v21;
  alignas(16) static std::array<float, 96> v22;
  alignas(16) static std::array<float, 18816> v23;
  alignas(16) static std::array<float, 7840> v24;
  alignas(16) static std::array<float, 47040> v25;
  alignas(16) static std::array<float, 47040> v26;
  alignas(16) static std::array<float, 47040> v27;
  alignas(16) static std::array<float, 47040> v28;
  alignas(16) static std::array<float, 240> v29;
  alignas(16) static std::array<float, 64> v30;
  alignas(16) static std::array<float, 240> v31;
  alignas(16) static std::array<float, 47040> v32;
  alignas(16) static std::array<float, 7840> v33;
  alignas(16) static std::array<float, 7840> v34;
  alignas(16) static std::array<float, 47040> v35;
  alignas(16) static std::array<float, 47040> v36;
  alignas(16) static std::array<float, 47040> v37;
  alignas(16) static std::array<float, 47040> v38;
  alignas(16) static std::array<float, 240> v39;
  alignas(16) static std::array<float, 64> v40;
  alignas(16) static std::array<float, 240> v41;
  alignas(16) static std::array<float, 47040> v42;
  alignas(16) static std::array<float, 7840> v43;
  alignas(16) static std::array<float, 7840> v44;
  alignas(16) static std::array<float, 23520> v45;
  alignas(16) static std::array<float, 23520> v46;
  alignas(16) static std::array<float, 23520> v47;
  alignas(16) static std::array<float, 23520> v48;
  alignas(16) static std::array<float, 120> v49;
  alignas(16) static std::array<float, 32> v50;
  alignas(16) static std::array<float, 120> v51;
  alignas(16) static std::array<float, 23520> v52;
  alignas(16) static std::array<float, 9408> v53;
  alignas(16) static std::array<float, 28224> v54;
  alignas(16) static std::array<float, 28224> v55;
  alignas(16) static std::array<float, 28224> v56;
  alignas(16) static std::array<float, 28224> v57;
  alignas(16) static std::array<float, 144> v58;
  alignas(16) static std::array<float, 40> v59;
  alignas(16) static std::array<float, 144> v60;
  alignas(16) static std::array<float, 28224> v61;
  alignas(16) static std::array<float, 9408> v62;
  alignas(16) static std::array<float, 9408> v63;
  alignas(16) static std::array<float, 56448> v64;
  alignas(16) static std::array<float, 56448> v65;
  alignas(16) static std::array<float, 14112> v66;
  alignas(16) static std::array<float, 14112> v67;
  alignas(16) static std::array<float, 288> v68;
  alignas(16) static std::array<float, 72> v69;
  alignas(16) static std::array<float, 288> v70;
  alignas(16) static std::array<float, 14112> v71;
  alignas(16) static std::array<float, 4704> v72;
  alignas(16) static std::array<float, 28224> v73;
  alignas(16) static std::array<float, 28224> v74;
  alignas(16) static std::array<float, 28224> v75;
  alignas(16) static std::array<float, 28224> v76;
  alignas(16) static std::array<float, 576> v77;
  alignas(16) static std::array<float, 144> v78;
  alignas(16) static std::array<float, 576> v79;
  alignas(16) static std::array<float, 28224> v80;
  alignas(16) static std::array<float, 4704> v81;
  alignas(16) static std::array<float, 4704> v82;
  alignas(16) static std::array<float, 28224> v83;
  alignas(16) static std::array<float, 28224> v84;
  alignas(16) static std::array<float, 28224> v85;
  alignas(16) static std::array<float, 28224> v86;
  alignas(16) static std::array<float, 576> v87;
  alignas(16) static std::array<float, 144> v88;
  alignas(16) static std::array<float, 576> v89;
  alignas(16) static std::array<float, 28224> v90;
  alignas(16) static std::array<float, 4704> v91;
  alignas(16) static std::array<float, 4704> v92;
  alignas(16) static std::array<float, 28224> v93;
  alignas(16) static std::array<float, 28224> v94;
  alignas(16) static std::array<float, 576> v95;
  alignas(16) static std::array<float, 1024> v96;
  alignas(16) static std::array<float, 1024> v97;
  alignas(16) static std::array<float, 1024> v98;
  alignas(16) static std::array<float, 1001> v99;
  alignas(16) static std::array<float, 432> w100;
  alignas(16) static std::array<float, 16> w101;
  alignas(16) static std::array<float, 144> w102;
  alignas(16) static std::array<float, 16> w103;
  alignas(16) static std::array<float, 128> w104;
  alignas(16) static std::array<float, 8> w105;
  alignas(16) static std::array<float, 128> w106;
  alignas(16) static std::array<float, 16> w107;
  alignas(16) static std::array<float, 256> w108;
  alignas(16) static std::array<float, 16> w109;
  alignas(16) static std::array<float, 1152> w110;
  alignas(16) static std::array<float, 72> w111;
  alignas(16) static std::array<float, 648> w112;
  alignas(16) static std::array<float, 72> w113;
  alignas(16) static std::array<float, 1728> w114;
  alignas(16) static std::array<float, 24> w115;
  alignas(16) static std::array<float, 2112> w116;
  alignas(16) static std::array<float, 88> w117;
  alignas(16) static std::array<float, 792> w118;
  alignas(16) static std::array<float, 88> w119;
  alignas(16) static std::array<float, 2112> w120;
  alignas(16) static std::array<float, 24> w121;
  alignas(16) static std::array<float, 2304> w122;
  alignas(16) static std::array<float, 96> w123;
  alignas(16) static std::array<float, 2400> w124;
  alignas(16) static std::array<float, 96> w125;
  alignas(16) static std::array<float, 2304> w126;
  alignas(16) static std::array<float, 24> w127;
  alignas(16) static std::array<float, 2304> w128;
  alignas(16) static std::array<float, 96> w129;
  alignas(16) static std::array<float, 3840> w130;
  alignas(16) static std::array<float, 40> w131;
  alignas(16) static std::array<float, 9600> w132;
  alignas(16) static std::array<float, 240> w133;
  alignas(16) static std::array<float, 6000> w134;
  alignas(16) static std::array<float, 240> w135;
  alignas(16) static std::array<float, 15360> w136;
  alignas(16) static std::array<float, 64> w137;
  alignas(16) static std::array<float, 15360> w138;
  alignas(16) static std::array<float, 240> w139;
  alignas(16) static std::array<float, 9600> w140;
  alignas(16) static std::array<float, 40> w141;
  alignas(16) static std::array<float, 9600> w142;
  alignas(16) static std::array<float, 240> w143;
  alignas(16) static std::array<float, 6000> w144;
  alignas(16) static std::array<float, 240> w145;
  alignas(16) static std::array<float, 15360> w146;
  alignas(16) static std::array<float, 64> w147;
  alignas(16) static std::array<float, 15360> w148;
  alignas(16) static std::array<float, 240> w149;
  alignas(16) static std::array<float, 9600> w150;
  alignas(16) static std::array<float, 40> w151;
  alignas(16) static std::array<float, 4800> w152;
  alignas(16) static std::array<float, 120> w153;
  alignas(16) static std::array<float, 3000> w154;
  alignas(16) static std::array<float, 120> w155;
  alignas(16) static std::array<float, 3840> w156;
  alignas(16) static std::array<float, 32> w157;
  alignas(16) static std::array<float, 3840> w158;
  alignas(16) static std::array<float, 120> w159;
  alignas(16) static std::array<float, 5760> w160;
  alignas(16) static std::array<float, 48> w161;
  alignas(16) static std::array<float, 6912> w162;
  alignas(16) static std::array<float, 144> w163;
  alignas(16) static std::array<float, 3600> w164;
  alignas(16) static std::array<float, 144> w165;
  alignas(16) static std::array<float, 5760> w166;
  alignas(16) static std::array<float, 40> w167;
  alignas(16) static std::array<float, 5760> w168;
  alignas(16) static std::array<float, 144> w169;
  alignas(16) static std::array<float, 6912> w170;
  alignas(16) static std::array<float, 48> w171;
  alignas(16) static std::array<float, 13824> w172;
  alignas(16) static std::array<float, 288> w173;
  alignas(16) static std::array<float, 7200> w174;
  alignas(16) static std::array<float, 288> w175;
  alignas(16) static std::array<float, 20736> w176;
  alignas(16) static std::array<float, 72> w177;
  alignas(16) static std::array<float, 20736> w178;
  alignas(16) static std::array<float, 288> w179;
  alignas(16) static std::array<float, 27648> w180;
  alignas(16) static std::array<float, 96> w181;
  alignas(16) static std::array<float, 55296> w182;
  alignas(16) static std::array<float, 576> w183;
  alignas(16) static std::array<float, 14400> w184;
  alignas(16) static std::array<float, 576> w185;
  alignas(16) static std::array<float, 82944> w186;
  alignas(16) static std::array<float, 144> w187;
  alignas(16) static std::array<float, 82944> w188;
  alignas(16) static std::array<float, 576> w189;
  alignas(16) static std::array<float, 55296> w190;
  alignas(16) static std::array<float, 96> w191;
  alignas(16) static std::array<float, 55296> w192;
  alignas(16) static std::array<float, 576> w193;
  alignas(16) static std::array<float, 14400> w194;
  alignas(16) static std::array<float, 576> w195;
  alignas(16) static std::array<float, 82944> w196;
  alignas(16) static std::array<float, 144> w197;
  alignas(16) static std::array<float, 82944> w198;
  alignas(16) static std::array<float, 576> w199;
  alignas(16) static std::array<float, 55296> w200;
  alignas(16) static std::array<float, 96> w201;
  alignas(16) static std::array<float, 55296> w202;
  alignas(16) static std::array<float, 576> w203;
  alignas(16) static std::array<float, 589824> w204;
  alignas(16) static std::array<float, 1024> w205;
  alignas(16) static std::array<float, 1025024> w206;
  alignas(16) static std::array<float, 1001> w207;

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
  std::generate(v65.begin(), v65.end(), std::ref(f32rng));
  std::generate(v66.begin(), v66.end(), std::ref(f32rng));
  std::generate(v67.begin(), v67.end(), std::ref(f32rng));
  std::generate(v68.begin(), v68.end(), std::ref(f32rng));
  std::generate(v69.begin(), v69.end(), std::ref(f32rng));
  std::generate(v70.begin(), v70.end(), std::ref(f32rng));
  std::generate(v71.begin(), v71.end(), std::ref(f32rng));
  std::generate(v72.begin(), v72.end(), std::ref(f32rng));
  std::generate(v73.begin(), v73.end(), std::ref(f32rng));
  std::generate(v74.begin(), v74.end(), std::ref(f32rng));
  std::generate(v75.begin(), v75.end(), std::ref(f32rng));
  std::generate(v76.begin(), v76.end(), std::ref(f32rng));
  std::generate(v77.begin(), v77.end(), std::ref(f32rng));
  std::generate(v78.begin(), v78.end(), std::ref(f32rng));
  std::generate(v79.begin(), v79.end(), std::ref(f32rng));
  std::generate(v80.begin(), v80.end(), std::ref(f32rng));
  std::generate(v81.begin(), v81.end(), std::ref(f32rng));
  std::generate(v82.begin(), v82.end(), std::ref(f32rng));
  std::generate(v83.begin(), v83.end(), std::ref(f32rng));
  std::generate(v84.begin(), v84.end(), std::ref(f32rng));
  std::generate(v85.begin(), v85.end(), std::ref(f32rng));
  std::generate(v86.begin(), v86.end(), std::ref(f32rng));
  std::generate(v87.begin(), v87.end(), std::ref(f32rng));
  std::generate(v88.begin(), v88.end(), std::ref(f32rng));
  std::generate(v89.begin(), v89.end(), std::ref(f32rng));
  std::generate(v90.begin(), v90.end(), std::ref(f32rng));
  std::generate(v91.begin(), v91.end(), std::ref(f32rng));
  std::generate(v92.begin(), v92.end(), std::ref(f32rng));
  std::generate(v93.begin(), v93.end(), std::ref(f32rng));
  std::generate(v94.begin(), v94.end(), std::ref(f32rng));
  std::generate(v95.begin(), v95.end(), std::ref(f32rng));
  std::generate(v96.begin(), v96.end(), std::ref(f32rng));
  std::generate(v97.begin(), v97.end(), std::ref(f32rng));
  std::generate(v98.begin(), v98.end(), std::ref(f32rng));
  std::generate(v99.begin(), v99.end(), std::ref(f32rng));
  std::generate(w100.begin(), w100.end(), std::ref(f32rng));
  std::generate(w101.begin(), w101.end(), std::ref(f32rng));
  std::generate(w102.begin(), w102.end(), std::ref(f32rng));
  std::generate(w103.begin(), w103.end(), std::ref(f32rng));
  std::fill(w104.begin(), w104.end(), 0.0f);
  std::generate(w104.begin(), w104.end() - size_t(sparsity * w104.size()), std::ref(f32rng));
  std::shuffle(w104.begin(), w104.end(), rng);
  std::generate(w105.begin(), w105.end(), std::ref(f32rng));
  std::fill(w106.begin(), w106.end(), 0.0f);
  std::generate(w106.begin(), w106.end() - size_t(sparsity * w106.size()), std::ref(f32rng));
  std::shuffle(w106.begin(), w106.end(), rng);
  std::generate(w107.begin(), w107.end(), std::ref(f32rng));
  std::fill(w108.begin(), w108.end(), 0.0f);
  std::generate(w108.begin(), w108.end() - size_t(sparsity * w108.size()), std::ref(f32rng));
  std::shuffle(w108.begin(), w108.end(), rng);
  std::generate(w109.begin(), w109.end(), std::ref(f32rng));
  std::fill(w110.begin(), w110.end(), 0.0f);
  std::generate(w110.begin(), w110.end() - size_t(sparsity * w110.size()), std::ref(f32rng));
  std::shuffle(w110.begin(), w110.end(), rng);
  std::generate(w111.begin(), w111.end(), std::ref(f32rng));
  std::generate(w112.begin(), w112.end(), std::ref(f32rng));
  std::generate(w113.begin(), w113.end(), std::ref(f32rng));
  std::fill(w114.begin(), w114.end(), 0.0f);
  std::generate(w114.begin(), w114.end() - size_t(sparsity * w114.size()), std::ref(f32rng));
  std::shuffle(w114.begin(), w114.end(), rng);
  std::generate(w115.begin(), w115.end(), std::ref(f32rng));
  std::fill(w116.begin(), w116.end(), 0.0f);
  std::generate(w116.begin(), w116.end() - size_t(sparsity * w116.size()), std::ref(f32rng));
  std::shuffle(w116.begin(), w116.end(), rng);
  std::generate(w117.begin(), w117.end(), std::ref(f32rng));
  std::generate(w118.begin(), w118.end(), std::ref(f32rng));
  std::generate(w119.begin(), w119.end(), std::ref(f32rng));
  std::fill(w120.begin(), w120.end(), 0.0f);
  std::generate(w120.begin(), w120.end() - size_t(sparsity * w120.size()), std::ref(f32rng));
  std::shuffle(w120.begin(), w120.end(), rng);
  std::generate(w121.begin(), w121.end(), std::ref(f32rng));
  std::fill(w122.begin(), w122.end(), 0.0f);
  std::generate(w122.begin(), w122.end() - size_t(sparsity * w122.size()), std::ref(f32rng));
  std::shuffle(w122.begin(), w122.end(), rng);
  std::generate(w123.begin(), w123.end(), std::ref(f32rng));
  std::generate(w124.begin(), w124.end(), std::ref(f32rng));
  std::generate(w125.begin(), w125.end(), std::ref(f32rng));
  std::fill(w126.begin(), w126.end(), 0.0f);
  std::generate(w126.begin(), w126.end() - size_t(sparsity * w126.size()), std::ref(f32rng));
  std::shuffle(w126.begin(), w126.end(), rng);
  std::generate(w127.begin(), w127.end(), std::ref(f32rng));
  std::fill(w128.begin(), w128.end(), 0.0f);
  std::generate(w128.begin(), w128.end() - size_t(sparsity * w128.size()), std::ref(f32rng));
  std::shuffle(w128.begin(), w128.end(), rng);
  std::generate(w129.begin(), w129.end(), std::ref(f32rng));
  std::fill(w130.begin(), w130.end(), 0.0f);
  std::generate(w130.begin(), w130.end() - size_t(sparsity * w130.size()), std::ref(f32rng));
  std::shuffle(w130.begin(), w130.end(), rng);
  std::generate(w131.begin(), w131.end(), std::ref(f32rng));
  std::fill(w132.begin(), w132.end(), 0.0f);
  std::generate(w132.begin(), w132.end() - size_t(sparsity * w132.size()), std::ref(f32rng));
  std::shuffle(w132.begin(), w132.end(), rng);
  std::generate(w133.begin(), w133.end(), std::ref(f32rng));
  std::generate(w134.begin(), w134.end(), std::ref(f32rng));
  std::generate(w135.begin(), w135.end(), std::ref(f32rng));
  std::fill(w136.begin(), w136.end(), 0.0f);
  std::generate(w136.begin(), w136.end() - size_t(sparsity * w136.size()), std::ref(f32rng));
  std::shuffle(w136.begin(), w136.end(), rng);
  std::generate(w137.begin(), w137.end(), std::ref(f32rng));
  std::fill(w138.begin(), w138.end(), 0.0f);
  std::generate(w138.begin(), w138.end() - size_t(sparsity * w138.size()), std::ref(f32rng));
  std::shuffle(w138.begin(), w138.end(), rng);
  std::generate(w139.begin(), w139.end(), std::ref(f32rng));
  std::fill(w140.begin(), w140.end(), 0.0f);
  std::generate(w140.begin(), w140.end() - size_t(sparsity * w140.size()), std::ref(f32rng));
  std::shuffle(w140.begin(), w140.end(), rng);
  std::generate(w141.begin(), w141.end(), std::ref(f32rng));
  std::fill(w142.begin(), w142.end(), 0.0f);
  std::generate(w142.begin(), w142.end() - size_t(sparsity * w142.size()), std::ref(f32rng));
  std::shuffle(w142.begin(), w142.end(), rng);
  std::generate(w143.begin(), w143.end(), std::ref(f32rng));
  std::generate(w144.begin(), w144.end(), std::ref(f32rng));
  std::generate(w145.begin(), w145.end(), std::ref(f32rng));
  std::fill(w146.begin(), w146.end(), 0.0f);
  std::generate(w146.begin(), w146.end() - size_t(sparsity * w146.size()), std::ref(f32rng));
  std::shuffle(w146.begin(), w146.end(), rng);
  std::generate(w147.begin(), w147.end(), std::ref(f32rng));
  std::fill(w148.begin(), w148.end(), 0.0f);
  std::generate(w148.begin(), w148.end() - size_t(sparsity * w148.size()), std::ref(f32rng));
  std::shuffle(w148.begin(), w148.end(), rng);
  std::generate(w149.begin(), w149.end(), std::ref(f32rng));
  std::fill(w150.begin(), w150.end(), 0.0f);
  std::generate(w150.begin(), w150.end() - size_t(sparsity * w150.size()), std::ref(f32rng));
  std::shuffle(w150.begin(), w150.end(), rng);
  std::generate(w151.begin(), w151.end(), std::ref(f32rng));
  std::fill(w152.begin(), w152.end(), 0.0f);
  std::generate(w152.begin(), w152.end() - size_t(sparsity * w152.size()), std::ref(f32rng));
  std::shuffle(w152.begin(), w152.end(), rng);
  std::generate(w153.begin(), w153.end(), std::ref(f32rng));
  std::generate(w154.begin(), w154.end(), std::ref(f32rng));
  std::generate(w155.begin(), w155.end(), std::ref(f32rng));
  std::fill(w156.begin(), w156.end(), 0.0f);
  std::generate(w156.begin(), w156.end() - size_t(sparsity * w156.size()), std::ref(f32rng));
  std::shuffle(w156.begin(), w156.end(), rng);
  std::generate(w157.begin(), w157.end(), std::ref(f32rng));
  std::fill(w158.begin(), w158.end(), 0.0f);
  std::generate(w158.begin(), w158.end() - size_t(sparsity * w158.size()), std::ref(f32rng));
  std::shuffle(w158.begin(), w158.end(), rng);
  std::generate(w159.begin(), w159.end(), std::ref(f32rng));
  std::fill(w160.begin(), w160.end(), 0.0f);
  std::generate(w160.begin(), w160.end() - size_t(sparsity * w160.size()), std::ref(f32rng));
  std::shuffle(w160.begin(), w160.end(), rng);
  std::generate(w161.begin(), w161.end(), std::ref(f32rng));
  std::fill(w162.begin(), w162.end(), 0.0f);
  std::generate(w162.begin(), w162.end() - size_t(sparsity * w162.size()), std::ref(f32rng));
  std::shuffle(w162.begin(), w162.end(), rng);
  std::generate(w163.begin(), w163.end(), std::ref(f32rng));
  std::generate(w164.begin(), w164.end(), std::ref(f32rng));
  std::generate(w165.begin(), w165.end(), std::ref(f32rng));
  std::fill(w166.begin(), w166.end(), 0.0f);
  std::generate(w166.begin(), w166.end() - size_t(sparsity * w166.size()), std::ref(f32rng));
  std::shuffle(w166.begin(), w166.end(), rng);
  std::generate(w167.begin(), w167.end(), std::ref(f32rng));
  std::fill(w168.begin(), w168.end(), 0.0f);
  std::generate(w168.begin(), w168.end() - size_t(sparsity * w168.size()), std::ref(f32rng));
  std::shuffle(w168.begin(), w168.end(), rng);
  std::generate(w169.begin(), w169.end(), std::ref(f32rng));
  std::fill(w170.begin(), w170.end(), 0.0f);
  std::generate(w170.begin(), w170.end() - size_t(sparsity * w170.size()), std::ref(f32rng));
  std::shuffle(w170.begin(), w170.end(), rng);
  std::generate(w171.begin(), w171.end(), std::ref(f32rng));
  std::fill(w172.begin(), w172.end(), 0.0f);
  std::generate(w172.begin(), w172.end() - size_t(sparsity * w172.size()), std::ref(f32rng));
  std::shuffle(w172.begin(), w172.end(), rng);
  std::generate(w173.begin(), w173.end(), std::ref(f32rng));
  std::generate(w174.begin(), w174.end(), std::ref(f32rng));
  std::generate(w175.begin(), w175.end(), std::ref(f32rng));
  std::fill(w176.begin(), w176.end(), 0.0f);
  std::generate(w176.begin(), w176.end() - size_t(sparsity * w176.size()), std::ref(f32rng));
  std::shuffle(w176.begin(), w176.end(), rng);
  std::generate(w177.begin(), w177.end(), std::ref(f32rng));
  std::fill(w178.begin(), w178.end(), 0.0f);
  std::generate(w178.begin(), w178.end() - size_t(sparsity * w178.size()), std::ref(f32rng));
  std::shuffle(w178.begin(), w178.end(), rng);
  std::generate(w179.begin(), w179.end(), std::ref(f32rng));
  std::fill(w180.begin(), w180.end(), 0.0f);
  std::generate(w180.begin(), w180.end() - size_t(sparsity * w180.size()), std::ref(f32rng));
  std::shuffle(w180.begin(), w180.end(), rng);
  std::generate(w181.begin(), w181.end(), std::ref(f32rng));
  std::fill(w182.begin(), w182.end(), 0.0f);
  std::generate(w182.begin(), w182.end() - size_t(sparsity * w182.size()), std::ref(f32rng));
  std::shuffle(w182.begin(), w182.end(), rng);
  std::generate(w183.begin(), w183.end(), std::ref(f32rng));
  std::generate(w184.begin(), w184.end(), std::ref(f32rng));
  std::generate(w185.begin(), w185.end(), std::ref(f32rng));
  std::fill(w186.begin(), w186.end(), 0.0f);
  std::generate(w186.begin(), w186.end() - size_t(sparsity * w186.size()), std::ref(f32rng));
  std::shuffle(w186.begin(), w186.end(), rng);
  std::generate(w187.begin(), w187.end(), std::ref(f32rng));
  std::fill(w188.begin(), w188.end(), 0.0f);
  std::generate(w188.begin(), w188.end() - size_t(sparsity * w188.size()), std::ref(f32rng));
  std::shuffle(w188.begin(), w188.end(), rng);
  std::generate(w189.begin(), w189.end(), std::ref(f32rng));
  std::fill(w190.begin(), w190.end(), 0.0f);
  std::generate(w190.begin(), w190.end() - size_t(sparsity * w190.size()), std::ref(f32rng));
  std::shuffle(w190.begin(), w190.end(), rng);
  std::generate(w191.begin(), w191.end(), std::ref(f32rng));
  std::fill(w192.begin(), w192.end(), 0.0f);
  std::generate(w192.begin(), w192.end() - size_t(sparsity * w192.size()), std::ref(f32rng));
  std::shuffle(w192.begin(), w192.end(), rng);
  std::generate(w193.begin(), w193.end(), std::ref(f32rng));
  std::generate(w194.begin(), w194.end(), std::ref(f32rng));
  std::generate(w195.begin(), w195.end(), std::ref(f32rng));
  std::fill(w196.begin(), w196.end(), 0.0f);
  std::generate(w196.begin(), w196.end() - size_t(sparsity * w196.size()), std::ref(f32rng));
  std::shuffle(w196.begin(), w196.end(), rng);
  std::generate(w197.begin(), w197.end(), std::ref(f32rng));
  std::fill(w198.begin(), w198.end(), 0.0f);
  std::generate(w198.begin(), w198.end() - size_t(sparsity * w198.size()), std::ref(f32rng));
  std::shuffle(w198.begin(), w198.end(), rng);
  std::generate(w199.begin(), w199.end(), std::ref(f32rng));
  std::fill(w200.begin(), w200.end(), 0.0f);
  std::generate(w200.begin(), w200.end() - size_t(sparsity * w200.size()), std::ref(f32rng));
  std::shuffle(w200.begin(), w200.end(), rng);
  std::generate(w201.begin(), w201.end(), std::ref(f32rng));
  std::fill(w202.begin(), w202.end(), 0.0f);
  std::generate(w202.begin(), w202.end() - size_t(sparsity * w202.size()), std::ref(f32rng));
  std::shuffle(w202.begin(), w202.end(), rng);
  std::generate(w203.begin(), w203.end(), std::ref(f32rng));
  std::fill(w204.begin(), w204.end(), 0.0f);
  std::generate(w204.begin(), w204.end() - size_t(sparsity * w204.size()), std::ref(f32rng));
  std::shuffle(w204.begin(), w204.end(), rng);
  std::generate(w205.begin(), w205.end(), std::ref(f32rng));
  std::generate(w206.begin(), w206.end(), std::ref(f32rng));
  std::generate(w207.begin(), w207.end(), std::ref(f32rng));

  ExecutionPlan operators;
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
    16 /* output_channels_per_group */,
    3 /* input pixel stride */,
    16 /* output pixel stride */,
    w100.data(), w101.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    XNN_FLAG_INPUT_NHWC /* flags */,
    &op0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #0" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op0, xnn_delete_operator);

  xnn_operator_t op1 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    16 /* channels */,
    16 /* input stride */,
    16 /* output stride */,
    0 /* flags */,
    &op1);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #1" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op1, xnn_delete_operator);

  xnn_operator_t op2 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
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
    &op2);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #2" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op2, xnn_delete_operator);

  xnn_operator_t op3 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    16 /* channels */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op3);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #3" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op3, xnn_delete_operator);

  xnn_operator_t op4 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    8 /* input channels per group */,
    16 /* output_channels_per_group */,
    8 /* input pixel stride */,
    16 /* output pixel stride */,
    w106.data(), w107.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
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
  status = xnn_create_convolution2d_nchw_f32(
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
    16 /* input channels per group */,
    72 /* output_channels_per_group */,
    16 /* input pixel stride */,
    72 /* output pixel stride */,
    w110.data(), w111.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op8);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #8" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op8, xnn_delete_operator);

  xnn_operator_t op9 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
    1 /* top padding */, 1 /* right padding */,
    1 /* bottom padding */, 1 /* left padding */,
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
    72 /* input channels per group */,
    24 /* output_channels_per_group */,
    72 /* input pixel stride */,
    24 /* output pixel stride */,
    w114.data(), w115.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op10);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #10" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op10, xnn_delete_operator);

  xnn_operator_t op11 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    &op11);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #11" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op11, xnn_delete_operator);

  xnn_operator_t op12 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    88 /* input channels per group */,
    24 /* output_channels_per_group */,
    88 /* input pixel stride */,
    24 /* output pixel stride */,
    w120.data(), w121.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
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
  status = xnn_create_convolution2d_nchw_f32(
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
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op15);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #15" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op15, xnn_delete_operator);

  xnn_operator_t op16 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    96 /* channels */,
    96 /* input stride */,
    96 /* output stride */,
    0 /* flags */,
    &op16);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #16" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op16, xnn_delete_operator);

  xnn_operator_t op17 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
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
    &op17);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #17" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op17, xnn_delete_operator);

  xnn_operator_t op18 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    96 /* channels */,
    96 /* input stride */,
    96 /* output stride */,
    0 /* flags */,
    &op18);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #18" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op18, xnn_delete_operator);

  xnn_operator_t op19 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    96 /* channels */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op19);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #19" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op19, xnn_delete_operator);

  xnn_operator_t op20 = nullptr;
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
    w126.data(), w127.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
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
    24 /* input channels per group */,
    96 /* output_channels_per_group */,
    24 /* input pixel stride */,
    96 /* output pixel stride */,
    w128.data(), w129.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
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
  status = xnn_create_convolution2d_nchw_f32(
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
    40 /* input channels per group */,
    240 /* output_channels_per_group */,
    40 /* input pixel stride */,
    240 /* output pixel stride */,
    w132.data(), w133.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op24);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #24" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op24, xnn_delete_operator);

  xnn_operator_t op25 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    0 /* flags */,
    &op25);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #25" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op25, xnn_delete_operator);

  xnn_operator_t op26 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    &op26);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #26" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op26, xnn_delete_operator);

  xnn_operator_t op27 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    0 /* flags */,
    &op27);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #27" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op27, xnn_delete_operator);

  xnn_operator_t op28 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    240 /* channels */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op28);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #28" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op28, xnn_delete_operator);

  xnn_operator_t op29 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    64 /* input channels per group */,
    240 /* output_channels_per_group */,
    64 /* input pixel stride */,
    240 /* output pixel stride */,
    w138.data(), w139.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
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
  status = xnn_create_convolution2d_nchw_f32(
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
  status = xnn_create_convolution2d_nchw_f32(
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
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op34);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #34" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op34, xnn_delete_operator);

  xnn_operator_t op35 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    0 /* flags */,
    &op35);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #35" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op35, xnn_delete_operator);

  xnn_operator_t op36 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    &op36);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #36" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op36, xnn_delete_operator);

  xnn_operator_t op37 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    240 /* channels */,
    240 /* input stride */,
    240 /* output stride */,
    0 /* flags */,
    &op37);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #37" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op37, xnn_delete_operator);

  xnn_operator_t op38 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    240 /* channels */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
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
    240 /* input channels per group */,
    64 /* output_channels_per_group */,
    240 /* input pixel stride */,
    64 /* output pixel stride */,
    w146.data(), w147.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op39);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #39" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op39, xnn_delete_operator);

  xnn_operator_t op40 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
  status = xnn_create_convolution2d_nchw_f32(
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
  status = xnn_create_convolution2d_nchw_f32(
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
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op44);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #44" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op44, xnn_delete_operator);

  xnn_operator_t op45 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    120 /* channels */,
    120 /* input stride */,
    120 /* output stride */,
    0 /* flags */,
    &op45);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #45" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op45, xnn_delete_operator);

  xnn_operator_t op46 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    &op46);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #46" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op46, xnn_delete_operator);

  xnn_operator_t op47 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    120 /* channels */,
    120 /* input stride */,
    120 /* output stride */,
    0 /* flags */,
    &op47);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #47" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op47, xnn_delete_operator);

  xnn_operator_t op48 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    120 /* channels */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
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
    120 /* input channels per group */,
    32 /* output_channels_per_group */,
    120 /* input pixel stride */,
    32 /* output pixel stride */,
    w156.data(), w157.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
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
    32 /* input channels per group */,
    120 /* output_channels_per_group */,
    32 /* input pixel stride */,
    120 /* output pixel stride */,
    w158.data(), w159.data(),
    0.0f /* output min */, +0x1.00014Fp+0 /* output max */,
    0 /* flags */,
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
  status = xnn_create_convolution2d_nchw_f32(
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
    &op52);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #52" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op52, xnn_delete_operator);

  xnn_operator_t op53 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op53);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #53" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op53, xnn_delete_operator);

  xnn_operator_t op54 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    144 /* channels */,
    144 /* input stride */,
    144 /* output stride */,
    0 /* flags */,
    &op54);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #54" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op54, xnn_delete_operator);

  xnn_operator_t op55 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    &op55);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #55" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op55, xnn_delete_operator);

  xnn_operator_t op56 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    144 /* channels */,
    144 /* input stride */,
    144 /* output stride */,
    0 /* flags */,
    &op56);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #56" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op56, xnn_delete_operator);

  xnn_operator_t op57 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    144 /* channels */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
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
    144 /* input channels per group */,
    40 /* output_channels_per_group */,
    144 /* input pixel stride */,
    40 /* output pixel stride */,
    w166.data(), w167.data(),
    0.0f /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op58);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #58" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op58, xnn_delete_operator);

  xnn_operator_t op59 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
  status = xnn_create_convolution2d_nchw_f32(
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
  status = xnn_create_convolution2d_nchw_f32(
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
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op63);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #63" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op63, xnn_delete_operator);

  xnn_operator_t op64 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    288 /* channels */,
    288 /* input stride */,
    288 /* output stride */,
    0 /* flags */,
    &op64);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #64" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op64, xnn_delete_operator);

  xnn_operator_t op65 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
    2 /* top padding */, 2 /* right padding */,
    2 /* bottom padding */, 2 /* left padding */,
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
    &op65);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #65" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op65, xnn_delete_operator);

  xnn_operator_t op66 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    288 /* channels */,
    288 /* input stride */,
    288 /* output stride */,
    0 /* flags */,
    &op66);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #66" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op66, xnn_delete_operator);

  xnn_operator_t op67 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    288 /* channels */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op67);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #67" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op67, xnn_delete_operator);

  xnn_operator_t op68 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    &op68);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #68" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op68, xnn_delete_operator);

  xnn_operator_t op69 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
  status = xnn_create_convolution2d_nchw_f32(
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
    &op71);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #71" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op71, xnn_delete_operator);

  xnn_operator_t op72 = nullptr;
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
    w182.data(), w183.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op72);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #72" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op72, xnn_delete_operator);

  xnn_operator_t op73 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    0 /* flags */,
    &op73);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #73" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op73, xnn_delete_operator);

  xnn_operator_t op74 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    &op74);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #74" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op74, xnn_delete_operator);

  xnn_operator_t op75 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    0 /* flags */,
    &op75);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #75" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op75, xnn_delete_operator);

  xnn_operator_t op76 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    576 /* channels */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op76);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #76" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op76, xnn_delete_operator);

  xnn_operator_t op77 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    &op77);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #77" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op77, xnn_delete_operator);

  xnn_operator_t op78 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    w190.data(), w191.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
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
    w192.data(), w193.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op82);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #82" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op82, xnn_delete_operator);

  xnn_operator_t op83 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    0 /* flags */,
    &op83);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #83" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op83, xnn_delete_operator);

  xnn_operator_t op84 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    &op84);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #84" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op84, xnn_delete_operator);

  xnn_operator_t op85 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    0 /* flags */,
    &op85);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #85" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op85, xnn_delete_operator);

  xnn_operator_t op86 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    576 /* channels */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op86);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #86" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op86, xnn_delete_operator);

  xnn_operator_t op87 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    &op87);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #87" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op87, xnn_delete_operator);

  xnn_operator_t op88 = nullptr;
  status = xnn_create_convolution2d_nchw_f32(
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
    w200.data(), w201.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
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
    w202.data(), w203.data(),
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op92);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #92" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op92, xnn_delete_operator);

  xnn_operator_t op93 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    576 /* channels */,
    576 /* input stride */,
    576 /* output stride */,
    0 /* flags */,
    &op93);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #93" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op93, xnn_delete_operator);

  xnn_operator_t op94 = nullptr;
  status = xnn_create_global_average_pooling_ncw_f32(
    576 /* channels */,
    -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
    0 /* flags */,
    &op94);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #94" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op94, xnn_delete_operator);

  xnn_operator_t op95 = nullptr;
  status = xnn_create_convolution2d_nhwc_f32(
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
    -std::numeric_limits<float>::infinity() /* output min */, std::numeric_limits<float>::infinity() /* output max */,
    0 /* flags */,
    &op95);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #95" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op95, xnn_delete_operator);

  xnn_operator_t op96 = nullptr;
  status = xnn_create_hardswish_nc_f32(
    1024 /* channels */,
    1024 /* input stride */,
    1024 /* output stride */,
    0 /* flags */,
    &op96);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #96" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op96, xnn_delete_operator);

  xnn_operator_t op97 = nullptr;
  status = xnn_create_global_average_pooling_nwc_f32(
    1024 /* channels */, 1024 /* input stride */, 1024 /* output stride */,
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
    &op98);
  if (status != xnn_status_success) {
    std::cerr << "failed to create operation #98" << std::endl;
    return ExecutionPlan();
  }
  operators.emplace_back(op98, xnn_delete_operator);



  status = xnn_setup_convolution2d_nchw_f32(
    op0,
    1 /* batch size */, 224 /* input height */, 224 /* input width */,
    v0.data() /* input */, v1.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #0" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op1,
    12544 /* batch size */,
    v1.data() /* input */, v2.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #1" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op2,
    1 /* batch size */, 112 /* input height */, 112 /* input width */,
    v2.data() /* input */, v3.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #2" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op3,
    1 /* batch size */, 3136 /* width */,
    v3.data() /* input */, v4.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #3" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op4,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v4.data() /* input */, v5.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #4" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op5,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v5.data() /* input */, v6.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #5" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 16, 56, 56 };
    const size_t b_shape[] = { 1, 16, 1, 1 };
    status = xnn_setup_multiply_nd_f32(
      op6,
      4, a_shape, 4, b_shape,
      v3.data() /* a */, v6.data() /* b */, v7.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #6" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op7,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v7.data() /* input */, v8.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #7" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op8,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v8.data() /* input */, v9.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #8" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op9,
    1 /* batch size */, 56 /* input height */, 56 /* input width */,
    v9.data() /* input */, v10.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #9" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op10,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v10.data() /* input */, v11.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #10" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op11,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v11.data() /* input */, v12.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #11" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op12,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v12.data() /* input */, v13.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #12" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op13,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v13.data() /* input */, v14.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #13" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 24, 28, 28 };
    const size_t b_shape[] = { 1, 24, 28, 28 };
    status = xnn_setup_add_nd_f32(
      op14,
      4, a_shape, 4, b_shape,
      v14.data() /* a */, v11.data() /* b */, v15.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #14" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op15,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v15.data() /* input */, v16.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #15" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op16,
    784 /* batch size */,
    v16.data() /* input */, v17.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #16" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op17,
    1 /* batch size */, 28 /* input height */, 28 /* input width */,
    v17.data() /* input */, v18.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #17" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op18,
    196 /* batch size */,
    v18.data() /* input */, v19.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #18" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op19,
    1 /* batch size */, 196 /* width */,
    v19.data() /* input */, v20.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #19" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op20,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v20.data() /* input */, v21.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #20" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op21,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v21.data() /* input */, v22.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #21" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 96, 14, 14 };
    const size_t b_shape[] = { 1, 96, 1, 1 };
    status = xnn_setup_multiply_nd_f32(
      op22,
      4, a_shape, 4, b_shape,
      v19.data() /* a */, v22.data() /* b */, v23.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #22" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op23,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v23.data() /* input */, v24.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #23" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op24,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v24.data() /* input */, v25.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #24" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op25,
    196 /* batch size */,
    v25.data() /* input */, v26.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #25" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op26,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v26.data() /* input */, v27.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #26" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op27,
    196 /* batch size */,
    v27.data() /* input */, v28.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #27" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op28,
    1 /* batch size */, 196 /* width */,
    v28.data() /* input */, v29.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #28" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op29,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v29.data() /* input */, v30.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #29" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op30,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v30.data() /* input */, v31.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #30" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 240, 14, 14 };
    const size_t b_shape[] = { 1, 240, 1, 1 };
    status = xnn_setup_multiply_nd_f32(
      op31,
      4, a_shape, 4, b_shape,
      v28.data() /* a */, v31.data() /* b */, v32.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #31" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op32,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v32.data() /* input */, v33.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #32" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 40, 14, 14 };
    const size_t b_shape[] = { 1, 40, 14, 14 };
    status = xnn_setup_add_nd_f32(
      op33,
      4, a_shape, 4, b_shape,
      v33.data() /* a */, v24.data() /* b */, v34.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #33" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op34,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v34.data() /* input */, v35.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #34" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op35,
    196 /* batch size */,
    v35.data() /* input */, v36.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #35" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op36,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v36.data() /* input */, v37.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #36" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op37,
    196 /* batch size */,
    v37.data() /* input */, v38.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #37" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op38,
    1 /* batch size */, 196 /* width */,
    v38.data() /* input */, v39.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #38" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op39,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v39.data() /* input */, v40.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #39" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op40,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v40.data() /* input */, v41.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #40" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 240, 14, 14 };
    const size_t b_shape[] = { 1, 240, 1, 1 };
    status = xnn_setup_multiply_nd_f32(
      op41,
      4, a_shape, 4, b_shape,
      v38.data() /* a */, v41.data() /* b */, v42.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #41" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op42,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v42.data() /* input */, v43.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #42" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 40, 14, 14 };
    const size_t b_shape[] = { 1, 40, 14, 14 };
    status = xnn_setup_add_nd_f32(
      op43,
      4, a_shape, 4, b_shape,
      v43.data() /* a */, v34.data() /* b */, v44.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #43" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op44,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v44.data() /* input */, v45.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #44" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op45,
    196 /* batch size */,
    v45.data() /* input */, v46.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #45" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op46,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v46.data() /* input */, v47.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #46" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op47,
    196 /* batch size */,
    v47.data() /* input */, v48.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #47" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op48,
    1 /* batch size */, 196 /* width */,
    v48.data() /* input */, v49.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #48" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op49,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v49.data() /* input */, v50.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #49" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op50,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v50.data() /* input */, v51.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #50" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 120, 14, 14 };
    const size_t b_shape[] = { 1, 120, 1, 1 };
    status = xnn_setup_multiply_nd_f32(
      op51,
      4, a_shape, 4, b_shape,
      v48.data() /* a */, v51.data() /* b */, v52.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #51" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op52,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v52.data() /* input */, v53.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #52" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op53,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v53.data() /* input */, v54.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #53" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op54,
    196 /* batch size */,
    v54.data() /* input */, v55.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #54" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op55,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v55.data() /* input */, v56.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #55" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op56,
    196 /* batch size */,
    v56.data() /* input */, v57.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #56" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op57,
    1 /* batch size */, 196 /* width */,
    v57.data() /* input */, v58.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #57" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op58,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v58.data() /* input */, v59.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #58" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op59,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v59.data() /* input */, v60.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #59" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 144, 14, 14 };
    const size_t b_shape[] = { 1, 144, 1, 1 };
    status = xnn_setup_multiply_nd_f32(
      op60,
      4, a_shape, 4, b_shape,
      v57.data() /* a */, v60.data() /* b */, v61.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #60" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op61,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v61.data() /* input */, v62.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #61" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 48, 14, 14 };
    const size_t b_shape[] = { 1, 48, 14, 14 };
    status = xnn_setup_add_nd_f32(
      op62,
      4, a_shape, 4, b_shape,
      v62.data() /* a */, v53.data() /* b */, v63.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #62" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op63,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v63.data() /* input */, v64.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #63" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op64,
    196 /* batch size */,
    v64.data() /* input */, v65.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #64" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op65,
    1 /* batch size */, 14 /* input height */, 14 /* input width */,
    v65.data() /* input */, v66.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #65" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op66,
    49 /* batch size */,
    v66.data() /* input */, v67.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #66" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op67,
    1 /* batch size */, 49 /* width */,
    v67.data() /* input */, v68.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #67" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op68,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v68.data() /* input */, v69.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #68" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op69,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v69.data() /* input */, v70.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #69" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 288, 7, 7 };
    const size_t b_shape[] = { 1, 288, 1, 1 };
    status = xnn_setup_multiply_nd_f32(
      op70,
      4, a_shape, 4, b_shape,
      v67.data() /* a */, v70.data() /* b */, v71.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #70" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op71,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v71.data() /* input */, v72.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #71" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op72,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v72.data() /* input */, v73.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #72" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op73,
    49 /* batch size */,
    v73.data() /* input */, v74.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #73" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op74,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v74.data() /* input */, v75.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #74" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op75,
    49 /* batch size */,
    v75.data() /* input */, v76.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #75" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op76,
    1 /* batch size */, 49 /* width */,
    v76.data() /* input */, v77.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #76" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op77,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v77.data() /* input */, v78.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #77" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op78,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v78.data() /* input */, v79.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #78" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 576, 7, 7 };
    const size_t b_shape[] = { 1, 576, 1, 1 };
    status = xnn_setup_multiply_nd_f32(
      op79,
      4, a_shape, 4, b_shape,
      v76.data() /* a */, v79.data() /* b */, v80.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #79" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op80,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v80.data() /* input */, v81.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #80" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 96, 7, 7 };
    const size_t b_shape[] = { 1, 96, 7, 7 };
    status = xnn_setup_add_nd_f32(
      op81,
      4, a_shape, 4, b_shape,
      v81.data() /* a */, v72.data() /* b */, v82.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #81" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op82,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v82.data() /* input */, v83.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #82" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op83,
    49 /* batch size */,
    v83.data() /* input */, v84.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #83" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op84,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v84.data() /* input */, v85.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #84" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op85,
    49 /* batch size */,
    v85.data() /* input */, v86.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #85" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op86,
    1 /* batch size */, 49 /* width */,
    v86.data() /* input */, v87.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #86" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op87,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v87.data() /* input */, v88.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #87" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op88,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v88.data() /* input */, v89.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #88" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 576, 7, 7 };
    const size_t b_shape[] = { 1, 576, 1, 1 };
    status = xnn_setup_multiply_nd_f32(
      op89,
      4, a_shape, 4, b_shape,
      v86.data() /* a */, v89.data() /* b */, v90.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #89" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op90,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v90.data() /* input */, v91.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #90" << std::endl;
    return ExecutionPlan();
  }

  {
    const size_t a_shape[] = { 1, 96, 7, 7 };
    const size_t b_shape[] = { 1, 96, 7, 7 };
    status = xnn_setup_add_nd_f32(
      op91,
      4, a_shape, 4, b_shape,
      v91.data() /* a */, v82.data() /* b */, v92.data() /* output */,
      threadpool /* threadpool */);
  }
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #91" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nchw_f32(
    op92,
    1 /* batch size */, 7 /* input height */, 7 /* input width */,
    v92.data() /* input */, v93.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #92" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op93,
    49 /* batch size */,
    v93.data() /* input */, v94.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #93" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_ncw_f32(
    op94,
    1 /* batch size */, 49 /* width */,
    v94.data() /* input */, v95.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #94" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op95,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v95.data() /* input */, v96.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #95" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_hardswish_nc_f32(
    op96,
    1 /* batch size */,
    v96.data() /* input */, v97.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #96" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_global_average_pooling_nwc_f32(
    op97,
    1 /* batch size */, 1 /* width */,
    v97.data() /* input */, v98.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #97" << std::endl;
    return ExecutionPlan();
  }

  status = xnn_setup_convolution2d_nhwc_f32(
    op98,
    1 /* batch size */, 1 /* input height */, 1 /* input width */,
    v98.data() /* input */, v99.data() /* output */,
    threadpool /* threadpool */);
  if (status != xnn_status_success) {
    std::cerr << "failed to setup operation #98" << std::endl;
    return ExecutionPlan();
  }

  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wpessimizing-move"
  return operators;
  #pragma clang diagnostic pop
}

}  // namespace models
