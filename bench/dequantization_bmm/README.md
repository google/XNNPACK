<!-- Copyright 2026 Google LLC.
     This source code is licensed under the BSD-style license found in the
     LICENSE file in the root directory of this source tree. -->

# Test plan: choosing whether to fuse INT8 dequantization into FP32 BMM

The goal is to collect enough comparable measurements across Android phones and
Linux x86-64 machines to design a per-BMM selection heuristic. This directory is
self-contained: it needs XNNPACK and its normal build dependencies, but no LiteRT
checkout, model weights, or files from the original experiment machine.

XNNPACK provides the node flag `XNN_FLAG_NO_BMM_DEQUANTIZATION_FUSION`. It provides explicit
control for this experiment; **it does not implement an automatic heuristic**.
Do not interpret a successful build or a fast smoke run as validation of a
production policy.

## 1. What the experiment compares

Both alternatives receive the same external FP32 A and symmetric INT8 B, with a
scalar scale and zero point zero. B is a mutable runtime input, not a constant
weight that can be packed once at model load time.

| Alternative | Graph lowering | Probe output |
|---|---|---|
| Default rewrite | INT8-to-QC8 adapter, then mixed FP32/QC8W BMM | `rewrite=1` |
| Opt out on this BMM | Separate INT8-to-FP32 conversion, then FP32 BMM | `rewrite=0` |

Only the new node flag differs. All other optimizations stay enabled, including
FP32 packing where the backend supports it. The mixed path does not dynamically
quantize A and does not use the INT4 FC/I8MM path. These are floating-point
matrix multiplications with differently represented right operands.

The primary measurement is the complete graph call, including repeated reshape,
setup, conversion/adapter work, packing and invocation. The probe also measures
invocation alone after setup. It does not measure softmax, masking, projections,
KV updates, sampling, or whole-model TTFT. **Do not label these numbers as model
prefill tokens/s or decode tokens/s.**

Each process constructs both alternatives, validates their dispatch and outputs,
and measures alternating on/off intervals. Separate process repetitions reverse
construction/initial measurement order and shuffle case order. The four warm
intervals inside one process are not four independent experiments.

## 2. Building the probe

The probe inspects private dispatch structs. Build it and XNNPACK together with
matching feature definitions, as the supplied build targets do; do not link a
separately configured system XNNPACK library into this diagnostic executable.
All commands start at the repository root.

### Bazel

```sh
# Host build
bazel build //bench/dequantization_bmm:probe

# Android arm64 build
bazel build --config=android_arm64 //bench/dequantization_bmm:probe
```

### Standalone CMake wrapper

```sh
# Host build
cmake -S bench/dequantization_bmm -B build/dequant-bmm-host -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/dequant-bmm-host --target dequant_bmm_probe

# Android arm64 build (requires ANDROID_NDK set)
cmake -S bench/dequantization_bmm -B build/dequant-bmm-android -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DANDROID_STL=c++_static
cmake --build build/dequant-bmm-android --target dequant_bmm_probe
```

These commands build only the probe and production dependencies. Keep the binary
unstripped so the report can resolve kernel offsets. Use the same binary for
both alternatives and, where practical, across devices of the same architecture.
Record build differences when different compilers or feature flags are necessary.
Do not add `-march=native` to a binary intended for other x86-64 machines.

## 3. Correctness gate

First test the analysis scripts:

```sh
python3 -m unittest discover -s bench/dequantization_bmm -p 'test_*.py'
```

To run the BMM regression suite:

```sh
# Bazel:
bazel test //test/subgraph:batch_matrix_multiply_test

# Or standalone CMake:
cmake -S bench/dequantization_bmm -B build/dequant-bmm-host -DXNNPACK_BUILD_TESTS=ON
cmake --build build/dequant-bmm-host --target batch-matrix-multiply-test
build/dequant-bmm-host/xnnpack/test/subgraph/batch-matrix-multiply-test
```

For Android, configure the Android CMake build the same way (`-DXNNPACK_BUILD_TESTS=ON`),
then deploy and run `batch-matrix-multiply-test`:

```sh
adb -s "$SERIAL" push \
  build/dequant-bmm-android/xnnpack/test/subgraph/batch-matrix-multiply-test \
  /data/local/tmp/bmm-dequant-test
adb -s "$SERIAL" shell /data/local/tmp/bmm-dequant-test
```

Set `SERIAL` to your device's adb serial before Android commands. Save the test
output with the results. This suite checks independent decisions within one
graph, both transpose layouts, broadcast batches, shape growth/shrinkage, and
power-of-two and non-power-of-two scales.

Every timing process also checks the selected BMM type, scale-buffer size, up to
256 FP64-reference output samples, and the difference between both paths over
all output elements. The reference first rounds dequantization to FP32, then
accumulates in FP64. Checks use `1e-4 + 5e-5 * abs(reference)`; path-to-path checks
use the larger output magnitude. Failures terminate that process and must be
investigated, not averaged into performance data.

**These checks establish synthetic operator correctness, not equal model
quality.** Real-scale whole-model experiments previously produced different
logits when the rewrite was disabled. A production policy still needs model
quality checks; identical results with scale 1/32 are insufficient.

## 4. Choose CPU placement deliberately

Choose CPU IDs using OS topology and maximum-frequency information. The runner
requires explicit `--cpus`; it does not guess which cores are big or little.
CPU IDs are not portable across devices.

On Linux, inspect `lscpu -e` and avoid counting SMT siblings as separate physical
cores in the initial study. On Android, inspect cpufreq policies' `related_cpus`
and `cpuinfo_max_freq`. The runner archives these, CPU capacity/topology when
available, and raw cache descriptions. Do not sum repeated descriptions of a
shared cache. Do not treat an aggregate CPUINFO L2 figure as the private L2 size
of the core executing a kernel.

Recommended configurations for each device:

1. One fast physical core, one worker: the best control for kernel behavior.
2. Two and four fast cores, matching worker counts where available.
3. A separate efficient-core configuration if that is an intended deployment.
4. A separate mixed-cluster configuration if the production runner uses one.

When measuring one worker, pin to one core instead of a mask that allows
migration. Set `HOST_CPUS` and `PHONE_CPUS` to comma-separated IDs for the specific
configuration, such as `0,2,4,6` on a suitable desktop or `4,5,6,7` on a suitable
phone. These are examples, not recommendations for an unknown machine.

Keep charging, screen state, power mode and ambient conditions consistent. Stop
other heavy work. Let the device cool before a repetition block. The runner
records temperature/cooling-service output and frequency snapshots before/after
each process and during longer processes; it does **not** lock clocks or enforce
a device-independent thermal threshold. `--cooldown-seconds` is a delay, not
proof that the device has cooled. Inspect telemetry before accepting a gain.

## 5. First run: verify the whole collection pipeline

Use one chosen core for `HOST_CPUS` or `PHONE_CPUS` in these smoke commands.

```sh
python3 bench/dequantization_bmm/run.py \
  --binary build/dequant-bmm-host/dequant_bmm_probe \
  --out results/host-smoke --label my-x86-host \
  --cpus "$HOST_CPUS" --threads 1 --profile smoke \
  --repetitions 1 --passes 2 --ms 30 --cooldown-seconds 1 \
  --build-info build/dequant-bmm-host/CMakeCache.txt
```

```sh
python3 bench/dequantization_bmm/run.py \
  --binary build/dequant-bmm-android/dequant_bmm_probe \
  --serial "$SERIAL" --out results/phone-smoke --label my-phone \
  --cpus "$PHONE_CPUS" --threads 1 --profile smoke \
  --repetitions 1 --passes 2 --ms 30 --cooldown-seconds 1 \
  --build-info build/dequant-bmm-android/CMakeCache.txt
```

The runner uploads and verifies the Android binary automatically. (If built
with Bazel, point `--binary` to `bazel-bin/bench/dequantization_bmm/probe`.) The smoke
profile covers QK, token-major PV and transposed PV for a 128-token prefill shape
and an 8K-history decode shape. For symbol tools not on PATH, pass `--nm` and
`--readelf`; the NDK's LLVM equivalents can read Android ELF binaries.

```sh
python3 bench/dequantization_bmm/summarize.py \
  results/host-smoke results/phone-smoke --out results/smoke-summary
```

Expect `insufficient_repetitions` in this summary. It verifies transport,
dispatch, numerical checks and data processing, rather than establishing a win.

## 6. Collect the comparison matrix

| Profile | Cases per thread count and scale | Purpose |
|---|---:|---|
| `smoke` | 6 | Quick prefill/decode pipeline checks |
| `core` | 33 | Decode, chunk16/chunk128, large prefill, local/global history, folded/broadcast heads, both V layouts |
| `boundary` | 52 | M = 1,2,4,5,6,7,8,10,12,16,32,64,128, at short/long QK and long PV shapes |
| `batch` | 24 | Shared RHS, independent RHS matrices, broadcast LHS, transpose, M = 1/8/128 |
| `history` | 144 | T = 1/16/128, S = 256 through 32768, D = 256/512, QK and both PV layouts |

Inspect a profile before starting:

```sh
python3 bench/dequantization_bmm/run.py --profile core \
  --threads 1 2 4 --repetitions 3 --dry-run
```

Start with `core`, at least three independent process repetitions, four
alternating paired intervals per process, and 60–100 ms per metric/interval.
Run each affinity configuration separately; a one-core run and a four-core run
should have different output directories. For example, with a four-core mask:

```sh
python3 bench/dequantization_bmm/run.py \
  --binary build/dequant-bmm-android/dequant_bmm_probe \
  --serial "$SERIAL" --out results/phone-core-fast4 --label my-phone \
  --cpus "$PHONE_CPUS" --threads 2 4 --profile core \
  --repetitions 3 --passes 4 --ms 100 --cooldown-seconds 5 \
  --max-pair-mib 512 --timeout-seconds 600 \
  --build-info build/dequant-bmm-android/CMakeCache.txt
```

For Linux use the host binary and omit `--serial`. Give each machine a distinct
label. The example keeps a four-core allowed mask for both worker counts; workers
can migrate within that mask. To restrict two workers to two cores, collect a
separate run with a two-core mask. The number of processes is cases × thread
counts × scales × repetitions.
The nominal timed work per process is passes × two alternatives × two metrics ×
`--ms`, plus warmups, validation, allocation and cooldown. Large single-thread
cases can take substantially longer because each interval performs at least
three calls. Start small before scheduling the full history sweep.

Then collect `boundary`, `batch`, and selected `history` cases. Cover points
around the **observed** MR/NR values and cache-size transitions, including sizes
just below/above aligned dimensions. `example_cases.json` demonstrates custom
matrix shapes, tail dimensions, and independent/broadcast RHS batches:

```sh
python3 bench/dequantization_bmm/run.py \
  --binary build/dequant-bmm-host/dequant_bmm_probe \
  --out results/host-custom --label my-x86-host \
  --cpus "$HOST_CPUS" --threads 1 \
  --cases bench/dequantization_bmm/example_cases.json \
  --scales 0.03125 0.037 --repetitions 3 --passes 4
```

The default performance scale is 1/32. Also run representative cases with
non-power-of-two scales, such as 0.037. Do not hide a numerical failure by
switching all tests back to an exactly representable scale.

### Shapes and head grouping

A is `[batch_a, M, K]`. B is `[batch_b, K, N]`, or `[batch_b, N, K]` when
`transpose_b=1`. Batch sizes must match or one must be 1. Attention presets use
8 query heads and one shared KV head:

| Operation | Folded-head shape | Broadcast-head shape |
|---|---|---|
| QK | M=8T, K=D, N=S; batches 1/1; transpose B | M=T, K=D, N=S; batches 8/1; transpose B |
| PV | M=8T, K=S, N=D; batches 1/1 | M=T, K=S, N=D; batches 8/1 |

PV is measured with both physical B layouts. Equal mathematical work does not
imply identical packing, scheduling or reuse. In particular, **one-token decode
can have M=8, not M=1**. A heuristic should use the actual BMM dimensions, not an
assumed correspondence between M and prompt length.

S here is the matrix's live extent, not an independently allocated KV capacity.
These probes have no attention masks, causal pruning, or sliding-window logic.
Their purpose is selecting the implementation of an already chosen dense BMM.

### Memory limits and cache conditions

`--max-pair-mib` applies a conservative preallocation estimate to the two
simultaneous runtimes and external arrays. It is not a hard RSS limit. Cases
above the estimate are recorded as skipped, with a reason; unsupported hardware
also returns skip status 77. An OOM kill or other failure is recorded as failed.
Preserve missing/failed coverage instead of substituting zero latency.

Start with warm buffers. To test sensitivity to cache pressure, repeat selected
cases with, for example, `--churn-mib 64` and a separate output directory. Before
each timed call the probe touches that many MiB, outside the measured interval.
This is reproducible cache pressure, **not a guaranteed cold-cache flush**. Choose
sizes relevant to each device and archive the value. Keep warm/churn results
separate. Neither condition reproduces all cache effects of a whole LLM graph.

Both paths retain external INT8 B. The opt-out path additionally materializes
FP32 B and its packing/workspace. A fourfold expansion of B does not mean total
runtime memory increases fourfold. `workspace_bytes` is the runtime's shared
workspace, `scale_bytes` is separate operator-owned scale storage, and
`external_bytes` describes logical A/B/C buffers. These exclude runtime metadata,
thread stacks, allocator overhead, guard bytes and process/library mappings.
Do not call their sum RSS, and do not infer one-runtime RSS from a process holding
both alternatives and an optional churn buffer.

## 7. Gather, summarize and inspect the evidence

Keep each complete output directory. It contains:

- `manifest.json`: binary/source identities, harness hashes, settings and every
  attempted process, including skipped and failed cases.
- `device.json`: CPU, cache/topology, frequency policies and available power data.
- `cases.json`, `source.patch`, copied build information, and matching ELF/symbol
  data. Preserve the original unstripped executable as well.
- One directory per process: raw `rows.csv`, `stderr.txt`, command/status and
  `telemetry.json`.

Produce a combined report without merging unlike binaries or affinity settings:

```sh
python3 bench/dequantization_bmm/summarize.py \
  results/phone-core-fast4 results/host-custom \
  --out results/comparison --margin 0.05
```

`summary.csv` groups by device label, binary hash, CPU placement, actual shapes,
thread count, scale, cache condition and dispatch identity. `processes.csv`
retains one result per independent process. `coverage.json` lists incomplete,
failed and skipped coverage. Different binaries remain separate, even if their
machine labels match. Do not collect two devices with the same label.

The ratio is **rewrite-on latency / rewrite-off latency**. Above 1 favors opting
out. For a ratio of 1.30, opt-out throughput is 30% higher, while latency is
23.1% lower. The report includes both percentages and labels the distinction.
It also reports effective BMM GFLOP/s including preparation; this is not the
microkernel-only arithmetic throughput.

Each process contributes the median of its paired interval ratios. The report
then takes the median across processes and shows their minimum and maximum.
With the default 5% margin, `prefer_off` requires at least three processes and
**every process median** above 1.05. `prefer_on` requires every median below
1/1.05. Other cases are `uncertain` or `insufficient_repetitions`. This is a
screening rule, not a statistical confidence interval or a final production
heuristic. Inspect interval variability and telemetry for important decisions.

Kernel names are resolved from configured microkernel function pointers. All
non-null microarchitecture slots are recorded. On heterogeneous CPUs these are
**dispatch candidates**, not proof that every worker executed every listed
kernel. Record unexpected packed-FP32 paths as separate groups; do not assume an
ISA was selected merely because the CPU supports it. Unresolved symbols require
the matching unstripped binary. ISA-specific tuning sweeps are not required by
this plan.


## 8. Turn the measurements into a heuristic

First ask which portion of the invocation changes. A useful decomposition is:

```text
rewrite off = dequantize B + pack FP32 B + FP32 GEMM + orchestration
rewrite on  = adapter      + pack INT8 B + mixed GEMM + orchestration
```

The decision must compare complete costs. Mixed GEMM can lose on arithmetic but
win overall by avoiding the larger FP32 operand. Conversely, repeated INT8
conversion in its inner loop can lose when many query rows reuse B. Different
tiles and schedules also contribute; do not attribute the entire difference to
conversion instructions without profiling.

For each hardware/kernel/thread group, examine:

1. M, K, N, physical transpose, and both batch sizes, keeping QK and PV separate.
2. Unique RHS size: `batch_b * K * N` bytes in INT8, four times that in FP32.
3. A rough reuse feature:
   `output_batches / batch_b * ceil(M / mixed_kernel_MR)`. This counts row-tile
   visits per unique RHS matrix; it is not a cycle model.
4. Workspace and cache boundaries, warm versus churn conditions, and packing
   type. Inspect OS cache-sharing information instead of trusting one aggregate
   L2 number. Timing dependence on cache pressure is evidence, not a measured
   memory-bandwidth roofline.
5. Total-call versus invoke-only ratios. Their difference may suggest reshape
   overhead, but subtracting independently timed medians is not a precise stage
   decomposition. Profile packing/conversion/GEMM separately when necessary.

Compare at least these candidate policies offline: always rewrite, never rewrite,
`M > MR` cutoff, and a shape/layout/kernel/thread-aware model. For each measured
case compute the chosen time relative to the faster measured alternative:

```text
regret = chosen_path_time / min(rewrite_on_time, rewrite_off_time) - 1
```

Report worst-case regret and regressions as well as a weighted aggregate. Choose
weights from intended application workloads; an equal average over an arbitrary
matrix does not predict model TTFT. Include memory limits as feasibility
constraints. For ambiguous cases retain the incumbent or a documented conservative
fallback; do not invent a crossover from noisy measurements.

Hold out whole devices and some shapes when fitting. Test whether rules trained
on one ARM core also work on another, and treat Intel/AMD or different x86 kernel
families separately when the evidence requires it. Check sensitivity to worker
count. A simple model that generalizes is more useful than a large lookup table
that memorizes these presets.

Before accepting an automatic policy, explicitly check that it does not discard
the observed long-history decode benefit while fixing prefill. Test it against
folded and broadcast heads, independent RHS batches, both transposes, tails and
memory-pressure cases. Preserve all counterexamples and missing coverage.

## 9. Final gates before changing defaults

The current flag is fixed when the graph is authored. Resizing is numerically
supported, but **does not reconsider the choice**. These timing probes keep a
single shape per process; repeated reshape calls measure steady-shape overhead,
not a growing-history request. A future automatic policy must separately cover
M changes, live-history growth/shrinkage, workspace replanning, and any change
in selected arithmetic. Construction-time shape alone is insufficient.

After choosing a candidate, run matched whole-model tests with representative
short/long prompts and live histories. Measure prefill latency, actual prefill
tokens/s, TTFT, decode latency and memory independently. Keep weights,
quantization, thread placement, prompt tokens and cache conditions matched.
Run quality tests with real quantization scales and multiple prompts. Synthetic
BMM success does not establish model equivalence, and a 30% BMM improvement does
not imply a 30% whole-model improvement.

The deliverable for heuristic review is: complete raw data and coverage,
reproducible build/run settings, per-configuration comparisons, numerical
validation, candidate-policy regret on held-out devices/shapes, and a clear
statement of where the proposed rule remains uncertain.
