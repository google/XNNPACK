#!/usr/bin/env python3
import subprocess
import json
import os
import sys
import math
import statistics
import time

# --- Configuration ---
BASE_SIZE = 240
SCALE_FACTOR = 1.3
NUM_STEPS = 12 
SIZES = [int(BASE_SIZE * (SCALE_FACTOR ** i)) for i in range(NUM_STEPS)]
if 4096 not in SIZES: SIZES.append(4096)
SIZES = sorted(list(set(SIZES)))

NUM_REPETITIONS = 5

BAZEL_FLAGS = [
    "--action_env=CLANG_COMPILER_PATH=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--repo_env=CC=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--repo_env=BAZEL_COMPILER=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--linkopt=--ld-path=/opt/homebrew/bin/ld64.lld"
]

TARGET = "ynnpack/kernels/dot:bench"
BINARY_PATH = "bazel-bin/ynnpack/kernels/dot/bench"

def build():
    print("[-] Building benchmark binary...", end="", flush=True)
    cmd = ["bazel", "build", "-c", "opt", TARGET] + BAZEL_FLAGS
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f" Failed!\n{result.stderr}")
        sys.exit(1)
    print(" Done.")

def run_benchmark(size):
    print(f"[-] Benchmarking size: {size}x{size}x{size} ...", end="", flush=True)
    cmd = [
        BINARY_PATH,
        "--benchmark_filter=dot_fp32_sme2",
        f"--shape={size}x{size}x{size}",
        f"--benchmark_repetitions={NUM_REPETITIONS}",
        "--benchmark_format=json",
        "--benchmark_min_time=0.1s",
        "--benchmark_display_aggregates_only=false"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = result.stdout
        start_idx = stdout.find("{")
        if start_idx == -1: return None
        print(" Done.")
        return json.loads(stdout[start_idx:])
    except Exception as e:
        print(f" Error: {e}")
        return None

def analyze_results(json_data):
    if not json_data or "benchmarks" not in json_data: return None
    times_orig, times_opt = [], []
    for b in json_data["benchmarks"]:
        if "aggregate_name" in b: continue
        if "error_occurred" in b and b["error_occurred"]: continue
        if "dot_fp32_sme2_opt" in b["name"]: times_opt.append(b["real_time"])
        elif "dot_fp32_sme2" in b["name"]: times_orig.append(b["real_time"])

    if not times_orig or not times_opt: return None
    mean_orig = statistics.mean(times_orig)
    mean_opt = statistics.mean(times_opt)
    speedup = mean_orig / mean_opt
    
    se_orig = statistics.stdev(times_orig) / math.sqrt(len(times_orig)) if len(times_orig) > 1 else 0
    se_opt = statistics.stdev(times_opt) / math.sqrt(len(times_opt)) if len(times_opt) > 1 else 0
    rel_error = math.sqrt((se_orig / (mean_orig or 1e-9))**2 + (se_opt / (mean_opt or 1e-9))**2)
    se_speedup = speedup * rel_error
    ci_95 = 1.96 * se_speedup

    return {
        "mean_orig_ns": mean_orig,
        "mean_opt_ns": mean_opt,
        "speedup": speedup,
        "ci_95": ci_95
    }

def main():
    build()
    results = []
    print(f"{'Size':<10} | {'Orig (ms)':<12} | {'Opt (ms)':<12} | {'Speedup':<10} | {'95% CI':<10}")
    print("-" * 65)

    for size in SIZES:
        data = run_benchmark(size)
        stats = analyze_results(data)
        if stats:
            results.append((size, stats))
            print(f"{size:<10} | {stats['mean_orig_ns']/1e6:<12.3f} | {stats['mean_opt_ns']/1e6:<12.3f} | {stats['speedup']:<10.3f} | +/-{stats['ci_95']:.3f}")
        else:
            print(f"{size:<10} | {'ERROR':<12} | {'ERROR':<12} | {'N/A':<10} | {'N/A':<10}")

    if results:
        try:
            import matplotlib.pyplot as plt
            sizes = [r[0] for r in results]
            speedups = [r[1]["speedup"] for r in results]
            errors = [r[1]["ci_95"] for r in results]
            plt.figure(figsize=(10, 6))
            plt.errorbar(sizes, speedups, yerr=errors, fmt='-o', color='b', ecolor='r', capsize=5, label='Optimized vs Original')
            plt.axhline(y=1.0, color='gray', linestyle='--')
            plt.title('ARM SME2 Dot Product Speedup')
            plt.xlabel('Matrix Dimension (M=N=K)')
            plt.ylabel('Speedup Factor')
            plt.xscale('log')
            plt.grid(True, which="both", ls="-", alpha=0.5)
            output_file = "sme2_benchmark.png"
            plt.savefig(output_file)
            print(f"\n[*] Plot saved to {output_file}")
        except Exception as e:
            print(f"\n[!] Plot error: {e}")

if __name__ == "__main__":
    main()
