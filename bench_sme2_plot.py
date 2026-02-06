#!/usr/bin/env python3
import subprocess
import json
import os
import sys
import math
import statistics
import time

# --- Configuration ---
# Geometric progression of sizes to test unaligned/power-of-2 cases
BASE_SIZE = 240
SCALE_FACTOR = 1.3
NUM_STEPS = 12 
SIZES = [int(BASE_SIZE * (SCALE_FACTOR ** i)) for i in range(NUM_STEPS)]
# Ensure we hit the large case explicitly if not included
if 4096 not in SIZES:
    SIZES.append(4096)
SIZES = sorted(list(set(SIZES)))

NUM_REPETITIONS = 10  # Number of times to run each benchmark for statistics

# Toolchain flags based on your environment
BAZEL_FLAGS = [
    "--action_env=CLANG_COMPILER_PATH=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--repo_env=CC=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--repo_env=BAZEL_COMPILER=/opt/homebrew/Cellar/llvm@20/20.1.8/bin/clang-20",
    "--linkopt=--ld-path=/opt/homebrew/bin/ld64.lld"
]

# Benchmark target
TARGET = "ynnpack/kernels/dot:bench"

def run_benchmark(size):
    """Runs the benchmark for a specific size and returns the JSON output."""
    print(f"[-] Benchmarking size: {size}x{size}x{size} ...", end="", flush=True)
    
    cmd = [
        "bazel", "run", "-c", "opt", TARGET
    ] + BAZEL_FLAGS + [
        "--",
        "--benchmark_filter=dot_fp32_sme2", # Matches both 'dot_fp32_sme2' and 'dot_fp32_sme2_opt'
        f"--shape={size}x{size}x{size}",
        f"--benchmark_repetitions={NUM_REPETITIONS}",
        "--benchmark_format=json",
        "--benchmark_min_time=0.1s", # Ensure each rep runs long enough to be stable
        "--benchmark_display_aggregates_only=false"
    ]

    try:
        # Capture stdout (JSON) and separate stderr (Build logs)
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(" Done.")
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"\n[!] Error running benchmark for size {size}:")
        print(e.stderr)
        return None
    except json.JSONDecodeError:
        print(f"\n[!] Failed to parse JSON output for size {size}.")
        return None

def analyze_results(json_data):
    """Parses JSON data to extract stats for original and optimized kernels."""
    if not json_data or "benchmarks" not in json_data:
        return None

    times_orig = []
    times_opt = []

    for b in json_data["benchmarks"]:
        name = b["name"]
        # Skip aggregate rows (mean, median, stddev) provided by Google Benchmark
        if "aggregate_name" in b:
            continue
            
        # Extract real_time (wall clock time) in nanoseconds
        time_ns = b["real_time"]

        if "dot_fp32_sme2_opt" in name:
            times_opt.append(time_ns)
        elif "dot_fp32_sme2" in name:
            times_orig.append(time_ns)

    if not times_orig or not times_opt:
        return None

    # Calculate statistics
    mean_orig = statistics.mean(times_orig)
    mean_opt = statistics.mean(times_opt)
    
    # Speedup
    speedup = mean_orig / mean_opt
    
    # Error Propagation for Speedup (Standard Error)
    # SE_speedup = Speedup * sqrt( (SE_orig/Mean_orig)^2 + (SE_opt/Mean_opt)^2 )
    se_orig = statistics.stdev(times_orig) / math.sqrt(len(times_orig)) if len(times_orig) > 1 else 0
    se_opt = statistics.stdev(times_opt) / math.sqrt(len(times_opt)) if len(times_opt) > 1 else 0
    
    rel_error = math.sqrt((se_orig / mean_orig)**2 + (se_opt / mean_opt)**2) if mean_orig > 0 and mean_opt > 0 else 0
    se_speedup = speedup * rel_error
    
    # 95% Confidence Interval (approx 1.96 * SE)
    ci_95 = 1.96 * se_speedup

    return {
        "mean_orig_ns": mean_orig,
        "mean_opt_ns": mean_opt,
        "speedup": speedup,
        "ci_95": ci_95
    }

def main():
    results = []
    print(f"[*] Starting Benchmark Suite")
    print(f"[*] Sizes: {SIZES}")
    print(f"[*] Repetitions: {NUM_REPETITIONS}\n")

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

    # Plotting
    try:
        import matplotlib.pyplot as plt
        
        sizes = [r[0] for r in results]
        speedups = [r[1]["speedup"] for r in results]
        errors = [r[1]["ci_95"] for r in results]

        plt.figure(figsize=(10, 6))
        plt.errorbar(sizes, speedups, yerr=errors, fmt='-o', color='b', ecolor='r', capsize=5, label='Optimized vs Original')
        plt.axhline(y=1.0, color='gray', linestyle='--', label='Baseline (1.0x)')
        
        plt.title('ARM SME2 Dot Product Optimization Speedup')
        plt.xlabel('Matrix Dimension (M=N=K)')
        plt.ylabel('Speedup (Factor)')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()
        
        # Annotate points
        for i, txt in enumerate(speedups):
            plt.annotate(f"{txt:.2f}x", (sizes[i], speedups[i]), textcoords="offset points", xytext=(0,10), ha='center')

        output_file = "sme2_benchmark.png"
        plt.savefig(output_file)
        print(f"\n[*] Plot saved to {output_file}")
        
    except ImportError:
        print("\n[!] matplotlib not found. Skipping plot generation.")
    except Exception as e:
        print(f"\n[!] Error generating plot: {e}")

if __name__ == "__main__":
    main()