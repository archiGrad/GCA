#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

NX, NY = 512, 256


def load_grid(csv_path, nx=NX, ny=NY):
    df = pd.read_csv(csv_path)
    required = {"x", "y", "vel"}
    if not required.issubset(df.columns):
        raise ValueError(f"{csv_path}: missing required columns {required}")

    grid = np.zeros((ny, nx), dtype=np.float32)
    x = df["x"].to_numpy(dtype=np.int32)
    y = df["y"].to_numpy(dtype=np.int32)
    v = df["vel"].to_numpy(dtype=np.float32)

    valid = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    grid[y[valid], x[valid]] = np.nan_to_num(v[valid], nan=0.0, posinf=0.0, neginf=0.0)
    return grid


def save_heatmap(grid, title, out_path, cmap="turbo", vmin=None, vmax=None):
    plt.figure(figsize=(14, 6))
    im = plt.imshow(grid, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Velocity magnitude")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_log_heatmap(grid, title, out_path):
    g = np.maximum(grid, 1e-8)
    nz = g[g > 1e-8]
    if len(nz) == 0:
        vmin, vmax = 1e-8, 1e-3
    else:
        vmin = max(np.percentile(nz, 5), 1e-8)
        vmax = max(np.percentile(nz, 99.5), vmin * 10)

    plt.figure(figsize=(14, 6))
    im = plt.imshow(g, origin="lower", cmap="turbo", aspect="auto", norm=LogNorm(vmin=vmin, vmax=vmax))
    plt.colorbar(im, label="Velocity magnitude (log)")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_diff_map(a, b, title, out_path):
    diff = np.abs(a - b)
    vmax = np.percentile(diff, 99.5) if np.any(diff > 0) else 1e-6
    plt.figure(figsize=(14, 6))
    im = plt.imshow(diff, origin="lower", cmap="magma", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(im, label="|Δ velocity|")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_performance_charts(gpu_s, cpu_single_s, cpu_multi_s, out_dir):
    labels = ["GPU", "CPU Single", "CPU Multi"]
    times = [gpu_s, cpu_single_s, cpu_multi_s]

    # Runtime chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, times)
    plt.ylabel("Runtime (seconds)")
    plt.title("LBM Runtime Comparison")
    for b, t in zip(bars, times):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{t:.3f}s", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "runtime_comparison.png"), dpi=220)
    plt.close()

    # Speedup chart (vs GPU)
    speedups = [1.0, cpu_single_s / gpu_s, cpu_multi_s / gpu_s]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, speedups)
    plt.ylabel("Speedup vs GPU (x)")
    plt.title("Relative Speedup")
    for b, s in zip(bars, speedups):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), f"{s:.2f}x", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "speedup_comparison.png"), dpi=220)
    plt.close()


def print_stats(name, grid):
    flat = grid.ravel()
    print(f"[{name}] min={flat.min():.6f}, max={flat.max():.6f}, mean={flat.mean():.6f}, nonzero={(flat > 1e-6).sum()}/{flat.size}")


def main():
    parser = argparse.ArgumentParser(description="Generate all velocity field and performance comparison plots.")
    parser.add_argument("--gpu-csv", default="velocity.csv")
    parser.add_argument("--cpu-single-csv", default="velocity_cpu_single.csv")
    parser.add_argument("--cpu-multi-csv", default="velocity_cpu_multi.csv")
    parser.add_argument("--gpu-time", type=float, default=0.832704)
    parser.add_argument("--cpu-single-time", type=float, default=66.3727)
    parser.add_argument("--cpu-multi-time", type=float, default=9.41565)
    parser.add_argument("--out-dir", default="images")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gpu = load_grid(args.gpu_csv)
    cpu_single = load_grid(args.cpu_single_csv)
    cpu_multi = load_grid(args.cpu_multi_csv)

    print_stats("GPU", gpu)
    print_stats("CPU single", cpu_single)
    print_stats("CPU multi", cpu_multi)

    # Common linear scaling for fair visual comparison
    vmax_common = max(np.max(gpu), np.max(cpu_single), np.max(cpu_multi))

    save_heatmap(gpu, "GPU Velocity Field", os.path.join(args.out_dir, "velocity_gpu.png"), vmin=0, vmax=vmax_common)
    save_heatmap(cpu_single, "CPU Single-Core Velocity Field", os.path.join(args.out_dir, "velocity_cpu_single.png"), vmin=0, vmax=vmax_common)
    save_heatmap(cpu_multi, "CPU Multi-Core Velocity Field", os.path.join(args.out_dir, "velocity_cpu_multi.png"), vmin=0, vmax=vmax_common)

    save_log_heatmap(gpu, "GPU Velocity Field (Log Scale)", os.path.join(args.out_dir, "velocity_gpu_log.png"))

    save_diff_map(gpu, cpu_single, "|GPU - CPU Single| Velocity Difference", os.path.join(args.out_dir, "diff_gpu_vs_cpu_single.png"))
    save_diff_map(gpu, cpu_multi, "|GPU - CPU Multi| Velocity Difference", os.path.join(args.out_dir, "diff_gpu_vs_cpu_multi.png"))

    save_performance_charts(args.gpu_time, args.cpu_single_time, args.cpu_multi_time, args.out_dir)

    print("\nSaved plots to:", args.out_dir)
    print("- velocity_gpu.png")
    print("- velocity_cpu_single.png")
    print("- velocity_cpu_multi.png")
    print("- velocity_gpu_log.png")
    print("- diff_gpu_vs_cpu_single.png")
    print("- diff_gpu_vs_cpu_multi.png")
    print("- runtime_comparison.png")
    print("- speedup_comparison.png")


if __name__ == "__main__":
    main()
