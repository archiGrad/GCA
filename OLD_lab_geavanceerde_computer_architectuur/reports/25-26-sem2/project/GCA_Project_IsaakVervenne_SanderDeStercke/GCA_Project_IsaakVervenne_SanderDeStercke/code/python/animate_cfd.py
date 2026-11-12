#!/usr/bin/env python3
import glob
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v3 as iio
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def load_vel_bin(path, nx, ny):
    # Supersnel binair inlezen
    v = np.fromfile(path, dtype=np.float32)
    g = v.reshape((ny, nx))
    return np.nan_to_num(g, nan=0.0)

def load_mask(path, nx, ny):
    d = pd.read_csv(path)
    g = np.zeros((ny, nx), dtype=np.uint8)
    x = d.x.to_numpy(np.int32)
    y = d.y.to_numpy(np.int32)
    o = d.occ.to_numpy(np.int32)
    m = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
    g[y[m], x[m]] = (o[m] != 0)
    return g

def process_single_frame(args):
    i, bin_file, mask, temp_dir, nx, ny = args
    png_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
    
    grid = load_vel_bin(bin_file, nx, ny)
    
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    im = ax.imshow(grid, origin="lower", cmap="turbo", aspect="equal", vmin=0, vmax=0.12, interpolation='bilinear')
    
    if mask is not None:
        ax.contour(mask, levels=[0.5], colors=["white"], linewidths=1.0, origin="lower")

    iter_num = bin_file.split('_iter')[-1].replace('.bin', '')
    ax.set_title(f"CFD Iteration: {iter_num}")
    
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temp-dir", default="temp_anim_data")
    parser.add_argument("--out-mp4", default="images/cfd_animation.mp4")
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    args = parser.parse_args()

    # Zoek naar .bin bestanden!
    bin_files = sorted(glob.glob(os.path.join(args.temp_dir, "*_iter*.bin")))
    if not bin_files:
        print(f"Geen iteratie BIN bestanden gevonden in {args.temp_dir}!")
        return

    mask_files = glob.glob("obstacle_mask_*.csv")
    mask = load_mask(mask_files[0], args.nx, args.ny) if mask_files else None

    args_list = [(i, f, mask, args.temp_dir, args.nx, args.ny) for i, f in enumerate(bin_files)]
    png_paths = []
    
    print("\n[Start rendering PNG frames]")
    with ProcessPoolExecutor() as executor:
        for path in tqdm(executor.map(process_single_frame, args_list), total=len(args_list), desc="   Progress", unit="frame"):
            png_paths.append(path)

    print(f"\n[Stitching {len(png_paths)} frames to MP4]")
    os.makedirs(os.path.dirname(args.out_mp4) or ".", exist_ok=True)
    # B4 FIX: stream frames one-by-one instead of loading all into RAM at once
    with iio.imopen(args.out_mp4, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=10)
        for p in tqdm(sorted(png_paths), desc="   Stitching", unit="frame"):
            frame = iio.imread(p)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]   # RGBA -> RGB (matplotlib saves RGBA PNGs)
            writer.write_frame(frame)

if __name__ == "__main__":
    main()