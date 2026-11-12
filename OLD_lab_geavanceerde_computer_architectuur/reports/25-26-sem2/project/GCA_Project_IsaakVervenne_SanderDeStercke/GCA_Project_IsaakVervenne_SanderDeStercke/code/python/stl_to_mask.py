#!/usr/bin/env python3
import argparse
import math
import os
import struct
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def deco(fn): return fn
        return deco


def load_stl_ascii(path: str) -> np.ndarray:
    verts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                p = line.split()
                if len(p) == 4:
                    verts.append([float(p[1]), float(p[2]), float(p[3])])
    if len(verts) % 3 != 0 or len(verts) == 0:
        raise ValueError("Invalid ASCII STL vertex count")
    return np.array(verts, dtype=np.float32).reshape(-1, 3, 3)


def load_stl_binary(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        f.read(80)
        tri_count = struct.unpack("<I", f.read(4))[0]
        data = f.read()
    tris = []
    off = 0
    for _ in range(tri_count):
        off += 12
        tri = []
        for _ in range(3):
            x, y, z = struct.unpack_from("<fff", data, off)
            off += 12
            tri.append([x, y, z])
        off += 2
        tris.append(tri)
    return np.array(tris, dtype=np.float32)


def load_stl(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        head = f.read(6)
    if head.lower().startswith(b"solid"):
        try:
            return load_stl_ascii(path)
        except Exception:
            return load_stl_binary(path)
    return load_stl_binary(path)


def project_triangle(tri: np.ndarray, view: str) -> np.ndarray:
    x, y, z = tri[:, 0], tri[:, 1], tri[:, 2]

    # NOTE: -z for front/back/side views fixes upside-down rendering for common STL conventions.
    if view in ("top", "bottom"):
        p = np.stack([x, y], axis=1)
    elif view in ("front", "back"):
        p = np.stack([x, -z], axis=1)
    elif view in ("side_left", "side_right"):
        p = np.stack([y, -z], axis=1)
    else:
        raise ValueError(f"Unknown view: {view}")

    if view in ("bottom", "back", "side_right"):
        p[:, 0] = -p[:, 0]

    return p


def rotate_3d(tris: np.ndarray, rx: float, ry: float, rz: float,
              flip_x: bool, flip_y: bool, flip_z: bool) -> np.ndarray:
    """Apply 3D flips then Rz·Ry·Rx rotation to all triangle vertices before projection."""
    pts = tris.reshape(-1, 3).copy()
    if flip_x: pts[:, 0] = -pts[:, 0]
    if flip_y: pts[:, 1] = -pts[:, 1]
    if flip_z: pts[:, 2] = -pts[:, 2]
    for deg, axis in ((rx, 0), (ry, 1), (rz, 2)):
        if abs(deg) < 1e-9:
            continue
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        if axis == 0:   R = np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)
        elif axis == 1: R = np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)
        else:           R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
        ctr = pts.mean(axis=0)
        pts = (pts - ctr) @ R.T + ctr
    return pts.reshape(tris.shape)


def rotate_2d(points: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg) < 1e-9:
        return points
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    ctr = points.reshape(-1, 2).mean(axis=0)
    p = points.reshape(-1, 2) - ctr
    p = p @ R.T
    p = p + ctr
    return p.reshape(points.shape)


def world_to_pixel_isotropic(tris2d: np.ndarray, nx: int, ny: int, margin: float = 0.5) -> np.ndarray:
    all_pts = tris2d.reshape(-1, 2)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-8)

    min_xy -= margin * span
    max_xy += margin * span
    span = np.maximum(max_xy - min_xy, 1e-8)

    sx = (nx - 1) / span[0]
    sy = (ny - 1) / span[1]
    s = min(sx, sy)

    used_w = span[0] * s
    used_h = span[1] * s
    ox = (nx - 1 - used_w) * 0.5
    oy = (ny - 1 - used_h) * 0.5

    pts = tris2d.reshape(-1, 2)
    px = (pts[:, 0] - min_xy[0]) * s + ox
    py = (pts[:, 1] - min_xy[1]) * s + oy
    return np.stack([px, py], axis=1).reshape(tris2d.shape).astype(np.float32)


@njit(cache=True)
def _edge_fn(ax, ay, bx, by, px, py):
    return (py - ay) * (bx - ax) - (px - ax) * (by - ay)


@njit(cache=True)
def rasterize_mask_numba(tris_px: np.ndarray, nx: int, ny: int) -> np.ndarray:
    mask = np.zeros((ny, nx), dtype=np.uint8)
    ntri = tris_px.shape[0]

    for t in range(ntri):
        x1, y1 = tris_px[t, 0, 0], tris_px[t, 0, 1]
        x2, y2 = tris_px[t, 1, 0], tris_px[t, 1, 1]
        x3, y3 = tris_px[t, 2, 0], tris_px[t, 2, 1]

        minx = int(max(0, math.floor(min(x1, x2, x3))))
        maxx = int(min(nx - 1, math.ceil(max(x1, x2, x3))))
        miny = int(max(0, math.floor(min(y1, y2, y3))))
        maxy = int(min(ny - 1, math.ceil(max(y1, y2, y3))))
        if maxx < minx or maxy < miny:
            continue

        area = _edge_fn(x1, y1, x2, y2, x3, y3)
        if abs(area) < 1e-12:
            continue

        for iy in range(miny, maxy + 1):
            py = iy + 0.5
            for ix in range(minx, maxx + 1):
                px = ix + 0.5
                w0 = _edge_fn(x2, y2, x3, y3, px, py)
                w1 = _edge_fn(x3, y3, x1, y1, px, py)
                w2 = _edge_fn(x1, y1, x2, y2, px, py)
                if area > 0:
                    if w0 >= 0 and w1 >= 0 and w2 >= 0:
                        mask[iy, ix] = 1
                else:
                    if w0 <= 0 and w1 <= 0 and w2 <= 0:
                        mask[iy, ix] = 1
    return mask


def write_mask_csv(mask: np.ndarray, out_path: str) -> None:
    """Write only occupied cells (sparse). The CUDA loader initialises the array to 0
    and only sets cells it reads, so non-present rows remain 0."""
    ys, xs = np.where(mask)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("x,y,occ\n")
        for x, y in zip(xs, ys):
            f.write(f"{x},{y},1\n")


def main():
    ap = argparse.ArgumentParser(description="STL -> obstacle mask CSV")
    ap.add_argument("--stl", required=True)
    ap.add_argument("--view", default="top")
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--heading-deg", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    # 3D orientation flags (applied before projection)
    ap.add_argument("--rotate3d-x", type=float, default=0.0, help="Pre-projection rotation around X axis (deg)")
    ap.add_argument("--rotate3d-y", type=float, default=0.0, help="Pre-projection rotation around Y axis (deg)")
    ap.add_argument("--rotate3d-z", type=float, default=0.0, help="Pre-projection rotation around Z axis (deg)")
    ap.add_argument("--flip-x", action="store_true", help="Mirror STL along X axis before projection")
    ap.add_argument("--flip-y", action="store_true", help="Mirror STL along Y axis before projection")
    ap.add_argument("--flip-z", action="store_true", help="Mirror STL along Z axis before projection")
    args = ap.parse_args()

    tris = load_stl(args.stl)

    # Apply 3D orientation corrections before projection
    tris = rotate_3d(tris, args.rotate3d_x, args.rotate3d_y, args.rotate3d_z,
                     args.flip_x, args.flip_y, args.flip_z)

    views = [args.view] if args.view != "all" else ["top", "bottom", "front", "back", "side_left", "side_right"]
    base = os.path.splitext(os.path.basename(args.stl))[0]

    print(f"Numba acceleration: {'ON' if NUMBA_AVAILABLE else 'OFF'}")

    for v in views:
        tris2d = np.array([project_triangle(t, v) for t in tris], dtype=np.float32)
        tris2d = rotate_2d(tris2d, args.heading_deg)
        tris_px = world_to_pixel_isotropic(tris2d, args.nx, args.ny, margin=args.margin)
        mask = rasterize_mask_numba(tris_px, args.nx, args.ny)

        if args.out and args.view != "all":
            out = args.out
        else:
            out = f"obstacle_mask_{base}_{v}_hdg{int(round(args.heading_deg))}.csv"

        write_mask_csv(mask, out)
        print(f"[{v}] wrote {out} | occupied: {int(mask.sum())}/{args.nx*args.ny}")


if __name__ == "__main__":
    main()
