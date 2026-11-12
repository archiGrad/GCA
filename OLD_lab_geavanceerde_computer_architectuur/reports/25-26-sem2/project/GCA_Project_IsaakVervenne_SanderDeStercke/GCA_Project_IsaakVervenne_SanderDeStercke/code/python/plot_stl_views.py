#!/usr/bin/env python3
import argparse, glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VIEWS=["top","bottom","front","back","side_left","side_right"]


def load_vel(path,nx,ny):
    d=pd.read_csv(path)
    g=np.zeros((ny,nx),dtype=np.float32)
    x=d.x.to_numpy(np.int32); y=d.y.to_numpy(np.int32); v=np.nan_to_num(d.vel.to_numpy(np.float32),nan=0.0,posinf=0.0,neginf=0.0)
    m=(x>=0)&(x<nx)&(y>=0)&(y<ny)
    g[y[m],x[m]]=v[m]
    return g


def load_mask(path,nx,ny):
    d=pd.read_csv(path)
    g=np.zeros((ny,nx),dtype=np.uint8)
    x=d.x.to_numpy(np.int32); y=d.y.to_numpy(np.int32); o=d.occ.to_numpy(np.int32)
    m=(x>=0)&(x<nx)&(y>=0)&(y<ny)
    g[y[m],x[m]]=(o[m]!=0)
    return g


def robust_limits(arrs):
    vals=np.concatenate([a.ravel() for a in arrs])
    vals=vals[np.isfinite(vals)]
    nz=vals[vals>1e-8]
    if len(nz)==0: return 0.0, 1e-3
    return 0.0, float(np.percentile(nz,99.5))


def draw(g,title,out,vmin,vmax,mask=None):
    # Maak de figure vierkanter, bijvoorbeeld 8 bij 8
    fig,ax=plt.subplots(figsize=(8,8),constrained_layout=True)
    # aspect="equal" zorgt ervoor dat pixels fysiek perfect vierkant blijven!
    im=ax.imshow(g,origin="lower",cmap="turbo",aspect="equal",vmin=vmin,vmax=vmax)
    if mask is not None and np.any(mask):
        ax.contour(mask,levels=[0.5],colors=["white"],linewidths=1.0,origin="lower")
    ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im,ax=ax,label="Velocity magnitude")
    fig.savefig(out,dpi=240); plt.close(fig)


def pick_matching_mask(view, vel_path, search_dir):
    """
    Tries to match mask by same view + same heading tag (hdgN) present in velocity filename.
    Falls back to latest mask for that view.
    """
    vel_name=os.path.basename(vel_path)
    m = re.search(r"hdg-?\d+", vel_name)
    heading_tag = m.group(0) if m else None

    candidates = glob.glob(os.path.join(search_dir, f"obstacle_mask_*_{view}*.csv"))
    if not candidates:
        return None

    if heading_tag:
        exact=[c for c in candidates if heading_tag in os.path.basename(c)]
        if exact:
            return sorted(exact)[-1]

    # fallback: most recent file
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--nx",type=int,default=512)
    ap.add_argument("--ny",type=int,default=256)
    ap.add_argument("--search-dir",default=".")
    ap.add_argument("--out-dir",default="images/stl_views")
    ap.add_argument("--base",default="velocity_stl")
    args=ap.parse_args()

    os.makedirs(args.out_dir,exist_ok=True)

    vel_files={}
    for v in VIEWS:
        patt = os.path.join(args.search_dir, f"{args.base}_{v}*.csv")
        matches=sorted(glob.glob(patt), key=lambda p: os.path.getmtime(p), reverse=True)
        if not matches:
            print(f"[skip] missing {patt}")
            continue
        vel_files[v]=matches[0]

    if not vel_files:
        raise FileNotFoundError("No velocity_stl_<view>*.csv files found")

    grids={v:load_vel(p,args.nx,args.ny) for v,p in vel_files.items()}
    vmin,vmax=robust_limits(list(grids.values()))

    for v,g in grids.items():
        draw(g,f"STL CFD Velocity - {v}",os.path.join(args.out_dir,f"velocity_{v}.png"),vmin,vmax)
        mf=pick_matching_mask(v,vel_files[v],args.search_dir)
        if mf:
            m=load_mask(mf,args.nx,args.ny)
            draw(g,f"STL CFD Velocity + Obstacle - {v}",os.path.join(args.out_dir,f"velocity_{v}_overlay.png"),vmin,vmax,mask=m)
            print(f"[ok] {v}: overlay {os.path.basename(mf)}")

    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    last_im=None
    for ax,v in zip(axes.ravel(),VIEWS):
        if v not in grids:
            ax.axis("off"); ax.set_title(f"{v} (missing)"); continue
        last_im=ax.imshow(grids[v],origin="lower",cmap="turbo",aspect="auto",vmin=vmin,vmax=vmax)
        ax.set_title(v); ax.set_xlabel("x"); ax.set_ylabel("y")
    if last_im is not None:
        fig.colorbar(last_im,ax=axes.ravel().tolist(),shrink=0.82,label="Velocity magnitude")
    fig.suptitle("STL Multi-view CFD (GPU)")
    fig.savefig(os.path.join(args.out_dir,"velocity_views_grid.png"),dpi=240)
    plt.close(fig)

    print(f"Saved figures in {args.out_dir}")


if __name__=="__main__":
    main()
