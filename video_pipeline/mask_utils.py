"""
mask_utils.py ─ ROI dilation, depth-threshold cleaning, union 등
"""
import cv2, numpy as np
from pathlib import Path
from scipy.ndimage import binary_fill_holes, binary_closing
from sklearn.mixture import GaussianMixture
from .config import DILATE_PX, TH_METHOD, OUTPUT_ROOT, SAVE_DEBUG
from .io_utils import load_numpy, save_numpy, ensure_dir

def auto_threshold(vals, method="gmm"):
    if method=="gmm":
        g=GaussianMixture(2,random_state=0).fit(vals.reshape(-1,1))
        mus=np.sort(g.means_.flatten()); return (mus[0]+mus[1])*0.5
    if method=="quantile": return np.percentile(vals,5)
    if method=="mean_std": m,std=vals.mean(),vals.std(); return m-std
    raise ValueError(method)

def refine(mask_path: Path, depth_path: Path, method=TH_METHOD):
    """
    • mask_path: (T,H,W) or (H,W) → 첫 프레임 사용
    • depth_path: (H,W) float32
    반환: refined_binary_mask (H,W)
    """
    mask = load_numpy(mask_path)
    mask2d = mask[0] if mask.ndim==3 else mask
    depth  = load_numpy(depth_path)
    if mask2d.shape!=depth.shape:
        mask2d=cv2.resize(mask2d,(depth.shape[1],depth.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # ① dilation
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(DILATE_PX*2+1,)*2)
    mask_d=cv2.dilate(mask2d,k)

    # ② threshold
    vals=depth[mask_d>0]
    lo = auto_threshold(vals, method)
    hi = vals.max()
    refined=(mask_d>0)&(depth>=lo)&(depth<=hi)

    # ③ hole-fill
    refined=binary_fill_holes(refined)
    refined=binary_closing(refined,iterations=1)

    return refined.astype(np.uint8), (lo,hi)

def union_masks(mask_dir: Path)->Path:
    """
    mask_dir 안의 *_refined.npy 파일들을 OR 연산해
    union_mask.npy 저장 후 경로 반환
    """
    masks=[load_numpy(p) for p in mask_dir.glob("*_refined.npy")]
    union=np.any(masks,axis=0).astype(np.uint8)
    out=mask_dir/"union_mask.npy"; save_numpy(union,out); return out
