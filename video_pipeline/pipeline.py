"""
pipeline.py ─ SAM2 → Depth → Mask 정제 → CoTracker 전체 묶기
"""
from pathlib import Path
from .config import OUTPUT_ROOT, SAVE_DEBUG
from .sam2_wrapper import Sam2Runner
from .depth_wrapper import DepthRunner
from .mask_utils import refine, union_masks, save_numpy, ensure_dir
from .cotrack_wrapper import CoTrackerRunner

def run(video_path: str|Path, seed_json: str|Path):
    sam = Sam2Runner()
    masks_dir = sam.segment_all(video_path, seed_json)

    depth_runner = DepthRunner()
    depth_npy   = depth_runner.run(video_path, frame_idx="auto")

    # -------- 마스크 정제 + 저장 --------------------------------
    refined_dir = OUTPUT_ROOT / "refined_masks"; ensure_dir(refined_dir)
    for m in masks_dir.glob("mask_obj*.npy"):
        refined, (lo,hi) = refine(m, depth_npy)
        out = refined_dir / (m.stem+"_refined.npy")
        save_numpy(refined, out)

    union_path = union_masks(refined_dir)

    # -------- Co-Tracker ---------------------------------------
    cot = CoTrackerRunner()
    cot_out = cot.run(video_path, union_path)

    return {
        "masks": masks_dir,
        "depth": depth_npy,
        "refined": refined_dir,
        "union": union_path,
        "cotracker": cot_out
    }
