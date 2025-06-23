"""
depth_wrapper.py ─ Depth-Anything-V2 추론 래퍼 (경로 패치版)
"""
from pathlib import Path
import cv2, torch, numpy as np, importlib, sys
from .config import DEPTH_CKPT_DIR, DEPTH_SRC_DIR, DEPTH_ENCODER, OUTPUT_ROOT
from .io_utils import ensure_dir, save_numpy

# ── ① 소스 경로를 PYTHONPATH에 추가 -------------------------
if str(DEPTH_SRC_DIR) not in sys.path:
    sys.path.append(str(DEPTH_SRC_DIR))

class DepthRunner:
    def __init__(self, encoder=DEPTH_ENCODER):
        # ② 이제 import 가 정상적으로 동작
        DepthAnythingV2 = importlib.import_module(
            "depth_anything_v2.dpt").DepthAnythingV2

        cfg = {
            "vits": dict(features=64, out_channels=[48, 96, 192, 384]),
            "vitb": dict(features=128, out_channels=[96, 192, 384, 768]),
            "vitl": dict(features=256, out_channels=[256, 512, 1024, 1024]),
            "vitg": dict(features=384, out_channels=[1536, 1536, 1536, 1536]),
        }[encoder]

        self.model = DepthAnythingV2(encoder=encoder, **cfg)
        ckpt = DEPTH_CKPT_DIR / f"depth_anything_v2_{encoder}.pth"
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.to(
            "cuda" if torch.cuda.is_available() else "cpu").eval()

    # 이하 run(), infer_frame() 코드는 이전과 동일 …


    def infer_frame(self, bgr, resize=518):
        depth = self.model.infer_image(bgr, input_size=resize)
        return depth    # float32 (H,W)

    def run(self, video_path: str|Path, frame_idx: int|str="auto"):
        """
        frame_idx:
            • int  → 해당 프레임 1장 추론
            • "auto" → 총프레임/2 위치 자동
        """
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = total//2 if frame_idx=="auto" else int(frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frm = cap.read(); cap.release()
        if not ret: raise RuntimeError("Failed to read frame")

        depth = self.infer_frame(frm)
        out_dir = OUTPUT_ROOT / "depth"
        ensure_dir(out_dir)
        npy_path = out_dir / f"depth_frame{idx:06d}.npy"
        save_numpy(depth, npy_path)
        return npy_path
