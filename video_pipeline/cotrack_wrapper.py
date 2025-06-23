"""
cotrack_wrapper.py ─ Co-Tracker 호출 (개괄)
※ 실제 Co-Tracker 모델 import 경로·API 는 프로젝트마다 달라
   아래 스텁을 여러분 환경에 맞춰 편집하세요.
"""
from pathlib import Path
from .config import GRID_RES, OUTPUT_ROOT
from .io_utils import ensure_dir

class CoTrackerRunner:
    def __init__(self):
        # from cotracker.predictor import CoTrackerPredictor  등등
        pass

    def run(self, video_path: str|Path, mask_path: Path):
        """
        mask_path : (H,W) binary union mask
        """
        ensure_dir(OUTPUT_ROOT/"cotracker")
        # ① mask → grid pts 추출
        # ② predictor 추론
        # ③ 변위 CSV·그래프 저장
        # ---- (사용 프로젝트에 맞춰 구현) ----
        return OUTPUT_ROOT/"cotracker"
