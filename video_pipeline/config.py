"""
config.py ─ 프로젝트 공통 설정 모음
⚠️ ❶ 모든 경로·하이퍼파라미터는 여기서만 수정하면 됩니다.
⚠️ ❷ 한글 주석은 의도적으로 길게 달아 사용 예시·주의점을 설명했습니다.
"""
from pathlib import Path
import datetime

# ── 파일·폴더 경로 ------------------------------------------------------------
PROJECT_ROOT   = Path("/DATA3/Minsoo")               # 프로젝트 루트
VIDEO_DIR      = PROJECT_ROOT / "Input_Video"        # 원본 영상 폴더
SAM2_CKPT      = PROJECT_ROOT / "sam2_repo/checkpoints/sam2.1_hiera_large.pt"
SAM2_CFG_DIR   = PROJECT_ROOT / "sam2_repo/sam2/configs/sam2.1"
DEPTH_CKPT_DIR = PROJECT_ROOT / "Depth-Anything-V2/checkpoints"
# ── Depth-Anything 소스 루트 ─────────────────────────────
DEPTH_SRC_DIR = PROJECT_ROOT / "Depth-Anything-V2" / "Depth-Anything-V2"


# ── 자동 생성되는 출력 루트 -----------------------------------------------------
RUN_ID         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_ROOT    = PROJECT_ROOT / "runs" / f"exp_{RUN_ID}"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── 기본 하이퍼파라미터 --------------------------------------------------------
GPU_INDEX      = 0
WIRES_PER_VIDEO= 3
DEPTH_ENCODER  = "vitl"        # vits/vitb/vitl/vitg
GRID_RES       = 6             # Co-Tracker 격자
DILATE_PX      = 5
TH_METHOD      = "gmm"         # quantile/mean_std/gmm/valley
SAVE_DEBUG     = True
