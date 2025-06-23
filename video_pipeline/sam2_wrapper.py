"""
sam2_wrapper.py ─ SAM2 비디오 추론 래퍼
* seed 인자는 Path 또는 dict 둘 다 허용
"""
from pathlib import Path
import torch, json, numpy as np, cv2
from hydra.core.global_hydra import GlobalHydra
import hydra
from .config import SAM2_CKPT, SAM2_CFG_DIR, GPU_INDEX, OUTPUT_ROOT
from .io_utils import ensure_dir, save_numpy, read_video
from sam2.build_sam import build_sam2_video_predictor

class Sam2Runner:
    def __init__(self):
        device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(version_base="1.3", config_dir=str(SAM2_CFG_DIR))
        self.predictor = build_sam2_video_predictor(
            config_file="sam2.1_hiera_l",
            ckpt_path=str(SAM2_CKPT),
            device=device,
            mode="eval",
            apply_postprocessing=True
        )

    # ----------------------------------------------------------------------
    def segment_all(self, video_path: str | Path,
                    seeds: str | Path | dict) -> Path:
        """
        video_path : 비디오 파일
        seeds      : (1) Path → JSON 파일
                     (2) dict → {"wire1":{frame:int, points:[[x,y,l],...]}, ...}
        반환값     : mask 저장 폴더 Path
        """
        # --- seed 로드 ------------------------------------------------------
        if isinstance(seeds, (str, Path)):
            with open(seeds, "r", encoding="utf-8") as f:
                seeds_dict = json.load(f)
        elif isinstance(seeds, dict):
            seeds_dict = seeds
        else:
            raise TypeError("seeds 인자는 Path 또는 dict 이어야 합니다.")

        # --- 출력폴더 -------------------------------------------------------
        out_dir = OUTPUT_ROOT / "sam2_masks"
        ensure_dir(out_dir)

        # --- 와이어별 추론 --------------------------------------------------
        for i, (wire, cfg) in enumerate(seeds_dict.items(), start=1):
            masks = self._segment_single(
                video_path, obj_id=i,
                frame_idx=cfg["frame"],
                pts=cfg["points"]
            )
            save_numpy(masks, out_dir / f"mask_obj{i}.npy")
        return out_dir

    # ------------------ 내부 전용 ------------------
    def _segment_single(self, video_path, obj_id, frame_idx, pts):
        """
        Path · str 모두 허용.
        SAM2 내부 로더는 str-path + ".mp4" 확장만 지원하므로
        ➜ 반드시 문자열로 변환하고 확장자를 확인한다.
        """
        from pathlib import Path
        import numpy as np, cv2

        video_path = Path(video_path).expanduser().resolve()
        if video_path.suffix.lower() != ".mp4":
            raise ValueError(f"SAM2는 현재 .mp4만 지원합니다 → {video_path.name}")

        video_path_str = str(video_path)      # ← 핵심 수정
        frames = read_video(video_path_str)
        H, W = frames[0].shape[:2]

        # ── SAM2 초기화 & seed 주입 ─────────────────────────────
        state = self.predictor.init_state(video_path=video_path_str)

        coords = np.asarray([[x, y] for x, y, _ in pts], np.float32)
        labels = np.asarray([lbl for *_, lbl in pts], np.int32)
        self.predictor.add_new_points_or_box(
            state, frame_idx, obj_id,
            points=coords, labels=labels,
            clear_old_points=True, normalize_coords=False
        )

        # ── 전 프레임 propagate 후 (T,H,W) mask 배열 생성 ─────
        video_masks = np.zeros((len(frames), H, W), np.uint8)
        for fidx, obj_ids, logits in self.predictor.propagate_in_video(state):
            for oid, logit in zip(obj_ids, logits):
                if logit.ndim == 3:
                    logit = logit[0]
                video_masks[fidx] = (logit > 0).cpu().numpy().astype(np.uint8)
        return video_masks
