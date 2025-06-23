"""
io_utils.py ─ 공통 I/O 함수
• 비디오 읽기, numpy 저장/로드, exp 폴더 관리
"""
from pathlib import Path
import cv2, numpy as np

def read_video(video_path: str|Path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frm = cap.read()
        if not ret: break
        frames.append(frm)
    cap.release()
    return frames

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_numpy(arr, path: str|Path):
    np.save(str(path), arr)

def load_numpy(path: str|Path):
    return np.load(str(path), allow_pickle=False)
