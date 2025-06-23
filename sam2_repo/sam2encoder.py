# %%
import os
import torch
from hydra import initialize, compose
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# (1) 작업 디렉토리 설정 → configs 폴더가 포함된 상위 폴더로 이동 (예: /DATA3/sj/sam2)
os.chdir("/DATA3/sj/sam2")

# (2) Hydra 초기화 전에 초기화 상태 제거
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# (3) 올바른 config 경로 지정 (실제 config 파일들이 있는 경로)
configs_dir = "sam2/configs"  # 실제 config 디렉토리 경로 확인
model_cfg_path = "sam2.1/sam2.1_hiera_l.yaml"  # config 내부의 상대 경로

# (4) Hydra 초기화 및 구성 로드
initialize(config_path=configs_dir, job_name="sam2_run")
cfg = compose(config_name=model_cfg_path)

# (5) 모델 생성 및 GPU로 이동
sam_model = instantiate(cfg.model, _recursive_=True)
sam_model = sam_model.cuda()  # 모델을 GPU로 이동
sam_model.eval()

# (6) 더미 이미지 입력 (이미 GPU에 있음)
dummy_img = torch.randn(1, 3, 512, 512).cuda()

with torch.no_grad():
    backbone_out = sam_model.forward_image(dummy_img)


with torch.no_grad():
    backbone_out = sam_model.forward_image(dummy_img)

# (7) 출력 확인
print("backbone_out의 키:", backbone_out.keys())

if "backbone_fpn" in backbone_out:
    print("backbone_fpn에 포함된 feature map 수:", len(backbone_out["backbone_fpn"]))
    for idx, feat in enumerate(backbone_out["backbone_fpn"]):
        print(f"Feature map {idx}: shape {feat.shape}")


# %% [markdown]
# gpt- sam Adapter

# %%
# [Cell 1] - SAMAdapter 클래스 구현
import torch
import torch.nn as nn

# (!!!) SAMAdapter 클래스: SAM2의 ViT-H Encoder 출력 형태를 변환하여 YOLO Neck에 연결
class SAMAdapter(nn.Module):
    def __init__(self, input_dim=1280, output_channels=256, spatial_size=16):
        super(SAMAdapter, self).__init__()
        # 1x1 convolution: 입력 채널 수(input_dim)를 출력 채널 수(output_channels)로 변환
        self.conv = nn.Conv2d(input_dim, output_channels, kernel_size=1)
        
    def forward(self, x):
        """
        입력: x shape = [B, 256, 1280]  (B: 배치 크기)
        변환 과정:
          1. view를 통해 [B, 16, 16, 1280]로 reshape
          2. permute를 통해 [B, 1280, 16, 16]로 차원 순서 변경
          3. 1x1 convolution 적용하여 최종 shape [B, 256, 16, 16] 생성
        """
        B, N, D = x.shape  # N는 256, D는 1280가 되어야 함
        # 확인: 입력이 올바른지 shape를 출력 (디버깅용)
        print(f"입력 텐서 shape: {x.shape}")  # 예: torch.Size([B, 256, 1280])
        
        # [B, 256, 1280] -> [B, 16, 16, 1280]
        x = x.view(B, 16, 16, D)
        # 차원 순서 변경: [B, 16, 16, 1280] -> [B, 1280, 16, 16]
        x = x.permute(0, 3, 1, 2)
        # 1x1 convolution 적용: [B, 1280, 16, 16] -> [B, 256, 16, 16]
        x = self.conv(x)
        
        # 디버깅: 최종 출력 shape 확인
        print(f"출력 텐서 shape: {x.shape}")  # 예: torch.Size([B, 256, 16, 16])
        return x

# 간단한 테스트: 더미 입력 데이터를 생성하여 Adapter 테스트
if __name__ == '__main__':
    # 배치 크기 1, 256, 1280의 더미 텐서 생성
    dummy_input = torch.randn(1, 256, 1280)
    adapter = SAMAdapter()
    output = adapter(dummy_input)



