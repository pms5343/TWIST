# %%
# [Cell X] - SAM2 Encoder 출력 shape 확인
import torch
from sam2.build_sam import build_sam2  # SAM2 설치 시 제공되는 모듈

# (!!!) 하이퍼파라미터 및 경로 설정
checkpoint = "/DATA3/sj/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "sam2.1.sam2.1_hiera_l"# 사용하고자 하는 모델 구성 파일

# SAM2 모델 로드 (SAM2Base를 상속받은 전체 모델을 반환)
sam_model = build_sam2(model_cfg, checkpoint)
sam_model.eval()  # evaluation 모드로 설정

# 1장의 512x512, 3채널 더미 이미지 생성 (실제 이미지에 맞춰 전처리 필요)
dummy_img = torch.randn(1, 3, 512, 512).cuda()  # GPU 사용 가정

with torch.no_grad():
    backbone_out = sam_model.forward_image(dummy_img)

# backbone_out은 딕셔너리 형태로 반환됨 (주로 "backbone_fpn"과 "vision_pos_enc" 포함)
print("backbone_out의 키:", backbone_out.keys())

# "backbone_fpn" 내 각 feature map의 shape 출력
if "backbone_fpn" in backbone_out:
    print("backbone_fpn에 포함된 feature map 수:", len(backbone_out["backbone_fpn"]))
    for idx, feat in enumerate(backbone_out["backbone_fpn"]):
        print(f"Feature map {idx}: shape {feat.shape}")

# "vision_pos_enc" 내 각 positional encoding의 shape 출력 (옵션)
if "vision_pos_enc" in backbone_out:
    print("vision_pos_enc에 포함된 positional encoding 수:", len(backbone_out["vision_pos_enc"]))
    for idx, pos_enc in enumerate(backbone_out["vision_pos_enc"]):
        print(f"Positional encoding {idx}: shape {pos_enc.shape}")


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



