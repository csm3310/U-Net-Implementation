"""U-Net model architecture.

Implements the U-Net convolutional network for semantic segmentation as
described in Ronneberger et al. (2015), "U-Net: Convolutional Networks for
Biomedical Image Segmentation" (https://arxiv.org/abs/1505.04597).
"""

import torch
import torch.nn as nn

# Default encoder channel progression: 64 -> 128 -> 256 -> 512.
DEFAULT_FEATURES: tuple[int, ...] = (64, 128, 256, 512)
# ...은 들어갈 원소 수가 정해지지 않았음을 의미함

class DoubleConv(nn.Module):
    """Two consecutive (Conv2d -> BatchNorm2d -> ReLU) blocks.

    This is the fundamental building block used in every level of the
    encoder, decoder, and bottleneck of the U-Net.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        padding: Spatial padding applied to each 3x3 convolution.
            Use 1 to preserve spatial dimensions (modern default)
            or 0 to reproduce the original paper's behavior.
    """

    def __init__(self, in_channels: int, out_channels: int, padding: int = 1) -> None:
        super().__init__()  # nn.Module 초기화: Layer 등록 및 Parameter 관리 기능 활성화
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Sequnetial()은 여러 Neural Network Layer를 순서대로 묶어서 실행하는 PyTorch 클래스
        # BN은 정규화 후 학습 가능한 Shift Parameter(β)를 적용하므로 Conv2d의 Bias가 일반적으로 불필요해

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net encoder-decoder architecture for semantic segmentation.

    Args:
        in_channels: Number of input image channels (e.g. 3 for RGB).
        out_channels: Number of output segmentation classes (1 for
            binary segmentation with BCEWithLogitsLoss).
        features: Channel counts for each encoder level. Defaults to
            (64, 128, 256, 512). The bottleneck automatically uses
            features[-1] * 2 channels.
        padding: Convolution padding. 1 preserves spatial dims
            (modern); 0 reproduces the original paper (output is
            spatially smaller than input).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] | None = None,
        padding: int = 1,
    ) -> None:
        super().__init__()

        if features is None:
            features = DEFAULT_FEATURES

        self.ups = nn.ModuleList() # Decoder에서 사용할 Layer들을 저장
        self.downs = nn.ModuleList() # Encoder에서 사용할 Layer들을 저장 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        ch = in_channels # 입력 이미지의 Channel 수 저장 (지금은 3)
        for feature in features:
            self.downs.append(DoubleConv(ch, feature, padding=padding))
            ch = feature # 다음 DoubleConv의 입력 Channel을 현재 출력 Channel로 변경

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, padding=padding) # Encoder에서 추출한 Feature를 더 깊게 처리하는 부분 (마지막 원소 → 마지막 원소 x2)

        # Decoder
        for feature in reversed(features):  # Encoder와 반대 순서로 Channel을 감소시키며 Decoder 구성
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2) # Channel 수를 줄이고, Spatial Resolution을 복원
            )
            self.ups.append(DoubleConv(feature * 2, feature, padding=padding))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1) # 최종 출력

    # Helpers
    @staticmethod # 클래스 내부에 정의하지만 self를 사용하지 않는 함수
    def center_crop(tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Center-crop a tensor to match target_shape spatially.

        Used when padding=0 to align the (larger) skip connection
        with the (smaller) up-sampled decoder feature map, exactly as
        described in the original paper.

        Args:
            tensor: Skip-connection tensor to crop, shape (B, C, H, W).
            target_shape: Desired output shape (B, C, H_t, W_t).

        Returns:
            Center-cropped view of tensor with spatial size (H_t, W_t).
        """
        _, _, h_target, w_target = target_shape
        _, _, h_skip, w_skip = tensor.shape

        dy = (h_skip - h_target) // 2
        dx = (w_skip - w_target) // 2

        return tensor[:, :, dy : dy + h_target, dx : dx + w_target] # Skip connection을 위해 size 맞추기 위한 Centor Crop

    # Forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the U-Net.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Raw (pre-sigmoid) logits of shape (B, out_channels, H', W').
            When padding=1 and the input spatial dims are divisible by
            16, H' == H and W' == W.
        """
        skip_connections: list[torch.Tensor] = [] # Skip Connection 저장공간

        # Encoder
        for down in self.downs:
            x = down(x) # 현재 Feature Map x를 DoubleConv에 입력 
            skip_connections.append(x) # DoubleConv 결과를 Skip Connection 리스트에 저장 (나중에 Decoder에서 Concat에 사용)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1] # 역순으로 슬라이싱 (Encoder에서 마미작에 저장한 Skip(가장 깊은 곳)을 Decoder가 가장 먼저 쓰므로)

        # Decoder
        for idx in range(0, len(self.ups), 2): # Decoder의 ConvTranspose2d와 DoubleConv를 한 쌍씩 실행하기 위해 Index를 2씩 증가시키는 구조
            # 0, 2, 4, 6
            x = self.ups[idx](x)  # 짝수 Index의 ConvTraspose2d를 실행해 upsampling
            skip = skip_connections[idx // 2] # index // 2를 이용해 해당 Decoder 단계에 맞는 Skip Connection을 가져옴 (한 쌍으로 실행시키려고)

            # When padding=0 the skip connection is larger than x;
            # when padding=1 they may still differ by 1 px if the
            # input size isn't divisible by 16.
            if x.shape != skip.shape:
                skip = self.center_crop(skip, x.shape)

            x = self.ups[idx + 1](torch.cat((skip, x), dim=1)) # 바로 다음 홀수 Index의 DoubleConv 실행 (Channel 차원을 기준으로 Concat)

        return self.final_conv(x)


def test() -> None:
    """Smoke-test both padded and unpadded forward passes."""
    # Test with padding=1 (modern behavior — preserves spatial dims)
    x = torch.randn((3, 1, 160, 160)) # Smoke Test를 위한 가상의 Tensor 정규분포에서 생성
    model = UNet(in_channels=1, out_channels=1, padding=1)
    preds = model(x) # 가상의 데이터를 U-Net에 입력하여 Forward Propagation 수행
    assert preds.shape == x.shape, f"Padded forward failed: {preds.shape} != {x.shape}"
    # assert는 조건이 참인지를 검사하는 문법

    # Test with padding=0 (original paper — 572×572 → 388×388)
    x = torch.randn((3, 1, 572, 572))
    model_no_pad = UNet(in_channels=1, out_channels=1, padding=0)
    preds_no_pad = model_no_pad(x)
    assert preds_no_pad.shape == (
        3,
        1,
        388,
        388,
    ), f"Unpadded forward failed: {preds_no_pad.shape}"

    print("All model architecture tests passed!")


if __name__ == "__main__":
    test()


'''
핵심 개념

1. nn.Module / ModuleList / Sequential
- nn.Module: PyTorch의 Layer 및 Parameter를 관리하는 기본 클래스.
- nn.ModuleList: 여러 Layer를 저장하고 등록하지만 자동으로 순차 실행하지 않음.
- nn.Sequential: 여러 Layer를 순서대로 연결하여 실행.
- __init__(): Network를 구성할 Layer 정의.
- forward(): 정의된 Layer에 실제 데이터를 입력하여 연산 수행.

2. Encoder / Bottleneck / Decoder
- Encoder: DoubleConv로 Feature 추출 후 MaxPool을 통해 Spatial Resolution 감소.
- Bottleneck: 가장 낮은 Spatial Resolution과 가장 많은 Channel을 갖는 Feature 처리.
- Decoder: ConvTranspose2d로 Upsampling 후 Skip Connection과 Concatenation.
- Channel 수는 Encoder에서 증가하고 Decoder에서 감소.

3. Skip Connection
- Encoder의 DoubleConv 출력을 MaxPool 이전에 저장.
- Decoder에서 대응하는 Encoder Feature Map을 Concatenation.
- Encoder의 Spatial Information을 Decoder에 직접 전달하여 정밀한 Localization 지원.
- Concatenation은 Channel 차원(dim=1)을 기준으로 수행.

4. Padding / Center Crop
- padding=1: 3×3 Convolution에서 Spatial Resolution 유지.
- padding=0: Valid Convolution으로 Feature Map의 Spatial Resolution 감소.
- Center Crop: Encoder와 Decoder Feature Map의 Spatial Size를 맞추기 위해 사용.
- 원본 U-Net은 Valid Convolution과 Crop 기반 Skip Connection 사용.

5. Forward / Output
- Encoder → Bottleneck → Decoder → Final Convolution 순서로 수행.
- Final 1×1 Convolution은 각 Pixel의 Feature Channel을 출력 Channel 수로 변환.
- Binary Segmentation에서는 out_channels=1 (Sigmoid), LiTS 3-class에서는 out_channels=3 (Softmax).
- 최종 출력은 Sigmoid/Softmax가 적용되지 않은 Raw Logits.
- Loss 계산 및 Metrics 계산 과정에서 목적에 맞게 Logits를 처리.

6. 역할 요약
- ConvTranspose2d: Upsampling을 통해 Spatial Resolution을 증가시키는 역할.
- DoubleConv: Skip Connection으로 결합된 Encoder와 Decoder의 Feature를 학습하고 정제하는 역할.
- LiTS 적용 시 model.py는 수정 없이 UNet(in_channels=1, out_channels=3)으로 생성만 바꿈.
'''