"""Utility helpers for checkpointing, segmentation metrics and losses.

- Checkpoint 저장 / 불러오기
- LiTS 3-class (0 Background / 1 Liver / 2 Tumor) Class별 Dice 계산용 통계
- Loss: Soft Dice Loss, CE + Dice Loss
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def save_checkpoint(
    state: dict[str, Any], filename: str = "checkpoint.pth.tar"
) -> None:
    """Serialize a training checkpoint to disk.

    Args:
        state: Dictionary containing at minimum "state_dict" (model
            weights) and optionally "optimizer" (optimizer state).
        filename: Destination file path.
    """
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    map_location: str | torch.device | None = None,
) -> None:
    """Restore model (and optionally optimizer) state from a checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        model: The model whose weights will be loaded.
        optimizer: If provided and the checkpoint contains an
            "optimizer" key, the optimizer state is also restored.
        map_location: Passed to torch.load to control device
            placement (e.g. "cpu" to load a GPU checkpoint on CPU).
    """
    print(f"=> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])


def dice_stats_per_class(logits, targets, classes=(1, 2)):
    """batch의 클래스별 (교집합, 예측 픽셀 수, 정답 픽셀 수) 반환 — 누적용"""
    # 관심 Class인 간(1)과 종양(2)만 계산
    # Background(0)는 전체 Pixel의 약 97%라 항상 0.99 근처 → 포함하면 평균 Dice가 부풀려짐

    # Batch마다 Dice를 바로 계산하지 않고 재료만 반환 → Epoch 전체를 누적한 뒤 한 번에 Dice 계산 (Global Dice)
    #   Batch별 Dice 평균은 종양이 없는 Batch에서 Dice = (0+ε)/(0+ε) = 1이 끼어 부풀려짐
    #   누적 방식에서는 종양 없는 Batch가 분자·분모에 0만 더해 결과에 영향 없음

    # 이진 분할의 sigmoid > 0.5 대신, Pixel마다 점수가 가장 큰 Channel 번호 = 예측 Class
    # dim=1: (B, C, H, W)의 Class Channel 축. 결과 shape가 targets와 같은 (B, H, W)
    pred = torch.argmax(logits, dim=1)
    stats = {}
    for c in classes:
        p = (pred == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()  # 교집합: 예측도 c, 정답도 c인 픽셀 수
        stats[c] = (inter.item(), p.sum().item(), t.sum().item())  # item(): 원소 하나인 Tensor → Python 숫자
    return stats


class DiceLoss(nn.Module):
    """Soft Dice Loss: 1 - mean(Dice_c), c = 간(1), 종양(2)

    지표용 Dice(argmax)와 달리 softmax 확률을 그대로 사용 → 미분 가능
    """

    def __init__(self, classes=(1, 2), smooth=1.0):
        super().__init__()
        self.classes = list(classes)
        self.smooth = smooth  # Loss에서는 1 정도로 크게 두어 분모가 작을 때 Gradient 폭발 방지

    def forward(self, logits, targets):
        # logits (B, C, H, W), targets (B, H, W) long
        probs = F.softmax(logits, dim=1)  # Class 방향으로 확률 합이 1이 되도록 dim=1
        # one_hot: (B, H, W, 3) → permute(0, 3, 1, 2): (B, 3, H, W)로 U-Net 출력과 Shape를 맞춤
        onehot = F.one_hot(targets, num_classes=3).permute(0, 3, 1, 2).float()

        probs = probs[:, self.classes]  # 간, 종양 채널만 → (B, 2, H, W)
        onehot = onehot[:, self.classes]

        dims = (0, 2, 3)  # batch, H, W 방향으로 합 → 클래스별 값 (2,)
        inter = (probs * onehot).sum(dim=dims)
        denom = probs.sum(dim=dims) + onehot.sum(dim=dims)

        dice = (2 * inter + self.smooth) / (denom + self.smooth)  # (2,) 클래스별 soft Dice
        return 1 - dice.mean()


class CEDiceLoss(nn.Module):
    """CE + Dice: 픽셀 단위 안정성(CE) + 클래스 불균형 대응(Dice)"""

    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return (self.ce_weight * self.ce(logits, targets)
                + self.dice_weight * self.dice(logits, targets))
