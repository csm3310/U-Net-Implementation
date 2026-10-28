"""U-Net training script for LiTS 3-class segmentation.

Trains a UNet (in 1ch / out 3ch: 0 Background, 1 Liver, 2 Tumor) on LiTS 2D slices.
Loss는 --loss로 선택 (ce / dice / ce_dice), CT 정규화는 --norm (window / minmax)과
--w_min / --w_max로 선택. The best model (by 0.3 × Liver Dice + 0.7 × Tumor Dice on
validation) is checkpointed to checkpoints/<exp_name>/ (기본값 = loss 이름).

Usage (data/ 폴더가 있는 위치에서 실행):
    # Loss 비교 (HU [-100, 200])
    python train.py --loss ce      --epochs 20 --batch_size 4 --seed 42
    python train.py --loss dice    --epochs 20 --batch_size 4 --seed 42
    python train.py --loss ce_dice --epochs 20 --batch_size 4 --seed 42

    # HU 전처리 비교 (CE + Dice 고정)
    python train.py --loss ce_dice --norm minmax                          --exp_name ce_dice_minmax
    python train.py --loss ce_dice --norm window --w_min -160 --w_max 240 --exp_name ce_dice_w160_240
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_loaders
from model import UNet
from utils import CEDiceLoss, DiceLoss, dice_stats_per_class, save_checkpoint


def train_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Train the model for a single epoch.

    Args:
        loader: Training data loader.
        model: The segmentation model.
        optimizer: Optimizer instance.
        loss_fn: Loss function (expects raw logits).
        device: Device to run on.

    Returns:
        (avg_loss, liver_dice, tumor_dice)
        loss는 Batch 평균, Dice는 Epoch 전체 누적 (Global Dice).
    """
    model.train()  # Training Mode로 전환
    loop = tqdm(loader, desc="Training")

    total_loss = 0.0
    inter = {1: 0.0, 2: 0.0}  # 간, 종양 따로 — Global Dice 계산용 누적 (교집합)
    denom = {1: 0.0, 2: 0.0}  # (예측 픽셀 수 + 정답 픽셀 수)

    for data, targets in loop:
        data = data.to(device)  # 모델 입력 데이터
        targets = targets.to(device)  # GT 마스크

        # Forward
        predictions = model(data)  # 예측 (Logit)
        loss = loss_fn(predictions, targets)  # Loss 구하기

        # Backward
        optimizer.zero_grad()  # 이전 Batch에서의 Gradient 초기화
        loss.backward()  # Loss에 대해 각 Weight의 Gradient 계산 (Chain rule을 이용한 Backpropagation)
        optimizer.step()  # Weight 업데이트

        # Metrics: Logit을 그대로 전달 (argmax로 Class 선택은 dice_stats_per_class 내부에서 처리)
        with torch.no_grad():  # 평가지표 계산이므로 gradient 추적 막기
            stats = dice_stats_per_class(predictions, targets)

        for c in (1, 2):
            i, p_sum, t_sum = stats[c]
            inter[c] += i
            denom[c] += p_sum + t_sum

        total_loss += loss.item()  # Batch당 1번 (for c 밖)

        # 지금까지 누적한 Dice (running Dice)
        liver = (2 * inter[1] + 1e-8) / (denom[1] + 1e-8)
        tumor = (2 * inter[2] + 1e-8) / (denom[2] + 1e-8)
        loop.set_postfix(loss=f"{loss.item():.4f}", liver=f"{liver:.3f}", tumor=f"{tumor:.3f}")

    n = len(loader)
    return total_loss / n, liver, tumor  # 마지막 running Dice = Epoch 전체 Global Dice


@torch.no_grad()  # Validation 과정에서 Gradient 추적 비활성화
def validate(
    loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the model on the validation set.

    Args:
        loader: Validation data loader.
        model: The segmentation model.
        loss_fn: Loss function (expects raw logits).
        device: Device to run on.

    Returns:
        (avg_loss, liver_dice, tumor_dice)
        loss는 Batch 평균, Dice는 Epoch 전체 누적 (Global Dice).
    """
    model.eval()  # Evaluation Mode: Dropout 비활성화, BN은 Running Statistics 사용
    loop = tqdm(loader, desc="Validation")

    total_loss = 0.0
    inter = {1: 0.0, 2: 0.0}
    denom = {1: 0.0, 2: 0.0}

    for data, targets in loop:
        data = data.to(device)
        targets = targets.to(device)

        predictions = model(data)
        loss = loss_fn(predictions, targets)

        stats = dice_stats_per_class(predictions, targets)
        for c in (1, 2):
            i, p_sum, t_sum = stats[c]
            inter[c] += i  # 교집합
            denom[c] += p_sum + t_sum
        total_loss += loss.item()

    smooth = 1e-8
    avg_loss = total_loss / len(loader)
    avg_liver = (2 * inter[1] + smooth) / (denom[1] + smooth)
    avg_tumor = (2 * inter[2] + smooth) / (denom[2] + smooth)

    print(
        f"  Validation - Loss: {avg_loss:.4f} | Liver Dice: {avg_liver:.4f} | Tumor Dice: {avg_tumor:.4f}"
    )
    return avg_loss, avg_liver, avg_tumor


def main(args: argparse.Namespace) -> None:
    """Set up data, model, and optimizer, then run the training loop.

    The best model checkpoint (by validation score = 0.3 × Liver Dice + 0.7 × Tumor Dice)
    is saved to checkpoints/<exp_name>/best_model.pth (exp_name 기본값 = loss 이름).
    """
    # 시드 고정: 초기 Weight와 DataLoader shuffle 순서를 실험끼리 동일하게 → 비교 변수 외 통제
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Seed: {args.seed}")

    # 실험별 저장 폴더: --exp_name이 있으면 그 이름, 없으면(None) loss 이름
    #   예) checkpoints/ce, checkpoints/ce_dice_minmax, checkpoints/ce_dice_w160_240
    # → 실험끼리 best_model.pth / history.csv가 덮어써지지 않음
    exp_name = args.exp_name or args.loss
    out_dir = os.path.join("checkpoints", exp_name)
    os.makedirs(out_dir, exist_ok=True)  # 체크포인트 저장 폴더 생성

    # 실험 설정(loss, norm, window, seed 등)을 함께 저장 → evaluate.py가 읽어 같은 전처리로 평가
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader, _ = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        norm=args.norm,
        w_min=args.w_min,
        w_max=args.w_max,
    )

    # GrayScale CT라서 in_channels=1 / Class 3개(배경, 간, 종양)라서 out_channels=3
    model = UNet(in_channels=1, out_channels=3, padding=1).to(device)

    # Loss 선택 (--loss 인자, 기본값 ce)
    #   ce      : 픽셀 단위 CE (내부 softmax, 정답은 (B, H, W) long)
    #   dice    : Soft Dice Loss (간·종양 클래스별 겹침 비율 → 불균형에 강함)
    #   ce_dice : CE + Dice (1:1)
    losses = {
        "ce": nn.CrossEntropyLoss(),
        "dice": DiceLoss(),
        "ce_dice": CEDiceLoss(),
    }
    loss_fn = losses[args.loss]
    print(f"Loss: {args.loss}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)  # 업데이트할 Weight와 학습률 지정

    best_val_score = 0.0

    for epoch in range(args.epochs):  # 지정한 Epoch만큼 Training과 Validation 반복
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_liver, train_tumor = train_epoch(
            train_loader, model, optimizer, loss_fn, device
        )
        val_loss, val_liver, val_tumor = validate(val_loader, model, loss_fn, device)

        # Best Checkpoint 기준: 0.3 × Liver Dice + 0.7 × Tumor Dice
        # 둘 다 중요하지만, 더 어렵고 Pixel 수가 적은 Tumor에 가중치를 더 줌
        val_score = 0.3 * val_liver + 0.7 * val_tumor
        print(f"  Val Score (0.3·Liver + 0.7·Tumor): {val_score:.4f}")
        print(f"  Train      - Loss: {train_loss:.4f} | Liver Dice: {train_liver:.4f} | Tumor Dice: {train_tumor:.4f}")

        # Epoch별 기록을 CSV로 저장 (그래프 그리기용)
        # 첫 Epoch는 "w"(새로 쓰기)로 이전 기록을 지우고, 이후 Epoch는 "a"(이어 쓰기)
        with open(os.path.join(out_dir, "history.csv"), "w" if epoch == 0 else "a") as f:
            if epoch == 0:
                f.write("epoch,train_loss,train_liver,train_tumor,val_loss,val_liver,val_tumor,val_score\n")
            f.write(f"{epoch + 1},{train_loss:.4f},{train_liver:.4f},{train_tumor:.4f},"
                    f"{val_loss:.4f},{val_liver:.4f},{val_tumor:.4f},{val_score:.4f}\n")

        # 마지막 Epoch가 아니라 Validation Score가 가장 높은 Epoch의 모델을 저장
        if val_score > best_val_score:
            best_val_score = val_score
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=os.path.join(out_dir, "best_model.pth"))

    print(f"\nTraining complete. Best validation score: {best_val_score:.4f}")


if __name__ == "__main__":  # 직접 실행할 때만 동작, 하이퍼파라미터는 명령줄 인자로 지정
    parser = argparse.ArgumentParser(
        description="Train a U-Net model on the LiTS dataset (2D slices, 3-class).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="directory for the LiTS dataset (default: data/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="mini-batch size (default: 4, RTX 3050 4GB / 256px 기준)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Adam learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of DataLoader workers (default: 2, Windows 기준)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "dice", "ce_dice"],  # 이 셋 외의 값을 주면 argparse가 에러를 냄
        help="loss function (default: ce)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for weight init / shuffle (default: 42)",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="window",
        choices=["window", "minmax"],
        help="CT 정규화 방식 (default: window)",
    )
    parser.add_argument(
        "--w_min",
        type=int,
        default=-100,
        help="HU window 하한 (default: -100)",
    )
    parser.add_argument(
        "--w_max",
        type=int,
        default=200,
        help="HU window 상한 (default: 200)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="저장 폴더 이름 (default: loss 이름)",
    )

    main(parser.parse_args())


'''
핵심 개념

1. Loss / Prediction (Multi-class)
- CrossEntropyLoss: 내부적으로 Softmax와 CE를 함께 계산하므로 모델 출력(Logit)을 그대로 입력.
- 예측 Mask는 Sigmoid + Threshold 대신 Argmax(dim=1)로 Class 번호 선택.
- Dice Loss: Softmax 확률 기반 Soft Dice → Class별 겹침 비율이라 Class Imbalance에 강함.
- CE + Dice: CE의 학습 안정성 + Dice의 불균형 대응 (Dice 단독은 Softmax 포화로 붕괴할 수 있음).

2. Forward / Backward / Optimizer
- Forward: 입력 데이터를 통해 Prediction 생성 및 Loss 계산.
- Backward: Chain Rule을 이용하여 각 Weight의 Gradient 계산.
- Optimizer: 계산된 Gradient를 기반으로 실제 Weight 업데이트.
- zero_grad(): Gradient는 누적되므로 Batch마다 초기화.

3. Training / Evaluation Mode
- model.train(): Dropout과 Batch Normalization 등을 Training Mode로 전환.
- model.eval(): Dropout 비활성화 및 Batch Normalization의 Running Statistics 사용.
- torch.no_grad(): Gradient 추적을 비활성화하여 메모리 사용 및 연산 비용 감소.
- model.eval() 자체는 Gradient 추적을 비활성화하지 않음.

4. Metric (Global Dice)
- 간(1)과 종양(2)의 Dice를 따로 계산 (Background는 97%라 평균을 부풀려 제외).
- Batch별 Dice 평균 대신 Epoch 전체의 교집합과 (예측 + 정답)을 누적한 뒤 한 번에 계산.
- 종양이 없는 Batch에서 Dice = 1로 과대평가되는 문제 방지.

5. Epoch / Batch
- Batch: 전체 Dataset을 나누어 한 번에 처리하는 데이터 묶음.
- Epoch: 전체 Training Dataset을 한 번 학습하는 과정.
- 각 Epoch에서 Weight를 초기화하지 않고 이전 학습 결과를 이어서 업데이트.

6. Checkpoint / 실험 관리
- Validation Score (0.3 × Liver Dice + 0.7 × Tumor Dice)가 최고일 때 모델 저장.
- 마지막 Epoch가 아닌 Validation 성능이 가장 좋았던 모델을 저장.
- 실험별 폴더(checkpoints/<exp_name>/)에 best_model.pth, history.csv, args.json 저장.
- Seed 고정으로 초기 Weight와 데이터 순서를 실험끼리 동일하게 맞춤.
'''
