"""U-Net evaluation script for LiTS 3-class segmentation (정량 평가).

학습된 checkpoint(checkpoints/<exp_name>/best_model.pth)를 불러와
Test set(Volume 10)의 간·종양 Global Dice / IoU를 출력.
전처리(norm, window)는 학습 때 저장한 checkpoints/<exp_name>/args.json을 읽어 자동으로 맞춤.
CT / GT / Prediction 시각화(정성 평가)는 노트북에서 진행.

Usage (data/, checkpoints/ 폴더가 있는 위치에서 실행):
    # Loss 비교 (HU [-100, 200])
    python evaluate.py --exp_name ce
    python evaluate.py --exp_name dice
    python evaluate.py --exp_name ce_dice

    # HU 전처리 비교 (CE + Dice)
    python evaluate.py --exp_name ce_dice_minmax
    python evaluate.py --exp_name ce_dice_w160_240
"""

import argparse
import json
import os

import torch
from tqdm import tqdm

from dataset import get_loaders
from model import UNet
from utils import dice_stats_per_class, load_checkpoint


@torch.no_grad()  # 평가만 하므로 Gradient 추적 비활성화
def evaluate_model(loader, model, device):
    """Test set의 간·종양 Global Dice / IoU 출력"""
    model.eval()  # BN은 학습 때 누적한 Running Statistics 사용
    loop = tqdm(loader, desc="Evaluating")

    # Test set 전체를 누적한 뒤 한 번에 계산 (Global Dice / IoU)
    # IoU의 분모(합집합 = 예측 + 정답 - 교집합) 계산을 위해 예측·정답 픽셀 수를 따로 누적
    inter = {1: 0.0, 2: 0.0}  # 교집합 누적
    p_tot = {1: 0.0, 2: 0.0}  # 예측 픽셀 수 누적
    t_tot = {1: 0.0, 2: 0.0}  # 정답 픽셀 수 누적

    for x, y in loop:
        x, y = x.to(device), y.to(device)
        stats = dice_stats_per_class(model(x), y)  # {클래스: (교집합, 예측 수, 정답 수)}
        for c in (1, 2):
            i, p_sum, t_sum = stats[c]
            inter[c] += i
            p_tot[c] += p_sum
            t_tot[c] += t_sum

    smooth = 1e-8  # 0으로 나누는 것을 방지
    print("Test Results (Global)")
    for c, name in [(1, "Liver"), (2, "Tumor")]:
        dice = (2 * inter[c] + smooth) / (p_tot[c] + t_tot[c] + smooth)
        iou = (inter[c] + smooth) / (p_tot[c] + t_tot[c] - inter[c] + smooth)  # 분모 = 합집합
        print(f"  {name:5s}  Dice: {dice:.4f}  |  IoU: {iou:.4f}")


def main(args: argparse.Namespace) -> None:
    """Load a checkpoint and evaluate on the test set."""
    # 실험 폴더의 args.json에서 학습 때 전처리 설정을 읽음
    exp_dir = os.path.join("checkpoints", args.exp_name)
    cfg_path = os.path.join(exp_dir, "args.json")

    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        cfg = {}  # args.json이 없던 예전 실험 (ce, dice, ce_dice) → 그때 쓴 기본값 사용

    # cfg.get(키, 기본값): 키가 없을 때만 기본값 (값이 0이어도 그대로 유지)
    norm = cfg.get("norm", "window")
    w_min = cfg.get("w_min", -100)
    w_max = cfg.get("w_max", 200)
    print(f"Experiment: {args.exp_name} | norm={norm}, window=[{w_min}, {w_max}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Only the test loader is needed for evaluation.
    # 전처리는 args.json에서 읽은 학습 때 설정 사용 (다르면 모델이 처음 보는 밝기 분포를 받음)
    _, _, test_loader = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        norm=norm,
        w_min=w_min,
        w_max=w_max,
    )

    # 학습 때와 구조가 정확히 같아야 checkpoint의 weight가 들어맞음 (다르면 size mismatch 에러)
    model = UNet(in_channels=1, out_channels=3, padding=1).to(device)

    # train.py의 저장 규칙(checkpoints/<exp_name>/)과 동일한 경로
    # 파일이 없으면 랜덤 모델로 평가되는 것을 막기 위해 에러로 중단
    ckpt_path = os.path.join(exp_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint를 찾을 수 없습니다: {ckpt_path}")
    load_checkpoint(ckpt_path, model, map_location=device)

    evaluate_model(test_loader, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained U-Net on the LiTS test set.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="directory for the LiTS dataset (default: data/)",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="ce",
        help="평가할 실험 폴더 (checkpoints/<exp_name>/)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="mini-batch size (default: 4)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of DataLoader workers (default: 2, Windows 기준)",
    )

    main(parser.parse_args())
