"""LiTS 2D slice 기반 Segmentation Dataset.

데이터 구조 (번호 N이 같은 파일끼리 짝):
    data/volume_pt1/volume-N.nii          CT Volume (HU 값), shape (512, 512, Z)
    data/segmentations/segmentation-N.nii GT Mask, shape (512, 512, Z)

Label: 0 = Background, 1 = Liver, 2 = Tumor
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def build_slice_index(data_dir, volume_ids, liver_only=True):
    """(volume_id, slice_idx) 리스트 반환"""
    index = []
    for vid in volume_ids:
        seg = np.asanyarray(nib.load(f"{data_dir}/segmentations/segmentation-{vid}.nii").dataobj)
        for k in range(seg.shape[2]):
            # 간이 없는 Slice는 제외: 데이터 양 감소 + 빈 Slice에서 Dice가 1로 나오는 문제 회피
            if liver_only and not seg[:, :, k].any():
                continue
            index.append((vid, k))
    return index


def hu_window(ct, w_min=-100, w_max=200):
    """CT slice (HU) → 0~1 float32"""
    # [-100, 200] = 간 Window. 범위 밖(-3024 채움값, 뼈 등)은 의도적으로 잘라내 간/종양 대비를 키움
    # ImageNet Normalize는 자연 영상 RGB 통계라 1채널 HU 데이터에는 맞지 않아 사용하지 않음
    ct = np.clip(ct, w_min, w_max)
    ct = (ct - w_min) / (w_max - w_min)
    ct = ct.astype("float32")
    return ct


def minmax_norm(ct):
    """CT slice → 자기 min/max 기준 0~1 float32 (windowing 없음, 대조군)"""
    ct = ct.astype("float32")
    return (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)  # 1e-8: 분모가 0일 때 NaN 방지


class LiTSDataset(Dataset):
    """LiTS 2D Slice Dataset.

    Args:
        data_dir: LiTS 데이터 폴더 (volume_pt1/, segmentations/ 포함)
        volume_ids: 사용할 Volume(환자) 번호 → Train/Val/Test를 환자 단위로 분리
        w_min, w_max: HU Window 범위 (norm="window"일 때 사용)
        img_size: 입력 해상도 (원본 512 → 기본 256으로 Resize)
        liver_only: True면 간이 있는 Slice만 사용
        norm: "window" (HU Windowing) 또는 "minmax" (Slice별 Min-Max)
    """

    def __init__(self, data_dir, volume_ids, w_min=-100, w_max=200,
                 img_size=256, liver_only=True, norm="window"):
        self.data_dir = data_dir
        self.w_min, self.w_max = w_min, w_max
        self.img_size = img_size
        self.norm = norm
        self.index = build_slice_index(data_dir, volume_ids, liver_only)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        vid, k = self.index[i]

        # .dataobj[:, :, k]: Volume 전체가 아니라 k번째 Slice만 디스크에서 읽음 (메모리 매핑)
        # get_fdata()로 전체를 읽으면 float64로 Volume당 최대 약 1.8GB
        ct = nib.load(f"{self.data_dir}/volume_pt1/volume-{vid}.nii").dataobj[:, :, k]  # numpy 배열
        msk = nib.load(f"{self.data_dir}/segmentations/segmentation-{vid}.nii").dataobj[:, :, k]

        # CT 정규화 → 0~1 float32
        if self.norm == "window":
            ct = hu_window(ct, self.w_min, self.w_max)
        elif self.norm == "minmax":
            ct = minmax_norm(ct)
        else:
            raise ValueError(f"알 수 없는 norm: {self.norm}")

        # 표준 판독 방향으로 회전 (CT와 Mask에 반드시 동일하게 적용)
        # .copy(): rot90은 View를 반환 → torch.from_numpy에서 negative stride 에러 방지
        ct, msk = np.rot90(ct).copy(), np.rot90(msk).copy()

        image = torch.from_numpy(ct).unsqueeze(0).float()  # (1, H, W): Conv 입력 형식, CT는 1채널
        mask = torch.from_numpy(msk).long()  # int64, 채널 없이 (H, W) → CrossEntropyLoss가 요구하는 형식

        # 원본 Slice가 512x512라고 가정 (받은 11개 Volume 모두 512)
        # F.interpolate는 4D (N, C, H, W)를 요구 → 차원을 붙였다가 Resize 후 다시 뗌
        # Mask는 interpolate가 long을 처리하지 못해 float로 바꿨다가 long으로 되돌림
        if self.img_size != 512:
            size = (self.img_size, self.img_size)
            # CT: bilinear (연속적인 밝기 값)
            image = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear",
                                  align_corners=False).squeeze(0)
            # Mask: nearest (Class 번호 보존, 0/1/2 외의 값이 생기지 않음)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=size,
                                 mode="nearest").squeeze(0).squeeze(0).long()

        return image, mask


def get_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
    train_ids=range(0, 8),
    val_ids=(8, 9),
    test_ids=(10,),
    w_min: int = -100,
    w_max: int = 200,
    img_size: int = 256,
    norm: str = "window",
):
    """Train / Validation / Test DataLoader 생성"""
    # Volume(환자) 단위 분할: 같은 환자의 인접 Slice가 Train/Val에 섞이는 데이터 누수 방지
    # 전처리 인자(norm, w_min, w_max, img_size)는 Train/Val/Test에 반드시 동일하게 적용
    common = dict(w_min=w_min, w_max=w_max, img_size=img_size, norm=norm)
    train_ds = LiTSDataset(data_dir, volume_ids=train_ids, **common)
    val_ds = LiTSDataset(data_dir, volume_ids=val_ids, **common)
    test_ds = LiTSDataset(data_dir, volume_ids=test_ids, **common)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,  # 학습 데이터만 Epoch마다 순서를 섞음
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


'''
핵심 개념

1. NIfTI / 2D Slice
- LiTS CT는 3D Volume (512, 512, Z), Axial Slice는 z축(axis=2) 방향으로 쌓여 있음
- Voxel Spacing이 비등방성(0.7 x 0.7 x 1~5 mm) → 해상도가 높은 단면 방향으로 2D 학습
- Slice는 (volume_id, slice_idx) 짝으로 관리 (Slice 번호만으로는 어느 환자인지 알 수 없음)
- 간이 있는 Slice만 사용 (전체 5,277장 중 1,688장)

2. CT 전처리
- HU Windowing: 관심 범위 [w_min, w_max] 밖은 잘라내고 0~1로 정규화
- Min-Max(대조군): Slice별 min/max 기준 → 간 주변 대비가 압축되고 Slice마다 기준이 달라짐
- Resize: CT는 Bilinear(밝기 값), Mask는 Nearest(Class 번호 보존)
- 회전 등 공간 변환은 CT와 Mask에 반드시 동일하게 적용

3. Mask 형식 (CrossEntropyLoss 기준)
- Binary(BCE): (B, 1, H, W) float 0/1
- Multi-class(CE): (B, H, W) long, 값은 Class 번호 0/1/2

4. Split
- Slice 단위가 아닌 Volume(환자) 단위 → 인접 Slice로 인한 Data Leakage 방지
- Train 0~7 (1,155장) / Val 8~9 (352장) / Test 10 (181장)

5. 데이터 흐름
- 개별 Sample: Image (1, 256, 256), Mask (256, 256)
- Batch: Image (B, 1, 256, 256), Mask (B, 256, 256)
- DataLoader가 맨 앞에 Batch 차원을 쌓음
'''
