import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import nibabel  # type: ignore[import-untyped]
import traceback
from typing import Any, cast, Dict, List, Optional, Tuple

try:
    from keras.applications import ResNet50  # type: ignore[import-untyped]
    from keras.applications.resnet50 import preprocess_input  # type: ignore[import-untyped]
    from keras.models import Sequential, load_model  # type: ignore[import-untyped]
    from keras.layers import GlobalAveragePooling2D, Dropout, Dense  # type: ignore[import-untyped]
except ImportError:
    from tensorflow.keras.applications import ResNet50  # type: ignore[import-untyped,no-redef]
    from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore[import-untyped,no-redef]
    from tensorflow.keras.models import Sequential, load_model  # type: ignore[import-untyped,no-redef]
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense  # type: ignore[import-untyped,no-redef]

IMG_SIZE: Tuple[int, int] = (224, 224)
PATCH_SIZE: Tuple[int, int, int] = (128, 128, 128)


def _nib_load(path: str) -> Any:
    return nibabel.load(path)  # type: ignore[no-untyped-call]


def _nib_save(img: Any, path: str) -> None:
    nibabel.save(img, path)  # type: ignore[no-untyped-call]


def _nib_nifti1image(data: np.ndarray, affine: np.ndarray) -> Any:
    return nibabel.Nifti1Image(data, affine)  # type: ignore[no-untyped-call]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.01, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(x)  # type: ignore[no-any-return]


class nnUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        f: List[int] = [32, 64, 128, 256]
        self.e1 = ConvBlock(4, f[0])
        self.e2 = ConvBlock(f[0], f[1])
        self.e3 = ConvBlock(f[1], f[2])
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(f[2], f[3])
        self.u3 = nn.ConvTranspose3d(f[3], f[2], 2, 2)
        self.u2 = nn.ConvTranspose3d(f[2], f[1], 2, 2)
        self.u1 = nn.ConvTranspose3d(f[1], f[0], 2, 2)
        self.d3 = ConvBlock(f[3], f[2])
        self.d2 = ConvBlock(f[2], f[1])
        self.d1 = ConvBlock(f[1], f[0])
        self.out = nn.Conv3d(f[0], 3, 1)
        self.ds2 = nn.Conv3d(f[1], 3, 1)
        self.ds3 = nn.Conv3d(f[2], 3, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b = self.bottleneck(self.pool(s3))
        d3 = self.d3(torch.cat([self.u3(b), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.out(d1), self.ds2(d2), self.ds3(d3)


class BrainProcessor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_model: Optional[Any] = None
        self.classification_model: Optional[Any] = None
        self.segmentation_model: Optional[nnUNet] = None
        self.classes: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]

    def _rebuild_classification_model(self) -> Any:
        base_model: Any = cast(
            Any,
            ResNet50(  # type: ignore[arg-type]
                weights=None,
                include_top=False,
                input_shape=(224, 224, 3),
            ),
        )
        model: Any = cast(
            Any,
            Sequential(
                [
                    base_model,
                    GlobalAveragePooling2D(name="global_average_pooling2d"),
                    Dropout(0.5, name="dropout"),
                    Dense(4, activation="softmax", name="dense"),
                ]
            ),
        )
        return model

    def load_models(
        self, detection_path: str, classification_path: str, segmentation_path: str
    ) -> None:
        print(f"Loading Detection model from {detection_path}...")
        try:
            self.detection_model = load_model(detection_path, compile=False)
            print("Detection model loaded.")
        except Exception as e:
            print(f"Error loading Detection model: {e}")
            traceback.print_exc()
            raise
        print(f"Loading Classification model from {classification_path}...")
        try:
            cls_model: Any = self._rebuild_classification_model()
            cls_model.load_weights(classification_path)
            self.classification_model = cls_model
            print("Classification model loaded successfully (rebuilt + weights).")
        except Exception as e:
            print(f"Error loading Classification model: {e}")
            traceback.print_exc()
            raise
        print(f"Loading Segmentation model from {segmentation_path}...")
        try:
            self.segmentation_model = nnUNet().to(self.device)
            state_dict = torch.load(segmentation_path, map_location=self.device)
            self.segmentation_model.load_state_dict(state_dict)
            self.segmentation_model.eval()
            print("Segmentation model loaded.")
        except Exception as e:
            print(f"Error loading Segmentation model: {e}")
            traceback.print_exc()
            raise

    def crop_brain(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            return cv2.resize(image, IMG_SIZE)  # type: ignore[return-value]
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return cv2.resize(image[y : y + h, x : x + w], IMG_SIZE)  # type: ignore[return-value]

    def preprocess_for_tf(self, image: np.ndarray) -> np.ndarray:
        cropped = self.crop_brain(image)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        result: np.ndarray = cast(
            np.ndarray,
            preprocess_input(np.expand_dims(rgb.astype(np.float32), axis=0)),
        )
        return result

    def nifti_to_2d(self, path: str) -> np.ndarray:
        img: Any = _nib_load(path)
        data: np.ndarray = np.asarray(img.get_fdata())
        if len(data.shape) == 3:
            mid: int = data.shape[2] // 2
            slice_data: np.ndarray = data[:, :, mid]
        else:
            slice_data = data[0]
        s_min, s_max = np.min(slice_data), np.max(slice_data)
        if s_max > s_min:
            slice_data = (slice_data - s_min) / (s_max - s_min) * 255
        else:
            slice_data = np.zeros_like(slice_data)
        slice_data = slice_data.astype(np.uint8)
        return cv2.cvtColor(slice_data, cv2.COLOR_GRAY2BGR)  # type: ignore[return-value]

    def get_slice_as_image(
        self, path: str, slice_idx: int
    ) -> Tuple[Optional[np.ndarray], int]:
        try:
            proxy: Any = _nib_load(path)
            data_shape: Tuple[int, ...] = tuple(proxy.shape)
            num_slices: int
            slice_data: np.ndarray
            if len(data_shape) == 4:
                num_slices = int(data_shape[2])
                slice_data = np.array(proxy.dataobj[:, :, slice_idx, 0])
            elif len(data_shape) == 3:
                num_slices = int(data_shape[2])
                slice_data = np.array(proxy.dataobj[:, :, slice_idx])
            else:
                return None, 1
            s_min = float(np.min(slice_data))
            s_max = float(np.percentile(slice_data, 99.5))
            if s_max > s_min:
                slice_data = np.clip(slice_data, s_min, s_max)
                slice_data = (slice_data - s_min) / (s_max - s_min) * 255
            else:
                slice_data = np.zeros_like(slice_data)
            slice_data = slice_data.astype(np.uint8)
            slice_data = cv2.rotate(slice_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return cv2.cvtColor(slice_data, cv2.COLOR_GRAY2BGR), num_slices  # type: ignore[return-value]
        except Exception as e:
            print(f"Error in get_slice_as_image: {e}")
            return None, 0

    def run_detection(self, image_data: np.ndarray) -> float:
        if self.detection_model is None:
            raise ValueError("Detection model not loaded")
        X = self.preprocess_for_tf(image_data)
        pred: np.ndarray = self.detection_model.predict(X)
        if pred.shape[1] == 1:
            return float(pred[0][0])
        else:
            return float(pred[0][1])

    def run_classification(self, image_data: np.ndarray) -> Dict[str, Any]:
        if self.classification_model is None:
            raise ValueError("Classification model not loaded")
        X = self.preprocess_for_tf(image_data)
        preds: np.ndarray = self.classification_model.predict(X)
        class_idx: int = int(np.argmax(preds[0]))
        return {
            "class": self.classes[class_idx],
            "confidence": float(preds[0][class_idx]),
        }

    def preprocess_modality(self, path: str) -> np.ndarray:
        img: Any = _nib_load(path)
        data: np.ndarray = np.asarray(img.get_fdata()).astype(np.float32)
        mask: np.ndarray = data != 0
        if mask.sum() > 0:
            data[mask] = (data[mask] - float(data[mask].mean())) / (
                float(data[mask].std()) + 1e-08
            )
        d, h, w = data.shape
        start_d, start_h, start_w = (d - 128) // 2, (h - 128) // 2, (w - 128) // 2
        cropped = np.zeros((128, 128, 128), dtype=np.float32)
        d_src = slice(max(0, start_d), min(d, start_d + 128))
        h_src = slice(max(0, start_h), min(h, start_h + 128))
        w_src = slice(max(0, start_w), min(w, start_w + 128))
        d_dst = slice(max(0, -start_d), min(128, d - start_d))
        h_dst = slice(max(0, -start_h), min(128, h - start_h))
        w_dst = slice(max(0, -start_w), min(128, w - start_w))
        cropped[d_dst, h_dst, w_dst] = data[d_src, h_src, w_src]
        return cropped

    def run_segmentation(
        self, modality_paths: Dict[str, str], save_path: Optional[str] = None
    ) -> Dict[str, float]:
        if self.segmentation_model is None:
            return {
                "tumorVolume": 0.0,
                "wtVolume": 0.0,
                "tcVolume": 0.0,
                "etVolume": 0.0,
            }
        imgs: List[np.ndarray] = []
        for k in ["flair", "t1", "t1ce", "t2"]:
            imgs.append(self.preprocess_modality(modality_paths[k]))
        stacked: np.ndarray = np.stack(imgs)
        X: torch.Tensor = cast(
            torch.Tensor,
            torch.from_numpy(stacked).unsqueeze(0).to(self.device),
        )
        self.segmentation_model.eval()
        with torch.no_grad():
            raw_output = self.segmentation_model(X)
            output: torch.Tensor = cast(
                torch.Tensor,
                raw_output[0] if isinstance(raw_output, tuple) else raw_output,
            )
            sigmoid_out: torch.Tensor = torch.sigmoid(output)
            probs: np.ndarray = sigmoid_out.cpu().numpy()[0]
            wt_vol = float(np.sum(probs[0] > 0.5) / 1e3)
            tc_vol = float(np.sum(probs[1] > 0.5) / 1e3)
            et_vol = float(np.sum(probs[2] > 0.5) / 1e3)
            if save_path:
                try:
                    ref_img: Any = _nib_load(modality_paths["flair"])
                    ref_shape: Tuple[int, ...] = tuple(ref_img.shape)
                    d, h, w = int(ref_shape[0]), int(ref_shape[1]), int(ref_shape[2])
                    affine: np.ndarray = np.asarray(ref_img.affine)
                    seg_mask = np.zeros((d, h, w), dtype=np.uint8)
                    start_d = (d - 128) // 2
                    start_h = (h - 128) // 2
                    start_w = (w - 128) // 2
                    d_dst = slice(max(0, start_d), min(d, start_d + 128))
                    h_dst = slice(max(0, start_h), min(h, start_h + 128))
                    w_dst = slice(max(0, start_w), min(w, start_w + 128))
                    d_src = slice(max(0, -start_d), min(128, d - start_d))
                    h_src = slice(max(0, -start_h), min(128, h - start_h))
                    w_src = slice(max(0, -start_w), min(128, w - start_w))
                    local_wt = (probs[0, d_src, h_src, w_src] > 0.5).astype(np.uint8)
                    local_tc = (probs[1, d_src, h_src, w_src] > 0.5).astype(np.uint8)
                    local_et = (probs[2, d_src, h_src, w_src] > 0.5).astype(np.uint8)
                    local_seg = np.zeros((128, 128, 128), dtype=np.uint8)
                    local_seg[local_wt == 1] = 1
                    local_seg[local_tc == 1] = 2
                    local_seg[local_et == 1] = 3
                    seg_mask[d_dst, h_dst, w_dst] = local_seg[d_src, h_src, w_src]
                    final_path = save_path.replace(".nii.gz", ".nii")
                    _nib_save(_nib_nifti1image(seg_mask, affine), final_path)
                    print(f"Segmentation mask saved to {final_path}")
                except Exception as e:
                    print(f"Failed to save segmentation mask: {e}")
                    traceback.print_exc()
            return {
                "tumorVolume": wt_vol,
                "wtVolume": wt_vol,
                "tcVolume": tc_vol,
                "etVolume": et_vol,
            }

    def get_segmentation_slice(self, path: str, slice_idx: int) -> Optional[np.ndarray]:
        try:
            if not os.path.exists(path):
                return None
            proxy: Any = _nib_load(path)
            proxy_shape: Tuple[int, ...] = tuple(proxy.shape)
            if len(proxy_shape) != 3:
                print(f"Unexpected mask shape: {proxy_shape}")
                return None
            slice_data: np.ndarray = np.array(proxy.dataobj[:, :, slice_idx])
            slice_data = np.rot90(slice_data, 1)
            h, w = slice_data.shape
            rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
            rgb_mask[slice_data == 1, 0] = 255
            rgb_mask[slice_data == 2, 0] = 255
            rgb_mask[slice_data == 2, 1] = 255
            rgb_mask[slice_data == 3, 0] = 255
            rgb_mask[slice_data == 3, 1] = 255
            rgb_mask[slice_data == 3, 2] = 255
            return rgb_mask
        except Exception as e:
            print(f"Error in get_segmentation_slice: {e}")
            return None
