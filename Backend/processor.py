import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import logging
import nibabel  # type: ignore[import-untyped]
import traceback
from typing import Any, cast, Dict, List, Optional, Tuple
from PIL import Image
from torchvision import transforms

# Removed Keras/Tensorflow imports as models are transitioned to PyTorch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

IMG_SIZE: Tuple[int, int] = (224, 224)
PATCH_SIZE: Tuple[int, int, int] = (128, 128, 128)


def _nib_load(path: str) -> Any:
    return nibabel.load(path)  # type: ignore[no-untyped-call]


def _nib_save(img: Any, path: str) -> None:
    nibabel.save(img, path)  # type: ignore[no-untyped-call]


def _nib_nifti1image(data: np.ndarray, affine: np.ndarray) -> Any:
    return nibabel.Nifti1Image(data, affine)  # type: ignore[no-untyped-call]


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_convs: int) -> None:
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(n_convs):
            layers += [
                nn.Conv3d(ch, out_ch, 3, padding=1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.PReLU(out_ch),
            ]
            ch = out_ch
        self.convs = nn.Sequential(*layers)
        self.proj = (
            nn.Conv3d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x) + self.proj(x)  # type: ignore[no-any-return]


class VNet3D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Encoder
        self.e1 = ResBlock(4, 16, 1)
        self.d1 = nn.Conv3d(16, 16, 2, stride=2, bias=False)
        self.e2 = ResBlock(16, 32, 1)
        self.d2 = nn.Conv3d(32, 32, 2, stride=2, bias=False)
        self.e3 = ResBlock(32, 64, 2)
        self.d3 = nn.Conv3d(64, 64, 2, stride=2, bias=False)
        # Bottleneck
        self.b = ResBlock(64, 128, 2)
        # Decoder
        self.u3 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.r3 = ResBlock(128, 64, 2)
        self.u2 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.r2 = ResBlock(64, 32, 1)
        self.u1 = nn.ConvTranspose3d(32, 16, 2, 2)
        self.r1 = ResBlock(32, 16, 1)
        # Outputs
        self.out = nn.Conv3d(16, 3, 1)
        self.ds2 = nn.Conv3d(32, 3, 1)
        self.ds3 = nn.Conv3d(64, 3, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.e1(x)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        b = self.b(self.d3(e3))
        d3 = self.r3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.r2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.r1(torch.cat([self.u1(d2), e1], dim=1))
        return self.out(d1), self.ds2(d2), self.ds3(d3)  # type: ignore[return-value]


class BrainProcessor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_model: Optional[nn.Module] = None
        self.classification_model: Optional[nn.Module] = None
        self.segmentation_model: Optional[VNet3D] = None
        self.classes: List[str] = ["glioma", "meningioma", "pituitary"]

    def _get_efficientnet_model(self, num_classes: int) -> nn.Module:
        model = efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    def load_models(
        self, detection_path: str, classification_path: str, segmentation_path: str
    ) -> None:
        try:
            self.detection_model = self._get_efficientnet_model(1).to(self.device)
            state_dict = torch.load(detection_path, map_location=self.device)
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.detection_model.load_state_dict(state_dict)
            self.detection_model.eval()
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
            raise

        try:
            self.classification_model = self._get_efficientnet_model(3).to(self.device)
            state_dict = torch.load(classification_path, map_location=self.device)
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.classification_model.load_state_dict(state_dict)
            self.classification_model.eval()
        except Exception as e:
            logger.error(f"Failed to load classification model: {e}")
            raise

        try:
            self.segmentation_model = VNet3D().to(self.device)
            state_dict = torch.load(segmentation_path, map_location=self.device)
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            self.segmentation_model.load_state_dict(state_dict)
            self.segmentation_model.eval()
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
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

    def preprocess_for_torch(self, image: np.ndarray) -> torch.Tensor:
        cropped = self.crop_brain(image)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # Using PIL Image to match training transforms exactly
        pil_img = Image.fromarray(rgb)
        tensor: torch.Tensor = transform(pil_img).unsqueeze(0).to(self.device)
        return tensor

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
        X = self.preprocess_for_torch(image_data)
        with torch.no_grad():
            output = self.detection_model(X)
            prob = torch.sigmoid(output).item()
            return float(prob)

    def run_classification(self, image_data: np.ndarray) -> Dict[str, Any]:
        if self.classification_model is None:
            raise ValueError("Classification model not loaded")
        X = self.preprocess_for_torch(image_data)
        with torch.no_grad():
            outputs = self.classification_model(X)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            class_idx: int = int(np.argmax(probs))
            return {
                "class": self.classes[class_idx],
                "confidence": float(probs[class_idx]),
            }

    def preprocess_modality(self, path: str) -> np.ndarray:
        img: Any = _nib_load(path)
        data: np.ndarray = np.asarray(img.get_fdata()).astype(np.float32)
        mask: np.ndarray = data != 0
        if mask.sum() > 0:
            data[mask] = (data[mask] - float(data[mask].mean())) / (
                float(data[mask].std()) + 1e-08
            )

        # Resizing using interpolation to match training logic
        t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t,
            size=(128, 128, 128),
            mode="trilinear",
            align_corners=False,
        )
        return t.squeeze().numpy().astype(np.float32)

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
                    affine: np.ndarray = np.asarray(ref_img.affine)

                    # Convert to categorical mask (1, 2, 3)
                    cat_probs = np.zeros((128, 128, 128), dtype=np.uint8)
                    cat_probs[probs[0] > 0.5] = 1  # WT
                    cat_probs[probs[1] > 0.5] = 2  # TC
                    cat_probs[probs[2] > 0.5] = 3  # ET

                    # Resize back to original shape using nearest
                    t_mask = (
                        torch.from_numpy(cat_probs.astype(np.float32))
                        .unsqueeze(0)
                        .unsqueeze(0)
                    )
                    t_mask = torch.nn.functional.interpolate(
                        t_mask, size=ref_shape, mode="nearest"
                    )
                    seg_mask = t_mask.squeeze().numpy().astype(np.uint8)

                    final_path = save_path.replace(".nii.gz", ".nii")
                    _nib_save(_nib_nifti1image(seg_mask, affine), final_path)
                    print(f"Segmentation mask saved to {final_path}")
                except Exception:
                    pass
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
