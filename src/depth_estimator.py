import numpy as np
from PIL import Image


class DepthEstimator:
    """Monocular depth estimator wrapper supporting dummy and MoGe modes."""

    def __init__(self, mode: str = "dummy"):
        self.mode = mode.lower()
        self.model = None
        self.device = "cpu"
        self.torch = None
        self.moge_available = False

        if self.mode == "moge":
            self._load_moge()

        if self.mode != "moge":
            self.mode = "dummy"

    def _load_moge(self) -> None:
        try:
            import torch
            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as exc:
            raise RuntimeError("Failed to import torch for MoGe mode") from exc

        try:
            # Use the official, exact MoGe import path as requested
            from moge.model.v2 import MoGeModel
        except Exception as exc:
            raise RuntimeError("Failed to import MoGeModel from moge.model.v2") from exc

        try:
            # Load the specified pretrained checkpoint
            self.model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
            if hasattr(self.model, "to"):
                self.model.to(self.device)
            if hasattr(self.model, "eval"):
                self.model.eval()

            self.moge_available = True
        except Exception as exc:
            raise RuntimeError("Failed to initialize MoGeModel.from_pretrained") from exc

    def _set_dummy_mode(self) -> None:
        self.mode = "dummy"
        self.model = None
        self.device = "cpu"
        self.torch = None
        self.moge_available = False

    def predict(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict a depth map and camera intrinsics from a single image.

        Args:
            image_path: Path to the input RGB image.

        Returns:
            depth: A positive depth map of shape (H, W).
            K: Pinhole camera intrinsic matrix of shape (3, 3).
        """
        if self.mode == "moge" and self.moge_available:
            try:
                return self._predict_moge(image_path)
            except Exception as exc:
                raise RuntimeError("MoGe prediction failed") from exc

        return self._predict_dummy(image_path)

    def _predict_dummy(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        xv, yv = np.meshgrid(np.linspace(0.0, 1.0, width), np.linspace(0.0, 1.0, height))
        depth = 0.5 + 1.5 * ((xv + yv) / 2.0)
        depth = depth.astype(np.float32)

        K = self._default_intrinsics(width, height)
        return depth, K

    def _predict_moge(self, image_path: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            import cv2
        except Exception as exc:
            print("Error importing cv2 for MoGe inference:", exc)
            raise

        # Load image with cv2 (BGR), convert to RGB and normalize to [0,1]
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image_np = rgb.astype(np.float32) / 255.0

        # Convert to tensor [B,3,H,W]
        tensor = self.torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

        try:
            output = None
            if hasattr(self.model, "infer"):
                output = self.model.infer(tensor)
            elif callable(self.model):
                output = self.model(tensor)
            elif hasattr(self.model, "forward"):
                output = self.model.forward(tensor)
            else:
                raise RuntimeError("Loaded MoGe model is not callable and has no 'infer' method.")
        except Exception as exc:
            print("Error during MoGe inference:", exc)
            raise

        if not isinstance(output, dict):
            raise ValueError("Expected MoGe model.infer(...) to return a dict with keys 'depth' and 'intrinsics'.")

        if "depth" not in output or "intrinsics" not in output:
            raise KeyError("MoGe output missing required keys 'depth' and/or 'intrinsics'.")

        depth_out = output["depth"]
        intr_norm = output["intrinsics"]

        # Convert tensors to numpy if needed, then squeeze batch dimensions
        if isinstance(depth_out, self.torch.Tensor):
            depth_np = depth_out.detach().cpu().numpy()
        else:
            depth_np = np.asarray(depth_out)

        if isinstance(intr_norm, self.torch.Tensor):
            intr_np = intr_norm.detach().cpu().numpy()
        else:
            intr_np = np.asarray(intr_norm)

        # Squeeze singleton batch dimensions (e.g., [1, H, W] -> [H, W]; [1,3,3] -> [3,3])
        depth_np = np.squeeze(depth_np)
        intr_np = np.squeeze(intr_np)

        # Debug prints for shapes
        try:
            print("MoGe depth shape:", depth_np.shape)
        except Exception:
            print("MoGe depth shape: unknown")
        try:
            print("MoGe intrinsics shape:", intr_np.shape)
        except Exception:
            print("MoGe intrinsics shape: unknown")

        # Ensure depth is 2D [H, W]
        if depth_np.ndim == 3 and depth_np.shape[0] in (1, 3):
            depth_np = depth_np[0]
        if depth_np.ndim == 3 and depth_np.shape[2] == 1:
            depth_np = depth_np[:, :, 0]
        if depth_np.ndim != 2:
            # Attempt best-effort conversion (e.g., mean across remaining axes)
            depth_np = np.squeeze(depth_np)
            if depth_np.ndim != 2:
                depth_np = np.mean(depth_np, axis=0)

        # Resize if necessary
        if depth_np.shape != (height, width):
            depth_np = self._resize_depth(depth_np, width, height)

        depth_np = depth_np.astype(np.float32)

        # Handle intrinsics: accept [3,3] or [1,3,3] (already squeezed) or flat length-9
        K_norm = intr_np
        if K_norm.ndim == 1 and K_norm.size == 9:
            K_norm = K_norm.reshape((3, 3))

        if K_norm.shape != (3, 3):
            print(f"Warning: unexpected MoGe intrinsics shape {K_norm.shape}; using default intrinsics instead of falling back.")
            K_pixel = self._default_intrinsics(width, height)
            return depth_np, K_pixel

        # Convert normalized intrinsics to pixel units
        K_pixel = K_norm.copy().astype(np.float32)
        K_pixel[0, 0] *= width
        K_pixel[0, 2] *= width
        K_pixel[1, 1] *= height
        K_pixel[1, 2] *= height

        return depth_np, K_pixel

    def _extract_depth_from_output(self, output: object) -> np.ndarray:
        if isinstance(output, dict):
            for key in ("depth", "pred_depth", "depth_pred", "predicted_depth", "depth_map"):
                if key in output:
                    depth = output[key]
                    break
            else:
                if len(output) == 1:
                    depth = next(iter(output.values()))
                else:
                    raise ValueError("MoGe output dict did not contain a recognized depth key.")
        elif isinstance(output, (tuple, list)):
            depth = output[0]
        else:
            depth = output

        if isinstance(depth, self.torch.Tensor):
            depth = depth.detach().cpu().numpy()

        if isinstance(depth, np.ndarray):
            return depth

        raise ValueError("Unable to convert MoGe output to a NumPy array.")

    def _resize_depth(self, depth: np.ndarray, width: int, height: int) -> np.ndarray:
        depth_image = Image.fromarray(depth.astype(np.float32))
        return np.asarray(depth_image.resize((width, height), resample=Image.BILINEAR), dtype=np.float32)

    def _default_intrinsics(self, width: int, height: int) -> np.ndarray:
        fx = fy = 0.8 * max(width, height)
        cx = width / 2.0
        cy = height / 2.0
        return np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
