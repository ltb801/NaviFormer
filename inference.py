"""
Fast inference script for TokenFusion RGB-D semantic segmentation.

Single pair:
    python inference.py --rgb path/to/rgb.png --depth path/to/depth.png \\
                        --ckpt ckpt/your_model/checkpoint.pth.tar \\
                        --backbone mit_b0 --num-classes 16 --output mask.png

Batch — matched directories (files with the same name are paired):
    python inference.py --rgb-dir data/rgb/ --depth-dir data/depth/ \\
                        --ckpt ckpt/your_model/checkpoint.pth.tar \\
                        --output-dir results/ --batch-size 4

Batch — explicit text file:
    python inference.py --pair-list pairs.txt \\
                        --ckpt ckpt/your_model/checkpoint.pth.tar \\
                        --output-dir results/ --batch-size 4

pairs.txt format (tab-separated, one pair per line):
    path/to/rgb1.png\\tpath/to/depth1.png
    path/to/rgb2.png\\tpath/to/depth2.png
    ...
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ImageNet stats (same as config.py NORMALISE_PARAMS)
_IMG_SCALE   = 1.0 / 255.0
_IMG_MEAN    = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD     = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DEPTH_SCALE = 1.0 / 1000.0


# ── preprocessing helpers ─────────────────────────────────────────────────────

def _load_rgb(path: str) -> np.ndarray:
    """Load RGB image as float32 HxWx3 numpy array, normalised for the model."""
    img = np.array(Image.open(path))
    if img.ndim == 2:                          # grayscale → 3-channel
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:                      # RGBA → RGB
        img = img[:, :, :3]
    img = img.astype(np.float32)
    img = (_IMG_SCALE * img - _IMG_MEAN) / _IMG_STD
    return img                                 # HxWx3


def _load_depth(path: str) -> np.ndarray:
    """Load depth image as float32 HxWx3 numpy array, scaled for the model."""
    img = np.array(Image.open(path))
    if img.ndim == 2:                          # single-channel → 3-channel (tile)
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = img.astype(np.float32) * _DEPTH_SCALE
    return img                                 # HxWx3


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """HxWxC numpy → 1xCxHxW float32 CUDA tensor."""
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return t.to(device=device, dtype=torch.float32, non_blocking=True)


def _resize_np(arr: np.ndarray, h: int, w: int, inter: int) -> np.ndarray:
    """Resize HxWxC array to (h, w)."""
    return cv2.resize(arr, (w, h), interpolation=inter)


def _load_pair(rgb_path: str, depth_path: str):
    """Load and preprocess one RGB-D pair. Returns (rgb_np, depth_np, orig_h, orig_w)."""
    rgb_np   = _load_rgb(rgb_path)
    depth_np = _load_depth(depth_path)
    return rgb_np, depth_np, rgb_np.shape[0], rgb_np.shape[1]


# ── main inferencer class ─────────────────────────────────────────────────────

class Inferencer:
    """
    Load once, run many times — single pair or true batched inference.

    Speed features:
      - model pinned to GPU in eval mode
      - torch.no_grad() + FP16 autocast
      - parallel CPU pre-processing with ThreadPoolExecutor
      - images resized to a common size so they can be stacked into one
        GPU tensor → single forward pass per batch
      - non_blocking host→device transfers
      - argmax on GPU; one CPU copy for the entire batch
    """

    def __init__(
        self,
        ckpt_path: str,
        backbone: str = "mit_b0",
        num_classes: int = 16,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        from models.segformer import WeTr

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and self.device.type == "cuda"

        # Build model
        self.model = WeTr(backbone, num_classes)
        self._load_weights(ckpt_path)
        self.model.to(self.device)
        self.model.eval()

        # Warm-up (amortise CUDA JIT compilation on first real call)
        self._warmup()

    # ── weight loading ────────────────────────────────────────────────────────

    def _load_weights(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        raw = ckpt.get("segmenter", ckpt)       # tolerate bare state-dict files
        # strip "module." prefix from DDP checkpoints
        state = {k.replace("module.", "", 1): v for k, v in raw.items()}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] missing keys ({len(missing)}): {missing[:5]} …")
        if unexpected:
            print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
        print(f"[info] loaded checkpoint: {ckpt_path}")

    # ── warm-up ───────────────────────────────────────────────────────────────

    def _warmup(self, size: int = 256):
        """Run one dummy forward pass so CUDA kernels are compiled before timing."""
        dummy_rgb   = torch.zeros(1, 3, size, size, device=self.device)
        dummy_depth = torch.zeros(1, 3, size, size, device=self.device)
        with torch.no_grad():
            ctx = torch.cuda.amp.autocast() if self.use_amp else _nullctx()
            with ctx:
                self.model([dummy_rgb, dummy_depth], return_embedding=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(
        self,
        rgb_path: str,
        depth_path: str,
        original_size: bool = True,
    ) -> np.ndarray:
        """
        Run inference on one RGB-D pair.

        Args:
            rgb_path:      path to RGB image file.
            depth_path:    path to depth image file.
            original_size: if True, upsample the mask back to the input resolution.

        Returns:
            mask: uint8 numpy array of shape (H, W) with class indices.
        """
        rgb_np   = _load_rgb(rgb_path)
        depth_np = _load_depth(depth_path)
        orig_h, orig_w = rgb_np.shape[:2]

        rgb_t   = _to_tensor(rgb_np,   self.device)
        depth_t = _to_tensor(depth_np, self.device)

        ctx = torch.cuda.amp.autocast() if self.use_amp else _nullctx()
        with ctx:
            preds = self.model([rgb_t, depth_t], return_embedding=False)

        seg = preds["seg"]                         # 1 x C x H' x W'

        if original_size and seg.shape[2:] != (orig_h, orig_w):
            seg = F.interpolate(
                seg, size=(orig_h, orig_w),
                mode="bilinear", align_corners=False
            )

        mask = seg.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return mask

    @torch.no_grad()
    def run_batch(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 4,
        original_size: bool = True,
        num_workers: int = 4,
    ) -> list[np.ndarray]:
        """
        Run inference on a list of (rgb_path, depth_path) pairs using true
        batched forward passes.

        Strategy for mixed-resolution inputs:
          - Within each mini-batch the images are resized to the size of the
            first image in that batch so they can be stacked into one tensor.
          - If original_size=True each mask is upsampled back to its own
            original resolution before being returned.

        Args:
            pairs:        list of (rgb_path, depth_path) tuples.
            batch_size:   number of pairs per GPU forward pass.
            original_size: upsample each mask to the image's original size.
            num_workers:  threads for parallel CPU pre-processing.

        Returns:
            list of uint8 numpy arrays (H_i, W_i) in the same order as pairs.
        """
        all_masks: list[np.ndarray] = []

        # Process in mini-batches
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start: start + batch_size]

            # ── 1. Parallel CPU loading ───────────────────────────────────────
            with ThreadPoolExecutor(max_workers=min(num_workers, len(chunk))) as pool:
                loaded = list(pool.map(lambda p: _load_pair(*p), chunk))
            # loaded: list of (rgb_np, depth_np, orig_h, orig_w)

            # ── 2. Common spatial size = size of first image in chunk ─────────
            ref_h, ref_w = loaded[0][2], loaded[0][3]
            orig_sizes = [(h, w) for _, _, h, w in loaded]

            rgb_batch   = []
            depth_batch = []
            for rgb_np, depth_np, h, w in loaded:
                if (h, w) != (ref_h, ref_w):
                    rgb_np   = _resize_np(rgb_np,   ref_h, ref_w, cv2.INTER_CUBIC)
                    depth_np = _resize_np(depth_np, ref_h, ref_w, cv2.INTER_NEAREST)
                # CxHxW
                rgb_batch.append(torch.from_numpy(rgb_np.transpose(2, 0, 1)))
                depth_batch.append(torch.from_numpy(depth_np.transpose(2, 0, 1)))

            # ── 3. Stack → NxCxHxW, send to GPU ──────────────────────────────
            rgb_t   = torch.stack(rgb_batch).to(
                device=self.device, dtype=torch.float32, non_blocking=True)
            depth_t = torch.stack(depth_batch).to(
                device=self.device, dtype=torch.float32, non_blocking=True)

            # ── 4. Single forward pass ────────────────────────────────────────
            ctx = torch.cuda.amp.autocast() if self.use_amp else _nullctx()
            with ctx:
                preds = self.model([rgb_t, depth_t], return_embedding=False)

            seg = preds["seg"]                     # N x C x H' x W'

            # ── 5. Per-image upsample + argmax ────────────────────────────────
            for i, (orig_h, orig_w) in enumerate(orig_sizes):
                s = seg[i: i + 1]                  # 1 x C x H' x W'
                if original_size and s.shape[2:] != (orig_h, orig_w):
                    s = F.interpolate(
                        s, size=(orig_h, orig_w),
                        mode="bilinear", align_corners=False,
                    )
                mask = s.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                all_masks.append(mask)

        return all_masks


# ── utility: null context manager ────────────────────────────────────────────

class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ── colour visualisation ──────────────────────────────────────────────────────

def mask_to_colour(mask: np.ndarray) -> np.ndarray:
    """Convert a class-index mask (H, W) to a colour image (H, W, 3) BGR."""
    colour = PALETTE[mask.clip(0, len(PALETTE) - 1)]
    return colour[:, :, ::-1].copy()             # RGB → BGR for cv2


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TokenFusion RGB-D inference")

    # ── single-pair mode ──────────────────────────────────────────────────────
    p.add_argument("--rgb",   default=None, help="Path to a single RGB image")
    p.add_argument("--depth", default=None, help="Path to a single depth image")
    p.add_argument("--output", default="output_mask.png",
                   help="Output path for the class-index mask (single-pair mode)")
    p.add_argument("--output-colour", default=None,
                   help="(optional) Output path for a colour-coded mask (single-pair mode)")

    # ── batch mode ────────────────────────────────────────────────────────────
    p.add_argument("--rgb-dir",   default=None,
                   help="Directory of RGB images (paired by matching filename with --depth-dir)")
    p.add_argument("--depth-dir", default=None,
                   help="Directory of depth images (paired by matching filename with --rgb-dir)")
    p.add_argument("--pair-list", default=None,
                   help="Text file with tab-separated rgb<TAB>depth paths, one per line")
    p.add_argument("--output-dir", default="results",
                   help="Directory to write masks in batch mode")
    p.add_argument("--save-colour", action="store_true",
                   help="Also save colour-coded masks in --output-dir")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Images per GPU forward pass (batch mode)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="CPU threads for parallel image loading (batch mode)")

    # ── shared ────────────────────────────────────────────────────────────────
    p.add_argument("--ckpt",        required=True, help="Path to .pth.tar checkpoint")
    p.add_argument("--backbone",    default="mit_b0")
    p.add_argument("--num-classes", type=int, default=16)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--no-amp",      action="store_true", help="Disable FP16 autocast")
    p.add_argument("--benchmark",   action="store_true",
                   help="Print inference time and FPS")
    return p.parse_args()


_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _pair_dirs(rgb_dir: str, depth_dir: str) -> list[tuple[str, str]]:
    """
    Match files in rgb_dir and depth_dir by stem (name without extension).
    Files present in both directories are paired; unmatched files are skipped
    with a warning.  The returned list is sorted by stem for reproducibility.
    """
    def stem_map(directory):
        return {
            os.path.splitext(f)[0]: os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in _IMG_EXTENSIONS
        }

    rgb_stems   = stem_map(rgb_dir)
    depth_stems = stem_map(depth_dir)

    common = sorted(set(rgb_stems) & set(depth_stems))

    only_rgb   = set(rgb_stems) - set(depth_stems)
    only_depth = set(depth_stems) - set(rgb_stems)
    if only_rgb:
        print(f"[warn] {len(only_rgb)} RGB file(s) have no matching depth — skipped")
    if only_depth:
        print(f"[warn] {len(only_depth)} depth file(s) have no matching RGB — skipped")

    return [(rgb_stems[s], depth_stems[s]) for s in common]


def main():
    args = parse_args()

    # ── resolve pairs ─────────────────────────────────────────────────────────
    n_sources = sum([
        args.rgb is not None,
        args.rgb_dir is not None,
        args.pair_list is not None,
    ])
    if n_sources == 0:
        raise ValueError(
            "Provide one of: --rgb/--depth  |  --rgb-dir/--depth-dir  |  --pair-list"
        )
    if n_sources > 1:
        raise ValueError(
            "--rgb, --rgb-dir, and --pair-list are mutually exclusive."
        )

    inferencer = Inferencer(
        ckpt_path   = args.ckpt,
        backbone    = args.backbone,
        num_classes = args.num_classes,
        device      = args.device,
        use_amp     = not args.no_amp,
    )

    # ── single-pair mode ──────────────────────────────────────────────────────
    if args.rgb is not None:
        if args.depth is None:
            raise ValueError("--depth is required when using --rgb.")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        mask = inferencer.run(args.rgb, args.depth, original_size=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if args.benchmark:
            print(f"Inference time: {elapsed * 1000:.1f} ms  ({1.0 / elapsed:.1f} FPS)")

        cv2.imwrite(args.output, mask)
        print(f"Saved mask → {args.output}")
        if args.output_colour:
            cv2.imwrite(args.output_colour, mask_to_colour(mask))
            print(f"Saved colour mask → {args.output_colour}")

    # ── batch mode ────────────────────────────────────────────────────────────
    else:
        if args.rgb_dir is not None:
            if args.depth_dir is None:
                raise ValueError("--depth-dir is required when using --rgb-dir.")
            pairs = _pair_dirs(args.rgb_dir, args.depth_dir)
            if not pairs:
                raise RuntimeError(
                    f"No matching filenames found between '{args.rgb_dir}' "
                    f"and '{args.depth_dir}'."
                )
        else:  # --pair-list
            with open(args.pair_list) as f:
                pairs = [
                    tuple(line.strip().split("\t"))
                    for line in f if line.strip()
                ]

        print(f"[info] {len(pairs)} pairs to process (batch_size={args.batch_size})")
        os.makedirs(args.output_dir, exist_ok=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        masks = inferencer.run_batch(
            pairs,
            batch_size    = args.batch_size,
            original_size = True,
            num_workers   = args.num_workers,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if args.benchmark:
            fps = len(pairs) / elapsed
            print(f"Total: {elapsed:.2f} s  |  {fps:.1f} FPS  "
                  f"({elapsed / len(pairs) * 1000:.1f} ms/image)")

        for mask, (rgb_path, _) in zip(masks, pairs):
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            cv2.imwrite(os.path.join(args.output_dir, f"{stem}_mask.png"), mask)
            if args.save_colour:
                cv2.imwrite(
                    os.path.join(args.output_dir, f"{stem}_colour.png"),
                    mask_to_colour(mask),
                )

        print(f"Saved {len(masks)} masks → {args.output_dir}/")


if __name__ == "__main__":
    main()
