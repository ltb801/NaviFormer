# NaviFormer: Multimodal Scene Segmentation for Assistive Navigation

by Ly Bui, Son Lam Phung, Yang Di, Soan Thi Minh Duong, Abdesselam Bouzerdoum

[Paper](https://doi.org/10.1016/j.cviu.2026.104793)

Official PyTorch implementation of **NaviFormer: Multimodal Scene Segmentation for Assistive Navigation**.
[Paper](https://doi.org/10.1016/j.cviu.2026.104793)

## Requirements

- Python 3.10 (recommended)
- CUDA 12.1

Create an environment and install the dependencies:

```bash
# Create an environment
conda create -n naviformer python=3.10 -y
conda activate naviformer

# Clone the repository
git clone https://github.comltb801/NaviFormer.git
cd NaviFormer

# Install the dependencies used by this repository
pip install -r requirements.txt
```

## Pretrained backbone

Place the ImageNet-pretrained MiT backbone weights in `pretrained/` folder:

```text
pretrained/
└── mit_b0.pth
```

## Dataset format

Store RGB images, semantic labels, and depth images below a common dataset root:

```text
dataset/
├── rgb/
│   └── sample_001.png
├── labels/
│   └── sample_001.png
└── depth/
    └── sample_001.png
```

Create separate training and validation lists. Each line must contain three paths relative to the dataset root, separated by tabs, in this order:

```text
rgb/sample_001.png<TAB>labels/sample_001.png<TAB>depth/sample_001.png
```

Labels must be single-channel class-index masks. The default ignored label is `255`. Depth images are converted to three channels when necessary and scaled by `1/1000`.

## Training

Run one process per GPU. A single-GPU example is:

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --train-dir /path/to/dataset \
  --val-dir /path/to/dataset \
  --train-list /path/to/train.txt \
  --val-list /path/to/val.txt \
  --backbone mit_b0 \
  --num-classes 16 \
  --batch-size 4 \
  --ckpt naviformer_mit_b0
```

Outputs are written to `ckpt/<name>/`, including the best model (`model-best.pth.tar`), latest non-best checkpoint (`checkpoint.pth.tar`), final checkpoint, logs, arguments, and per-step losses. Resume training with `--resume path/to/checkpoint.pth.tar`.

## Evaluation

Evaluate a checkpoint on a validation split with:

```bash
torchrun --standalone --nproc_per_node=1 main.py \
  --evaluate \
  --train-dir /path/to/dataset \
  --val-dir /path/to/dataset \
  --train-list /path/to/train.txt \
  --val-list /path/to/val.txt \
  --backbone mit_b0 \
  --num-classes 16 \
  --resume ckpt/naviformer_mit_b0/model-best.pth.tar
```

The script reports global accuracy, mean accuracy, mean IoU, and validation FPS. Use `--save-image N --save-img-path results/validation` to save `N` predicted masks.

## Inference

Use the same backbone and number of classes that were used to train the checkpoint.

Single RGB-depth pair:

```bash
python inference.py \
  --rgb path/to/rgb.png \
  --depth path/to/depth.png \
  --ckpt ckpt/naviformer_mit_b0/model-best.pth.tar \
  --backbone mit_b0 \
  --num-classes 16 \
  --output mask.png \
  --output-colour mask_colour.png
```

Batch inference pairs files with the same filename stem:

```bash
python inference.py \
  --rgb-dir path/to/rgb \
  --depth-dir path/to/depth \
  --ckpt ckpt/naviformer_mit_b0/model-best.pth.tar \
  --backbone mit_b0 \
  --num-classes 16 \
  --output-dir results \
  --batch-size 4 \
  --save-colour \
  --benchmark
```

Alternatively, pass `--pair-list pairs.txt`, where each line is `rgb_path<TAB>depth_path`. The raw output masks contain class indices; colour masks are visualizations only.

## Citation

If you find our work useful for your research, please consider citing the following paper.

```bibtex
@article{Bui2026naviformer,
  title   = {NaviFormer: Multimodal scene segmentation for assistive navigation},
  author  = {Ly Bui and Son Lam Phung and Yang Di and Soan Thi Minh Duong and Abdesselam Bouzerdoum},
  journal = {Computer Vision and Image Understanding},
  volume  = {269},
  pages   = {104793},
  year    = {2026},
  doi     = {10.1016/j.cviu.2026.104793},
  url     = {https://doi.org/10.1016/j.cviu.2026.104793}
}
```
