# NaviFormer: Multimodal Scene Segmentation for Assistive Navigation

by Ly Bui, Son Lam Phung, Yang Di, Soan Thi Minh Duong, Abdesselam Bouzerdoum

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
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
pip install -r requirements.txt
```

## Pretrained backbone

Download the SegFormer pretrained model (pretrained on ImageNet) from [segformer_weights](https://drive.google.com/drive/folders/1JsrHNAAL5zc5XKPbgZNgk68AhGNlJyML?usp=sharing), e.g., mit_b0.pth. Place the ImageNet-pretrained MiT backbone weights in `pretrained/` folder:

```text
pretrained/
└── mit_b0.pth
```

## Dataset format

The datasets we used in this paper include [Cityscapes](https://www.cityscapes-dataset.com/) and [TrueSight](https://documents.uow.edu.au/~phung/TrueSight.html). 
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
rgb/sample_001.png labels/sample_001.png depth/sample_001.png
```

Labels must be single-channel class-index masks. The default ignored label is `255`. Depth images are converted to three channels when necessary and scaled by `1/1000`.

## Training

Training script for semantic segmentation with RGB and Depth input:

```bash
python main.py --backbone mit_b0 -c exp_name --gpu 0 1 2
```

Outputs are written to `ckpt/<name>/`, including the best model (`model-best.pth.tar`), latest non-best checkpoint (`checkpoint.pth.tar`), final checkpoint, logs, arguments, and per-step losses. Resume training with `--resume path/to/checkpoint.pth.tar`.

## Evaluation

Evaluation script:

```bash
python main.py --gpu 0 --resume path_to_pth --evaluate  
```

The script reports global accuracy, mean accuracy, mean IoU, and validation FPS. Use `--save-image N --save-img-path results/validation` to save `N` predicted masks.

## Inference

Inference script:

```bash
python inference.py \
  --rgb-dir path/to/rgb \
  --depth-dir path/to/depth \
  --ckpt path/to/ckpt \
  --output-dir results \
  --save-colour \
  --benchmark
```

## Acknowledgement
Our implementation is mainly based on [TokenFusion](https://github.com/yikaiw/TokenFusion#multimodal-token-fusion-for-vision-transformers). Thanks to the authors for their excellent work!


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
