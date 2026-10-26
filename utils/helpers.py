import os
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

IMG_SCALE  = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
logger = None


def plot_confusion_matrix(conf_mat, save_dir='visualizations', prefix=''):
    """Plot and save both raw and normalized confusion matrices."""
    os.makedirs(save_dir, exist_ok=True)
    class_ids = list(range(conf_mat.shape[0]))

    # Normalized Confusion Matrix
    conf_mat_norm = conf_mat.astype(np.float32) / conf_mat.sum(axis=1, keepdims=True)
    conf_mat_norm = np.nan_to_num(conf_mat_norm)  # Replace NaNs with 0

    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_mat_norm, annot=True, fmt=".2f", cmap='YlGnBu',
                xticklabels=class_ids, yticklabels=class_ids, annot_kws={"size": 13}, cbar=False)
    plt.xlabel('Predicted labels', fontsize=18)
    plt.ylabel('True labels', fontsize=18)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix.png'), dpi=600)
    plt.close()


def print_log(message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')
        logger.flush()  # Flush immediately for real-time logging


def maybe_download(model_name, model_url, model_dir=None, map_location=None):
    import os, sys
    from six.moves import urllib
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = '{}.pth.tar'.format(model_name)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        url = model_url
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urllib.request.urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD


label_colours = [
    (0, 0, 0),
    (128, 64, 128),     # Road
    (244, 35, 232),     # Sidewalk
    (70, 70, 70),       # Building
    (102, 102, 156),    # Wall
    (190, 153, 153),    # Fence
    (153, 153, 153),    # Pole
    (250, 170, 30),     # Traffic Light
    (220, 220, 0),      # Traffic Sign
    (107, 142, 35),     # Vegetation
    (152, 251, 152),    # Terrain
    (70, 130, 180),     # Sky
    (220, 20, 60),      # Person
    (255, 0, 0),        # Rider
    (0, 0, 142),        # Car
    (0, 0, 70),         # Truck
    (0, 60, 100),       # Bus
    (0, 80, 100),       # Train
    (0, 0, 230),        # Motorcycle
    (119, 11, 32),      # Bicycle
]

label_colours_array = np.array(label_colours, dtype=np.uint8)

def make_segmentation_mask(predictions):
    cmap = label_colours_array
    mask = np.concatenate(predictions)
    colored_mask = np.array([cmap[i.astype(np.uint8) + 1] for i in mask])
    return colored_mask.astype(np.uint8)


def make_validation_img(img_, depth_, lab, pre):
    cmap = label_colours_array
    
    img = np.array([i * IMG_STD.reshape((3, 1, 1)) + IMG_MEAN.reshape((3, 1, 1)) for i in img_])
    img *= 255
    img = img.astype(np.uint8)
    img = np.concatenate(img, axis=1)

    depth_ = depth_[0].transpose(1, 2, 0) / max(depth_.max(), 10)
    vmax = np.percentile(depth_, 95)
    normalizer = mpl.colors.Normalize(vmin=depth_.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    depth = (mapper.to_rgba(depth_)[:,:,:3] * 255).astype(np.uint8)
    lab = np.concatenate(lab)
    lab = np.array([cmap[i.astype(np.uint8) + 1] for i in lab])

    pre = np.concatenate(pre)
    pre = np.array([cmap[i.astype(np.uint8) + 1] for i in pre])
    img = img.transpose(1, 2, 0)

    return np.concatenate([img, depth, lab, pre], 1)
