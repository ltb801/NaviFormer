import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset


def line_to_paths_fn(x, input_names):
    return x.decode('utf-8').strip().split('\t')


class SegDataset(Dataset):
    def __init__(self, dataset, data_file, data_dir, input_names, input_mask_idxs,
                 transform_trn=None, transform_val=None, stage='train', ignore_label=255):
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = [line_to_paths_fn(l, input_names) for l in datalist]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = stage
        self.input_names = input_names
        self.input_mask_idxs = input_mask_idxs
        self.ignore_label = ignore_label

    def set_stage(self, stage):
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        idxs = self.input_mask_idxs
        image_names = [os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]]
        mask_names = [os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]]
        sample = {}

        for i, key in enumerate(self.input_names):
            sample[key] = self.read_image(image_names[idxs[i]], key)

        try:
            mask = np.array(Image.open(mask_names[idxs[-1]]))
        except FileNotFoundError:
            path = image_names[idxs[-1]]
            num_idx = int(path[-10:-4]) + 5050
            path = path[:-10] + '%06d' % num_idx + path[-4:]
            mask = np.array(Image.open(path))

        assert len(mask.shape) == 2, 'Masks must be encoded without colourmap'

        # Convert 0 → 255 (ignore), and shift all other labels down by 1
        # mask = np.where(mask == 0, self.ignore_label, mask - 1).astype(np.uint8)

        sample['inputs'] = self.input_names
        sample['mask'] = mask

        if self.stage == 'train' and self.transform_trn:
            sample = self.transform_trn(sample)
        elif self.stage == 'val' and self.transform_val:
            sample = self.transform_val(sample)

        del sample['inputs']
        return sample

    @staticmethod
    def read_image_(x, key):
        img = cv2.imread(x)
        if key == 'depth':
            img = cv2.applyColorMap(cv2.convertScaleAbs(255 - img, alpha=1), cv2.COLORMAP_JET)
        return img

    @staticmethod
    def read_image(x, key):
        img_arr = np.array(Image.open(x))
        if len(img_arr.shape) == 2:
            img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
        return img_arr

