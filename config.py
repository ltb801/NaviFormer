import numpy as np

# DATASET PARAMETERS
DATASET = 'cityscapes'
TRAIN_DIR = '/path/to/dataset'  # 'Modify data path'
VAL_DIR = TRAIN_DIR
TRAIN_LIST = '/path/to/train.txt'
VAL_LIST = '/path/to/val.txt'

SHORTER_SIDE = 480
CROP_SIZE = 640
RESIZE_SIZE = None

NORMALISE_PARAMS = [1./255,  # Image SCALE
                    np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)),  # Image MEAN
                    np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)),  # Image STD
                    1./1000]  # Depth SCALE

BATCH_SIZE = 4
NUM_WORKERS = 4
NUM_CLASSES = 19
LOW_SCALE = 0.5
HIGH_SCALE = 2.0
IGNORE_LABEL = 255

# ENCODER PARAMETERS
ENC = '50'  # ResNet101
ENC_PRETRAINED = True  # pre-trained on ImageNet or randomly initialised

# GENERAL
FREEZE_BN = True
NUM_SEGM_EPOCHS = [100] * 3  # [150] * 3 if using ResNet152 as backbone
PRINT_EVERY = 100
RANDOM_SEED = 42
VAL_EVERY = 5  # how often to record validation scores

LR_ENC = [5e-4, 2.5e-4, 1e-4]  # TO FREEZE, PUT 0
LR_DEC = [3e-3, 1.5e-3, 7e-4]
MOM_ENC = 0.9  # TO FREEZE, PUT 0
MOM_DEC = 0.9
WD_ENC = 1e-5  # TO FREEZE, PUT 0
WD_DEC = 1e-5
LAMDA = 1e-4  # slightly better
OPTIM_DEC = 'sgd'

# PC Loss
WARMUP_EPOCH = 2
CONTRAST_WEIGHT = 0.4
TEMP = 0.1
BASE_TEMP = 0.07
MAX_SAMPLES = 1024
MAX_VIEWS = 100
