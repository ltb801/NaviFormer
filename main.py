# TokenFusion
import os, sys, argparse, re
import random, time
import warnings
import csv

warnings.filterwarnings('ignore')
import cv2
import datetime
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from config import *
from utils import *
import utils.helpers as helpers
from utils.optimizer import PolyWarmupAdamW
from models.segformer import WeTr
from utils.loss import PixelContrastLoss

from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def setup_ddp():
    """Set up distributed data parallel training"""
    # Handle both SLURM and direct launching
    if "SLURM_PROCID" in os.environ and not "RANK" in os.environ:
        # Multi-node setup
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["SLURM_PROCID"])
        gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        gpu = rank - gpus_per_node * (rank // gpus_per_node)
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=7200),
        )
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Regular DDP setup
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            "nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=7200),
        )
        dist.barrier()
    else:
        # Single GPU setup
        gpu = 0
    return gpu


def cleanup_ddp():
    """Clean up distributed data parallel setup"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Full Pipeline Training for NaviFormer')

    # Dataset
    parser.add_argument('-d', '--train-dir', type=str, default=TRAIN_DIR,
                        help='Path to the training set directory.')
    parser.add_argument('--val-dir', type=str, default=VAL_DIR,
                        help='Path to the validation set directory.')
    parser.add_argument('--train-list', type=str, default=TRAIN_LIST,
                        help='Path to the training set list.')
    parser.add_argument('--val-list', type=str, default=VAL_LIST,
                        help='Path to the validation set list.')
    parser.add_argument('--shorter-side', type=int, default=SHORTER_SIDE,
                        help='Shorter side transformation.')
    parser.add_argument('--crop-size', type=int, default=CROP_SIZE,
                        help='Crop size for training,')
    parser.add_argument('--input-size', type=int, default=RESIZE_SIZE,
                        help='Final input size of the model')
    parser.add_argument('--normalise-params', type=list, default=NORMALISE_PARAMS,
                        help='Normalisation parameters [scale, mean, std],')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Batch size to train the segmenter model.')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES,
                        help='Number of output classes for each task.')
    parser.add_argument('--low-scale', type=float, default=LOW_SCALE,
                        help='Lower bound for random scale')
    parser.add_argument('--high-scale', type=float, default=HIGH_SCALE,
                        help='Upper bound for random scale')
    parser.add_argument('--ignore-label', type=int, default=IGNORE_LABEL,
                        help='Label to ignore during training')

    # Encoder
    parser.add_argument('--enc', type=str, default=ENC,
                        help='Encoder net type.')
    parser.add_argument('--enc-pretrained', type=bool, default=ENC_PRETRAINED,
                        help='Whether to init with imagenet weights.')
    # General
    parser.add_argument('--name', default='', type=str,
                        help='model name')
    parser.add_argument('--gpu', type=int, nargs='+', default=[1],
                        help='select gpu.')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='If true, only validate segmentation.')
    parser.add_argument('--freeze-bn', type=bool, nargs='+', default=FREEZE_BN,
                        help='Whether to keep batch norm statistics intact.')
    parser.add_argument('--num-epoch', type=int, nargs='+', default=NUM_SEGM_EPOCHS,
                        help='Number of epochs to train for segmentation network.')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument('-c', '--ckpt', default='model', type=str, metavar='PATH',
                        help='path to save checkpoint (default: model)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--val-every', type=int, default=VAL_EVERY,
                        help='How often to validate current architecture.')
    parser.add_argument('--print-network', action='store_true', default=False,
                        help='Whether print newtork paramemters.')
    parser.add_argument('--print-loss', action='store_true', default=False,
                        help='Whether print losses during training.')
    parser.add_argument('--save-image', type=int, default=0,
                        help='Number to save images during evaluating, -1 to save all.')
    parser.add_argument('--save-img-path', type=str, default=None,
                        help='Path to save evaluated images.')
    parser.add_argument('-i', '--input', default=['rgb', 'depth'], type=str, nargs='+',
                        help='input type (image, depth)')

    # Optimisers
    parser.add_argument('--lr-enc', type=float, nargs='+', default=LR_ENC,
                        help='Learning rate for encoder.')
    parser.add_argument('--lr-dec', type=float, nargs='+', default=LR_DEC,
                        help='Learning rate for decoder.')
    parser.add_argument('--mom-enc', type=float, default=MOM_ENC,
                        help='Momentum for encoder.')
    parser.add_argument('--mom-dec', type=float, default=MOM_DEC,
                        help='Momentum for decoder.')
    parser.add_argument('--wd-enc', type=float, default=WD_ENC,
                        help='Weight decay for encoder.')
    parser.add_argument('--wd-dec', type=float, default=WD_DEC,
                        help='Weight decay for decoder.')
    parser.add_argument('--optim-dec', type=str, default=OPTIM_DEC,
                        help='Optimiser algorithm for decoder.')
    parser.add_argument('--lamda', type=float, default=LAMDA,
                        help='Lamda for L1 norm.')
    # contrastive loss
    parser.add_argument('--warmup-epochs', type=float, default=WARMUP_EPOCH,
                        help='warmup epoch for PC loss.')
    parser.add_argument('--contrast-weight', type=float, default=CONTRAST_WEIGHT,
                        help='temperature for PC loss.')
    parser.add_argument('--temp', type=float, default=TEMP,
                        help='temperature for PC loss.')
    parser.add_argument('--base_temp', type=float, default=BASE_TEMP,
                        help='base temperature for PC loss.')
    parser.add_argument('--max_samples', type=float, default=MAX_SAMPLES,
                        help='maximum samples for PC loss.')
    parser.add_argument('--max_views', type=float, default=MAX_VIEWS,
                        help='maximum views for PC loss.')

    parser.add_argument('--backbone', default='mit_b0', type=str)
    return parser.parse_args()


def create_segmenter(num_classes, gpu, backbone):
    """Create Encoder; for now only ResNet [50,101,152]"""
    segmenter = WeTr(backbone, num_classes)
    param_groups = segmenter.get_param_groups()
    assert(torch.cuda.is_available())
    segmenter.to("cuda:" + str(gpu))
    # segmenter = DistributedDataParallel(wetr, device_ids=[-1], find_unused_parameters=True)
    return segmenter, param_groups


def create_loaders(dataset, inputs, train_dir, val_dir, train_list, val_list,
                   shorter_side, crop_size, input_size, low_scale, high_scale,
                   normalise_params, batch_size, num_workers, ignore_label):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      crop_size (int) : square crop to apply during the training.
      low_scale (float) : lowest scale ratio for augmentations.
      high_scale (float) : highest scale ratio for augmentations.
      normalise_params (list / tuple) : img_scale, img_mean, img_std.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.
      ignore_label (int) : label to pad segmentation masks with

    Returns:
      train_loader, val loader

    """
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    # Custom libraries
    from utils.datasets import SegDataset as Dataset
    from utils.transforms import Normalise, Pad, RandomCrop, RandomMirror, ResizeAndScale, \
                                 CropAlignToMask, ToTensor, ResizeInputs

    input_names, input_mask_idxs = ['rgb', 'depth'], [0, 2, 1]

    composed_trn = transforms.Compose([
        CropAlignToMask(),
        ResizeAndScale(shorter_side, low_scale, high_scale),
        Pad(crop_size, [123.675, 116.28, 103.53], ignore_label),
        RandomMirror(),
        RandomCrop(crop_size),
        ResizeInputs(input_size),
        Normalise(*normalise_params),
        ToTensor()
    ])
    composed_val = transforms.Compose([
        CropAlignToMask(),
        ResizeInputs(input_size),
        Normalise(*normalise_params),
        ToTensor()
    ])
    # Training and validation sets
    trainset = Dataset(dataset=dataset, data_file=train_list, data_dir=train_dir,
                       input_names=input_names, input_mask_idxs=input_mask_idxs,
                       transform_trn=composed_trn, transform_val=composed_val,
                       stage='train', ignore_label=ignore_label)

    validset = Dataset(dataset=dataset, data_file=val_list, data_dir=val_dir,
                       input_names=input_names, input_mask_idxs=input_mask_idxs,
                       transform_trn=None, transform_val=composed_val, stage='val',
                       ignore_label=ignore_label)
    print_log('Created train set {} examples, val set {} examples'.format(len(trainset), len(validset)))
    # Training and validation loaders
    train_sampler = DistributedSampler(
        trainset, dist.get_world_size(), dist.get_rank(), shuffle=True
    )
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=False, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, train_sampler


def create_optimisers(lr_enc, lr_dec, mom_enc, mom_dec, wd_enc, wd_dec, param_enc, param_dec, optim_dec):
    """Create optimisers for encoder, decoder and controller"""
    optim_enc = torch.optim.SGD(param_enc, lr=lr_enc, momentum=mom_enc, weight_decay=wd_enc)
    if optim_dec == 'sgd':
        optim_dec = torch.optim.SGD(param_dec, lr=lr_dec, momentum=mom_dec, weight_decay=wd_dec)
    elif optim_dec == 'adam':
        optim_dec = torch.optim.Adam(param_dec, lr=lr_dec, weight_decay=wd_dec, eps=1e-3)

    return optim_enc, optim_dec


def load_ckpt(ckpt_path, ckpt_dict, is_pretrain_finetune=False):
    print("----------------")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    new_segmenter_ckpt = dict()
    if is_pretrain_finetune:
        for ckpt_k, ckpt_v in ckpt["segmenter"].items():
            if "linear_pred" in ckpt_k:
                print(ckpt_k, " is Excluded!")
            else:
                if "module." in ckpt_k:
                    new_segmenter_ckpt[ckpt_k[7:]] = ckpt_v
    else:
        for ckpt_k, ckpt_v in ckpt["segmenter"].items():
            new_segmenter_ckpt[ckpt_k] = ckpt_v
            if "module." in ckpt_k:
                new_segmenter_ckpt[ckpt_k[7:]] = ckpt_v
    ckpt["segmenter"] = new_segmenter_ckpt

    for k, v in ckpt_dict.items():
        if k in ckpt:
            v.load_state_dict(ckpt[k], strict=False)
        else:
            print(v, " is  missed!")
    best_val = ckpt.get("best_val", 0)
    epoch_start = ckpt.get("epoch_start", 0)
    if is_pretrain_finetune:
        print_log(
            "Found [Pretrain] checkpoint at {} with best_val {:.4f} at epoch {}".format(
                ckpt_path, best_val, epoch_start
            )
        )
        return 0, 0
    else:

        print_log(
            "Found checkpoint at {} with best_val {:.4f} at epoch {}".format(
                ckpt_path, best_val, epoch_start
            )
        )
        return best_val, epoch_start


def train(segmenter, input_types, train_loader, optimizer, epoch, warmup_epochs,
          segm_crit, contrast_crit, contrast_weight, freeze_bn, print_loss=False, csv_writer=None):
    """Training segmenter

    Args:
      segmenter (nn.Module) : segmentation network
      train_loader (DataLoader) : training data iterator
      optim_enc (optim) : optimiser for encoder
      optim_dec (optim) : optimiser for decoder
      epoch (int) : current epoch
      segm_crit (nn.Loss) : segmentation criterion
      freeze_bn (bool) : whether to keep BN params intact

    """
    train_loader.dataset.set_stage('train')
    segmenter.train()
    if freeze_bn:
        for module in segmenter.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    loss_weight = 0 if epoch <= warmup_epochs else contrast_weight

    for i, sample in tqdm(enumerate(train_loader), total=len(train_loader)):
        # print('train input:', sample['rgb'].shape, sample['depth'].shape, sample['mask'].shape)
        start = time.time()
        inputs = [sample[key].cuda().float() for key in input_types]
        target = sample['mask'].cuda().long()
        # Compute outputs
        output = segmenter(inputs, return_embedding=True)
        seg_output = F.interpolate(output['seg'], size=target.size()[1:],
                                    mode='bilinear', align_corners=False)
        soft_output = nn.LogSoftmax()(seg_output)

        # Compute loss and backpropagate
        ce_loss = segm_crit(soft_output, target)

        embedding = output['embed']
        _, predict_label = torch.max(output['seg'], dim=1)

        contrast_loss = contrast_crit(embedding, labels=target, predict=predict_label)

        # Total loss
        loss = ce_loss + loss_weight * contrast_loss

        optimizer.zero_grad()
        loss.backward()
        if print_loss:
            print('step: %-3d: loss=%.2f' % (i, loss), flush=True)
        optimizer.step()
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        # Write loss to CSV
        if csv_writer is not None:
            csv_writer.writerow([epoch, i, loss.item()])

    print_log('Epoch %d, Avg Loss: %.4f' % (epoch, losses.avg))


def validate(segmenter, input_types, val_loader, epoch, num_classes=-1, save_image=0, save_path=''):
    global best_iou
    val_loader.dataset.set_stage('val')
    segmenter.eval()
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64, device='cuda')

    inference_times = []  # Store inference time per image

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            inputs = [sample[key].cuda().float() for key in input_types]
            # print(inputs[0].shape)
            target = sample['mask'].cuda()

            # Timing inference start
            torch.cuda.synchronize()
            start_time = time.time()

            # Forward pass
            preds = segmenter(inputs, return_embedding=False)

            torch.cuda.synchronize()
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            seg_output = preds['seg']

            if seg_output.shape[2:] != target.shape[1:]:
                seg_output = F.interpolate(seg_output, size=target.shape[1:], mode='bilinear', align_corners=False)

            output_label = seg_output.argmax(dim=1)
            gt = target[0].cpu().numpy().astype(np.uint8)
            pred = output_label[0].cpu().numpy().astype(np.uint8)

            gt_idx = gt < num_classes
            conf_mat += torch.tensor(confusion_matrix(gt[gt_idx], pred[gt_idx], num_classes), device='cuda')

            if save_image > 0 and i < save_image:
                print("predicted labels: ", np.unique(pred))
                img = make_segmentation_mask(pred[np.newaxis, :])
                os.makedirs(save_path, exist_ok=True)
                cv2.imwrite(f'{save_path}/validate_{i}.png', img[:, :, ::-1])
                print(f'Saved validation image: {save_path}/validate_{i}.png')

    conf_mat = conf_mat.cpu().numpy()

    glob, mean, iou = getScores(conf_mat)

    best_iou_note = ''
    if iou > best_iou:
        best_iou = iou
        best_iou_note = ' (best)'

    print_log('Epoch %-4d %-7s   glob_acc=%-5.2f    mean_acc=%-5.2f    IoU=%-5.2f%s' %
              (epoch, '(val)', glob, mean, iou, best_iou_note))
    # ADD FPS printing
    total_inference_time = sum(inference_times)
    num_samples = len(inference_times)
    fps = num_samples / total_inference_time
    print_log(f"Validation FPS: {fps:.2f}")

    return iou


def main():
    global args, best_iou
    best_iou = 0
    args = get_arguments()
    args.num_stages = len(args.lr_enc)
    gpu = setup_ddp()

    ckpt_dir = os.path.join('ckpt', args.ckpt)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Only copy files on rank 0 to avoid conflicts
    if dist.is_initialized() and dist.get_rank() == 0:
        os.system('cp -r *py models utils data %s' % ckpt_dir)
    helpers.logger = open(os.path.join(ckpt_dir, 'log.txt'), 'w+')
    print_log(' '.join(sys.argv))

    # Set random seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Generate Segmenter
    segmenter, param_groups = create_segmenter(args.num_classes, gpu, args.backbone)

    if args.print_network:
        print_log('')

    print_log('Loaded Segmenter {}, ImageNet-Pre-Trained={}, #PARAMS={:3.2f}M'
          .format(args.backbone, args.enc_pretrained, compute_params(segmenter) / 1e6))
    # Restore if any
    best_val, epoch_start = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            best_val, epoch_start = load_ckpt(args.resume, {'segmenter': segmenter})
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume))
            return

    # Keep a non-DDP version of model for validation
    no_ddp_segmenter = segmenter

    # Wrap model with DDP
    if dist.is_initialized():
        segmenter = DDP(
            segmenter,
            device_ids=[gpu],
            output_device=gpu,
            find_unused_parameters=True
        )

    epoch_current = epoch_start
    # Criterion
    segm_crit = nn.CrossEntropyLoss(reduction='mean', ignore_index=args.ignore_label).cuda()
    contrast_crit = PixelContrastLoss(args.temp, args.base_temp, args.max_samples, args.max_views, args.ignore_label)

    # Saver
    csv_file_path = os.path.join(ckpt_dir, 'losses.csv')
    csv_file = open(csv_file_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'step', 'loss'])  # CSV header

    saver = Saver(args=vars(args), ckpt_dir=ckpt_dir, best_val=best_val,
                  condition=lambda x, y: x > y)  # keep checkpoint with the best validation score

    lrs = [6e-5, 3e-5, 1.5e-5]

    for task_idx in range(args.num_stages):
        optimizer = PolyWarmupAdamW(
        params=[
                {
                    "params": param_groups[0],
                    "lr": lrs[task_idx],
                    "weight_decay": 0.01,
                },
                {
                    "params": param_groups[1],
                    "lr": lrs[task_idx],
                    "weight_decay": 0.0,
                },
                {
                    "params": param_groups[2],
                    "lr": lrs[task_idx] * 5,
                    "weight_decay": 0.01,
                },
            ],
            lr = lrs[task_idx],
            weight_decay = 0.01,
            betas = [0.9, 0.999],
            warmup_iter = 3000,
            max_iter = 45000,
            warmup_ratio = 1e-6,
            power = 0.9
        )
        total_epoch = sum([args.num_epoch[idx] for idx in range(task_idx + 1)])
        if epoch_start >= total_epoch:
            continue
        start = time.time()
        torch.cuda.empty_cache()
        # Create dataloaders
        train_loader, val_loader, train_sampler = create_loaders(
            DATASET, args.input, args.train_dir, args.val_dir, args.train_list, args.val_list,
            args.shorter_side, args.crop_size, args.input_size, args.low_scale, args.high_scale,
            args.normalise_params, args.batch_size, args.num_workers, args.ignore_label)
        if args.evaluate:
            miou = validate(no_ddp_segmenter, args.input, val_loader, 0, num_classes=args.num_classes,
                            save_image=args.save_image, save_path=args.save_img_path)

            return miou

        # Optimisers
        print_log('Training Stage {}'.format(str(task_idx)))

        for epoch in range(min(args.num_epoch[task_idx], total_epoch - epoch_start)):
            train(segmenter, args.input, train_loader, optimizer, epoch_current, args.warmup_epochs,
                  segm_crit, contrast_crit, args.contrast_weight, args.freeze_bn, args.print_loss,
                  csv_writer)
            if (epoch + 1) % (args.val_every) == 0:
                miou = validate(no_ddp_segmenter, args.input, val_loader, epoch_current, args.num_classes,
                                args.save_image, args.save_img_path)
                saver.save(miou, {'segmenter' : segmenter.state_dict(), 'epoch_start' : epoch_current})
            epoch_current += 1

        print_log('Stage {} finished, time spent {:.3f}min\n'.format(task_idx, (time.time() - start) / 60.))

    final_dict = {
        "segmenter": segmenter.state_dict(),
        "epoch_start": epoch_current,
        "best_val": saver.best_val  # Ensure to include the best_val
    }

    csv_file.close()

    # Save the final checkpoint
    torch.save(final_dict, os.path.join(ckpt_dir, 'final_checkpoint.pth.tar'))

    print_log('All stages are now finished. Best Val is {:.3f}'.format(saver.best_val))
    helpers.logger.close()
    cleanup_ddp()


if __name__ == '__main__':
    main()

