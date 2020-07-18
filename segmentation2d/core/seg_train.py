import importlib
import numpy as np
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from segmentation2d.dataset.dataset import SegmentationDataset
from segmentation2d.loss.focal_loss import FocalLoss
from segmentation2d.loss.multi_dice_loss import MultiDiceLoss
from segmentation2d.utils.file_io import load_config, setup_logger
from segmentation2d.utils.image_tools import save_intermediate_results, convert_tensor_to_image
from segmentation2d.utils.model_io import load_checkpoint, save_checkpoint
from segmentation2d.utils.metrics import cal_dsc


def train_one_epoch(model, optimizer, data_loader, loss_func, num_gpus, epoch, logger, writer, print_freq,
                    save_inputs, save_folder):
    """ Train one epoch
    """
    if num_gpus > 0:
        model = nn.parallel.DataParallel(model, device_ids=list(range(num_gpus)))
        model = model.cuda()

    model.train()

    avg_loss = 0
    for batch_idx, (crops, masks, frames, filenames) in enumerate(data_loader):
        begin_t = time.time()

        if num_gpus > 0:
            crops, masks = crops.cuda(), masks.cuda()

        # clear previous gradients
        optimizer.zero_grad()

        # network forward and backward
        outputs = model(crops)
        train_loss = loss_func(outputs, masks)
        train_loss.backward()

        avg_loss += train_loss.item()

        # update weights
        optimizer.step()

        # save training crops for visualization
        if save_inputs:
            batch_size = crops.size(0)
            save_intermediate_results(list(range(batch_size)), crops, masks, outputs, frames, filenames,
                                      os.path.join(save_folder, 'batch_{}'.format(batch_idx)))

        batch_duration = time.time() - begin_t

        # print training loss per batch
        msg = 'epoch: {}, batch: {}, train_loss: {:.4f}, time: {:.4f} s/vol'
        msg = msg.format(epoch, batch_idx, train_loss.item(), batch_duration)

        if batch_idx % print_freq:
            logger.info(msg)

    writer.add_scalar('Train/Loss', avg_loss / len(data_loader), epoch)


def train(train_config_file):
    """ Medical image segmentation training engine
    :param train_config_file: the input configuration file
    :return: None
    """
    assert os.path.isfile(train_config_file), 'Config not found: {}'.format(train_config_file)

    # load config file
    train_cfg = load_config(train_config_file)

    # clean the existing folder if training from scratch
    model_folder = os.path.join(train_cfg.general.save_dir, train_cfg.general.model_scale)
    if os.path.isdir(model_folder):
        if train_cfg.general.resume_epoch < 0:
            shutil.rmtree(model_folder)
            os.makedirs(model_folder)
    else:
        os.makedirs(model_folder)

    # copy training and inference config files to the model folder
    shutil.copy(train_config_file, os.path.join(model_folder, 'train_config.py'))
    infer_config_file = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'infer_config.py'))
    shutil.copy(infer_config_file, os.path.join(train_cfg.general.save_dir, 'infer_config.py'))

    # enable logging
    log_file = os.path.join(model_folder, 'train_log.txt')
    logger = setup_logger(log_file, 'seg2d')

    # control randomness during training
    np.random.seed(train_cfg.general.seed)
    torch.manual_seed(train_cfg.general.seed)
    if train_cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(train_cfg.general.seed)

    # training dataset
    train_dataset = SegmentationDataset(
                'train',
                imlist_file=train_cfg.general.train_im_list,
                labels=train_cfg.dataset.labels,
                spacing=train_cfg.dataset.spacing,
                crop_size=train_cfg.dataset.crop_size,
                sampling_method=train_cfg.dataset.sampling_method,
                random_translation=train_cfg.dataset.random_translation,
                random_scale=train_cfg.dataset.random_scale,
                random_hori_flip=train_cfg.dataset.random_horizontal_flip,
                random_vert_flip=train_cfg.dataset.random_vertical_flip,
                interpolation=train_cfg.dataset.interpolation,
                crop_normalizers=train_cfg.dataset.crop_normalizers)

    train_data_loader = DataLoader(train_dataset, batch_size=train_cfg.train.batchsize,
                                   num_workers=train_cfg.train.num_threads, pin_memory=True, shuffle=True)

    # validation dataset
    val_dataset = SegmentationDataset(
                'val',
                imlist_file=train_cfg.general.val_im_list,
                labels=train_cfg.dataset.labels,
                spacing=train_cfg.dataset.spacing,
                crop_size=train_cfg.dataset.crop_size,
                sampling_method=train_cfg.dataset.sampling_method,
                random_translation=train_cfg.dataset.random_translation,
                random_scale=train_cfg.dataset.random_scale,
                random_hori_flip=False,
                random_vert_flip=False,
                interpolation=train_cfg.dataset.interpolation,
                crop_normalizers=train_cfg.dataset.crop_normalizers)

    val_data_loader = DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=False)

    net_module = importlib.import_module('segmentation2d.network.' + train_cfg.net.name)
    net = net_module.SegmentationNet(train_dataset.num_modality(), train_cfg.dataset.num_classes)
    max_stride = net.max_stride()
    net_module.parameters_kaiming_init(net)

    assert np.all(np.array(train_cfg.dataset.crop_size) % max_stride == 0), 'crop size not divisible by max stride'

    # training optimizer
    opt = optim.Adam(net.parameters(), lr=train_cfg.train.lr, betas=train_cfg.train.betas)

    # load checkpoint if resume epoch > 0
    if train_cfg.general.resume_epoch >= 0:
        last_save_epoch, batch_start = load_checkpoint(train_cfg.general.resume_epoch, net, opt, model_folder)
    else:
        last_save_epoch, batch_start = 0, 0

    if train_cfg.loss.name == 'Focal':
        loss_func = FocalLoss(class_num=train_cfg.dataset.num_classes, alpha=train_cfg.loss.obj_weight,
                              gamma=train_cfg.loss.focal_gamma, use_gpu=train_cfg.general.num_gpus > 0)

    elif train_cfg.loss.name == 'Dice':
        loss_func = MultiDiceLoss(weights=train_cfg.loss.obj_weight, num_class=train_cfg.dataset.num_classes,
                                  use_gpu=train_cfg.general.num_gpus > 0)
    else:
        raise ValueError('Unknown loss function')

    writer = SummaryWriter(os.path.join(model_folder, 'tensorboard'))

    # loop over epochs
    for epoch_idx in range(train_cfg.train.save_epochs):

        train_one_epoch(net, opt, train_data_loader, loss_func, train_cfg.general.num_gpus, epoch_idx+last_save_epoch,
            logger, writer, train_cfg.train.print_freq, train_cfg.debug.save_inputs, train_cfg.general.save_dir)

        if epoch_idx % train_cfg.train.save_epochs:
            # test on validation dataset
            net.eval()
            max_avg_dice, dice_res = 0, []
            for batch_idx, (crop, mask, _, _) in enumerate(val_data_loader):
                if train_cfg.general.num_gpus > 0:
                    crop = crop.cuda()

                probs = net(crop)
                pred_mask = probs.squeeze(0)
                _, pred_mask = pred_mask.max(0)

                mask = mask.squeeze()
                if train_cfg.general.num_gpus > 0:
                    mask = mask.cpu()

                mask = convert_tensor_to_image(mask, dtype=np.int32)
                pred_mask = convert_tensor_to_image(pred_mask, dtype=np.int32)

                # compute dice
                dice, _ = cal_dsc(mask, pred_mask, 1, 10)
                dice_res.append(dice)

            if np.mean(dice_res) > max_avg_dice:
                # save model
                save_checkpoint(net, opt, epoch_idx + last_save_epoch, 0, train_cfg, max_stride, train_dataset.num_modality())

                max_avg_dice = np.mean(dice_res)
                logger.info('Best epoch: {}, mean dice: {}'.format(epoch_idx + last_save_epoch, max_avg_dice))

    writer.close()