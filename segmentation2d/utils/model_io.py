import os
import glob
import torch
import shutil


def get_checkpoint_folder(chk_root, epoch):
  """
  Get the checkpoint's folder with the specified epoch.
  :param chk_root: the check point root directory, which may contain multiple checkpoints.
  :param epoch: the epoch of the checkpoint, set -1 to get the latest epoch.
  :return: the folder containing the checkpoint with the specified epoch.
  """
  assert os.path.isdir(chk_root), 'The folder does not exist: {}'.format(chk_root)

  if epoch < 0:
    latest_epoch = -1
    chk_folders = os.path.join(chk_root, 'chk_*')
    for folder in glob.glob(chk_folders):
      folder_name = os.path.basename(folder)
      tokens = folder_name.split('_')
      cur_epoch = int(tokens[-1])
      if cur_epoch > latest_epoch:
        latest_epoch = cur_epoch

    epoch = latest_epoch

  return os.path.join(chk_root, 'chk_{}'.format(epoch))


def load_checkpoint(epoch_idx, net, opt, save_dir):
    """ load network parameters from directory

    :param epoch_idx: the epoch idx of model to load
    :param net: the network object
    :param opt: the optimizer object
    :param save_dir: the save directory
    :return: loaded epoch index, loaded batch index
    """
    # load network parameters
    chk_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'params.pth')
    assert os.path.isfile(chk_file), 'checkpoint file not found: {}'.format(chk_file)

    state = torch.load(chk_file, map_location=torch.device('cpu'))
    net.load_state_dict(state['state_dict'])

    # load optimizer state
    opt_file = os.path.join(save_dir, 'checkpoints', 'chk_{}'.format(epoch_idx), 'optimizer.pth')
    assert os.path.isfile(opt_file), 'optimizer file not found: {}'.format(chk_file)

    opt_state = torch.load(opt_file, map_location=torch.device('cpu'))
    opt.load_state_dict(opt_state)

    return state['epoch'], state['batch']


def save_checkpoint(net, opt, epoch_idx, batch_idx, cfg, max_stride, num_modality):
    """ save model and parameters into a checkpoint file (.pth)

    :param net: the network object
    :param opt: the optimizer object
    :param epoch_idx: the epoch index
    :param batch_idx: the batch index
    :param cfg: the configuration object
    :param config_file: the configuration file path
    :param max_stride: the maximum stride of network
    :param num_modality: the number of image modalities
    :return: None
    """
    model_folder = os.path.join(cfg.general.save_dir, cfg.general.model_scale)
    chk_folder = os.path.join(model_folder, 'checkpoints', 'chk_{}'.format(epoch_idx))
    if not os.path.isdir(chk_folder):
        os.makedirs(chk_folder)

    state = {'epoch':             epoch_idx,
             'batch':             batch_idx,
             'net':               cfg.net.name,
             'max_stride':        max_stride,
             'state_dict':        net.state_dict(),
             'spacing':           cfg.dataset.spacing,
             'interpolation':     cfg.dataset.interpolation,
             'in_channels':       num_modality,
             'out_channels':      cfg.dataset.num_classes,
             'crop_normalizers':  [normalizer.to_dict() for normalizer in cfg.dataset.crop_normalizers]}

    # save python check point
    parm_filename = os.path.join(chk_folder, 'params.pth')
    optm_filename = os.path.join(chk_folder, 'optimizer.pth')
    torch.save(state, parm_filename)
    torch.save(opt.state_dict(), optm_filename)

    # save training configuration files
    shutil.copy(os.path.join(model_folder, 'train_config.py'), os.path.join(chk_folder, 'train_config.py'))