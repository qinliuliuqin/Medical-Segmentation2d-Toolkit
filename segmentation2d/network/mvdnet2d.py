import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation2d.network.module.weight_init import kaiming_weight_init


def parameters_init(net):
  net.apply(kaiming_weight_init)


class Dense(nn.Module):
  """ fully connected layer """

  def __init__(self, in_channels, out_channels):
    super(Dense, self).__init__()
    self.linear = nn.Linear(in_channels, out_channels)

  def forward(self, x):
    x = self.linear(x)
    return F.relu(x, inplace=True)


class InputBlock(nn.Module):
  """ input block of 2d vnet """

  def __init__(self, in_channels, out_channels):
    super(InputBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.act = nn.ReLU(inplace=True)

  def forward(self, input):
    out = self.act(self.bn(self.conv(input)))
    return out


class ConvBnRelu2(nn.Module):
  """ classic combination: conv + batch normalization [+ relu] """

  def __init__(self, in_channels, out_channels, ksize, padding, do_act=True):
    super(ConvBnRelu2, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=padding, groups=1)
    self.bn = nn.BatchNorm2d(out_channels)
    self.do_act = do_act
    if do_act:
      self.act = nn.ReLU(inplace=True)

  def forward(self, input):
    out = self.bn(self.conv(input))
    if self.do_act:
      out = self.act(out)
    return out


class BottConvBnRelu2(nn.Module):
  """Bottle neck structure"""

  def __init__(self, channels, ratio, do_act=True):
    super(BottConvBnRelu2, self).__init__()
    self.conv1 = ConvBnRelu2(channels, channels / ratio, ksize=1, padding=0, do_act=True)
    self.conv2 = ConvBnRelu2(channels / ratio, channels / ratio, ksize=3, padding=1, do_act=True)
    self.conv3 = ConvBnRelu2(channels / ratio, channels, ksize=1, padding=0, do_act=do_act)

  def forward(self, input):
    out = self.conv3(self.conv2(self.conv1(input)))
    return out


class ResidualBlock2(nn.Module):
  """ 2d residual block with variable number of convolutions """

  def __init__(self, channels, ksize, padding, num_convs):
    super(ResidualBlock2, self).__init__()

    layers = []
    for i in range(num_convs):
      if i != num_convs - 1:
        layers.append(ConvBnRelu2(channels, channels, ksize, padding, do_act=True))
      else:
        layers.append(ConvBnRelu2(channels, channels, ksize, padding, do_act=False))

    self.ops = nn.Sequential(*layers)
    self.act = nn.ReLU(inplace=True)

  def forward(self, input):

    output = self.ops(input)
    return self.act(input + output)


class BottResidualBlock2(nn.Module):
  """ block with bottle neck conv"""

  def __init__(self, channels, ratio, num_convs):
    super(BottResidualBlock2, self).__init__()
    layers = []
    for i in range(num_convs):
      if i != num_convs - 1:
        layers.append(BottConvBnRelu2(channels, ratio, True))
      else:
        layers.append(BottConvBnRelu2(channels, ratio, False))

    self.ops = nn.Sequential(*layers)
    self.act = nn.ReLU(inplace=True)

  def forward(self, input):
    output = self.ops(input)
    return self.act(input + output)


class DownBlock(nn.Module):
  """ downsample block of 2d v-net """

  def __init__(self, in_channels, num_convs, use_bottle_neck=False):
    super(DownBlock, self).__init__()
    out_channels = in_channels * 2
    self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, groups=1)
    self.down_bn = nn.BatchNorm2d(out_channels)
    self.down_act = nn.ReLU(inplace=True)
    if use_bottle_neck:
      self.rblock = BottResidualBlock2(out_channels, 4, num_convs)
    else:
      self.rblock = ResidualBlock2(out_channels, 3, 1, num_convs)

  def forward(self, input):
    out = self.down_act(self.down_bn(self.down_conv(input)))
    out = self.rblock(out)
    return out


class DimOrSizeReduceBlock(nn.Module):
  """ dimension reduction block of 2d v-net """

  def __init__(self, in_channels, out_channels, kernel_size, stride):
    super(DimOrSizeReduceBlock, self).__init__()
    self.dim_or_size_reduce_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=1)
    self.dim_or_size_reduce_bn = nn.BatchNorm2d(out_channels)
    self.dim_or_size_reduce_act = nn.ReLU(inplace=True)

  def forward(self, input):
    out = self.dim_or_size_reduce_act(self.dim_or_size_reduce_bn(self.dim_or_size_reduce_conv(input)))
    return out


class ConcatAndConvBlock(nn.Module):
  """ dimension reduction block of 2d v-net """

  def __init__(self, in_channels_after_concat, out_channels, kernel_size, stride):
    super(ConcatAndConvBlock, self).__init__()
    self.dim_or_size_reduce_conv = nn.Conv2d(in_channels_after_concat, out_channels, kernel_size=kernel_size, stride=stride, groups=1)
    self.dim_or_size_reduce_bn = nn.BatchNorm2d(out_channels)
    self.dim_or_size_reduce_act = nn.ReLU(inplace=True)

  def forward(self, input1, input2, input3):
    inputs = torch.cat((input1, input2, input3), 1)
    out = self.dim_or_size_reduce_act(self.dim_or_size_reduce_bn(self.dim_or_size_reduce_conv(inputs)))
    return out


class UpBlock(nn.Module):
  """ Upsample block of 2d v-net """

  def __init__(self, in_channels, out_channels, num_convs, use_bottle_neck=False):
    super(UpBlock, self).__init__()
    self.up_conv = nn.ConvTranspose2d(in_channels, out_channels // 2, kernel_size=2, stride=2, groups=1)
    self.up_bn = nn.BatchNorm2d(out_channels // 2)
    self.up_act = nn.ReLU(inplace=True)
    if use_bottle_neck:
      self.rblock = BottResidualBlock2(out_channels, 4, num_convs)
    else:
      self.rblock = ResidualBlock2(out_channels, 3, 1, num_convs)

  def forward(self, input, skip):
    out = self.up_act(self.up_bn(self.up_conv(input)))
    out = torch.cat((out, skip), 1)
    out = self.rblock(out)
    return out


class OutputBlock(nn.Module):
    """ output block of 2d v-net """

    def __init__(self, in_channels, num_classes):
      super(OutputBlock, self).__init__()
      self.num_classes = num_classes

      out_channels = num_classes
      self.class_conv1 = nn.Conv2d(
         in_channels, out_channels, kernel_size=3, padding=1)
      self.class_bn1 = nn.BatchNorm2d(out_channels)
      self.class_act1 = nn.ReLU(inplace=True)
      self.class_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
      self.softmax = nn.Softmax(1)

    def forward(self, input):
      out = self.class_act1(self.class_bn1(self.class_conv1(input)))
      out = self.class_conv2(out)
      out_size = out.size()
      # out_size = [batch_size, out_channels, dim_y, dim_x]

      # move channels to last dimension and softmax
      out = out.permute(0, 2, 3, 1).contiguous()
      out = out.view(out.numel() // out_size[1], out_size[1])
      out = self.softmax(out)

      # reshape back to output size
      out = out.view(out_size[0], out_size[2], out_size[3], out_size[1])
      out = out.permute(0, 3, 1, 2)

      out = out.contiguous()
      return out


class RegressionNet(nn.Module):
  """ multi-task vd-net for regression

  num_classes: include background
  """

  def __init__(self, num_in_channels, num_classes, multitask=False):
    super(RegressionNet, self).__init__()

    self.multitask = multitask

    # input processing block
    self.in_block   =   InputBlock(num_in_channels, 16)

    # down conv path
    self.down_32    =   DownBlock(16, 1, use_bottle_neck=False)
    self.down_64    =   DownBlock(32, 2, use_bottle_neck=False)
    self.down_128   =   DownBlock(64, 3, use_bottle_neck=False)
    self.down_256   =   DownBlock(128, 3, use_bottle_neck=False)

    # conv and fc for landmark classfication
    self.lm_cls_conv=   DimOrSizeReduceBlock(256, 4, 2, 2)
    self.lm_cls_fc =   nn.Linear(1024, num_classes - 1)

    # up conv path
    self.up_256     =   UpBlock(256, 256, 3, use_bottle_neck=False)
    self.up_128     =   UpBlock(256, 128, 3, use_bottle_neck=False)
    self.up_64      =   UpBlock(128, 64, 2, use_bottle_neck=False)
    self.up_32      =   UpBlock(64, 32, 1, use_bottle_neck=False)

    # down conv for bone age regression
    self.ba_reg_conv_512_1 =   DimOrSizeReduceBlock(32, 16, 4, 4)
    self.ba_reg_conv_512_2 =   DimOrSizeReduceBlock(16, 2, 4, 4)

    self.ba_reg_conv_128_1 = DimOrSizeReduceBlock(128, 32, 2, 2)
    self.ba_reg_conv_128_2 = DimOrSizeReduceBlock(32, 4, 2, 2)

    self.ba_reg_conv_32_1 = DimOrSizeReduceBlock(256, 32, 1, 1)
    self.ba_reg_conv_32_2 = DimOrSizeReduceBlock(32, 6, 1, 1)

    self.cat_and_conv = ConcatAndConvBlock(12, 2, 1, 1)

    self.fc_1 = Dense(1, 32)
    self.fc_2 = Dense(2080, 1000)
    self.fc_3 = Dense(1000, 1000)
    self.fc_4 = nn.Linear(1000, 1)

    # output processing block
    self.out_block  =   OutputBlock(32, num_classes)

  def forward(self, input, gender):
    out16  =  self.in_block(input)
    out32  =  self.down_32(out16)
    out64  =  self.down_64(out32)
    out128 = self.down_128(out64)
    out256 = self.down_256(out128)

    lm_cls = self.lm_cls_conv(out256)
    lm_cls = lm_cls.view(lm_cls.size(0), -1)
    lm_cls = self.lm_cls_fc(lm_cls)

    up_out256 = self.up_256(out256, out128)
    up_out128 = self.up_128(up_out256, out64)
    up_out64  =  self.up_64(up_out128, out32)
    up_out32  =  self.up_32(up_out64, out16)

    lm_prob_maps = self.out_block(up_out32)

    ba_reg_512 = self.ba_reg_conv_512_1(up_out32)
    ba_reg_512 = self.ba_reg_conv_512_2(ba_reg_512)

    ba_reg_128 = self.ba_reg_conv_128_1(up_out128)
    ba_reg_128 = self.ba_reg_conv_128_2(ba_reg_128)

    ba_reg_32  = self.ba_reg_conv_32_1(out256)
    ba_reg_32  = self.ba_reg_conv_32_2(ba_reg_32)

    ba_reg = self.cat_and_conv(ba_reg_32, ba_reg_128, ba_reg_512)

    ba_reg = ba_reg.view(ba_reg.size(0), -1)

    gender = gender.view(gender.size(0), -1)
    gender = self.fc_1(gender)

    ba_reg = torch.cat((ba_reg, gender), 1)

    ba_reg = self.fc_2(ba_reg)
    ba_reg = F.dropout(ba_reg, training=self.training)
    ba_reg = self.fc_3(ba_reg)
    ba_reg = F.dropout(ba_reg, training=self.training)
    ba_reg = self.fc_4(ba_reg)

    if self.multitask:
      return ba_reg, lm_cls, lm_prob_maps
    else:
      return ba_reg

  def max_stride(self):
    return 16