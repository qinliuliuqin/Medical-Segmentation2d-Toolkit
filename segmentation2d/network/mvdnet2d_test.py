import torch
import unittest

from segmentation2d.network.mvdnet2d import RegressionNet


class test_mvdnet2d(unittest.TestCase):

  def setUp(self):
    self.kMega = 1e6

  def test_mvdnet2d_model_parameters(self):
    batch_size, num_classes, in_channels = 1, 10, 1
    model = RegressionNet(in_channels, num_classes)
    if torch.cuda.is_available():
      model = model.cuda()
    model_params = (sum(p.numel() for p in model.parameters()) / self.kMega)

    self.assertLess(abs(model_params - 8.124668), 1e-6)


  def test_mvdnet2d_output_channels(self):
    batch_size, num_classes, in_channels = 1, 10, 1
    (dim_x, dim_y) = (512, 512)
    model = RegressionNet(in_channels, num_classes, True)
    model = model.cuda()

    in_images = torch.zeros([batch_size, in_channels, dim_y, dim_x])
    if torch.cuda.is_available():
      in_images = in_images.cuda()

    gender = torch.ones([batch_size, 1])
    if torch.cuda.is_available():
      gender = gender.cuda()
    reg, cls, prob = model(in_images, gender)

    self.assertEqual(cls.shape[0], batch_size)


if __name__ == '__main__':
  unittest.main()