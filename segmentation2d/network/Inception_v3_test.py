import torch
import unittest

from segmentation2d.network.Inception_v3 import RegressionNet


class test_inception3(unittest.TestCase):

  def setUp(self):
    self.kMega = 1e6

  def test_inception3_model_parameters(self):
    batch_size, num_classes, in_channels = 1, 10, 1
    model = RegressionNet(in_channels, num_classes)
    if torch.cuda.is_available():
      model = model.cuda()
    model_params = (sum(p.numel() for p in model.parameters()) / self.kMega)

    self.assertLess(abs(model_params - 24.894282), 1e-6)


  def test_inception3_output_channels(self):
    batch_size, num_classes, in_channels = 1, 10, 1
    (dim_x, dim_y) = (500, 500)

    model = RegressionNet(in_channels, num_classes)
    if torch.cuda.is_available():
      model = model.cuda()

    in_images = torch.zeros([batch_size, in_channels, dim_y, dim_x])
    if torch.cuda.is_available():
      in_images = in_images.cuda()

    in_gender = torch.FloatTensor([0] * batch_size)
    if torch.cuda.is_available():
      in_gender = in_gender.cuda()
    outputs = model(in_images, in_gender)

    self.assertEqual(outputs.size()[0], batch_size)
    self.assertEqual(outputs.size()[1], num_classes)


if __name__ == '__main__':
  unittest.main()