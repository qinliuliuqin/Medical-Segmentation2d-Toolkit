import torch
import unittest

from segmentation2d.network.vdnet2d import SegmentationNet


class test_vdnet2d(unittest.TestCase):

  def setUp(self):
    self.kMega = 1e6

  def test_vdnet2d_model_parameters(self):
    in_channels, num_classes = 1, 10
    model = SegmentationNet(in_channels, num_classes)
    if torch.cuda.is_available():
      model = model.cuda()
    model_params = (sum(p.numel() for p in model.parameters()) / self.kMega)

    self.assertLess(abs(model_params - 4.99398), 1e-6)


  def test_vdnet2d_output_channels(self):
    batch_size, in_channels, num_classes = 1, 1, 10
    (dim_x, dim_y) = (512, 512)

    model = SegmentationNet(in_channels, num_classes)
    if torch.cuda.is_available():
      model = model.cuda()

    in_images = torch.zeros([batch_size, in_channels, dim_y, dim_x])
    if torch.cuda.is_available():
      in_images = in_images.cuda()
    outputs = model(in_images)

    self.assertEqual(outputs.size()[0], batch_size)
    self.assertEqual(outputs.size()[1], num_classes)


if __name__ == '__main__':

  unittest.main()