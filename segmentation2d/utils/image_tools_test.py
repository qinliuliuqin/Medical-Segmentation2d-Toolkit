import SimpleITK as sitk
from segmentation2d.utils.image_tools import crop_image, resample_spacing, \
  pick_largest_connected_component, get_bounding_box


def test_copy_image():
  seg_path = '/home/qinliu/debug/seg.mha'
  seg = sitk.ReadImage(seg_path)

  assert isinstance(seg, sitk.Image)

  seg_empty = sitk.Image(seg.GetSize(), seg.GetPixelID())
  seg_empty.CopyInformation(seg)

  # crop from seg
  cropping_center_voxel = [int(seg.GetSize()[idx] // 2) for idx in range(3)]
  cropping_center_world = seg.TransformIndexToPhysicalPoint(cropping_center_voxel)
  cropping_size = [128, 128, 128]
  cropping_spacing = [1.0, 1.0, 1.0]
  interp_method = 'NN'
  seg_cropped = crop_image(seg, cropping_center_world, cropping_size, cropping_spacing, interp_method)

  seg_cropped_path = '/home/qinliu/debug/seg_cropped.mha'
  sitk.WriteImage(seg_cropped, seg_cropped_path)

  # copy_image(seg_cropped, cropping_center_world, cropping_size, seg_empty)
  # seg_copy_path = '/home/qinliu/debug/seg_empty_copy.mha'
  # sitk.WriteImage(seg_empty, seg_copy_path)

  seg_origin = seg.GetOrigin()
  seg_empty_origin = list(map(int, seg_empty.GetOrigin()))
  seg_cropped_size = list(map(int, seg_cropped.GetSize()))
  seg_cropped_origin = list(map(int, seg_cropped.GetSize()))
  seg_pasted = sitk.Paste(seg_empty, seg_cropped, seg_cropped_size, [0, 0, 0], [100, 100, 100])
  seg_paste_path = '/home/qinliu/debug/seg_empty_paste.mha'
  sitk.WriteImage(seg_pasted, seg_paste_path)


def test_resample_spacing():
  seg_path = '/home/qinliu/debug/org.mha'
  seg = sitk.ReadImage(seg_path)

  resampled_seg = resample_spacing(seg, [0.5, 0.5, 0.5], 'LINEAR')
  resampled_seg_path = '/home/qinliu/debug/resampled_seg.mha'
  sitk.WriteImage(resampled_seg, resampled_seg_path)


def test_crop_image():
  image_path = '/mnt/projects/image.nii.gz'
  seg_path = '/mnt/projects/mask.nii.gz'

  image = sitk.ReadImage(image_path)
  seg = sitk.ReadImage(seg_path)

  cropping_center = [96, 96]
  cropping_size = [32, 32]
  cropping_spacing = [3.0, 3.0]
  cropped_image = crop_image(image, cropping_center, cropping_size, cropping_spacing, 'LINEAR', 2)
  cropped_seg = crop_image(seg, cropping_center, cropping_size, cropping_spacing, 'NN', 2)

  cropped_image_path = '/mnt/projects/image_cropped.nii.gz'
  cropped_seg_path = '/mnt/projects/mask_cropped.nii.gz'
  sitk.WriteImage(cropped_image, cropped_image_path, True)
  sitk.WriteImage(cropped_seg, cropped_seg_path, True)


def test_pick_largest_connected_component():

    seg =sitk.ReadImage('/mnt/projects/mask.nii.gz')
    labels = [255]

    seg_cc_path = '/mnt/projects/mask_cc.nii.gz'
    seg_cc = pick_largest_connected_component(seg, labels)
    sitk.WriteImage(seg_cc, seg_cc_path, True)


def test_get_bounding_box():

  seg_path = '/mnt/projects/mask_cc.nii.gz'
  seg = sitk.ReadImage(seg_path)

  bbox = get_bounding_box(seg, None)
  print(bbox)


if __name__ == '__main__':

  # test_copy_image()

  # test_resample_spacing()

  # test_crop_image()

  # test_pick_largest_connected_component()

  test_get_bounding_box()