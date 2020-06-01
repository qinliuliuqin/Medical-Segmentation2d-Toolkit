import SimpleITK as sitk
from segmentation2d.utils.metrics import cal_dsc


seg_path = '/mnt/projects/mask.nii.gz'
gt_path = '/mnt/projects/mask.nii.gz'

seg = sitk.ReadImage(seg_path)
gt = sitk.ReadImage(gt_path)

dsc, seg_type = cal_dsc(gt, seg, 0, 10)
print(dsc, seg_type)