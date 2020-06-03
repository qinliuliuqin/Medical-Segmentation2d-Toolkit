import os

from segmentation2d.core.seg_eval import cal_dsc_batch
from segmentation2d.dataset.dataset import read_image_list


def test_cal_dsc_batch():
    test_file = '/mnt/projects/PIC_TNSCUI2020/datasets/test.csv'
    gt_folder = '/mnt/projects/PIC_TNSCUI2020/TNSCUI2020_train/mask'
    seg_folder = '/mnt/projects/PIC_TNSCUI2020/results/model_0601_2020/test'
    result_file = '/mnt/projects/PIC_TNSCUI2020/results/model_0601_2020/test.csv'

    file_name_list, file_path_list, _ = read_image_list(test_file, 'test')
    gt_files = []
    for case_name in file_name_list:
        gt_files.append(os.path.join(gt_folder, case_name))

    seg_files = []
    for case_name in file_name_list:
        seg_files.append(os.path.join(seg_folder, case_name))

    labels = [255]
    cal_dsc_batch(gt_files, seg_files, labels, 10, result_file)


if __name__ == '__main__':
    test_cal_dsc_batch()
