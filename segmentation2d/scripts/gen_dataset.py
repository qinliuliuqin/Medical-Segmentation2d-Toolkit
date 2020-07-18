import numpy as np
import os
import pandas as pd
import random
import SimpleITK as sitk

from segmentation2d.utils.image_tools import read_picture


def split_dataset(image_list, image_folder, mask_folder, output_folder):
    """
    Generate dataset
    """
    seed = 0
    random.Random(seed).shuffle(image_list)

    num_training_images = int(len(image_list) * 4 // 5)
    training_images = image_list[:num_training_images]
    test_images = image_list[num_training_images:]

    # generate dataset for the training set
    content = []
    training_images.sort()
    print('Generating training set ...')
    for name in training_images:
        print(name)
        image_path = os.path.join(image_folder, name)
        mask_path = os.path.join(mask_folder, name)
        content.append([name, image_path, mask_path])

    csv_file_path = os.path.join(output_folder, 'train.csv')
    columns = ['image_name', 'image_path', 'mask_path']
    df = pd.DataFrame(data=content, columns=columns)
    df.to_csv(csv_file_path, index=False)

    # generate dataset for the test set
    content = []
    test_images.sort()
    print('Generating training set ...')
    for name in test_images:
        print(name)
        image_path = os.path.join(image_folder, name)
        mask_path = os.path.join(mask_folder, name)
        content.append([name, image_path, mask_path])

    csv_file_path = os.path.join(output_folder, 'test.csv')
    columns = ['image_name', 'image_path', 'mask_path']
    df = pd.DataFrame(data=content, columns=columns)
    df.to_csv(csv_file_path, index=False)


def dataset_statistics(image_list, image_folder, mask_folder):
    """
    Generate dataset
    """
    for name in image_list:
        print(name)
        image_path = os.path.join(image_folder, name)
        if image_path.endswith('PNG'):
            image = read_picture(image_path, np.float32)
        else:
            image = sitk.ReadImage(image_path, sitk.sitkFloat32)

        mask_path = os.path.join(mask_folder, name)
        if mask_path.endswith('PNG'):
            mask = read_picture(mask_path, np.float32)
        else:
            mask = sitk.ReadImage(mask_path, sitk.sitkFloat32)

        image_npy = sitk.GetArrayFromImage(image)
        mask_npy = sitk.GetArrayFromImage(mask)
        print(image_npy.shape, mask_npy.shape)


def get_image_list(image_folder):
    """
    Get image list from the image folder
    """
    image_list = []

    images = os.listdir(image_folder)
    for image in images:
        if image.endswith('PNG'):
            image_list.append(image)

    return image_list


if __name__ == '__main__':

    image_list = get_image_list('/mnt/projects/PIC_TNSCUI2020/TNSCUI2020_train/image')
    # split_dataset(image_list,
    #               '/shenlab/lab_stor6/projects/PIC_TNSCUI2020/TNSCUI2020_train/image',
    #               '/shenlab/lab_stor6/projects/PIC_TNSCUI2020/TNSCUI2020_train/mask',
    #               '/mnt/projects/PIC_TNSCUI2020/TNSCUI2020_train')

    dataset_statistics(image_list,
                  '/mnt/projects/PIC_TNSCUI2020/TNSCUI2020_train/image',
                  '/mnt/projects/PIC_TNSCUI2020/TNSCUI2020_train/mask')