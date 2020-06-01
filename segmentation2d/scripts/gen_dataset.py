import os
import pandas as pd
import random


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
    split_dataset(image_list,
                  '/shenlab/lab_stor6/projects/PIC_TNSCUI2020/TNSCUI2020_train/image',
                  '/shenlab/lab_stor6/projects/PIC_TNSCUI2020/TNSCUI2020_train/mask',
                  '/mnt/projects/PIC_TNSCUI2020/TNSCUI2020_train')

