import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset

from segmentation2d.utils.image_tools import select_random_voxels_in_multi_class_mask, crop_image, \
    convert_image_to_tensor, get_image_frame, read_picture, resample_spacing


def read_image_list(image_list_file, mode):
  """
  Reads the training image list file and returns a list of image file names.
  """
  images_df = pd.read_csv(image_list_file)
  image_name_list = images_df['image_name'].tolist()
  image_path_list = images_df['image_path'].tolist()
  mask_path_list = None

  if mode == 'train' or 'val':
    mask_path_list = images_df['mask_path'].tolist()

  return image_name_list, image_path_list, mask_path_list


class SegmentationDataset(Dataset):
    """ training data set for volumetric segmentation """

    def __init__(self, mode, imlist_file, labels, spacing, crop_size, sampling_method,
                 random_translation, random_scale, interpolation, crop_normalizers):
        """ constructor
        :param imlist_file: image-segmentation list file
        :param spacing: the resolution, e.g., [1, 1, 1]
        :param crop_size: crop size, e.g., [96, 96, 96]
        :param sampling_method: 'GLOBAL', 'MASK'
        :param random_translation: random translation
        :param interpolation: 'LINEAR' for linear interpolation, 'NN' for nearest neighbor
        :param crop_normalizers: used to normalize the image crops, one for one image modality
        """
        if imlist_file.endswith('csv'):
            self.im_name_list, self.im_path_list, self.seg_list = read_image_list(imlist_file, mode)
        else:
            raise ValueError('imseg_list must be a csv file')

        self.mode = mode

        self.labels = labels

        self.num_classes = len(self.labels.keys()) + 1

        self.spacing = np.array(spacing, dtype=np.double)
        assert self.spacing.size == 2, 'only 2-element of spacing is supported'

        self.crop_size = np.array(crop_size, dtype=np.int32)
        assert self.crop_size.size == 2, 'only 2-element of crop size is supported'

        self.sampling_method = sampling_method
        assert self.sampling_method in ('CENTER', 'GLOBAL', 'MASK', 'HYBRID'), \
            'sampling_method must be CENTER, GLOBAL, MASK or HYBRID'

        self.random_translation = np.array(random_translation, dtype=np.double)
        assert self.random_translation.size == 2, 'Only 2-element of random translation is supported'

        self.random_scale = np.array(random_scale, dtype=np.double)
        assert self.random_scale.size == 2, 'Only 2-element of random scale is supported'

        self.interpolation = interpolation
        assert self.interpolation in ('LINEAR', 'NN'), 'interpolation must either be a LINEAR or NN'

        self.crop_normalizers = crop_normalizers
        assert isinstance(self.crop_normalizers, list), 'crop normalizers must be a list'

    def __len__(self):
        """ get the number of images in this data set """
        return len(self.im_path_list)

    def num_modality(self):
        """ get the number of input image modalities """
        return 1

    def global_sample(self, image):
        """ random sample a position in the image
        :param image: a SimpleITK image object which should be in the RAI coordinate
        :return: a world position in the RAI coordinate
        """
        assert isinstance(image, sitk.Image)

        origin = image.GetOrigin()
        im_size_mm = [image.GetSize()[idx] * image.GetSpacing()[idx] for idx in range(2)]
        crop_size_mm = self.crop_size * self.spacing

        sp = np.array(origin, dtype=np.double)
        for i in range(2):
            if im_size_mm[i] > crop_size_mm[i]:
                sp[i] = origin[i] + np.random.uniform(0, im_size_mm[i] - crop_size_mm[i])
        center = sp + crop_size_mm / 2
        return center

    def center_sample(self, image):
        """ return the world coordinate of the image center
        :param image: a image object
        :return: the image center in world coordinate
        """
        assert isinstance(image, sitk.Image)

        origin = image.GetOrigin()
        end_point_voxel = [int(image.GetSize()[idx] - 1) for idx in range(2)]
        end_point_world = image.TransformIndexToPhysicalPoint(end_point_voxel)

        center = np.array([(origin[idx] + end_point_world[idx]) / 2.0 for idx in range(2)], dtype=np.double)
        return center

    def __getitem__(self, index):
        """ get a training sample - image(s) and segmentation pair
        :param index:  the sample index
        :return cropped image, cropped mask, crop frame, case name
        """
        image_name, image_path, seg_path = \
            self.im_name_list[index], self.im_path_list[index], self.seg_list[index]

        # image IO
        images = []
        if image_path.endswith('PNG'):
            image = read_picture(image_path, np.float32)
        else:
            image = sitk.ReadImage(image_path, sitk.sitkFloat32)
        images.append(image)

        if seg_path.endswith('PNG'):
            seg = read_picture(seg_path, np.float32)
        else:
            seg = sitk.ReadImage(seg_path, sitk.sitkFloat32)

        # select labels from the seg
        seg_npy = sitk.GetArrayFromImage(seg)
        reordered_seg_npy = np.zeros_like(seg_npy)
        for idx, key in enumerate(list(self.labels.keys())):
            label = self.labels[key]
            reordered_seg_npy[abs(seg_npy - label) < 1e-1] = idx + 1

        reordered_seg = sitk.GetImageFromArray(reordered_seg_npy)
        reordered_seg.CopyInformation(seg)
        seg = reordered_seg

        if self.mode == 'train':
            # sampling a crop center
            if self.sampling_method == 'CENTER':
                center = self.center_sample(seg)

            elif self.sampling_method == 'GLOBAL':
                center = self.global_sample(seg)

            elif self.sampling_method == 'MASK':
                centers = select_random_voxels_in_multi_class_mask(seg, 1, np.random.randint(1, self.num_classes))
                if len(centers) > 0:
                    center = seg.TransformIndexToPhysicalPoint([int(centers[0][idx]) for idx in range(2)])
                else:  # if no segmentation
                    center = self.global_sample(seg)

            elif self.sampling_method == 'HYBRID':
                if index % 2:
                    center = self.global_sample(seg)
                else:
                    centers = select_random_voxels_in_multi_class_mask(seg, 1, np.random.randint(1, self.num_classes))
                    if len(centers) > 0:
                        center = seg.TransformIndexToPhysicalPoint([int(centers[0][idx]) for idx in range(2)])
                    else:  # if no segmentation
                        center = self.global_sample(seg)

            else:
                raise ValueError('Only CENTER, GLOBAL, MASK and HYBRID are supported as sampling methods')

            # random translation
            center += np.random.uniform(-self.random_translation, self.random_translation, size=[2])

            # random resampling
            crop_spacing = self.spacing * np.random.uniform(self.random_scale[0], self.random_scale[1])

            # sample a crop from image and normalize it
            for idx in range(len(images)):
                images[idx] = crop_image(images[idx], center, self.crop_size, crop_spacing, self.interpolation, 2)

                if self.crop_normalizers[idx] is not None:
                    images[idx] = self.crop_normalizers[idx](images[idx])

            seg = crop_image(seg, center, self.crop_size, crop_spacing, 'NN', 2)

        elif self.mode == 'val':
            for idx in range(len(images)):
                images[idx] = resample_spacing(images[idx], self.spacing, 16, self.interpolation)
            seg = resample_spacing(seg, self.spacing, 16, 'NN')

        else:
            raise ValueError('Unsupported mode type.')


        # image frame
        frame = get_image_frame(seg)

        # convert to tensors
        im = convert_image_to_tensor(images)
        seg = convert_image_to_tensor(seg)

        return im, seg, frame, image_name