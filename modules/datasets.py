import cv2
import os
from os.path import join
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import Sequence

cv2.setNumThreads(0)

# IMG_EXTN = '.png'


def filename_without_ext(file_path):
    basename = os.path.basename(file_path)
    filename = os.path.splitext(basename)[0]
    return filename


def samples_from_dirs(data_dirs, IMG_EXTN, TRGT_EXTN):
    images_lst = []
    target_lst = []

    for data_dir in data_dirs:
        images_data_dir = join(data_dir, 'images')
        target_data_dir = join(data_dir, 'target')
        file_names = [filename_without_ext(f) for f
                      in os.listdir(images_data_dir)]
        file_names.sort()
        for name in file_names:
            image_name = join(images_data_dir, name + IMG_EXTN)
            target_name = join(target_data_dir, name + TRGT_EXTN)
            images_lst.append(image_name)
            target_lst.append(target_name)

    return images_lst, target_lst


class ImageTargetDataset(Sequence):
    def __init__(self, data_dirs,
                 batch_size,
                 shuffle=True,
                 device='GPU:0',
                 transform=None,
                 image_transform=None,
                 target_transform=None,
                 IMG_EXTN=None,
                 TRGT_EXTN=None):
        super().__init__()

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        self.transform = transform
        self.target_transform = target_transform
        self.image_transform = image_transform
        self.images_lst, self.target_lst = \
            samples_from_dirs(data_dirs, IMG_EXTN, TRGT_EXTN)
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_lst))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images_lst) / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        list_imgs_temp = [self.images_lst[k] for k in indexes]
        list_targets_temp = [self.target_lst[k] for k in indexes]
        image, target = self.__data_generation(list_imgs_temp, list_targets_temp)

        return image, target

    def __data_generation(self, list_imgs_temp, list_targets_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        with tf.device(self.device):
            input_list = []
            target_list = []
            # Generate data
            for i, j in zip(list_imgs_temp, list_targets_temp):
                # Store sample
                #print(i)
                #print(type(i))
                image = cv2.imread(i)
                image = image[:,:,::-1]
                target = cv2.imread(j, cv2.IMREAD_GRAYSCALE)
                if self.transform is not None:
                    image, target = self.transform(image, target)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                if self.image_transform is not None:
                    image = self.image_transform(image)
                input_list.append(image)
                target_list.append(target)

        return tf.stack(input_list), tf.stack(target_list)


class RandomConcatDataset(Sequence):
    def __init__(self, datasets, probs, size=10000):
        super().__init__()
        self.datasets = list(datasets)
        self.probs = probs
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        np.random.seed(seed=int(time.time() + idx))
        dataset_idx = np.random.choice(range(len(self.datasets)), p=self.probs)
        dataset = self.datasets[dataset_idx]
        idx = np.random.randint(len(dataset))
        return dataset[idx]
