import os, sys, pdb
import glob, random
import numpy as np
from skimage import io
import itertools
from collections import Counter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Adding local Keras
KERAS_PATH = '/media/manish/Data/keras/keras'
sys.path.insert(0, KERAS_PATH)
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras'))
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras', 'layers'))
from keras.preprocessing.image import ImageDataGenerator


def norm_mean_img(img):
    assert img.shape[3] == 3
    rgb_mean = np.mean(img, axis=(1, 2))
    rgb_mean = np.expand_dims(rgb_mean, axis=1)
    rgb_mean = np.expand_dims(rgb_mean, axis=1)

    img = (img - rgb_mean) / 255.0
    return img


"""
data_weigthed_loader consider the label for specific treatmentss
"""
def data_weighted_loader(path, batch_size, ignore_val=44, pos_val=255, neg_val=155, pos_class=3, neg_class=4):
    # pos_class and neg_class in the folder name for keras ImageDataGenerator input
    # 0,1,2 is for previous data. 3 and 4 is for new tcga data, which needs specific treatment

    def imerge(img_gen, mask_gen):
        for (img, img_labels), (mask, mask_labels) in itertools.izip(img_gen, mask_gen):
            # weight
            mask = np.expand_dims(mask[:,:,:,0], axis=3)

            weight = np.ones(mask.shape, np.float)
            weight[mask==ignore_val] = 0
            # In mask, ignored pixel has value ignore_val.
            # The weight of these pixel is set to zero, so they do not contribute to loss
            # The returned mask is still binary.

            # compute per sample
            for c, mask_label in enumerate(mask_labels):
                assert(mask_labels[c] == img_labels[c])
                mask_sample = mask[c]
                if mask_label == pos_class:
                    assert(np.where(mask_sample == neg_val)[0].size == 0)
                    mask_sample[mask_sample==pos_val] = 1
                elif mask_label == neg_class:
                    assert(np.where(mask_sample == pos_val)[0].size == 0)
                    mask_sample[mask_sample==neg_val] = 0
                else:
                	mask_sample /= 255

                mask_sample[mask_sample==ignore_val] = 0

            assert set(np.unique(mask)).issubset([0, 1])
            assert set(np.unique(weight)).issubset([0, 1])

            yield img/255.0, mask, weight, img_labels
            # yield norm_mean_img(img), mask, weight, img_labels

    train_data_gen_args = dict()

    seed = 1234
    train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'train/img',
                                class_mode="sparse",
                                target_size=(512, 512),
                                batch_size=batch_size,
                                seed=seed)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                path+'train/groundtruth',
                                class_mode="sparse",
                                target_size=(512, 512),
                                batch_size=batch_size,
                                seed=seed)

    test_image_datagen = ImageDataGenerator().flow_from_directory(
                                path+'val/img',
                                class_mode="sparse",
                                target_size=(512, 512),
                                batch_size=batch_size,
                                seed=seed)
    test_mask_datagen = ImageDataGenerator().flow_from_directory(
                                path+'val/groundtruth',
                                class_mode="sparse",
                                target_size=(512, 512),
                                batch_size=batch_size,
                                seed=seed)

    train_generator = imerge(train_image_datagen, train_mask_datagen)
    test_generator = imerge(test_image_datagen, test_mask_datagen)

    return train_generator, test_generator
