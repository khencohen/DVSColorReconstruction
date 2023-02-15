import numpy as np
import os, os.path
from matplotlib import pyplot as plt

# paths for training_samples and training_labels
loc_samples = 'C:/Users/Public/Desktop/LONGDIC/training_samples'
loc_labels = 'C:/Users/Public/Desktop/LONGDIC/training_labels'

path, dirs, files = next(os.walk(loc_samples))
batch_size = len(files)  # the length of training_samples directory should match the length of training_labels directory


def extractPatch(arr_samples, arr_label, w, h):
    # This method creates one patch
    arr_width = len(arr_label[0])
    arr_height = len(arr_label)

    # We pick a point (x,y) which represents the upper left corner of the patch
    x = np.random.randint(0, arr_height - h)
    y = np.random.randint(0, arr_width - w)

    # (x,y) represent the upper left corner of the patch

    samples_patch = arr_samples[:, x:x + h, y:y + w]  # this is the samples patch
    labels_patch = arr_label[x:x + h, y:y + w, :]  # this is the labels patch

    return samples_patch, labels_patch, x, y


def extractPatches(arr_samples, arr_label, w, h, quantity=1):
    # This method is used to extract a certain amount of patches from a single video (i.e. with a single labels frame)
    # The default amount of patches is quantity = 1.
    samples_patches_list = []
    labels_patches_list = []

    for i in range(quantity):
        samples_patch_temp, labels_patch_temp, x, y = extractPatch(arr_samples, arr_label, w, h)
        samples_patches_list.append(samples_patch_temp)
        labels_patches_list.append(labels_patch_temp)

    return samples_patches_list, labels_patches_list


def get_batch(loc_samples, loc_labels, index, w, h, quantity):
    ''' This method handles the augmentation and preparation of numpy files at paths loc_samples, loc_labels
        It is very crucial to make sure all files have the following structure:
        training_dataset_K.npy
        Where K starts from 1 and ends at the length of the directory.
        In other words, if we have 5 training np arrays in the directory loc_samples, then
        each file should be called training_dataset_1, training_dataset_2, ...., training_dataset_5.
        The same idea goes for the labels files
    '''
    samples_batch = []
    labels_batch = []

    samples_file = np.load(loc_samples + '/training_dataset_' + str(index) + '.npy')
    labels_file = np.load(loc_labels + '/labels_dataset_' + str(index) + '.npy')

    # Rotation augmentation (by 90 deg and mirroring, apparently. I used transpose() function here)
    samples_file_rotated = samples_file.transpose((0, 2, 1))
    labels_file_rotated = labels_file.transpose((1, 0, 2))

    # Mirroring augmentation (I used np.flip(arr,(0,1)))
    samples_file_mirrored = np.flip(samples_file, (1, 2))
    labels_file_mirrored = np.flip(labels_file, (0, 1))

    # getting patch lists for augmented data and non-augmented data
    non_augmented_samples, non_augmented_labels = extractPatches(samples_file, labels_file, w, h, quantity)
    rotation_augmented_samples, rotation_augmented_labels = extractPatches(samples_file_rotated, labels_file_rotated, w,
                                                                           h, quantity)
    mirroring_augmented_samples, mirroring_augmented_labels = extractPatches(samples_file_mirrored,
                                                                             labels_file_mirrored, w, h, quantity)

    # we don't care if the data is augmented or not, so we put all patches at the same list
    for j in range(len(non_augmented_samples)):
        samples_batch.append(non_augmented_samples[j])
        labels_batch.append(non_augmented_labels[j])

    for j in range(len(rotation_augmented_samples)):
        samples_batch.append(rotation_augmented_samples[j])
        labels_batch.append(rotation_augmented_labels[j])

    for j in range(len(mirroring_augmented_samples)):
        samples_batch.append(mirroring_augmented_samples[j])
        labels_batch.append(mirroring_augmented_labels[j])

    # samples_batch, labels_batch should be numpy arrays in order to later convert them to pytorch tensors
    samples_batch = np.array(samples_batch)
    labels_batch = np.array(labels_batch)

    plt.imshow(labels_batch[0])
    plt.show()

    return samples_batch, labels_batch


# That's how one should call to the method get_batch, for 20 patches x 3 augmentation types (including non-augmented data)

# samples_batch, labels_batch = get_batch(loc_samples, loc_labels, 1, 128, 96, 20)




