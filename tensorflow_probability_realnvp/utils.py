import tensorflow as tf
import numpy as np


def load_image(img):
    img = tf.image.random_flip_left_right(img)
    return img, img

def load_dataset(split):
    train_list_ds = tf.data.Dataset.from_tensor_slices(np.load('./data/{}.npy'.format(split)))
    train_ds = train_list_ds.map(load_image)
    return train_ds