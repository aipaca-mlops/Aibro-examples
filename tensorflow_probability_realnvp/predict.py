import numpy as np
import matplotlib.pyplot as plt
from utils import *


batch_size = 64


def load_model():
    # Portuguese to English translator
    rf_model = tf.saved_model.load('model')
    return rf_model
    

def run(realnvp_model):
    test_ds = load_dataset('test')
    test_ds = test_ds.batch(batch_size)

    result = realnvp_model.evaluate(test_ds)
    samples = realnvp_model.sample(8).numpy()
    n_img = 8
    f, axs = plt.subplots(2, n_img // 2, figsize=(14, 7))

    for k, image in enumerate(samples):
        i = k % 2
        j = k // 2
        axs[i, j].imshow(np.clip(image, 0., 1.))
        axs[i, j].axis('off')
    f.subplots_adjust(wspace=0.01, hspace=0.03)

    return result