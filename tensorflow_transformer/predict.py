import json

import tensorflow as tf
import tensorflow_text


def load_model():
    # Portuguese to English translator
    translator = tf.saved_model.load('model')
    return translator

def run(model):
    fp = open("./data/data.json", "r")
    data = json.load(fp)
    sentence = data["data"]

    result = {"data": model(sentence).numpy().decode("utf-8")}
    return result

if __name__ == "__main__":
    run(load_model())
